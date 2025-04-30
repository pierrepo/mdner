# run_scoring_analysis.py

import os
import json
import re
from typing import Any, Dict, List, Tuple, Union
from pathlib import Path
import pandas as pd
from loguru import logger

# === Configuration ===

# DATE_TIME_STR = "2025-04-30_15-50-52"
DATE_TIME_STR = input("Enter the date and time string to analyse (YYYY-MM-DD_HH-MM-SS): ")
ANNOTATIONS_FOLDER = "../annotations/"
TAGS = ["MOL", "SOFTNAME", "SOFTVERS", "STIME", "TEMP", "FFM"]

QC_RESULTS_PATH = f"../llm_outputs/{DATE_TIME_STR}/stats/quality_control_results.csv"
SCORE_RESULTS_FOLDER = f"../llm_outputs/{DATE_TIME_STR}/stats/"
SCORE_RESULTS_PATH = os.path.join(SCORE_RESULTS_FOLDER, "scoring_results.csv")

# === Extraction Functions ===

def extract_entities_from_annotation(text: str, entities: list) -> Dict[str, List[str]]:
    result = {key: [] for key in TAGS}
    for start, end, entity_type in entities:
        if entity_type == 'SOFT':
            entity_type = 'SOFTNAME'
        if entity_type in result:
            result[entity_type].append(text[start:end])
    return result


def extract_entities_from_llm_text(text: str) -> Dict[str, List[str]]:
    result = {tag: [] for tag in TAGS}
    pattern = re.compile(r"<([A-Z]+)>(.*?)</\1>")
    for tag, content in pattern.findall(text):
        if tag in result:
            result[tag].append(content.strip())
    return result


# === Scoring Metrics ===

def exact_match_score(gt: Dict[str, List[str]], pred: Dict[str, List[str]]) -> Tuple[int, int, float]:
    matched = sum(1 for k in gt for e in gt[k] if e in set(pred.get(k, [])))
    total = sum(len(v) for v in gt.values())
    return matched, total, matched / total if total > 0 else 0


def detection_ratio(gt: Dict[str, List[str]], pred: Dict[str, List[str]]) -> Dict[str, float]:
    return {
        k: sum(1 for e in gt[k] if e in set(pred.get(k, []))) / len(gt[k]) if gt[k] else None
        for k in TAGS
    }


def false_positives(gt: Dict[str, List[str]], pred: Dict[str, List[str]]) -> Dict[str, List[str]]:
    return {
        k: [e for e in pred.get(k, []) if e not in set(gt.get(k, []))] for k in TAGS
    }


def false_negatives(gt: Dict[str, List[str]], pred: Dict[str, List[str]]) -> Dict[str, List[str]]:
    return {
        k: [e for e in gt.get(k, []) if e not in set(pred.get(k, []))] for k in TAGS
    }


def per_type_breakdown(gt: Dict[str, List[str]], pred: Dict[str, List[str]]) -> Dict[str, Dict[str, Any]]:
    breakdown = {}
    for tag in TAGS:
        gt_set = set(gt.get(tag, []))
        pred_set = set(pred.get(tag, []))
        correct = len(gt_set & pred_set)
        total = len(gt_set)
        breakdown[tag] = {
            "exact_matches": correct,
            "total_gt": total,
            "detection_ratio": correct / total if total > 0 else None,
            "false_positives": len(pred_set - gt_set),
            "false_negatives": len(gt_set - pred_set)
        }
    return breakdown


# === File Utilities ===

def extract_annotations_to_score(csv_file: Union[str, Path]) -> Tuple[pd.DataFrame, List[str], List[str]]:
    df = pd.read_csv(csv_file)
    filtered = df[df["one_entity_verified"]]
    return filtered[["prompt", "model", "filename", "full_path"]], \
           filtered["filename"].tolist(), \
           filtered["full_path"].tolist()


def process_json_file(json_file: Union[str, Path]) -> Tuple[str, List]:
    with open(json_file, "r") as f:
        data = json.load(f)
    ann = data["annotations"][0]
    return ann[0], ann[1]["entities"]


def process_llm_json_file(json_file: Union[str, Path]) -> Tuple[str, str, str]:
    with open(json_file, "r") as f:
        data = json.load(f)
    return data["text_to_annotate"], data["response"], data["model"]


def save_scoring_results_to_csv(rows: List[Dict[str, Any]], output_dir: Union[str, Path]) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "scoring_results.csv"
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False, mode="a", header=not csv_path.exists())


# === Main Evaluation Logic ===

def score_annotations():
    logger.info("Starting scoring analysis...")

    if Path(SCORE_RESULTS_PATH).exists():
        logger.warning(f"Overwriting previous scoring file: {SCORE_RESULTS_PATH}")
        os.remove(SCORE_RESULTS_PATH)

    df, filenames, llm_paths = extract_annotations_to_score(QC_RESULTS_PATH)
    filenames = [os.path.join(ANNOTATIONS_FOLDER, name) for name in filenames]

    for i, (gt_file, llm_file) in enumerate(zip(filenames, llm_paths)):
        input_text, gt_entities = process_json_file(gt_file)
        gt_extracted = extract_entities_from_annotation(input_text, gt_entities)

        _, llm_response, _ = process_llm_json_file(llm_file)
        llm_extracted = extract_entities_from_llm_text(llm_response)

        matched, total, score_ratio = exact_match_score(gt_extracted, llm_extracted)
        fps = false_positives(gt_extracted, llm_extracted)
        fns = false_negatives(gt_extracted, llm_extracted)
        detect_ratio = detection_ratio(gt_extracted, llm_extracted)
        breakdown = per_type_breakdown(gt_extracted, llm_extracted)

        prompt_name, model, filename, file_path = df.iloc[i]

        row = {
            "prompt": prompt_name,
            "model": model,
            "filename": filename,
            "percentage_correct": round(score_ratio * 100, 2),
            "total_correct": matched,
            "total": total,
            "total_fp": sum(len(v) for v in fps.values()),
            "full path": str(file_path),
        }

        for tag in TAGS:
            row.update({
                f"{tag}_correct": breakdown[tag]["exact_matches"],
                f"{tag}_total": breakdown[tag]["total_gt"],
                f"{tag}_FP": "; ".join(fps[tag]),
                f"{tag}_FN": "; ".join(fns[tag]),
            })

        logger.info(f"[{i+1}/{len(df)}] Scored {filename:^25} | {prompt_name:^13} | {model:>20} — {row['percentage_correct']}% accuracy")
        save_scoring_results_to_csv([row], SCORE_RESULTS_FOLDER)

    logger.success(f"Scoring analysis complete. Results saved to {SCORE_RESULTS_PATH}")


# === Run script ===

if __name__ == "__main__":
    score_annotations()
