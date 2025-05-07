# run_qc_analysis.py

import os
import json
import re
from typing import Any, Dict, List, Union
from pathlib import Path
import pandas as pd
from loguru import logger

# === Configuration ===

# DATE_TIME_STR = "2025-04-30_16-19-10"
DATE_TIME_STR = input("Enter the date and time string to analyse (YYYY-MM-DD_HH-MM-SS): ")
LLM_ANNOTATIONS = f"../llm_outputs/{DATE_TIME_STR}/annotations/"
QC_RESULTS_FOLDER = f"../llm_outputs/{DATE_TIME_STR}/stats/"
QC_RESULTS_PATH = os.path.join(QC_RESULTS_FOLDER, "quality_control_results.csv")

TAGS = ["MOL", "SOFTNAME", "SOFTVERS", "STIME", "TEMP", "FFM"]

# === Helper Functions ===

def strip_tags(text: str, tags: List[str] = TAGS) -> str:
    for tag in tags:
        text = re.sub(f"</?{re.escape(tag)}>", "", text)
    return text.strip()


def compare_annotated_to_original(original: str, annotated: str) -> bool:
    return strip_tags(annotated).strip().lower() == original.strip().lower()


def process_llm_json_file(json_file: Union[str, Path]) -> tuple:
    with open(json_file, "r") as f:
        data = json.load(f)
    return data["text_to_annotate"], data["response"], data["model"]


def extract_entities_from_llm_text(text: str) -> Dict[str, List[str]]:
    result = {tag: [] for tag in TAGS}
    pattern = re.compile(r"<([A-Z]+)>(.*?)</\1>")
    for tag, content in pattern.findall(text):
        if tag in result:
            result[tag].append(content.strip())
    return result


def find_one_valid_llm_entity(llm_entities: Dict[str, List[str]], input_text: str) -> bool:
    for values in llm_entities.values():
        for value in values:
            if value in input_text:
                return True
    return False


def define_quality_entities(llm_entities: Dict[str, List[str]], input_text: str) -> bool:
    """Looks at each entity, and checks if it belong to one of the three following groups:
    - fully_valid: the entity is present in the input text
    - partially_valid: the entity is present in the input text but not fully (part of the entity was hallucinated)
    - invalid: the entity is not present in the input text

    Args:
        llm_entities (Dict[str, List[str]]): entities extracted from the LLM response
        input_text (str): original text to compare with

    Returns:
        fully_valid: (int) count of all the valid entities
        partially_valid: (int) count of all the partially valid entities
        invalid: (int) count of all the invalid entities
    """
    fully_valid = 0
    partially_valid = 0
    invalid = 0

    text_lc = input_text.lower()

    fully_valid = partially_valid = invalid = 0

    for values in llm_entities.values():
        for entity in values:
            ent_lc = entity.lower().strip()

            # 1) FULL match
            if ent_lc and ent_lc in text_lc:
                fully_valid += 1
                continue

            # 2) PARTIAL match (whole‑word token overlap)
            tokens = re.findall(r"\w+", ent_lc)
            if tokens and any(
                re.search(rf"\b{re.escape(tok)}\b", text_lc) for tok in tokens
            ):
                partially_valid += 1
            else:
                invalid += 1

    return fully_valid, partially_valid, invalid


def save_qc_results_to_csv(rows: List[Dict[str, Any]], output_dir: Union[str, Path]) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "quality_control_results.csv"
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False, mode="a", header=not csv_path.exists())


# === Main Quality Control Logic ===

def quality_control(path_to_test: Union[str, Path]) -> None:
    path_to_test = Path(path_to_test)
    results_file = Path(QC_RESULTS_PATH)

    if results_file.exists():
        logger.info(f"Overwriting existing file: {results_file}")
        os.remove(results_file)

    rows: List[Dict[str, Any]] = []

    for prompt in os.listdir(path_to_test):
        prompt_folder = path_to_test / prompt

        for model in os.listdir(prompt_folder):
            model_path = prompt_folder / model

            if model.startswith("meta-llama"):
                subdirs = [dir.name for dir in model_path.iterdir() if dir.is_dir()]
                if len(subdirs) > 1:
                    logger.warning(f"Multiple submodels found in {model_path}, using the first.")
                only_model = subdirs[0]
                model_folder = model_path / only_model
                model = f"{model}/{only_model}"
            else:
                model_folder = model_path

            for filename in os.listdir(model_folder):
                file_path = model_folder / filename

                input_text, response, _ = process_llm_json_file(file_path)
                llm_entities = extract_entities_from_llm_text(response)

                exact_text_result = compare_annotated_to_original(input_text, response)
                entities_result = find_one_valid_llm_entity(llm_entities, input_text)

                fully_valid, partially_valid, invalid = define_quality_entities(llm_entities, input_text)

                rows.append({
                    "prompt": prompt,
                    "model": model,
                    "filename": filename,
                    "text_unchanged": exact_text_result,
                    "one_entity_verified": entities_result,
                    "fully_valid": fully_valid,
                    "partially_valid": partially_valid,
                    "invalid": invalid,
                    "total_entities": fully_valid + partially_valid + invalid,
                    "full_path": str(file_path),
                })

    save_qc_results_to_csv(rows, QC_RESULTS_FOLDER)
    logger.success(f"Quality control completed. Results saved to {QC_RESULTS_PATH}")


# === Run script ===

if __name__ == "__main__":
    quality_control(LLM_ANNOTATIONS)
