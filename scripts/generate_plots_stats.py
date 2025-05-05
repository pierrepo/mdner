# generate_plots_stats.py

from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from loguru import logger


# === Configuration ===

# DATE_TIME_STR = "2025-04-30_15-50-52"
DATE_TIME_STR = input("Enter the DATE_TIME_STR and time string to make plots (YYYY-MM-DD_HH-MM-SS): ")

BASE_DIR = Path("../llm_outputs")

MODEL_ORDER = [
    "o3-2025-04-16",
    "gpt-4.1-2025-04-14",
    "gpt-4o-2024-11-20",
]

PROMPT_ORDER = [
    "zero_shot",
    "one_shot",
    "few_shot",
]

TAGS = ["MOL", "SOFTNAME", "SOFTVERS", "STIME", "TEMP", "FFM"]

# ---------------------------------------------------------------------------
# Plot builders
# ---------------------------------------------------------------------------

def plot_one_entity_verified(qc_df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(12, 8))
    ax = sns.countplot(
        data=qc_df[qc_df["one_entity_verified"]],
        x="prompt",
        hue="model",
        palette="viridis",
        order=PROMPT_ORDER,
        hue_order=MODEL_ORDER,
    )
    for c in ax.containers:
        ax.bar_label(c, fmt="%.0f", padding=5)
    plt.title("LLM responses with ≥1 entity from the input text (100 texts)")
    plt.xlabel("Prompt")
    plt.ylabel("Responses with one entity verified")
    plt.legend(title="Model")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    logger.info(f"Saved entity‑verified count plot → {out_path}")


def plot_text_unchanged(qc_df: pd.DataFrame, out_path: Path) -> None:
    combos = (
        pd.MultiIndex.from_product(
            [qc_df["prompt"].unique(), qc_df["model"].unique()],
            names=["prompt", "model"],
        ).to_frame(index=False)
    )
    counts = (
        qc_df[qc_df["text_unchanged"]]
        .groupby(["prompt", "model"])
        .size()
        .reset_index(name="count")
    )
    df = pd.merge(combos, counts, on=["prompt", "model"], how="left").fillna(0)
    df["count"] = df["count"].astype(int)

    plt.figure(figsize=(12, 8))
    ax = sns.barplot(
        data=df,
        x="prompt",
        y="count",
        hue="model",
        palette="viridis",
        order=PROMPT_ORDER,
        hue_order=MODEL_ORDER,
    )
    for c in ax.containers:
        ax.bar_label(c, fmt="%.0f", padding=5)
    plt.title("LLM responses where output text equals input text (100 texts)")
    plt.xlabel("Prompt")
    plt.ylabel("Unchanged responses")
    plt.legend(title="Model")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    logger.info(f"Saved unchanged‑text plot → {out_path}")


def plot_precision(scoring_df: pd.DataFrame, out_path: Path) -> None:
    agg = (
        scoring_df.groupby(["model", "prompt"])[["total_correct", "total_fp"]]
        .sum()
        .reset_index()
    )
    agg["precision"] = agg["total_correct"] / (
        agg["total_correct"] + agg["total_fp"]
    )
    agg["model"] = pd.Categorical(agg["model"], MODEL_ORDER, ordered=True)
    agg["prompt"] = pd.Categorical(agg["prompt"], PROMPT_ORDER, ordered=True)

    plt.figure(figsize=(12, 6))
    ax = sns.barplot(
        data=agg,
        x="prompt",
        y="precision",
        hue="model",
        palette="viridis",
        order=PROMPT_ORDER,
        hue_order=MODEL_ORDER,
    )
    for c in ax.containers:
        ax.bar_label(c, fmt="%.2f", padding=5)
    plt.title("Average Precision by Model and Prompt")
    plt.xlabel("Prompt")
    plt.ylabel("Precision")
    plt.ylim(0, 1)
    plt.legend(title="Model")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    logger.info(f"Saved precision plot → {out_path}")


def plot_recall(scoring_df: pd.DataFrame, out_path: Path) -> None:
    agg = (
        scoring_df.groupby(["model", "prompt"])[["total_correct", "total"]]
        .sum()
        .reset_index()
    )
    agg["recall"] = agg["total_correct"] / agg["total"]
    agg["model"] = pd.Categorical(agg["model"], MODEL_ORDER, ordered=True)
    agg["prompt"] = pd.Categorical(agg["prompt"], PROMPT_ORDER, ordered=True)

    plt.figure(figsize=(12, 6))
    ax = sns.barplot(
        data=agg,
        x="prompt",
        y="recall",
        hue="model",
        palette="viridis",
        order=PROMPT_ORDER,
        hue_order=MODEL_ORDER,
    )
    for c in ax.containers:
        ax.bar_label(c, fmt="%.2f", padding=5)
    plt.title("Average Recall by Model and Prompt")
    plt.xlabel("Prompt")
    plt.ylabel("Recall")
    plt.ylim(0, 1)
    plt.legend(title="Model")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    logger.info(f"Saved recall plot → {out_path}")


def plot_entity_contingency(df: pd.DataFrame, entity: str, out_path: Path) -> None:
    """Save a contingency‑metrics bar plot for *entity*."""
    out_path = out_path / f"contingency_{entity}.png"

    correct_col = f"{entity}_correct"
    total_col = f"{entity}_total"
    fp_col = f"{entity}_FP"
    fn_col = f"{entity}_FN"

    # Parse FP / FN semicolon lists → counts
    df[f"{entity}_FP_count"] = (
        df[fp_col].fillna("").apply(lambda x: len([v for v in x.split(";") if v.strip()]))
    )
    df[f"{entity}_FN_count"] = (
        df[fn_col].fillna("").apply(lambda x: len([v for v in x.split(";") if v.strip()]))
    )
    df[f"{entity}_pred_total"] = df[correct_col] + df[f"{entity}_FP_count"]

    plot_df = pd.DataFrame(
        {
            "True Positives": df[correct_col],
            "False Positives": df[f"{entity}_FP_count"],
            "False Negatives": df[f"{entity}_FN_count"],
            "Ground Truth Total": df[total_col],
            "Predicted Total": df[f"{entity}_pred_total"],
        }
    )

    agg = plot_df.sum().reset_index()
    agg.columns = ["Metric", "Count"]

    plt.figure(figsize=(12, 6))
    ax = sns.barplot(data=agg, x="Metric", y="Count", hue="Metric", palette="viridis")
    for c in ax.containers:
        ax.bar_label(c, fmt="%.0f", padding=5)
    plt.title(f"Contingency Metrics for '{entity}' Entity")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    logger.info(f"Saved {entity} contingency plot → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    stats_dir = BASE_DIR / DATE_TIME_STR / "stats"
    images_dir = BASE_DIR / DATE_TIME_STR / "images"

    scoring_path = stats_dir / "scoring_results.csv"
    qc_path = stats_dir / "quality_control_results.csv"

    if not scoring_path.exists() or not qc_path.exists():
        logger.error(f"CSV files not found in {stats_dir}")
        raise SystemExit(1)

    logger.info("Reading CSV files …")
    scoring_df = pd.read_csv(scoring_path)
    qc_df = pd.read_csv(qc_path)

    plot_one_entity_verified(qc_df, images_dir / "one_entity_verified_count.png")
    plot_text_unchanged(qc_df, images_dir / "text_unchanged_count.png")
    plot_precision(scoring_df, images_dir / "precision.png")
    plot_recall(scoring_df, images_dir / "recall.png")

    for entity in TAGS:
        plot_entity_contingency(scoring_df, entity, images_dir)

    logger.success(f"All plots written to {images_dir.resolve()}")


if __name__ == "__main__":
    main()
