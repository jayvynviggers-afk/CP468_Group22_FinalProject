import json
from pathlib import Path

import pandas as pd
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer

PREDICTIONS_PATH = Path("outputs/predictions/llm_predictions_test.csv")
METRICS_DIR = Path("outputs/metrics")
METRICS_DIR.mkdir(parents=True, exist_ok=True)

LLM_COLUMNS = [
    "zero_shot_v1",
    "zero_shot_v2",
    "few_shot_v1",
    "few_shot_v2",
]


def clean_text(text):
    if pd.isna(text):
        return ""
    return str(text).replace("\n", " ").strip()


def compute_rouge(predictions, references):
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=True
    )

    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []

    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        rouge1_scores.append(scores["rouge1"].fmeasure)
        rouge2_scores.append(scores["rouge2"].fmeasure)
        rougeL_scores.append(scores["rougeL"].fmeasure)

    return {
        "rouge1_f1": sum(rouge1_scores) / len(rouge1_scores),
        "rouge2_f1": sum(rouge2_scores) / len(rouge2_scores),
        "rougeL_f1": sum(rougeL_scores) / len(rougeL_scores),
    }


def compute_bleu(predictions, references):
    smoothie = SmoothingFunction().method4

    tokenized_predictions = [pred.split() for pred in predictions]
    tokenized_references = [[ref.split()] for ref in references]

    return corpus_bleu(
        tokenized_references,
        tokenized_predictions,
        smoothing_function=smoothie
    )


def main():
    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {PREDICTIONS_PATH}. Run run_llm_baseline.py first."
        )

    df = pd.read_csv(PREDICTIONS_PATH)

    references = [clean_text(x) for x in df["reference_headline"].tolist()]

    all_metrics = {}

    for col in LLM_COLUMNS:
        if col not in df.columns:
            print(f"Skipping missing column: {col}")
            continue

        predictions = [clean_text(x) for x in df[col].tolist()]

        # Count failed/empty outputs, but still evaluate all rows.
        empty_outputs = sum(1 for x in predictions if x == "")
        error_outputs = sum(1 for x in predictions if x.startswith("ERROR:"))

        rouge = compute_rouge(predictions, references)
        bleu = compute_bleu(predictions, references)

        all_metrics[col] = {
            "num_examples": len(predictions),
            "empty_outputs": empty_outputs,
            "error_outputs": error_outputs,
            "bleu": bleu,
            **rouge,
        }

    output_path = METRICS_DIR / "llm_metrics_test.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)

    summary_rows = []

    for setting, metrics in all_metrics.items():
        summary_rows.append({
            "model_setting": setting,
            "bleu": metrics["bleu"],
            "rouge1_f1": metrics["rouge1_f1"],
            "rouge2_f1": metrics["rouge2_f1"],
            "rougeL_f1": metrics["rougeL_f1"],
            "empty_outputs": metrics["empty_outputs"],
            "error_outputs": metrics["error_outputs"],
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_csv_path = METRICS_DIR / "llm_metrics_test.csv"
    summary_df.to_csv(summary_csv_path, index=False)

    print("\nLLM evaluation complete.")
    print(f"Saved JSON metrics to: {output_path}")
    print(f"Saved CSV metrics to: {summary_csv_path}")

    print("\nMetrics summary:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
