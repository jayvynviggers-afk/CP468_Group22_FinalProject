import argparse
import json
import time
from pathlib import Path

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

DATA_DIR = Path("data/processed")
PREDICTIONS_DIR = Path("outputs/predictions")
METRICS_DIR = Path("outputs/metrics")

PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)


def truncate_article(text, max_words=250):
    words = str(text).split()
    return " ".join(words[:max_words])


def build_zero_shot_prompt(article, variant):
    article = truncate_article(article)

    if variant == "v1":
        return f"""Generate a concise news headline for the following article.

Article:
{article}

Headline:"""

    if variant == "v2":
        return f"""You are a professional news editor. Write one clear, factual headline for the article below.
Rules:
- Output only the headline.
- Do not add extra explanation.
- Do not invent facts.
- Keep it under 15 words.

Article:
{article}

Headline:"""

    raise ValueError(f"Unknown prompt variant: {variant}")


def build_few_shot_prompt(article, few_shot_examples, variant):
    article = truncate_article(article)
    examples_text = ""

    for i, row in enumerate(few_shot_examples, start=1):
        example_article = truncate_article(row["source"])
        example_headline = row["target"]

        examples_text += f"""Example {i}
Article:
{example_article}
Headline:
{example_headline}

"""

    if variant == "v1":
        return f"""Generate a concise news headline for each article.

{examples_text}Now generate a headline for this article.

Article:
{article}

Headline:"""

    if variant == "v2":
        return f"""You are a professional news editor. Based on the examples, write one clear, factual headline.
Rules:
- Output only the headline.
- Do not add extra explanation.
- Do not invent facts.
- Keep it under 15 words.

{examples_text}Article:
{article}

Headline:"""

    raise ValueError(f"Unknown prompt variant: {variant}")


def call_llm(client, model, prompt):
    response = client.responses.create(
        model=model,
        input=prompt,
        max_output_tokens=120,
    )

    text = response.output_text.strip()
    text = text.replace("\n", " ").strip()

    usage = response.usage
    input_tokens = getattr(usage, "input_tokens", 0) if usage else 0
    output_tokens = getattr(usage, "output_tokens", 0) if usage else 0

    return text, input_tokens, output_tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt-4.1-nano")
    parser.add_argument("--max_examples", type=int, default=5)
    parser.add_argument("--few_shot_k", type=int, default=3)
    parser.add_argument("--sleep", type=float, default=0.2)
    args = parser.parse_args()

    client = OpenAI()

    test_df = pd.read_csv(DATA_DIR / "test.csv")
    train_df = pd.read_csv(DATA_DIR / "train.csv")

    test_df = test_df.head(args.max_examples)
    few_shot_examples = train_df.head(args.few_shot_k).to_dict("records")

    rows = []
    total_input_tokens = 0
    total_output_tokens = 0
    total_requests = 0

    start_time = time.time()

    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Running LLM baseline"):
        source = row["source"]
        reference = row["target"]

        result_row = {
            "source": source,
            "reference_headline": reference,
        }

        configs = [
            ("zero_shot_v1", build_zero_shot_prompt(source, "v1")),
            ("zero_shot_v2", build_zero_shot_prompt(source, "v2")),
            ("few_shot_v1", build_few_shot_prompt(source, few_shot_examples, "v1")),
            ("few_shot_v2", build_few_shot_prompt(source, few_shot_examples, "v2")),
        ]

        for config_name, prompt in configs:
            try:
                output, input_tokens, output_tokens = call_llm(client, args.model, prompt)
                result_row[config_name] = output
                total_input_tokens += input_tokens
                total_output_tokens += output_tokens
                total_requests += 1
                time.sleep(args.sleep)

            except Exception as e:
                result_row[config_name] = f"ERROR: {e}"

        rows.append(result_row)

    predictions_df = pd.DataFrame(rows)
    predictions_path = PREDICTIONS_DIR / "llm_predictions_test.csv"
    predictions_df.to_csv(predictions_path, index=False)

    summary = {
        "model": args.model,
        "num_examples": len(test_df),
        "few_shot_k": args.few_shot_k,
        "prompt_settings": [
            "zero_shot_v1",
            "zero_shot_v2",
            "few_shot_v1",
            "few_shot_v2"
        ],
        "total_requests": total_requests,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "runtime_seconds": end_time - start_time if False else None,
        "note": "Use token counts with current provider pricing to estimate API cost."
    }

    end_time = time.time()
    summary["runtime_seconds"] = end_time - start_time
    summary["runtime_minutes"] = (end_time - start_time) / 60

    summary_path = METRICS_DIR / "llm_baseline_summary.json"

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nLLM baseline complete.")
    print(f"Saved predictions to: {predictions_path}")
    print(f"Saved summary to: {summary_path}")
    print(f"Total requests: {total_requests}")
    print(f"Input tokens: {total_input_tokens}")
    print(f"Output tokens: {total_output_tokens}")
    print(f"Runtime minutes: {summary['runtime_minutes']:.2f}")


if __name__ == "__main__":
    main()
