from datasets import load_dataset
from pathlib import Path
import pandas as pd
import random

SEED = 468
random.seed(SEED)

DATASET_NAME = "faisaltareque/XL-HeadTags"

N_TRAIN = 10000
N_VAL = 1000
N_TEST = 1000
TOTAL_NEEDED = N_TRAIN + N_VAL + N_TEST

OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def clean_text(text):
    return str(text).replace("\n", " ").replace("\r", " ").strip()

def main():
    print(f"Loading dataset in streaming mode: {DATASET_NAME}")
    dataset = load_dataset(DATASET_NAME, split="train", streaming=True)

    examples = []

    print("Collecting English headline-generation examples...")

    for example in dataset:
        language = example.get("Language Code", "")

        if language != "eng":
            continue

        article = clean_text(example.get("Article", ""))
        headline = clean_text(example.get("Headline", ""))

        if not article or not headline:
            continue

        source_len = len(article.split())
        target_len = len(headline.split())

        if source_len < 30 or source_len > 400:
            continue

        if target_len < 3 or target_len > 20:
            continue

        examples.append({
            "source": article,
            "target": headline
        })

        if len(examples) % 1000 == 0:
            print(f"Collected {len(examples)} examples...")

        if len(examples) >= TOTAL_NEEDED:
            break

    if len(examples) < TOTAL_NEEDED:
        raise ValueError(f"Only found {len(examples)} examples, but need {TOTAL_NEEDED}.")

    random.shuffle(examples)

    train = examples[:N_TRAIN]
    val = examples[N_TRAIN:N_TRAIN + N_VAL]
    test = examples[N_TRAIN + N_VAL:N_TRAIN + N_VAL + N_TEST]

    pd.DataFrame(train).to_csv(OUT_DIR / "train.csv", index=False)
    pd.DataFrame(val).to_csv(OUT_DIR / "val.csv", index=False)
    pd.DataFrame(test).to_csv(OUT_DIR / "test.csv", index=False)

    print("\nSaved processed splits:")
    print(f"Train: {len(train)} examples")
    print(f"Validation: {len(val)} examples")
    print(f"Test: {len(test)} examples")

    print("\nExample training row:")
    print("SOURCE:", train[0]["source"][:500])
    print("TARGET:", train[0]["target"])

if __name__ == "__main__":
    main()
