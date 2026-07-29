from datasets import load_dataset
from itertools import islice

DATASET_NAME = "faisaltareque/XL-HeadTags"

def main():
    print(f"Loading dataset in streaming mode: {DATASET_NAME}")

    # Streaming avoids downloading the full multi-GB dataset.
    dataset = load_dataset(DATASET_NAME, split="train", streaming=True)

    print("\nShowing first 3 English examples:")

    count = 0
    for example in dataset:
        if example.get("Language Code") == "eng":
            count += 1
            print("\n" + "=" * 80)
            print(f"Example {count}")
            print("Columns:", list(example.keys()))
            print("Headline:", str(example.get("Headline", ""))[:300])
            print("Article:", str(example.get("Article", ""))[:800])

            if count >= 3:
                break

    print("\nDataset inspection complete.")

if __name__ == "__main__":
    main()
