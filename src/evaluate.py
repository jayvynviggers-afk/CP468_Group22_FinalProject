import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer

from dataset import get_dataloaders
from model import Seq2SeqAttentionModel
from utils import set_seed, get_device


MODEL_PATH = Path("outputs/models/best_lstm_attention.pt")
PREDICTIONS_DIR = Path("outputs/predictions")
METRICS_DIR = Path("outputs/metrics")

PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)


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


def evaluate_model(model, dataloader, target_vocab, device, max_examples=None):
    model.eval()

    sources = []
    references = []
    predictions = []

    total_seen = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating predictions"):
            source_ids = batch["source_ids"].to(device)

            generated_ids = model.generate(source_ids, max_len=32)

            batch_predictions = [
                target_vocab.decode(ids.cpu())
                for ids in generated_ids
            ]

            batch_size = len(batch_predictions)

            sources.extend(batch["source_text"])
            references.extend(batch["target_text"])
            predictions.extend(batch_predictions)

            total_seen += batch_size

            if max_examples is not None and total_seen >= max_examples:
                sources = sources[:max_examples]
                references = references[:max_examples]
                predictions = predictions[:max_examples]
                break

    return sources, references, predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_examples", type=int, default=None)
    args = parser.parse_args()

    set_seed()
    device = get_device()

    print(f"Using device: {device}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Could not find trained model at {MODEL_PATH}. "
            f"Run python src/train.py first."
        )

    train_loader, val_loader, test_loader, source_vocab, target_vocab = get_dataloaders(
        batch_size=args.batch_size
    )

    if args.split == "train":
        dataloader = train_loader
    elif args.split == "val":
        dataloader = val_loader
    else:
        dataloader = test_loader

    checkpoint = torch.load(MODEL_PATH, map_location="cpu")

    model = Seq2SeqAttentionModel(
        source_vocab_size=checkpoint["source_vocab_size"],
        target_vocab_size=checkpoint["target_vocab_size"],
        embedding_dim=checkpoint["embedding_dim"],
        encoder_hidden_dim=checkpoint["hidden_dim"],
        dropout=checkpoint["dropout"],
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    print("Loaded trained LSTM attention model.")
    print(f"Evaluating on split: {args.split}")

    sources, references, predictions = evaluate_model(
        model=model,
        dataloader=dataloader,
        target_vocab=target_vocab,
        device=device,
        max_examples=args.max_examples
    )

    rouge = compute_rouge(predictions, references)
    bleu = compute_bleu(predictions, references)

    metrics = {
        "split": args.split,
        "num_examples": len(predictions),
        "bleu": bleu,
        **rouge
    }

    predictions_df = pd.DataFrame({
        "source": sources,
        "reference_headline": references,
        "lstm_output": predictions
    })

    pred_path = PREDICTIONS_DIR / f"lstm_predictions_{args.split}.csv"
    metrics_path = METRICS_DIR / f"lstm_metrics_{args.split}.json"

    predictions_df.to_csv(pred_path, index=False)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\nEvaluation complete.")
    print(f"Examples evaluated: {len(predictions)}")
    print(f"BLEU: {bleu:.4f}")
    print(f"ROUGE-1 F1: {rouge['rouge1_f1']:.4f}")
    print(f"ROUGE-2 F1: {rouge['rouge2_f1']:.4f}")
    print(f"ROUGE-L F1: {rouge['rougeL_f1']:.4f}")

    print(f"\nSaved predictions to: {pred_path}")
    print(f"Saved metrics to: {metrics_path}")

    print("\nSample predictions:")
    for i in range(min(5, len(predictions))):
        print("\n" + "=" * 80)
        print("SOURCE:", sources[i][:400])
        print("REFERENCE:", references[i])
        print("LSTM:", predictions[i])


if __name__ == "__main__":
    main()
