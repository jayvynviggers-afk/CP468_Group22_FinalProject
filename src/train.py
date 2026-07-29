import argparse
import time
import json
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

from dataset import get_dataloaders
from model import Seq2SeqAttentionModel, count_parameters
from utils import PAD_IDX, set_seed, get_device


OUTPUT_DIR = Path("outputs")
MODEL_DIR = OUTPUT_DIR / "models"
METRICS_DIR = OUTPUT_DIR / "metrics"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)


def train_one_epoch(model, dataloader, optimizer, criterion, device, clip=1.0, limit_batches=None):
    model.train()
    total_loss = 0.0

    progress_bar = tqdm(dataloader, desc="Training", leave=False)

    for batch_idx, batch in enumerate(progress_bar):
        if limit_batches is not None and batch_idx >= limit_batches:
            break

        source_ids = batch["source_ids"].to(device)
        target_input_ids = batch["target_input_ids"].to(device)
        target_output_ids = batch["target_output_ids"].to(device)

        optimizer.zero_grad()

        outputs = model(source_ids, target_input_ids)

        # outputs shape: [batch_size, target_len, target_vocab_size]
        # target_output_ids shape: [batch_size, target_len]
        output_dim = outputs.shape[-1]

        loss = criterion(
            outputs.reshape(-1, output_dim),
            target_output_ids.reshape(-1)
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        total_loss += loss.item()

        progress_bar.set_postfix(loss=loss.item())

    num_batches = limit_batches if limit_batches is not None else len(dataloader)
    return total_loss / max(1, num_batches)


def evaluate_loss(model, dataloader, criterion, device, limit_batches=None):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Validation", leave=False)

        for batch_idx, batch in enumerate(progress_bar):
            if limit_batches is not None and batch_idx >= limit_batches:
                break

            source_ids = batch["source_ids"].to(device)
            target_input_ids = batch["target_input_ids"].to(device)
            target_output_ids = batch["target_output_ids"].to(device)

            outputs = model(source_ids, target_input_ids)
            output_dim = outputs.shape[-1]

            loss = criterion(
                outputs.reshape(-1, output_dim),
                target_output_ids.reshape(-1)
            )

            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

    num_batches = limit_batches if limit_batches is not None else len(dataloader)
    return total_loss / max(1, num_batches)


def preview_generation(model, dataloader, target_vocab, device):
    model.eval()

    batch = next(iter(dataloader))

    source_ids = batch["source_ids"][:1].to(device)
    source_text = batch["source_text"][0]
    reference = batch["target_text"][0]

    with torch.no_grad():
        generated_ids = model.generate(source_ids, max_len=32)

    generated = target_vocab.decode(generated_ids[0].cpu())

    print("\nSample generation:")
    print("SOURCE:", source_text[:500])
    print("REFERENCE:", reference)
    print("LSTM OUTPUT:", generated)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--limit_batches", type=int, default=None)
    args = parser.parse_args()

    set_seed()
    device = get_device()

    print(f"Using device: {device}")

    train_loader, val_loader, test_loader, source_vocab, target_vocab = get_dataloaders(
        batch_size=args.batch_size
    )

    model = Seq2SeqAttentionModel(
        source_vocab_size=len(source_vocab),
        target_vocab_size=len(target_vocab),
        embedding_dim=args.embedding_dim,
        encoder_hidden_dim=args.hidden_dim,
        dropout=args.dropout
    ).to(device)

    param_count = count_parameters(model)

    print(f"Source vocab size: {len(source_vocab)}")
    print(f"Target vocab size: {len(target_vocab)}")
    print(f"Trainable parameters: {param_count:,}")

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float("inf")
    history = []

    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            limit_batches=args.limit_batches
        )

        val_loss = evaluate_loss(
            model,
            val_loader,
            criterion,
            device,
            limit_batches=args.limit_batches
        )

        print(f"Train loss: {train_loss:.4f}")
        print(f"Validation loss: {val_loss:.4f}")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "source_vocab_size": len(source_vocab),
                "target_vocab_size": len(target_vocab),
                "embedding_dim": args.embedding_dim,
                "hidden_dim": args.hidden_dim,
                "dropout": args.dropout,
                "best_val_loss": best_val_loss,
                "param_count": param_count
            }

            torch.save(checkpoint, MODEL_DIR / "best_lstm_attention.pt")
            print("Saved new best model.")

        preview_generation(model, val_loader, target_vocab, device)

    end_time = time.time()
    training_time_seconds = end_time - start_time

    summary = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "embedding_dim": args.embedding_dim,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "device": str(device),
        "trainable_parameters": param_count,
        "best_val_loss": best_val_loss,
        "training_time_seconds": training_time_seconds,
        "training_time_minutes": training_time_seconds / 60,
        "history": history
    }

    with open(METRICS_DIR / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nTraining complete.")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Training time: {training_time_seconds / 60:.2f} minutes")
    print(f"Saved model to: {MODEL_DIR / 'best_lstm_attention.pt'}")
    print(f"Saved metrics to: {METRICS_DIR / 'training_summary.json'}")


if __name__ == "__main__":
    main()
