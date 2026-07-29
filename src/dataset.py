import re
import json
from collections import Counter
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from utils import (
    PAD_TOKEN,
    UNK_TOKEN,
    SOS_TOKEN,
    EOS_TOKEN,
    PAD_IDX,
    UNK_IDX,
    EOS_IDX,
    SPECIAL_TOKENS,
    set_seed,
)

DATA_DIR = Path("data/processed")
VOCAB_DIR = Path("data/processed/vocab")
VOCAB_DIR.mkdir(parents=True, exist_ok=True)

MAX_SOURCE_LEN = 256
MAX_TARGET_LEN = 32
MIN_FREQ = 2
MAX_SOURCE_VOCAB = 20000
MAX_TARGET_VOCAB = 10000
BATCH_SIZE = 16


def tokenize(text):
    """
    Simple lowercase tokenizer.
    Splits words and punctuation.
    Example: "Trump's speech." -> ["trump", "'", "s", "speech", "."]
    """
    text = str(text).lower().strip()
    return re.findall(r"\w+|[^\w\s]", text)


class Vocabulary:
    def __init__(self, max_size=None, min_freq=1):
        self.max_size = max_size
        self.min_freq = min_freq
        self.token_to_idx = {}
        self.idx_to_token = {}

        for token in SPECIAL_TOKENS:
            self.add_token(token)

    def add_token(self, token):
        if token not in self.token_to_idx:
            idx = len(self.token_to_idx)
            self.token_to_idx[token] = idx
            self.idx_to_token[idx] = token

    def build(self, texts):
        counter = Counter()

        for text in texts:
            counter.update(tokenize(text))

        words = [
            token for token, freq in counter.most_common()
            if freq >= self.min_freq
        ]

        if self.max_size is not None:
            words = words[: max(0, self.max_size - len(SPECIAL_TOKENS))]

        for word in words:
            self.add_token(word)

    def encode(self, text, max_len, add_sos=False, add_eos=True):
        tokens = tokenize(text)

        if add_sos:
            tokens = [SOS_TOKEN] + tokens

        if add_eos:
            tokens = tokens + [EOS_TOKEN]

        ids = [
            self.token_to_idx.get(token, UNK_IDX)
            for token in tokens
        ]

        if len(ids) > max_len:
            ids = ids[:max_len]
            if add_eos:
                ids[-1] = EOS_IDX

        padding_needed = max_len - len(ids)
        ids = ids + [PAD_IDX] * padding_needed

        return ids

    def decode(self, ids):
        tokens = []

        for idx in ids:
            token = self.idx_to_token.get(int(idx), UNK_TOKEN)

            if token == EOS_TOKEN:
                break

            if token not in SPECIAL_TOKENS:
                tokens.append(token)

        return " ".join(tokens)

    def __len__(self):
        return len(self.token_to_idx)

    def save(self, path):
        data = {
            "token_to_idx": self.token_to_idx,
            "idx_to_token": {str(k): v for k, v in self.idx_to_token.items()},
            "max_size": self.max_size,
            "min_freq": self.min_freq,
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        vocab = cls(max_size=data["max_size"], min_freq=data["min_freq"])
        vocab.token_to_idx = data["token_to_idx"]
        vocab.idx_to_token = {int(k): v for k, v in data["idx_to_token"].items()}
        return vocab


class HeadlineDataset(Dataset):
    def __init__(self, csv_path, source_vocab, target_vocab):
        self.df = pd.read_csv(csv_path)
        self.source_vocab = source_vocab
        self.target_vocab = target_vocab

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        source_text = row["source"]
        target_text = row["target"]

        source_ids = self.source_vocab.encode(
            source_text,
            max_len=MAX_SOURCE_LEN,
            add_sos=False,
            add_eos=True,
        )

        # Decoder input gets <sos> but no <eos>
        target_input_ids = self.target_vocab.encode(
            target_text,
            max_len=MAX_TARGET_LEN,
            add_sos=True,
            add_eos=False,
        )

        # Decoder output gets <eos> but no <sos>
        target_output_ids = self.target_vocab.encode(
            target_text,
            max_len=MAX_TARGET_LEN,
            add_sos=False,
            add_eos=True,
        )

        return {
            "source_ids": torch.tensor(source_ids, dtype=torch.long),
            "target_input_ids": torch.tensor(target_input_ids, dtype=torch.long),
            "target_output_ids": torch.tensor(target_output_ids, dtype=torch.long),
            "source_text": source_text,
            "target_text": target_text,
        }


def build_vocabs():
    train_df = pd.read_csv(DATA_DIR / "train.csv")

    source_vocab = Vocabulary(max_size=MAX_SOURCE_VOCAB, min_freq=MIN_FREQ)
    target_vocab = Vocabulary(max_size=MAX_TARGET_VOCAB, min_freq=MIN_FREQ)

    source_vocab.build(train_df["source"].tolist())
    target_vocab.build(train_df["target"].tolist())

    source_vocab.save(VOCAB_DIR / "source_vocab.json")
    target_vocab.save(VOCAB_DIR / "target_vocab.json")

    return source_vocab, target_vocab


def load_or_build_vocabs():
    source_path = VOCAB_DIR / "source_vocab.json"
    target_path = VOCAB_DIR / "target_vocab.json"

    if source_path.exists() and target_path.exists():
        source_vocab = Vocabulary.load(source_path)
        target_vocab = Vocabulary.load(target_path)
    else:
        source_vocab, target_vocab = build_vocabs()

    return source_vocab, target_vocab


def get_dataloaders(batch_size=BATCH_SIZE):
    source_vocab, target_vocab = load_or_build_vocabs()

    train_dataset = HeadlineDataset(DATA_DIR / "train.csv", source_vocab, target_vocab)
    val_dataset = HeadlineDataset(DATA_DIR / "val.csv", source_vocab, target_vocab)
    test_dataset = HeadlineDataset(DATA_DIR / "test.csv", source_vocab, target_vocab)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, source_vocab, target_vocab


def main():
    set_seed()

    train_loader, val_loader, test_loader, source_vocab, target_vocab = get_dataloaders()

    print("Dataset/DataLoader setup complete.")
    print(f"Source vocab size: {len(source_vocab)}")
    print(f"Target vocab size: {len(target_vocab)}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    batch = next(iter(train_loader))

    print("\nBatch shapes:")
    print("source_ids:", batch["source_ids"].shape)
    print("target_input_ids:", batch["target_input_ids"].shape)
    print("target_output_ids:", batch["target_output_ids"].shape)

    print("\nExample source text:")
    print(batch["source_text"][0][:500])

    print("\nExample target headline:")
    print(batch["target_text"][0])

    print("\nDecoded target input:")
    print(target_vocab.decode(batch["target_input_ids"][0]))

    print("\nDecoded target output:")
    print(target_vocab.decode(batch["target_output_ids"][0]))


if __name__ == "__main__":
    main()
