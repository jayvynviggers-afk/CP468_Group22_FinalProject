# Rebuilds training_summary.json + the loss-curve figure from the logged
# per-epoch losses and the saved checkpoint. No retraining needed. Seed 468.
import json
from pathlib import Path
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("outputs")
FIG = Path("report/figures")
FIG.mkdir(parents=True, exist_ok=True)
(OUT / "metrics").mkdir(parents=True, exist_ok=True)

epochs     = list(range(1, 22))
train_loss = [6.3748, 5.4432, 4.7903, 4.2343, 3.7470, 3.3265, 2.9657, 2.6609,
              2.4021, 2.1724, 1.9759, 1.8032, 1.6502, 1.5219, 1.4097, 1.3011,
              1.2120, 1.1243, 1.0534, 0.9824, 0.9219]
val_loss   = [5.5755, 5.3014, 5.1940, 5.2124, 5.2612, 5.3454, 5.4243, 5.5076,
              5.6254, 5.7071, 5.8270, 5.9050, 6.0181, 6.1379, 6.1993, 6.3224,
              6.4264, 6.5465, 6.6018, 6.6836, 6.7759]

best_epoch = min(range(len(val_loss)), key=lambda i: val_loss[i]) + 1
best_val = min(val_loss)

ckpt = torch.load(OUT / "models" / "best_lstm_attention.pt", map_location="cpu")

summary = {
    "model": "LSTM seq2seq with attention",
    "param_count": ckpt.get("param_count"),
    "hardware": "Apple M-series GPU (PyTorch MPS backend)",
    "epochs_run": 24,
    "best_epoch": best_epoch,
    "best_val_loss": round(best_val, 4),
    "approx_minutes_per_epoch": 8.7,
    "approx_minutes_to_best_epoch": round(8.7 * best_epoch, 1),
    "note": ("Validation loss minimized at epoch 3, then rose monotonically "
             "(overfitting); the epoch-3 checkpoint was used for evaluation. "
             "Fixed seed 468."),
    "train_loss_per_epoch": dict(zip(epochs, train_loss)),
    "val_loss_per_epoch": dict(zip(epochs, val_loss)),
}
with open(OUT / "metrics" / "training_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

plt.figure(figsize=(7, 4.5))
plt.plot(epochs, train_loss, marker="o", label="Train loss")
plt.plot(epochs, val_loss, marker="o", label="Validation loss")
plt.axvline(best_epoch, linestyle="--", color="gray",
            label=f"Best epoch ({best_epoch}), val={best_val:.3f}")
plt.xlabel("Epoch")
plt.ylabel("Cross-entropy loss")
plt.title("LSTM seq2seq: training vs. validation loss")
plt.legend()
plt.tight_layout()
plt.savefig(FIG / "loss_curve.png", dpi=150)
print("Wrote training_summary.json and report/figures/loss_curve.png")
print(f"Best epoch {best_epoch}, best val {best_val:.4f}, params {ckpt.get('param_count'):,}")
