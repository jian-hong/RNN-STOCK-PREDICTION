"""
03b_train_quantile.py — train QuantileLSTM (q05/q50/q95) for all tickers.
"""

import sys
from pathlib import Path

import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import PROC_DIR, PROJECT_ROOT, TICKERS
from models.quantile_lstm import QuantileLSTM, pinball_loss
from models.train_utils import DEVICE

OUT_DIR = PROJECT_ROOT / "outputs"

loss_q05 = pinball_loss(0.05)
loss_q50 = pinball_loss(0.50)
loss_q95 = pinball_loss(0.95)


def train_quantile(ticker):
    print(f"\nQuantileLSTM — {ticker}")
    d = PROC_DIR / ticker
    X_train = np.load(d / "X_train.npy")
    y_train = np.load(d / "y_train.npy")
    X_val = np.load(d / "X_val.npy")
    y_val = np.load(d / "y_val.npy")

    y_t = torch.tensor(y_train, dtype=torch.float32)
    y_v = torch.tensor(y_val, dtype=torch.float32)
    X_t = torch.tensor(X_train, dtype=torch.float32)
    X_v = torch.tensor(X_val, dtype=torch.float32)

    train_ds = TensorDataset(X_t, y_t)
    val_ds = TensorDataset(X_v, y_v)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    model = QuantileLSTM().to(DEVICE)
    optimizer = Adam(model.parameters(), lr=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=7, min_lr=1e-6)

    best_val, patience_count, best_state = float("inf"), 0, None

    for epoch in range(100):
        model.train()
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            optimizer.zero_grad()
            q05, q50, q95 = model(X_b)
            q05 = q05.squeeze(-1)
            q50 = q50.squeeze(-1)
            q95 = q95.squeeze(-1)
            loss = loss_q05(q05, y_b) + loss_q50(q50, y_b) + loss_q95(q95, y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
                q05, q50, q95 = model(X_b)
                q05 = q05.squeeze(-1)
                q50 = q50.squeeze(-1)
                q95 = q95.squeeze(-1)
                val_losses.append(
                    (
                        loss_q05(q05, y_b) + loss_q50(q50, y_b) + loss_q95(q95, y_b)
                    ).item()
                )
        v = np.mean(val_losses)
        scheduler.step(v)

        if v < best_val - 1e-5:
            best_val, patience_count = v, 0
            best_state = {k: v2.clone() for k, v2 in model.state_dict().items()}
        else:
            patience_count += 1
            if patience_count >= 15:
                print(f"  EarlyStopping epoch {epoch+1}")
                break

    model.load_state_dict(best_state)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    save_path = OUT_DIR / f"best_{ticker}_quantile_lstm.pt"
    torch.save(model.state_dict(), save_path)
    print(f"  Saved -> {save_path}")


def main():
    for ticker in TICKERS:
        train_quantile(ticker)
    print("\nQuantile LSTM training complete.")


if __name__ == "__main__":
    main()
