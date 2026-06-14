"""
03_train.py — train RNN models on preprocessed ticker data (PyTorch).

Usage:
  python scripts/03_train.py --model all
  python scripts/03_train.py --model slim_lstm --tickers AAPL MSFT
  python scripts/03_train.py --all              # Phase 1 models x 10 tickers
  python scripts/03_train.py --all-slim        # SlimLSTM x 10 tickers
  python scripts/03_train.py --ticker AAPL --model lstm
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import METRICS_DIR, PROJECT_ROOT, TICKERS
from models import MODEL_REGISTRY, DEVICE, make_loaders, train_model

PHASE1_MODELS = ["vanilla_rnn", "lstm", "gru"]
PROC_DIR = PROJECT_ROOT / "data" / "processed"
OUT_DIR = PROJECT_ROOT / "outputs"


def train_one(model_name, ticker):
    d = PROC_DIR / ticker
    X_train = np.load(d / "X_train.npy")
    y_train = np.load(d / "y_train.npy")
    X_val = np.load(d / "X_val.npy")
    y_val = np.load(d / "y_val.npy")

    train_loader, val_loader = make_loaders(X_train, y_train, X_val, y_val)

    model_class = MODEL_REGISTRY[model_name]
    model = model_class().to(DEVICE)
    optimizer = Adam(model.parameters(), lr=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=7, min_lr=1e-6)
    criterion = torch.nn.MSELoss()

    print(
        f"  Training {model_name} on {ticker} | device: {DEVICE} | "
        f"params: {sum(p.numel() for p in model.parameters()):,}"
    )
    history = train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        epochs=100,
        patience=15,
        scheduler=scheduler,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    save_path = OUT_DIR / f"best_{ticker}_{model_name}.pt"
    torch.save(model.state_dict(), save_path)
    print(f"  Saved -> {save_path}")

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    history_path = METRICS_DIR / f"{ticker}_{model_name}_history.csv"
    pd.DataFrame(history).to_csv(history_path, index=False)
    return history


def main():
    parser = argparse.ArgumentParser(
        description="Train RNN stock prediction models (PyTorch)"
    )
    parser.add_argument(
        "--model", default=None, choices=list(MODEL_REGISTRY.keys()) + ["all"]
    )
    parser.add_argument("--tickers", nargs="+", default=TICKERS)
    parser.add_argument(
        "--ticker", type=str, default=None, help="Single ticker (legacy)"
    )
    parser.add_argument(
        "--all", action="store_true", help="Train Phase 1 models x 10 tickers"
    )
    parser.add_argument(
        "--all-slim", action="store_true", help="Train SlimLSTM for all tickers"
    )
    args = parser.parse_args()

    if args.all:
        for ticker in TICKERS:
            for model_name in PHASE1_MODELS:
                print(f"\n{'='*50}\nModel: {model_name}  |  Ticker: {ticker}")
                train_one(model_name, ticker)
        return

    if args.all_slim:
        for ticker in TICKERS:
            print(f"\n{'='*50}\nModel: slim_lstm  |  Ticker: {ticker}")
            train_one("slim_lstm", ticker)
        return

    if args.model == "all":
        models_to_run = ["vanilla_rnn", "lstm", "gru", "slim_lstm"]
        tickers = [args.ticker] if args.ticker else args.tickers
    elif args.model:
        models_to_run = [args.model]
        tickers = [args.ticker] if args.ticker else args.tickers
    else:
        parser.error("Provide --model, --all, or --all-slim")

    for model_name in models_to_run:
        for ticker in tickers:
            if ticker not in TICKERS:
                parser.error(f"Unknown ticker: {ticker}")
            print(f"\n{'='*50}\nModel: {model_name}  |  Ticker: {ticker}")
            train_one(model_name, ticker)


if __name__ == "__main__":
    main()
