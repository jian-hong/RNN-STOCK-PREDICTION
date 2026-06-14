"""
04e_evaluate_slim.py — evaluate SlimLSTM and append rows to all_metrics.csv.
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import METRICS_DIR, PROC_DIR, PROJECT_ROOT, TICKERS
from metrics_utils import evaluate_model
from models.slim_lstm import SlimLSTM
from models.train_utils import load_model_weights, predict_array


def load_scaler(ticker):
    with open(PROC_DIR / ticker / "scaler.pkl", "rb") as f:
        return pickle.load(f)


def evaluate_slim(ticker):
    pt_path = PROJECT_ROOT / "outputs" / f"best_{ticker}_slim_lstm.pt"
    if not pt_path.exists():
        raise FileNotFoundError(f"Model not found: {pt_path}")

    X_test = np.load(PROC_DIR / ticker / "X_test.npy")
    y_test = np.load(PROC_DIR / ticker / "y_test.npy")
    scaler = load_scaler(ticker)
    model = load_model_weights(SlimLSTM, pt_path)
    y_pred = predict_array(model, X_test)

    metrics = evaluate_model(y_test, y_pred, scaler)
    metrics["Model"] = "SlimLSTM"
    metrics["Ticker"] = ticker
    metrics["model_key"] = "slim_lstm"
    return metrics


def main():
    rows = []
    for ticker in TICKERS:
        pt_path = PROJECT_ROOT / "outputs" / f"best_{ticker}_slim_lstm.pt"
        if not pt_path.exists():
            print(f"  SKIP {ticker} slim_lstm (not trained yet)")
            continue
        print(f"Evaluating SlimLSTM on {ticker}...")
        rows.append(evaluate_slim(ticker))

    if not rows:
        print("No SlimLSTM models found.")
        return

    path = METRICS_DIR / "all_metrics.csv"
    new_df = pd.DataFrame(rows)
    if path.exists():
        existing = pd.read_csv(path)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["Model", "Ticker"], keep="last")
    else:
        combined = new_df
    combined.to_csv(path, index=False)
    print(f"\nSlimLSTM evaluation complete. Updated {path}")


if __name__ == "__main__":
    main()
