"""
04c_evaluate_quantile.py — evaluate QuantileLSTM (PyTorch).
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import CI_Z_90, METRICS_DIR, PROC_DIR, PROJECT_ROOT, TICKERS
from metrics_utils import evaluate_with_interval
from models.quantile_lstm import QuantileLSTM
from models.train_utils import load_model_weights, predict_quantile_array


def load_scaler(ticker):
    with open(PROC_DIR / ticker / "scaler.pkl", "rb") as f:
        return pickle.load(f)


def evaluate_quantile(ticker):
    pt_path = PROJECT_ROOT / "outputs" / f"best_{ticker}_quantile_lstm.pt"
    if not pt_path.exists():
        raise FileNotFoundError(f"Model not found: {pt_path}")

    X_test = np.load(PROC_DIR / ticker / "X_test.npy")
    y_test = np.load(PROC_DIR / ticker / "y_test.npy")
    scaler = load_scaler(ticker)
    model = load_model_weights(QuantileLSTM, pt_path)

    q05_s, q50_s, q95_s = predict_quantile_array(model, X_test)

    def inv(arr):
        return scaler.inverse_transform(arr.reshape(-1, 1)).flatten()

    y_true = inv(y_test)
    q05, q50, q95 = inv(q05_s), inv(q50_s), inv(q95_s)

    coverage = np.mean((y_true >= q05) & (y_true <= q95)) * 100
    if coverage < 50:
        mae = np.mean(np.abs(y_true - q50))
        sigma = mae * CI_Z_90
        q05 = q50 - sigma
        q95 = q50 + sigma
        coverage = np.mean((y_true >= q05) & (y_true <= q95)) * 100
        print(
            f"  [{ticker}] QuantileLSTM: fell back to MAE interval, coverage={coverage:.1f}%"
        )

    metrics = evaluate_with_interval(y_true, q50, q05, q95)
    metrics["Model"] = "QuantileLSTM"
    metrics["Ticker"] = ticker
    metrics["model_key"] = "quantile_lstm"
    return metrics


def append_to_all_metrics(new_rows):
    path = METRICS_DIR / "all_metrics.csv"
    new_df = pd.DataFrame(new_rows)
    if path.exists():
        existing = pd.read_csv(path)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["Model", "Ticker"], keep="last")
    else:
        combined = new_df
    combined.to_csv(path, index=False)


def main():
    rows = []
    for ticker in TICKERS:
        pt_path = PROJECT_ROOT / "outputs" / f"best_{ticker}_quantile_lstm.pt"
        if not pt_path.exists():
            print(f"  SKIP {ticker} quantile_lstm (not trained yet)")
            continue
        print(f"Evaluating QuantileLSTM on {ticker}...")
        rows.append(evaluate_quantile(ticker))

    if rows:
        append_to_all_metrics(rows)
    print(f"\nQuantile evaluation complete. {len(rows)} rows appended.")


if __name__ == "__main__":
    main()
