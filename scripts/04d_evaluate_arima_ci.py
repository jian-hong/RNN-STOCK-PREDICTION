"""
04d_evaluate_arima_ci.py
ARIMA walk-forward with statistical confidence intervals.
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import METRICS_DIR, PROC_DIR, PROJECT_ROOT, TICKERS, TRAIN_R, VAL_R
from metrics_utils import evaluate_with_interval
from models.baseline_arima import run_arima_walkforward_with_ci


def load_scaler(ticker):
    with open(PROC_DIR / ticker / "scaler.pkl", "rb") as f:
        return pickle.load(f)


def get_train_test_prices(ticker):
    df = pd.read_csv(
        PROJECT_ROOT / "data" / "raw" / f"{ticker}.csv",
        index_col="Date",
        parse_dates=True,
    )
    df.sort_index(inplace=True)
    prices = df["Close"].replace(0, np.nan).ffill().dropna().values
    n = len(prices)
    test_start = int(n * (TRAIN_R + VAL_R))
    return prices[:test_start], prices[test_start:]


def evaluate_arima_ci(ticker):
    train_series, test_series = get_train_test_prices(ticker)
    scaler = load_scaler(ticker)
    _, y_test_scaled = (
        np.load(PROC_DIR / ticker / "X_test.npy"),
        np.load(PROC_DIR / ticker / "y_test.npy"),
    )

    print(f"  ARIMA+CI walk-forward for {ticker} ({len(test_series)} steps)...")
    y_pred, lower, upper = run_arima_walkforward_with_ci(train_series, test_series)

    y_true = scaler.inverse_transform(y_test_scaled.reshape(-1, 1)).flatten()
    min_len = min(len(y_true), len(y_pred))
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]
    lower = lower[:min_len]
    upper = upper[:min_len]

    metrics = evaluate_with_interval(y_true, y_pred, lower, upper)
    metrics["Model"] = "ARIMA(5,1,0) + CI"
    metrics["Ticker"] = ticker
    metrics["model_key"] = "arima_ci"

    result_df = pd.DataFrame(
        {
            "Actual": y_true,
            "Predicted": y_pred,
            "Lower_90": lower,
            "Upper_90": upper,
        }
    )
    result_df.to_csv(METRICS_DIR / f"{ticker}_arima_ci_results.csv", index=False)
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
        print(f"Evaluating ARIMA+CI on {ticker}...")
        rows.append(evaluate_arima_ci(ticker))
    append_to_all_metrics(rows)
    print("\nARIMA CI evaluation complete.")


if __name__ == "__main__":
    main()
