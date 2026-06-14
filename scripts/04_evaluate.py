"""
04_evaluate.py — evaluate trained PyTorch models on test sets.
Writes to outputs/metrics/all_metrics.csv
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    METRICS_DIR,
    MODEL_CHOICES,
    PROC_DIR,
    PROJECT_ROOT,
    TICKERS,
    TRAIN_R,
    VAL_R,
)
from metrics_utils import evaluate_from_usd, evaluate_model
from models import MODEL_REGISTRY, predict_array, load_model_weights
from models.baseline_arima import run_arima_walkforward

RNN_MODELS = ["vanilla_rnn", "lstm", "gru", "slim_lstm"]


def load_scaler(ticker):
    with open(PROC_DIR / ticker / "scaler.pkl", "rb") as f:
        return pickle.load(f)


def load_test_data(ticker):
    return (
        np.load(PROC_DIR / ticker / "X_test.npy"),
        np.load(PROC_DIR / ticker / "y_test.npy"),
    )


def get_train_test_prices(ticker):
    df = pd.read_csv(
        PROJECT_ROOT / "data" / "raw" / f"{ticker}.csv",
        index_col="Date",
        parse_dates=True,
    )
    df.sort_index(inplace=True)
    df = df[["Close"]].replace(0, np.nan).ffill().dropna()
    prices = df["Close"].values
    n = len(prices)
    test_start = int(n * (TRAIN_R + VAL_R))
    return prices[:test_start], prices[test_start:]


def evaluate_rnn(ticker, model_key):
    pt_path = PROJECT_ROOT / "outputs" / f"best_{ticker}_{model_key}.pt"
    if not pt_path.exists():
        raise FileNotFoundError(f"Model not found: {pt_path}")

    X_test, y_test = load_test_data(ticker)
    scaler = load_scaler(ticker)
    model = load_model_weights(MODEL_REGISTRY[model_key], pt_path)
    y_pred = predict_array(model, X_test)

    metrics = evaluate_model(y_test, y_pred, scaler)
    metrics["Model"] = MODEL_CHOICES.get(model_key, model_key)
    metrics["Ticker"] = ticker
    metrics["model_key"] = model_key
    metrics["y_pred"] = y_pred
    metrics["y_true_scaled"] = y_test
    return metrics


def evaluate_arima(ticker):
    train_series, test_series = get_train_test_prices(ticker)
    scaler = load_scaler(ticker)

    print(f"  Running ARIMA walk-forward for {ticker} ({len(test_series)} steps)...")
    y_pred = run_arima_walkforward(train_series, test_series, order=(5, 1, 0))

    _, y_test_scaled = load_test_data(ticker)
    y_true = scaler.inverse_transform(y_test_scaled.reshape(-1, 1)).flatten()

    min_len = min(len(y_true), len(y_pred))
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]

    metrics = evaluate_from_usd(y_true, y_pred)
    metrics["Model"] = "ARIMA(5,1,0)"
    metrics["Ticker"] = ticker
    metrics["model_key"] = "arima"
    metrics["y_pred_usd"] = y_pred
    metrics["y_true_usd"] = y_true
    return metrics


def run_all():
    rows = []
    predictions_cache = {}

    for ticker in TICKERS:
        for model_key in RNN_MODELS:
            pt_path = PROJECT_ROOT / "outputs" / f"best_{ticker}_{model_key}.pt"
            if not pt_path.exists():
                print(f"  SKIP {ticker} {model_key} (not trained yet)")
                continue
            print(f"Evaluating {model_key} on {ticker}...")
            result = evaluate_rnn(ticker, model_key)
            predictions_cache[(ticker, model_key)] = {
                "y_pred": result.pop("y_pred"),
                "y_true_scaled": result.pop("y_true_scaled"),
            }
            rows.append(
                {
                    k: v
                    for k, v in result.items()
                    if k not in ("y_pred", "y_true_scaled")
                }
            )

        print(f"Evaluating ARIMA on {ticker}...")
        result = evaluate_arima(ticker)
        predictions_cache[(ticker, "arima")] = {
            "y_pred_usd": result.pop("y_pred_usd"),
            "y_true_usd": result.pop("y_true_usd"),
        }
        rows.append(
            {k: v for k, v in result.items() if k not in ("y_pred_usd", "y_true_usd")}
        )

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    out_path = METRICS_DIR / "all_metrics.csv"
    df.to_csv(out_path, index=False)
    print(f"\nMetrics saved → {out_path}")

    import pickle as pkl

    pred_path = METRICS_DIR / "predictions_cache.pkl"
    with open(pred_path, "wb") as f:
        pkl.dump(predictions_cache, f)
    print(f"Predictions cache → {pred_path}")

    agg = df.groupby("Model")[["RMSE", "MAE", "MAPE", "R2", "DA"]].mean()
    agg_path = METRICS_DIR / "aggregated_metrics.csv"
    agg.to_csv(agg_path)
    print(f"Aggregated metrics → {agg_path}")
    print("\n--- Aggregated Results ---")
    print(agg.to_string())


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate stock prediction models (PyTorch)"
    )
    parser.add_argument(
        "--all", action="store_true", help="Evaluate all models and tickers"
    )
    parser.add_argument("--ticker", type=str, default=None)
    parser.add_argument(
        "--model", type=str, default=None, choices=RNN_MODELS + ["arima"]
    )
    args = parser.parse_args()

    if args.all:
        run_all()
        return

    if not args.ticker or not args.model:
        parser.error("Provide --ticker and --model, or use --all")

    if args.model == "arima":
        result = evaluate_arima(args.ticker)
    else:
        result = evaluate_rnn(args.ticker, args.model)

    for k, v in result.items():
        if not isinstance(v, (np.ndarray, list)):
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
