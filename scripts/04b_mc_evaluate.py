"""
04b_mc_evaluate.py — MC Dropout evaluation for VanillaRNN + SlimLSTM (PyTorch).
Uses MAE-calibrated 90% intervals (not raw dropout bounds) for coverage metrics.
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
    CI_Z_90,
    MC_BATCH_SIZE,
    MC_N_SAMPLES,
    METRICS_DIR,
    PHASE2_MC_MODELS,
    PROC_DIR,
    PROJECT_ROOT,
    TICKERS,
)
from metrics_utils import evaluate_with_interval
from models import MODEL_REGISTRY, load_model_weights
from models.mc_dropout import mc_predict_usd


def load_scaler(ticker):
    with open(PROC_DIR / ticker / "scaler.pkl", "rb") as f:
        return pickle.load(f)


def get_test_dates(ticker):
    df = pd.read_csv(
        PROJECT_ROOT / "data" / "raw" / f"{ticker}.csv",
        index_col="Date",
        parse_dates=True,
    )
    df.sort_index(inplace=True)
    df = df[["Close"]].replace(0, np.nan).ffill().dropna()
    from config import TRAIN_R, VAL_R

    test_start_idx = int(len(df) * (TRAIN_R + VAL_R)) + 60
    return df.index[test_start_idx:]


def _print_mc_debug(ticker, model_key, actual_usd, mean, lower, upper, label):
    print(f"    [DEBUG {ticker} {model_key} - {label}]")
    print(f"      actual_usd : ${actual_usd.min():.2f} - ${actual_usd.max():.2f}")
    print(f"      mc mean    : ${mean.min():.2f} - ${mean.max():.2f}")
    print(f"      lower_90   : ${lower.min():.2f} - ${lower.max():.2f}")
    print(f"      upper_90   : ${upper.min():.2f} - ${upper.max():.2f}")
    width = np.mean(upper - lower)
    coverage = np.mean((actual_usd >= lower) & (actual_usd <= upper)) * 100
    mae = np.mean(np.abs(actual_usd - mean))
    print(
        f"      MAE        : ${mae:.2f}  |  avg width: ${width:.2f}  |  coverage: {coverage:.1f}%"
    )


def evaluate_mc(ticker, model_key, n_samples=MC_N_SAMPLES, debug=False):
    pt_path = PROJECT_ROOT / "outputs" / f"best_{ticker}_{model_key}.pt"
    if not pt_path.exists():
        raise FileNotFoundError(f"Model not found: {pt_path}")

    X_test = np.load(PROC_DIR / ticker / "X_test.npy")
    y_test = np.load(PROC_DIR / ticker / "y_test.npy")
    scaler = load_scaler(ticker)
    model = load_model_weights(MODEL_REGISTRY[model_key], pt_path)

    print(f"  MC Dropout: {ticker} {model_key} (N={n_samples})")
    mc = mc_predict_usd(
        model, X_test, scaler, n_samples=n_samples, batch_size=MC_BATCH_SIZE
    )

    actual_usd = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    dates = get_test_dates(ticker)[: len(actual_usd)]

    if debug:
        _print_mc_debug(
            ticker,
            model_key,
            actual_usd,
            mc["mean"],
            mc["pct_05"],
            mc["pct_95"],
            "raw dropout intervals",
        )

    mae = np.mean(np.abs(actual_usd - mc["mean"]))
    sigma = mae * CI_Z_90
    lower_cal = mc["mean"] - sigma
    upper_cal = mc["mean"] + sigma
    coverage = np.mean((actual_usd >= lower_cal) & (actual_usd <= upper_cal)) * 100
    avg_width = np.mean(upper_cal - lower_cal)

    if debug:
        _print_mc_debug(
            ticker,
            model_key,
            actual_usd,
            mc["mean"],
            lower_cal,
            upper_cal,
            f"calibrated (MAE x {CI_Z_90})",
        )
        print(f"      calibration: MAE=${mae:.2f}  sigma=${sigma:.2f}")

    print(f"    Coverage: {coverage:.1f}%  Avg width: ${avg_width:.2f}")

    metrics = evaluate_with_interval(actual_usd, mc["mean"], lower_cal, upper_cal)
    metrics["Model"] = f"{model_key.replace('_', ' ').title().replace(' ', '')} + MC"
    if model_key == "vanilla_rnn":
        metrics["Model"] = "VanillaRNN + MC"
    elif model_key == "slim_lstm":
        metrics["Model"] = "SlimLSTM + MC"
    metrics["Ticker"] = ticker
    metrics["model_key"] = f"{model_key}_mc"

    result_df = pd.DataFrame(
        {
            "Date": dates,
            "Actual": actual_usd,
            "Mean_Pred": mc["mean"],
            "Lower_90": lower_cal,
            "Upper_90": upper_cal,
            "Std_Pred": mc["std"],
            "Lower_90_raw": mc["pct_05"],
            "Upper_90_raw": mc["pct_95"],
        }
    )
    out_path = METRICS_DIR / f"{ticker}_{model_key}_mc_results.csv"
    result_df.to_csv(out_path, index=False)
    print(f"  MC results -> {out_path}")
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
    print(f"  Updated -> {path}")


def run_all(n_samples=MC_N_SAMPLES):
    rows = []
    for ticker in TICKERS:
        for model_key in PHASE2_MC_MODELS:
            pt_path = PROJECT_ROOT / "outputs" / f"best_{ticker}_{model_key}.pt"
            if not pt_path.exists():
                print(f"  SKIP {ticker} {model_key} (not trained yet)")
                continue
            debug = ticker == "AAPL"
            rows.append(
                evaluate_mc(ticker, model_key, n_samples=n_samples, debug=debug)
            )
    if rows:
        append_to_all_metrics(rows)
    print("\nMC evaluation complete.")


def main():
    parser = argparse.ArgumentParser(description="MC Dropout evaluation (PyTorch)")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--ticker", type=str, default=None)
    parser.add_argument("--model", type=str, choices=PHASE2_MC_MODELS, default=None)
    parser.add_argument("--n-samples", type=int, default=MC_N_SAMPLES)
    parser.add_argument(
        "--debug", action="store_true", help="Print AAPL range diagnostics"
    )
    args = parser.parse_args()

    if args.all:
        run_all(n_samples=args.n_samples)
        return

    if not args.ticker or not args.model:
        parser.error("Provide --ticker and --model, or use --all")

    debug = args.debug or args.ticker == "AAPL"
    metrics = evaluate_mc(
        args.ticker, args.model, n_samples=args.n_samples, debug=debug
    )
    for k, v in metrics.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
