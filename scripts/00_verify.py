"""
00_verify.py
Data validation checklist (Master Plan Appendix A) + model parameter verification.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import PROC_DIR, PROJECT_ROOT, RAW_DIR, TICKERS, WINDOW
from models.gru import StackedGRU
from models.lstm import StackedLSTM
from models.slim_lstm import SlimLSTM
from models.vanilla_rnn import VanillaRNN

EXPECTED_PARAMS = {
    "VanillaRNN": (10000, 15000),
    "StackedLSTM": (100000, 125000),
    "StackedGRU": (75000, 95000),
    "SlimLSTM": (20000, 35000),
}


def check_raw_data():
    print("=" * 60)
    print("APPENDIX A — Data Validation Checklist")
    print("=" * 60)
    passed = 0
    total = 0

    for ticker in TICKERS:
        total += 1
        csv_path = RAW_DIR / f"{ticker}.csv"
        ok = csv_path.exists()
        if ok:
            df = pd.read_csv(csv_path, index_col="Date", parse_dates=True)
            rows = len(df)
            cols = set(df.columns)
            expected_cols = {"Open", "High", "Low", "Close", "Volume"}
            ok = rows > 2400 and expected_cols.issubset(cols)
            status = f"PASS ({rows} rows)" if ok else f"FAIL ({rows} rows, cols={cols})"
        else:
            status = "FAIL (missing file)"
        print(f"  [{'x' if ok else ' '}] {ticker}.csv >2400 rows: {status}")
        passed += int(ok)

    for ticker in TICKERS:
        proc_dir = PROC_DIR / ticker
        arrays = ["X_train", "y_train", "X_val", "y_val", "X_test", "y_test"]
        for name in arrays:
            total += 1
            arr = np.load(proc_dir / f"{name}.npy")
            ok = not np.isnan(arr).any()
            print(
                f"  [{'x' if ok else ' '}] {ticker}/{name}: no NaN — {'PASS' if ok else 'FAIL'}"
            )
            passed += int(ok)

        total += 1
        x_train = np.load(proc_dir / "X_train.npy")
        ok = x_train.ndim == 3 and x_train.shape[1] == WINDOW and x_train.shape[2] == 1
        print(
            f"  [{'x' if ok else ' '}] {ticker}/X_train shape {x_train.shape} "
            f"(N,60,1): {'PASS' if ok else 'FAIL'}"
        )
        passed += int(ok)

        total += 1
        splits = pd.read_csv(proc_dir / "splits.csv", index_col=0).squeeze("columns")
        train_end = pd.Timestamp(splits["train_end"])
        val_end = pd.Timestamp(splits["val_end"])
        ok = train_end < val_end
        print(
            f"  [{'x' if ok else ' '}] {ticker} chronological split: {'PASS' if ok else 'FAIL'}"
        )
        passed += int(ok)

    print(f"\nData checks: {passed}/{total} passed")
    return passed, total


def check_model_params():
    print("\n" + "=" * 60)
    print("Model Parameter Count Verification (PyTorch)")
    print("=" * 60)

    models = {
        "VanillaRNN": VanillaRNN(),
        "StackedLSTM": StackedLSTM(),
        "StackedGRU": StackedGRU(),
        "SlimLSTM": SlimLSTM(),
    }
    for name, model in models.items():
        params = sum(p.numel() for p in model.parameters())
        lo, hi = EXPECTED_PARAMS[name]
        ok = lo <= params <= hi
        print(
            f"  {name}: {params:,} params — {'PASS' if ok else 'FAIL'} (expected {lo:,}–{hi:,})"
        )


def check_trained_models():
    print("\n" + "=" * 60)
    print("Trained Model Artifacts (.pt)")
    print("=" * 60)
    models = ["vanilla_rnn", "lstm", "gru"]
    found = 0
    expected = len(TICKERS) * len(models)
    for ticker in TICKERS:
        for model in models:
            path = PROJECT_ROOT / "outputs" / f"best_{ticker}_{model}.pt"
            if path.exists():
                found += 1
    print(f"  Found {found}/{expected} trained RNN model files (.pt)")
    metrics_path = PROJECT_ROOT / "outputs" / "metrics" / "all_metrics.csv"
    if metrics_path.exists():
        df = pd.read_csv(metrics_path)
        print(
            f"  Metrics file: {len(df)} rows ({df['Model'].nunique()} models x {df['Ticker'].nunique()} tickers)"
        )
    else:
        print("  Metrics file: NOT FOUND")


def main():
    check_raw_data()
    check_model_params()
    check_trained_models()


if __name__ == "__main__":
    main()
