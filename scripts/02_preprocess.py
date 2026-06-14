"""
02_preprocess.py
For each ticker:
  1. Load raw CSV
  2. Check for missing dates (market holidays OK; fill NaN with ffill)
  3. MinMaxScale the Close price
  4. Create sliding windows of shape (N, 60, 1) -> (N, 1)
  5. Split chronologically: 70/10/20
  6. Save arrays and scaler
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import PROC_DIR, RAW_DIR, TICKERS, TRAIN_R, VAL_R, WINDOW

PROC_DIR.mkdir(parents=True, exist_ok=True)


def make_sequences(data, window):
    X, y = [], []
    for i in range(window, len(data)):
        X.append(data[i - window : i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)


for ticker in TICKERS:
    print(f"Processing {ticker}...")
    df = pd.read_csv(RAW_DIR / f"{ticker}.csv", index_col="Date", parse_dates=True)

    df.sort_index(inplace=True)
    df = df[["Close"]]
    df.replace(0, np.nan, inplace=True)
    df.ffill(inplace=True)
    df.dropna(inplace=True)

    prices = df["Close"].values.reshape(-1, 1)
    n = len(prices)
    print(f"  Total rows after cleaning: {n}")

    train_end = int(n * TRAIN_R)
    val_end = int(n * (TRAIN_R + VAL_R))

    train_prices = prices[:train_end]
    val_prices = prices[train_end:val_end]
    test_prices = prices[val_end:]

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_prices)

    train_scaled = scaler.transform(train_prices)
    val_scaled = scaler.transform(val_prices)
    test_scaled = scaler.transform(test_prices)

    X_train, y_train = make_sequences(train_scaled, WINDOW)
    X_val, y_val = make_sequences(val_scaled, WINDOW)
    X_test, y_test = make_sequences(test_scaled, WINDOW)

    X_train = X_train.reshape(X_train.shape[0], WINDOW, 1)
    X_val = X_val.reshape(X_val.shape[0], WINDOW, 1)
    X_test = X_test.reshape(X_test.shape[0], WINDOW, 1)

    print(f"  X_train: {X_train.shape}  X_val: {X_val.shape}  X_test: {X_test.shape}")

    out_dir = PROC_DIR / ticker
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "X_train.npy", X_train)
    np.save(out_dir / "y_train.npy", y_train)
    np.save(out_dir / "X_val.npy", X_val)
    np.save(out_dir / "y_val.npy", y_val)
    np.save(out_dir / "X_test.npy", X_test)
    np.save(out_dir / "y_test.npy", y_test)

    with open(out_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    split_info = {
        "train_start": str(df.index[0].date()),
        "train_end": str(df.index[train_end].date()),
        "val_end": str(df.index[val_end].date()),
        "test_end": str(df.index[-1].date()),
        "total_rows": n,
    }
    pd.Series(split_info).to_csv(out_dir / "splits.csv")

print("\nPreprocessing complete.")
