"""
01_download_data.py
Downloads daily OHLCV data for 10 top-cap tech stocks from Yahoo Finance.
Saves one CSV per ticker into data/raw/.
Date range: 2015-01-01 to 2024-12-31
"""

import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import END_DATE, RAW_DIR, START_DATE, TICKER_NAMES

RAW_DIR.mkdir(parents=True, exist_ok=True)

for ticker, name in TICKER_NAMES.items():
    print(f"Downloading {name} ({ticker})...")
    df = yf.download(
        ticker,
        start=START_DATE,
        end=END_DATE,
        auto_adjust=True,
        progress=False,
    )
    if df.empty:
        print(f"  WARNING: No data returned for {ticker}. Skipping.")
        continue

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.index.name = "Date"
    out_path = RAW_DIR / f"{ticker}.csv"
    df.to_csv(out_path)
    print(f"  Saved {len(df)} rows -> {out_path}")

print("\nAll downloads complete.")
print("Columns in each CSV: Open, High, Low, Close, Volume")
