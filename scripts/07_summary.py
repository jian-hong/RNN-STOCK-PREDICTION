"""
07_summary.py
Merge Phase 1 + Phase 2 metrics into a single summary table.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import METRICS_DIR

SUMMARY_COLS = [
    "Model",
    "Ticker",
    "RMSE",
    "MAPE",
    "R2",
    "DA",
    "Coverage_90pct",
    "Avg_Width_USD",
]


def main():
    path = METRICS_DIR / "all_metrics.csv"
    if not path.exists():
        print(f"ERROR: {path} not found. Run evaluation scripts first.")
        sys.exit(1)

    df = pd.read_csv(path)
    for col in SUMMARY_COLS:
        if col not in df.columns:
            df[col] = np.nan if col not in ("Model", "Ticker") else None

    summary = df[SUMMARY_COLS].copy()
    summary = summary.sort_values(
        by=["Coverage_90pct", "RMSE"],
        ascending=[False, True],
        na_position="last",
    )

    out_path = METRICS_DIR / "phase2_summary.csv"
    summary.to_csv(out_path, index=False)
    print(f"Summary saved -> {out_path}\n")
    print("--- Phase 2 Summary (sorted by Coverage, then RMSE) ---")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
