"""
06_walkforward.py — expanding-window walk-forward validation (PyTorch SlimLSTM).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from torch.optim import Adam

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    CI_Z_90,
    METRICS_DIR,
    PROJECT_ROOT,
    QUARTER_DAYS,
    TICKERS,
    WALKFORWARD_INITIAL_TRAIN,
    WALKFORWARD_TICKERS,
    WINDOW,
)
from models.slim_lstm import SlimLSTM
from models.train_utils import DEVICE, make_loaders, train_model


def run_walkforward(ticker):
    print(f"\nWalk-forward: {ticker} (device={DEVICE})")
    df = pd.read_csv(
        PROJECT_ROOT / "data" / "raw" / f"{ticker}.csv",
        index_col="Date",
        parse_dates=True,
    )
    df.sort_index(inplace=True)
    prices = df["Close"].values.reshape(-1, 1)
    dates = df.index

    cutoff_idx = np.searchsorted(dates, pd.Timestamp(WALKFORWARD_INITIAL_TRAIN))
    window_end = cutoff_idx

    total_quarters = sum(
        1
        for _ in range(cutoff_idx, len(prices), QUARTER_DAYS)
        if _ + QUARTER_DAYS <= len(prices)
    )

    all_preds, all_actuals, all_dates = [], [], []
    all_lower, all_upper = [], []
    quarterly_rmse = []
    q = 0

    while window_end + QUARTER_DAYS <= len(prices):
        train_prices = prices[:window_end]

        scaler = MinMaxScaler((0, 1))
        scaler.fit(train_prices)
        scaled = scaler.transform(prices)

        X_tr, y_tr = [], []
        for i in range(WINDOW, window_end):
            X_tr.append(scaled[i - WINDOW : i, 0])
            y_tr.append(scaled[i, 0])
        X_tr = np.array(X_tr).reshape(-1, WINDOW, 1)
        y_tr = np.array(y_tr)

        model = SlimLSTM()
        model.to(DEVICE)
        optimizer = Adam(model.parameters(), lr=1e-3)
        train_loader, _ = make_loaders(X_tr, y_tr, X_tr[:1], y_tr[:1])
        train_model(
            model,
            train_loader,
            train_loader,
            optimizer,
            torch.nn.MSELoss(),
            epochs=15,
            patience=15,
            verbose=False,
        )

        n_steps = min(QUARTER_DAYS, len(prices) - window_end)
        seqs = np.array(
            [
                scaled[window_end + s - WINDOW : window_end + s, 0]
                for s in range(n_steps)
            ]
        )
        seqs_tensor = torch.tensor(seqs, dtype=torch.float32).unsqueeze(-1).to(DEVICE)

        model.eval()
        with torch.no_grad():
            preds_scaled = model(seqs_tensor).squeeze(-1).cpu().numpy()

        pred_indices = range(window_end, window_end + n_steps)
        quarter_preds = scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
        quarter_actuals = prices[window_end : window_end + n_steps, 0]
        quarter_dates = [dates[i] for i in pred_indices]

        if len(all_preds) >= 60:
            recent_errors = np.array(all_actuals[-60:]) - np.array(all_preds[-60:])
            sigma = np.std(recent_errors)
        else:
            sigma = (
                np.std(quarter_actuals - quarter_preds) if len(quarter_preds) else 0.0
            )

        q_rmse = np.sqrt(mean_squared_error(quarter_actuals, quarter_preds))
        quarterly_rmse.append(
            {
                "Quarter_End": str(quarter_dates[-1].date()),
                "RMSE": q_rmse,
            }
        )

        all_preds.extend(quarter_preds.tolist())
        all_actuals.extend(quarter_actuals.tolist())
        all_dates.extend(quarter_dates)
        all_lower.extend([p - CI_Z_90 * sigma for p in quarter_preds])
        all_upper.extend([p + CI_Z_90 * sigma for p in quarter_preds])

        print(
            f"  [{ticker}] Quarter {q + 1}/{total_quarters} | "
            f"window_end={dates[window_end].date()} | RMSE=${q_rmse:.2f}"
        )
        window_end += QUARTER_DAYS
        q += 1

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    result_df = pd.DataFrame(
        {
            "Date": all_dates,
            "Actual": all_actuals,
            "Predicted": all_preds,
            "Lower_90": all_lower,
            "Upper_90": all_upper,
        }
    )
    out_path = METRICS_DIR / f"{ticker}_walkforward.csv"
    result_df.to_csv(out_path, index=False)

    q_path = METRICS_DIR / f"{ticker}_walkforward_quarterly_rmse.csv"
    pd.DataFrame(quarterly_rmse).to_csv(q_path, index=False)
    print(f"  Saved: {out_path}")
    print(f"  Saved: {q_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Walk-forward expanding-window validation"
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=WALKFORWARD_TICKERS,
        help="Tickers to run (default: AAPL MSFT TSLA)",
    )
    parser.add_argument("--all", action="store_true", help="Run all 10 tickers")
    args = parser.parse_args()

    tickers = TICKERS if args.all else args.tickers
    for ticker in tickers:
        if ticker not in TICKERS:
            raise ValueError(f"Unknown ticker: {ticker}")
        run_walkforward(ticker)

    print("\nWalk-forward validation complete.")


if __name__ == "__main__":
    main()
