"""
05_visualise.py
Generate all 13 plots specified in the master plan Section 7.
"""

import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    FIGURES_DIR,
    METRICS_DIR,
    PROC_DIR,
    PROJECT_ROOT,
    TICKERS,
    TRAIN_R,
    VAL_R,
)
from models import (
    MODEL_REGISTRY,
    load_model_weights,
    predict_array,
    predict_quantile_array,
)
from models.quantile_lstm import QuantileLSTM

plt.style.use("seaborn-v0_8-whitegrid")
FIGSIZE = (12, 6)
FONTSIZE = 12
DPI = 300

RNN_MODELS = ["vanilla_rnn", "lstm", "gru"]
MODEL_LABELS = {
    "vanilla_rnn": "VanillaRNN",
    "lstm": "StackedLSTM",
    "gru": "StackedGRU",
    "arima": "ARIMA(5,1,0)",
}


def setup_plot(title, xlabel, ylabel):
    plt.figure(figsize=FIGSIZE)
    plt.title(title, fontsize=FONTSIZE + 2)
    plt.xlabel(xlabel, fontsize=FONTSIZE)
    plt.ylabel(ylabel, fontsize=FONTSIZE)
    plt.xticks(fontsize=FONTSIZE - 1)
    plt.yticks(fontsize=FONTSIZE - 1)


def load_raw_close(ticker):
    df = pd.read_csv(
        PROJECT_ROOT / "data" / "raw" / f"{ticker}.csv",
        index_col="Date",
        parse_dates=True,
    )
    df.sort_index(inplace=True)
    return df["Close"]


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
    n = len(df)
    test_start_idx = int(n * (TRAIN_R + VAL_R)) + 60
    return df.index[test_start_idx:]


def plot_01_normalised_prices():
    setup_plot("Normalised Closing Prices (Base=100)", "Date", "Normalised Price")
    for ticker in TICKERS:
        close = load_raw_close(ticker)
        norm = close / close.iloc[0] * 100
        plt.plot(norm.index, norm.values, label=ticker, linewidth=1.2)
    plt.legend(loc="upper left", fontsize=9, ncol=2)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "01_normalised_price_history.png", dpi=DPI)
    plt.close()


def plot_02_split_boundaries():
    for ticker in TICKERS:
        close = load_raw_close(ticker)
        splits = pd.read_csv(PROC_DIR / ticker / "splits.csv", index_col=0).squeeze(
            "columns"
        )
        train_end = pd.Timestamp(splits["train_end"])
        val_end = pd.Timestamp(splits["val_end"])

        setup_plot(f"{ticker} — Train/Val/Test Split", "Date", "Close Price (USD)")
        plt.plot(close.index, close.values, color="black", linewidth=0.8, alpha=0.7)
        plt.axvspan(close.index[0], train_end, alpha=0.2, color="blue", label="Train")
        plt.axvspan(train_end, val_end, alpha=0.2, color="orange", label="Validation")
        plt.axvspan(val_end, close.index[-1], alpha=0.2, color="green", label="Test")
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"02_{ticker}_split_boundaries.png", dpi=DPI)
        plt.close()


def plot_03_rolling_volatility():
    setup_plot("Rolling 30-Day Annualised Volatility", "Date", "Volatility")
    for ticker in TICKERS:
        close = load_raw_close(ticker)
        daily_return = close.pct_change()
        rolling_vol = daily_return.rolling(30).std() * np.sqrt(252)
        plt.plot(
            rolling_vol.index,
            rolling_vol.values,
            label=ticker,
            linewidth=1.0,
            alpha=0.8,
        )
    plt.legend(loc="upper right", fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "03_rolling_volatility.png", dpi=DPI)
    plt.close()


def plot_04_correlation_heatmap():
    closes = pd.DataFrame({t: load_raw_close(t) for t in TICKERS})
    corr = closes.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Close Price Correlation Heatmap", fontsize=FONTSIZE + 2)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "04_correlation_heatmap.png", dpi=DPI)
    plt.close()


def plot_05_loss_curves():
    for ticker in TICKERS:
        for model_key in RNN_MODELS:
            hist_path = METRICS_DIR / f"{ticker}_{model_key}_history.csv"
            if not hist_path.exists():
                continue
            hist = pd.read_csv(hist_path)
            setup_plot(
                f"{ticker} — {MODEL_LABELS[model_key]} Loss",
                "Epoch",
                "MSE Loss",
            )
            plt.plot(hist["loss"], label="Train Loss", color="blue")
            plt.plot(hist["val_loss"], label="Val Loss", color="orange")
            best_epoch = hist["val_loss"].idxmin()
            plt.axvline(
                best_epoch,
                color="red",
                linestyle="--",
                label=f"Best Epoch ({best_epoch})",
            )
            plt.legend()
            plt.tight_layout()
            plt.savefig(
                FIGURES_DIR / f"05_{ticker}_{model_key}_loss_curve.png", dpi=DPI
            )
            plt.close()


def plot_06_lr_schedule():
    for model_key in RNN_MODELS:
        lr_data = []
        for ticker in TICKERS:
            hist_path = METRICS_DIR / f"{ticker}_{model_key}_history.csv"
            if hist_path.exists() and "lr" in pd.read_csv(hist_path, nrows=1).columns:
                hist = pd.read_csv(hist_path)
                lr_data.append(hist["lr"].values)
        if not lr_data:
            continue
        avg_lr = np.mean(lr_data, axis=0)
        setup_plot(
            f"{MODEL_LABELS[model_key]} — Learning Rate Schedule",
            "Epoch",
            "Learning Rate",
        )
        plt.semilogy(avg_lr, marker="o", markersize=3)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"06_{model_key}_lr_schedule.png", dpi=DPI)
        plt.close()


def load_predictions_cache():
    cache_path = METRICS_DIR / "predictions_cache.pkl"
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    return {}


PREDICTIONS_CACHE = None


def get_predictions(ticker, model_key):
    global PREDICTIONS_CACHE
    if PREDICTIONS_CACHE is None:
        PREDICTIONS_CACHE = load_predictions_cache()

    scaler = load_scaler(ticker)
    y_test = np.load(PROC_DIR / ticker / "y_test.npy")
    dates = get_test_dates(ticker)
    y_true = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    if model_key == "arima":
        cached = PREDICTIONS_CACHE.get((ticker, "arima"), {})
        y_pred = cached.get("y_pred_usd")
        if y_pred is None:
            raise FileNotFoundError(
                "ARIMA predictions not cached. Run scripts/04_evaluate.py --all first."
            )
        min_len = min(len(y_true), len(y_pred))
        return dates[:min_len], y_true[:min_len], y_pred[:min_len]

    cached = PREDICTIONS_CACHE.get((ticker, model_key), {})
    y_pred_scaled = cached.get("y_pred")
    if y_pred_scaled is not None:
        y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        return dates[: len(y_true)], y_true, y_pred

    model_path = PROJECT_ROOT / "outputs" / f"best_{ticker}_{model_key}.pt"
    model = load_model_weights(MODEL_REGISTRY[model_key], model_path)
    X_test = np.load(PROC_DIR / ticker / "X_test.npy")
    y_pred_scaled = predict_array(model, X_test)
    y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    return dates[: len(y_true)], y_true, y_pred


def plot_07_predictions():
    for ticker in TICKERS:
        for model_key in RNN_MODELS:
            dates, y_true, y_pred = get_predictions(ticker, model_key)
            setup_plot(
                f"{ticker} — {MODEL_LABELS[model_key]} Test Predictions",
                "Date",
                "Price (USD)",
            )
            plt.plot(dates, y_true, label="Actual", color="blue", linewidth=1.2)
            plt.plot(
                dates,
                y_pred,
                label="Predicted",
                color="red",
                linestyle="--",
                linewidth=1.2,
            )
            plt.legend()
            plt.tight_layout()
            plt.savefig(
                FIGURES_DIR / f"07_{ticker}_{model_key}_prediction.png", dpi=DPI
            )
            plt.close()


def plot_08_scatter():
    for ticker in TICKERS:
        for model_key in RNN_MODELS:
            _, y_true, y_pred = get_predictions(ticker, model_key)
            setup_plot(
                f"{ticker} — {MODEL_LABELS[model_key]} Scatter",
                "Actual Price (USD)",
                "Predicted Price (USD)",
            )
            colors = np.linspace(0, 1, len(y_true))
            plt.scatter(y_true, y_pred, c=colors, cmap="viridis", alpha=0.6, s=15)
            lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
            plt.plot(lims, lims, "k--", linewidth=1, label="Perfect Prediction")
            plt.legend()
            plt.tight_layout()
            plt.savefig(FIGURES_DIR / f"08_{ticker}_{model_key}_scatter.png", dpi=DPI)
            plt.close()


def plot_09_residuals():
    for ticker in TICKERS:
        for model_key in RNN_MODELS:
            dates, y_true, y_pred = get_predictions(ticker, model_key)
            residuals = y_true - y_pred
            setup_plot(
                f"{ticker} — {MODEL_LABELS[model_key]} Residuals",
                "Date",
                "Residual (USD)",
            )
            plt.plot(dates, residuals, color="purple", linewidth=0.8)
            plt.axhline(0, color="black", linestyle="--", linewidth=0.8)
            plt.tight_layout()
            plt.savefig(FIGURES_DIR / f"09_{ticker}_{model_key}_residuals.png", dpi=DPI)
            plt.close()


def plot_10_rmse_comparison():
    metrics = pd.read_csv(METRICS_DIR / "all_metrics.csv")
    pivot = metrics.pivot(index="Ticker", columns="Model", values="RMSE")
    pivot.plot(kind="bar", figsize=FIGSIZE, width=0.8)
    plt.title("RMSE Comparison by Ticker and Model", fontsize=FONTSIZE + 2)
    plt.xlabel("Ticker", fontsize=FONTSIZE)
    plt.ylabel("RMSE ($)", fontsize=FONTSIZE)
    plt.xticks(rotation=45)
    plt.legend(title="Model", fontsize=9)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "10_rmse_comparison.png", dpi=DPI)
    plt.close()


def plot_11_mape_heatmap():
    metrics = pd.read_csv(METRICS_DIR / "all_metrics.csv")
    pivot = metrics.pivot(index="Model", columns="Ticker", values="MAPE")
    plt.figure(figsize=(12, 5))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="RdYlGn_r")
    plt.title("MAPE (%) Heatmap — Models × Tickers", fontsize=FONTSIZE + 2)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "11_mape_heatmap.png", dpi=DPI)
    plt.close()


def plot_12_directional_accuracy():
    metrics = pd.read_csv(METRICS_DIR / "all_metrics.csv")
    da_avg = metrics.groupby("Model")["DA"].mean().sort_values(ascending=False)
    setup_plot("Directional Accuracy (Avg Across Tickers)", "Model", "DA (%)")
    da_avg.plot(kind="bar", color="steelblue", edgecolor="black")
    plt.axhline(50, color="red", linestyle="--", label="Random Baseline (50%)")
    plt.xticks(rotation=30)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "12_directional_accuracy.png", dpi=DPI)
    plt.close()


def plot_13_tsla_all_models():
    ticker = "TSLA"
    setup_plot("TSLA — All Models Comparison (Test Period)", "Date", "Price (USD)")
    dates_ref = None
    y_true_ref = None

    for model_key in RNN_MODELS + ["arima"]:
        dates, y_true, y_pred = get_predictions(ticker, model_key)
        if dates_ref is None:
            dates_ref, y_true_ref = dates, y_true
            plt.plot(dates_ref, y_true_ref, label="Actual", color="black", linewidth=2)
        plt.plot(
            dates, y_pred, linestyle="--", linewidth=1.2, label=MODEL_LABELS[model_key]
        )

    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "13_TSLA_all_models_comparison.png", dpi=DPI)
    plt.close()


def plot_14_mc_dropout_ci():
    """MC Dropout 90% prediction intervals."""
    for ticker in TICKERS:
        for model_key in ["vanilla_rnn", "slim_lstm"]:
            mc_path = METRICS_DIR / f"{ticker}_{model_key}_mc_results.csv"
            if not mc_path.exists():
                continue
            df = pd.read_csv(mc_path, parse_dates=["Date"])
            label = "VanillaRNN" if model_key == "vanilla_rnn" else "SlimLSTM"
            setup_plot(
                f"{ticker} — {label} MC Dropout 90% Prediction Interval (N=200)",
                "Date",
                "Price (USD)",
            )
            plt.plot(
                df["Date"], df["Actual"], color="black", linewidth=1.2, label="Actual"
            )
            plt.plot(
                df["Date"],
                df["Mean_Pred"],
                color="blue",
                linewidth=1.2,
                label="MC Mean",
            )
            plt.fill_between(
                df["Date"],
                df["Lower_90"],
                df["Upper_90"],
                color="blue",
                alpha=0.2,
                label="90% CI",
            )
            plt.legend(fontsize=9)
            plt.tight_layout()
            plt.savefig(
                FIGURES_DIR / f"14_{ticker}_{model_key}_mc_dropout_CI.png", dpi=DPI
            )
            plt.close()


def plot_15_quantile_bands():
    """Quantile LSTM q05/q50/q95 bands."""
    for ticker in TICKERS:
        model_path = PROJECT_ROOT / "outputs" / f"best_{ticker}_quantile_lstm.pt"
        if not model_path.exists():
            continue
        scaler = load_scaler(ticker)
        X_test = np.load(PROC_DIR / ticker / "X_test.npy")
        y_test = np.load(PROC_DIR / ticker / "y_test.npy")
        model = load_model_weights(QuantileLSTM, model_path)
        q05_s, q50_s, q95_s = predict_quantile_array(model, X_test)

        def inv(arr):
            return scaler.inverse_transform(arr.reshape(-1, 1)).flatten()

        dates = get_test_dates(ticker)
        y_true = inv(y_test)
        q05, q50, q95 = inv(q05_s), inv(q50_s), inv(q95_s)

        setup_plot(f"{ticker} — QuantileLSTM Prediction Bands", "Date", "Price (USD)")
        plt.plot(dates, y_true, color="black", linewidth=1.2, label="Actual")
        plt.plot(dates, q50, color="blue", linewidth=1.2, label="q50 (median)")
        plt.plot(dates, q05, color="red", linestyle="--", linewidth=1.0, label="q05")
        plt.plot(dates, q95, color="green", linestyle="--", linewidth=1.0, label="q95")
        plt.fill_between(dates, q05, q95, alpha=0.1, color="blue")
        plt.legend(fontsize=9)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"15_{ticker}_quantile_bands.png", dpi=DPI)
        plt.close()


def plot_16_coverage_rate():
    """Coverage rate bar chart by model."""
    path = METRICS_DIR / "all_metrics.csv"
    if not path.exists() or "Coverage_90pct" not in pd.read_csv(path, nrows=1).columns:
        return
    metrics = pd.read_csv(path).dropna(subset=["Coverage_90pct"])
    if metrics.empty:
        return

    pivot = metrics.pivot(index="Ticker", columns="Model", values="Coverage_90pct")
    pivot.plot(kind="bar", figsize=(14, 6), width=0.85)
    plt.title("90% Interval Coverage Rate by Ticker and Model", fontsize=FONTSIZE + 2)
    plt.xlabel("Ticker", fontsize=FONTSIZE)
    plt.ylabel("Coverage (%)", fontsize=FONTSIZE)
    plt.axhline(90, color="red", linestyle="--", label="Target 90%")
    plt.xticks(rotation=45)
    plt.legend(title="Model", fontsize=8, bbox_to_anchor=(1.02, 1))
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "16_coverage_rate.png", dpi=DPI)
    plt.close()


def plot_17_width_vs_rmse():
    """Interval width vs RMSE scatter."""
    path = METRICS_DIR / "all_metrics.csv"
    if not path.exists():
        return
    metrics = pd.read_csv(path).dropna(subset=["Avg_Width_USD", "RMSE"])
    if metrics.empty:
        return

    setup_plot("Interval Width vs RMSE", "Avg Interval Width ($)", "RMSE ($)")
    for model in metrics["Model"].unique():
        sub = metrics[metrics["Model"] == model]
        plt.scatter(sub["Avg_Width_USD"], sub["RMSE"], label=model, alpha=0.7, s=40)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "17_width_vs_rmse.png", dpi=DPI)
    plt.close()


def plot_18_walkforward_quarterly_rmse():
    """Plot 18 — walk-forward quarterly RMSE with regime-shift annotations."""
    walkforward_tickers = ["AAPL", "MSFT", "TSLA"]
    colors = {"AAPL": "blue", "MSFT": "orange", "TSLA": "green"}

    series = {}
    for ticker in walkforward_tickers:
        q_path = METRICS_DIR / f"{ticker}_walkforward_quarterly_rmse.csv"
        if not q_path.exists():
            print(f"  SKIP plot 18: {q_path.name} not found")
            return
        df = pd.read_csv(q_path, parse_dates=["Quarter_End"])
        series[ticker] = df

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.set_title(
        "Walk-Forward RMSE by Quarter — Regime Shift Visualised", fontsize=FONTSIZE + 2
    )
    ax.set_xlabel("Quarter End", fontsize=FONTSIZE)
    ax.set_ylabel("RMSE ($)", fontsize=FONTSIZE)

    ax.axvspan(
        pd.Timestamp("2021-10-01"),
        pd.Timestamp("2022-03-31"),
        color="mistyrose",
        alpha=0.5,
        label="Rate hike begins",
    )
    ax.axvspan(
        pd.Timestamp("2023-07-01"),
        pd.Timestamp("2024-12-31"),
        color="lightyellow",
        alpha=0.6,
        label="AI rally",
    )

    for ticker in walkforward_tickers:
        df = series[ticker]
        ax.plot(
            df["Quarter_End"],
            df["RMSE"],
            marker="o",
            markersize=4,
            label=ticker,
            color=colors[ticker],
            linewidth=1.5,
        )

    ax.tick_params(axis="both", labelsize=FONTSIZE - 1)
    ax.legend(fontsize=FONTSIZE - 1)
    fig.autofmt_xdate()
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "18_walkforward_quarterly_rmse.png", dpi=DPI)
    plt.close(fig)


def plot_19_tsla_mc_annotated():
    """TSLA MC Dropout with news event annotations."""
    mc_path = METRICS_DIR / "TSLA_vanilla_rnn_mc_results.csv"
    if not mc_path.exists():
        mc_path = METRICS_DIR / "TSLA_slim_lstm_mc_results.csv"
    if not mc_path.exists():
        return

    df = pd.read_csv(mc_path, parse_dates=["Date"])
    setup_plot("TSLA — MC Dropout 90% CI with Key Events", "Date", "Price (USD)")
    plt.plot(df["Date"], df["Actual"], color="black", linewidth=1.5, label="Actual")
    plt.plot(df["Date"], df["Mean_Pred"], color="blue", linewidth=1.2, label="MC Mean")
    plt.fill_between(
        df["Date"],
        df["Lower_90"],
        df["Upper_90"],
        color="blue",
        alpha=0.2,
        label="90% CI",
    )

    events = [
        ("2023-04-20", "Q1 miss"),
        ("2024-01-03", "Delivery miss"),
        ("2024-10-10", "Robotaxi"),
        ("2024-11-06", "Post-election"),
    ]
    for date_str, label in events:
        evt = pd.Timestamp(date_str)
        if df["Date"].min() <= evt <= df["Date"].max():
            plt.axvline(evt, color="gray", linestyle=":", alpha=0.8)
            plt.text(
                evt, df["Actual"].max() * 0.95, label, rotation=90, fontsize=8, va="top"
            )

    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "19_TSLA_mc_annotated.png", dpi=DPI)
    plt.close()


def run_phase1_plots():
    print("Generating EDA plots (1-4)...")
    plot_01_normalised_prices()
    plot_02_split_boundaries()
    plot_03_rolling_volatility()
    plot_04_correlation_heatmap()

    print("Generating training plots (5-6)...")
    plot_05_loss_curves()
    plot_06_lr_schedule()

    print("Generating prediction plots (7-9)...")
    plot_07_predictions()
    plot_08_scatter()
    plot_09_residuals()

    print("Generating comparison plots (10-13)...")
    plot_10_rmse_comparison()
    plot_11_mape_heatmap()
    plot_12_directional_accuracy()
    plot_13_tsla_all_models()


def run_phase2_plots():
    print("Generating Phase 2 plots (14-19)...")
    plot_14_mc_dropout_ci()
    plot_15_quantile_bands()
    plot_16_coverage_rate()
    plot_17_width_vs_rmse()
    plot_18_walkforward_quarterly_rmse()
    plot_19_tsla_mc_annotated()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate visualisation plots")
    parser.add_argument(
        "--phase1", action="store_true", help="Phase 1 plots only (1-13)"
    )
    parser.add_argument(
        "--phase2", action="store_true", help="Phase 2 plots only (14-19)"
    )
    args = parser.parse_args()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if args.phase2:
        run_phase2_plots()
    elif args.phase1:
        run_phase1_plots()
    else:
        run_phase1_plots()
        run_phase2_plots()

    n_figures = len(list(FIGURES_DIR.glob("*.png")))
    print(f"\nAll visualisations complete. {n_figures} figures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
