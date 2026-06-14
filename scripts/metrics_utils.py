"""Shared evaluation metrics utilities."""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_from_usd(y_true, y_pred):
    """Compute metrics directly in USD price space."""
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    r2 = r2_score(y_true, y_pred)

    actual_dir = np.sign(np.diff(y_true))
    predicted_dir = np.sign(np.diff(y_pred))
    da = np.mean(actual_dir == predicted_dir) * 100

    return {
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape,
        "R2": r2,
        "DA": da,
    }


def evaluate_model(y_true_scaled, y_pred_scaled, scaler):
    """
    y_true_scaled, y_pred_scaled: normalised [0,1] values
    Returns metrics in ORIGINAL price space (USD)
    """
    y_true = scaler.inverse_transform(y_true_scaled.reshape(-1, 1)).flatten()
    y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    return evaluate_from_usd(y_true, y_pred)


def coverage_rate(y_true, lower, upper):
    """
    Fraction of actual prices within [lower, upper].
    Target for 90% CI: ~90%.
    """
    y_true = np.asarray(y_true).flatten()
    lower = np.asarray(lower).flatten()
    upper = np.asarray(upper).flatten()
    in_interval = np.sum((y_true >= lower) & (y_true <= upper))
    return (in_interval / len(y_true)) * 100


def interval_width(lower, upper):
    """Average prediction interval width in USD."""
    return float(np.mean(np.asarray(upper).flatten() - np.asarray(lower).flatten()))


def evaluate_with_interval(y_true, y_pred, lower, upper):
    """Point metrics plus interval calibration metrics."""
    metrics = evaluate_from_usd(y_true, y_pred)
    metrics["Coverage_90pct"] = coverage_rate(y_true, lower, upper)
    metrics["Avg_Width_USD"] = interval_width(lower, upper)
    return metrics
