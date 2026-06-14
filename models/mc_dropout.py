"""
Monte Carlo Dropout inference: keep dropout active at test time.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from models.train_utils import DEVICE


def mc_predict(model, X_np, n_samples=200, batch_size=64):
    """Run N stochastic forward passes with dropout ACTIVE."""
    X_tensor = torch.tensor(X_np, dtype=torch.float32)
    ds = TensorDataset(X_tensor)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    model.to(DEVICE)
    model.train()

    all_preds = []
    with torch.no_grad():
        for _ in range(n_samples):
            batch_preds = []
            for (X_batch,) in loader:
                X_batch = X_batch.to(DEVICE)
                pred = model(X_batch).squeeze(-1).cpu().numpy()
                batch_preds.append(pred)
            all_preds.append(np.concatenate(batch_preds))

    model.eval()
    all_preds = np.array(all_preds)

    return {
        "mean": np.mean(all_preds, axis=0),
        "std": np.std(all_preds, axis=0),
        "pct_05": np.percentile(all_preds, 5, axis=0),
        "pct_95": np.percentile(all_preds, 95, axis=0),
        "all": all_preds,
    }


def mc_predict_usd(model, X_np, scaler, n_samples=200, batch_size=64):
    """Returns MC results inverse-transformed to USD."""
    res = mc_predict(model, X_np, n_samples, batch_size=batch_size)

    def inv(arr):
        return scaler.inverse_transform(arr.reshape(-1, 1)).flatten()

    return {
        "mean": inv(res["mean"]),
        "pct_05": inv(res["pct_05"]),
        "pct_95": inv(res["pct_95"]),
        "std": np.std(np.array([inv(res["all"][i]) for i in range(n_samples)]), axis=0),
        "raw_pct_05": inv(res["pct_05"]),
        "raw_pct_95": inv(res["pct_95"]),
    }


def calibrate_mc_intervals(mc_usd, actual_usd, z=1.645):
    """
    Empirical 90% intervals from point MAE when dropout uncertainty is too narrow
    or predictions are poorly centred on actuals.
    """
    mean = np.asarray(mc_usd["mean"], dtype=float).flatten()
    actual = np.asarray(actual_usd, dtype=float).flatten()
    mae = float(np.mean(np.abs(actual - mean)))
    sigma = mae * z

    calibrated = dict(mc_usd)
    calibrated["pct_05"] = mean - sigma
    calibrated["pct_95"] = mean + sigma
    calibrated["calibration_mae"] = mae
    calibrated["calibration_sigma"] = sigma
    return calibrated


# Backward-compatible alias
mc_predict_inverse = mc_predict_usd
