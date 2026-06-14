# Phase 2 — Range Forecasting & Model Hardening

## What's new

| Component | File | Purpose |
|-----------|------|---------|
| SlimLSTM | `models/slim_lstm.py` | LayerNorm, ~28K params — fixes Phase 1 LSTM regime shift |
| QuantileLSTM | `models/quantile_lstm.py` | q05/q50/q95 via pinball loss |
| MC Dropout | `models/mc_dropout.py` | Epistemic uncertainty at inference |
| ARIMA + CI | `models/baseline_arima.py` | `run_arima_walkforward_with_ci()` |
| Train Slim | `scripts/03_train.py --all-slim` | 10 tickers |
| Train Quantile | `scripts/03b_train_quantile.py` | 10 tickers |
| Eval MC | `scripts/04b_mc_evaluate.py --all` | 20 MC runs |
| Eval Quantile | `scripts/04c_evaluate_quantile.py` | Coverage + width |
| Eval ARIMA CI | `scripts/04d_evaluate_arima_ci.py` | Statistical intervals |
| Eval Slim | `scripts/04e_evaluate_slim.py` | Point metrics |
| Walk-forward | `scripts/06_walkforward.py` | Quarterly expanding window |
| Summary | `scripts/07_summary.py` | `phase2_summary.csv` |
| GPU guide | `docs/GPU_SETUP.md` | WSL2 CUDA + AMD notes |

## Run order (WSL2 + PyTorch cu121, RTX 4050/5060)

```bash
cd ~/projects/rnn-stock-prediction
source .venv/bin/activate

# Install PyTorch (see docs/GPU_SETUP.md)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

python scripts/check_gpu.py
python scripts/run_phase2.py --step all

# Or step-by-step:
python scripts/run_phase2.py --step slim
python scripts/run_phase2.py --step quantile
python scripts/run_phase2.py --step mc
```

Models save as `outputs/best_{ticker}_{model}.pt` (PyTorch state dicts).

## New metrics

- **Coverage_90pct** — % of actuals inside [lower, upper]; target ~90%
- **Avg_Width_USD** — mean interval width in dollars

## New plots (14–19)

Saved to `outputs/figures/` — see master plan Part H.

## CPU fallback

Phase 2 on CPU is slower (~2–3× Phase 1 for training; MC Dropout ~8+ hours).
Use RTX 5060 + WSL2 for MC Dropout in particular.
