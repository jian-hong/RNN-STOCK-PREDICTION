# RNN Stock Prediction — Progress Summary (for Claude review)

**Status: PIPELINE COMPLETE** — all steps finished on CPU, June 10 2026.

---

## Timing (this machine, CPU-only)

| Step | Duration | Notes |
|------|----------|-------|
| Data download (10 tickers) | ~1 min | yfinance, 2515 rows each |
| Preprocessing | ~20 sec | 70/10/20 split, 60-day windows |
| Training (30 RNN runs) | **~26 min** | ~52 sec/model avg, EarlyStopping ~20–80 epochs |
| Evaluation (30 RNN + 10 ARIMA) | **~11 min** | ARIMA ~50 sec/ticker at ~10 steps/sec |
| Visualisation (137 PNGs) | **~2 min** | All 13 plot types |
| **Total end-to-end** | **~40 min** | Well within 30–60 min acceptable range |

---

## What was built

Full modular project at `rnn-stock-prediction/` per master plan:

- `scripts/01_download_data.py` → `02_preprocess.py` → `03_train.py` → `04_evaluate.py` → `05_visualise.py`
- `scripts/00_verify.py`, `scripts/run_pipeline.py`, `scripts/config.py`
- `models/vanilla_rnn.py`, `lstm.py`, `gru.py`, `baseline_arima.py`
- Python 3.12 venv at `.venv/` (TensorFlow requires 3.12; system had 3.14)

---

## Validation (Appendix A)

- **90/90 data checks PASSED**
- **30/30** trained `.keras` models saved
- **40/40** metric rows (4 models × 10 tickers)
- **137** figures in `outputs/figures/`
- Model params: VanillaRNN 7,873 | StackedLSTM 119,361 | StackedGRU 90,433

---

## Benchmark Results (aggregated across 10 tickers)

| Model | RMSE ($) | MAE ($) | MAPE (%) | R² | DA (%) |
|-------|----------|---------|----------|-----|--------|
| **VanillaRNN** | **18.72** | **14.73** | **7.29** | **0.76** | 49.6 |
| ARIMA(5,1,0) | 33.29 | 26.76 | 14.30 | 0.35 | 49.6 |
| StackedGRU | 41.82 | 37.62 | 17.32 | 0.20 | 49.7 |
| StackedLSTM | 51.43 | 45.78 | 19.30 | -0.40 | 49.7 |

**Note for report discussion:** LSTM/GRU underperformed Vanilla RNN unexpectedly. All models show DA ≈ 50% (random baseline). LSTM negative R² suggests severe generalisation failure — likely overfitting (112K params vs ~1700 training windows) and/or BatchNorm + validation distribution shift. Worth discussing in critical analysis section.

Full per-ticker metrics: `outputs/metrics/all_metrics.csv`

---

## Problems encountered

1. **Python 3.14 incompatible with TensorFlow** — installed Python 3.12 via winget, recreated venv.
2. **GPU not used on native Windows** — TensorFlow ≥2.11 logs: *"GPU support is not available on native Windows"*. RTX 5060 laptop will **not** accelerate unless using **WSL2 + CUDA** or **tensorflow-directml-plugin**.
3. **Vanilla RNN param count** — master plan cites ~26K; Keras 3 actual count is 7,873 (architecture matches spec exactly).
4. **LSTM val_loss spikes during training** — observed val_loss >> train_loss (e.g. AAPL LSTM val_loss ~12 vs train ~0.02). Models still saved via EarlyStopping; results reflect this.

---

## Key output files

```
outputs/metrics/all_metrics.csv          # all 40 rows
outputs/metrics/aggregated_metrics.csv   # model averages
outputs/metrics/predictions_cache.pkl    # for ARIMA plot reuse
outputs/figures/                         # 137 PNGs (plots 01–13)
outputs/best_{TICKER}_{model}.keras      # 30 trained models
data/raw/*.csv                           # 10 tickers, 2515 rows
data/processed/{TICKER}/                 # X_train (1700,60,1), etc.
```

---

## What to do next (for report / Claude review)

1. **Review `all_metrics.csv`** — fill report Table 4.3 with actual numbers.
2. **Insert figures** from `outputs/figures/` into report (especially 07, 10, 11, 12, 13).
3. **Discuss LSTM underperformance** — overfitting, regime shift (2022–2024 rate-hike era in test set).
4. **Discuss DA ≈ 50%** — models predict level/trend, not direction (EMH argument).
5. **Optional:** re-run on RTX 5060 via WSL2 for faster iteration (not required — results already complete).
6. **Optional:** write `notebooks/exploration.ipynb` for EDA (plots 1–4 already generated).

---

## Re-run commands

```powershell
cd rnn-stock-prediction
.\.venv\Scripts\Activate.ps1
python scripts/run_pipeline.py --step all   # full pipeline
python scripts/00_verify.py               # validation only
```
