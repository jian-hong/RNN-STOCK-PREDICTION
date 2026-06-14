# RNN Stock Price Prediction (KIE4031)

Modular pipeline for predicting daily stock close prices using Vanilla RNN, Stacked LSTM, Stacked GRU, and an ARIMA(5,1,0) baseline.

## Project Structure

```
rnn-stock-prediction/
├── data/raw/              # Downloaded CSVs (yfinance)
├── data/processed/        # Scaled windowed .npy arrays + scalers
├── models/                # Model architectures
├── scripts/               # Pipeline scripts (01–05 + utilities)
├── outputs/figures/       # All PNG plots
├── outputs/metrics/       # metrics CSV + training histories
├── requirements.txt
└── .venv/                 # Python virtual environment
```

## Setup (Windows)

**Requires Python 3.12** (TensorFlow does not support 3.14 yet).

```powershell
cd rnn-stock-prediction
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run Pipeline

Run steps individually (recommended for debugging):

```powershell
python scripts/01_download_data.py
python scripts/02_preprocess.py
python scripts/03_train.py --ticker AAPL --model lstm    # single model
python scripts/03_train.py --all                         # all 30 RNN runs
python scripts/04_evaluate.py --all
python scripts/05_visualise.py
python scripts/00_verify.py
```

Or run the full pipeline:

```powershell
python scripts/run_pipeline.py --step all
```

## Models

| Model | Script key | ~Params |
|---|---|---|
| Vanilla RNN | `vanilla_rnn` | ~26K |
| Stacked LSTM | `lstm` | ~112K |
| Stacked GRU | `gru` | ~84K |
| ARIMA(5,1,0) | `arima` | statistical baseline |

## Outputs

- `outputs/metrics/all_metrics.csv` — per-ticker metrics (RMSE, MAE, MAPE, R², DA)
- `outputs/metrics/aggregated_metrics.csv` — model averages
- `outputs/figures/` — 13 plot types per master plan
