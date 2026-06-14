"""Shared configuration for the RNN stock prediction pipeline."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TICKERS = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "TSLA",
    "NVDA",
    "META",
    "AVGO",
    "ORCL",
    "AMD",
]

TICKER_NAMES = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "GOOGL": "Alphabet",
    "AMZN": "Amazon",
    "TSLA": "Tesla",
    "NVDA": "NVIDIA",
    "META": "Meta Platforms",
    "AVGO": "Broadcom",
    "ORCL": "Oracle",
    "AMD": "Advanced Micro Devices",
}

START_DATE = "2015-01-01"
END_DATE = "2024-12-31"

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROC_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"
MODELS_DIR = PROJECT_ROOT / "models"

WINDOW = 60
TRAIN_R = 0.70
VAL_R = 0.10

MODEL_CHOICES = {
    "vanilla_rnn": "VanillaRNN",
    "lstm": "StackedLSTM",
    "gru": "StackedGRU",
    "slim_lstm": "SlimLSTM",
    "quantile_lstm": "QuantileLSTM",
}

# Phase 2 defaults
MC_N_SAMPLES = 200
MC_BATCH_SIZE = 64
WALKFORWARD_TICKERS = ["AAPL", "MSFT", "TSLA"]
WALKFORWARD_INITIAL_TRAIN = "2020-12-31"
QUARTER_DAYS = 63
CI_Z_90 = 1.645

PHASE1_RNN_MODELS = ["vanilla_rnn", "lstm", "gru"]
PHASE2_MC_MODELS = ["vanilla_rnn", "slim_lstm"]
