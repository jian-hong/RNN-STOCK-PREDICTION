from .vanilla_rnn import VanillaRNN
from .lstm import StackedLSTM
from .gru import StackedGRU
from .slim_lstm import SlimLSTM
from .quantile_lstm import QuantileLSTM, pinball_loss
from .baseline_arima import run_arima_walkforward, run_arima_walkforward_with_ci
from .mc_dropout import mc_predict, mc_predict_usd, mc_predict_inverse
from .train_utils import (
    DEVICE,
    make_loaders,
    train_model,
    predict_array,
    predict_quantile_array,
    load_model_weights,
)

MODEL_REGISTRY = {
    "vanilla_rnn": VanillaRNN,
    "lstm": StackedLSTM,
    "stacked_lstm": StackedLSTM,
    "gru": StackedGRU,
    "stacked_gru": StackedGRU,
    "slim_lstm": SlimLSTM,
    "quantile_lstm": QuantileLSTM,
}

__all__ = [
    "VanillaRNN",
    "StackedLSTM",
    "StackedGRU",
    "SlimLSTM",
    "QuantileLSTM",
    "pinball_loss",
    "MODEL_REGISTRY",
    "DEVICE",
    "make_loaders",
    "train_model",
    "predict_array",
    "predict_quantile_array",
    "load_model_weights",
    "run_arima_walkforward",
    "run_arima_walkforward_with_ci",
    "mc_predict",
    "mc_predict_usd",
    "mc_predict_inverse",
]
