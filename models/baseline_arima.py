import warnings

import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from tqdm import tqdm

warnings.filterwarnings("ignore")


def run_arima_walkforward(train_series, test_series, order=(5, 1, 0)):
    """
    Walk-forward validation: retrain ARIMA on each test step.
    order=(p,d,q):
      p=5  autoregressive lags (last 5 days influence today)
      d=1  differencing order (make series stationary)
      q=0  moving average order (no MA term; simpler model)
    """
    history = list(train_series)
    predictions = []

    for t in tqdm(range(len(test_series)), desc="ARIMA walk-forward"):
        model = ARIMA(history, order=order)
        fitted = model.fit()
        yhat = fitted.forecast(steps=1)[0]
        predictions.append(yhat)
        history.append(test_series[t])

    return np.array(predictions)


def run_arima_walkforward_with_ci(
    train_series, test_series, order=(5, 1, 0), alpha=0.10
):
    """
    Walk-forward ARIMA with normal-theory confidence intervals per step.
    Returns predictions, lower, upper arrays (same length as test_series).
    """
    history = list(train_series)
    predictions, lowers, uppers = [], [], []

    for t in tqdm(range(len(test_series)), desc="ARIMA walk-forward + CI"):
        model = ARIMA(history, order=order)
        fitted = model.fit()
        forecast = fitted.get_forecast(steps=1)
        yhat = forecast.predicted_mean.iloc[0]
        conf = forecast.conf_int(alpha=alpha)
        lower = conf.iloc[0, 0]
        upper = conf.iloc[0, 1]

        predictions.append(yhat)
        lowers.append(lower)
        uppers.append(upper)
        history.append(test_series[t])

    return np.array(predictions), np.array(lowers), np.array(uppers)
