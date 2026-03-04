import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def classification_metrics(y_true, y_pred):
    """
    Returns basic classification metrics.
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred)
    }


def strategy_returns(y_true, y_pred, daily_returns):
    """
    Simulate simple long-only strategy:
    - If prediction == 1 → take market return
    - If prediction == 0 → stay in cash (0 return)
    """

    y_pred = np.array(y_pred)
    daily_returns = np.array(daily_returns)

    strat_returns = y_pred * daily_returns

    return strat_returns


def cumulative_returns(returns):
    """
    Computes cumulative growth curve.
    """
    returns = np.array(returns)
    return np.cumprod(1 + returns)


def sharpe_ratio(returns, risk_free_rate=0.0):
    """
    Computes annualized Sharpe ratio.
    Assumes daily returns (252 trading days).
    """
    returns = np.array(returns)

    excess_returns = returns - risk_free_rate / 252

    mean = np.mean(excess_returns)
    std = np.std(excess_returns)

    if std == 0:
        return 0

    return np.sqrt(252) * mean / std


def evaluate_model(y_true, y_pred, daily_returns):
    """
    Full evaluation bundle.
    """

    cls_metrics = classification_metrics(y_true, y_pred)

    strat_ret = strategy_returns(y_true, y_pred, daily_returns)

    cum_ret = cumulative_returns(strat_ret)

    sharpe = sharpe_ratio(strat_ret)

    return {
        "classification": cls_metrics,
        "sharpe_ratio": sharpe,
        "final_cumulative_return": cum_ret[-1],
        "strategy_returns": strat_ret,
        "equity_curve": cum_ret
    }