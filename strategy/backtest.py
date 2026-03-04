"""
backtest.py

Handles trading strategy simulation and performance calculations.
"""

import numpy as np


def long_only_strategy(predictions, daily_returns):
    """
    Simple strategy:
    - prediction = 1 → take market return
    - prediction = 0 → stay in cash (0 return)
    """

    predictions = np.array(predictions)
    daily_returns = np.array(daily_returns)

    if len(predictions) != len(daily_returns):
        raise ValueError("Predictions and returns must have same length")

    strategy_returns = predictions * daily_returns

    return strategy_returns


def buy_and_hold(daily_returns):
    """
    Buy & Hold strategy returns.
    """
    return np.array(daily_returns)


def cumulative_returns(returns):
    """
    Computes cumulative growth curve from returns.
    
    If starting capital = 1:
    1 * (1+r1) * (1+r2) * ...
    """

    returns = np.array(returns)

    cumulative_curve = np.cumprod(1 + returns)

    return cumulative_curve


def final_return(returns):
    """
    Returns total return multiplier.
    """

    curve = cumulative_returns(returns)
    return curve[-1]


def max_drawdown(returns):
    """
    Computes maximum drawdown.
    """

    curve = cumulative_returns(returns)
    peak = np.maximum.accumulate(curve)
    drawdown = (curve - peak) / peak

    return drawdown.min()