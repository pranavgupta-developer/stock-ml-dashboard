"""
engineer.py

Builds ML features for stock trend classification.
All features must use only past data (no future leakage).
"""

import pandas as pd
from ta.momentum import RSIIndicator

# Feature column list (used later in training)
FEATURE_COLS = [
    'MA5', 'MA20', 'MA_Diff',
    'Close_to_MA5', 'Close_to_MA20',
    'RSI', 'Daily_Return',
    'Volatility_20',
    'Lag1', 'Lag2', 'Lag3'
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build trend, momentum, volatility, and lag features.

    Assumes input df contains:
    ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    """

    df = df.copy()

    # --- Target (next day direction) ---
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

    # --- Trend Features ---
    df['MA5'] = df['Close'].rolling(5).mean()
    df['MA20'] = df['Close'].rolling(20).mean()
    df['MA_Diff'] = df['MA5'] - df['MA20']
    df['Close_to_MA5'] = df['Close'] / df['MA5']
    df['Close_to_MA20'] = df['Close'] / df['MA20']

    # --- Momentum Feature (RSI 14) ---
    rsi_indicator = RSIIndicator(close=df['Close'], window=14)
    df['RSI'] = rsi_indicator.rsi()

    # --- Volatility ---
    df['Daily_Return'] = df['Close'].pct_change()
    df['Volatility_20'] = df['Daily_Return'].rolling(20).std()

    # --- Lagged Returns (purely backward-looking) ---
    df['Lag1'] = df['Daily_Return'].shift(1)
    df['Lag2'] = df['Daily_Return'].shift(2)
    df['Lag3'] = df['Daily_Return'].shift(3)

    # --- Drop last row (target has NaN due to shift(-1)) ---
    df = df.iloc[:-1]

    # --- Drop all rows with NaNs (from rolling windows) ---
    df = df.dropna()

    # Final validation
    if df.isna().sum().sum() != 0:
        raise ValueError("NaN values still present after feature engineering.")

    return df