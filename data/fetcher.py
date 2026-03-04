import pandas as pd
import yfinance as yf


def fetch_stock_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Fetch historical OHLCV stock data from Yahoo Finance.
    """

    df = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False
    )

    if df.empty:
        raise ValueError(f"No data found for ticker '{ticker}'.")

    # If columns are multi-index (happens in newer yfinance versions), flatten them
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()

    required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']

    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[required_cols]

    df = df.sort_values(by='Date')

    df = df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'])

    if df.empty:
        raise ValueError(f"All rows dropped after cleaning for '{ticker}'.")

    return df