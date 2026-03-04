from data.fetcher import fetch_stock_data
from features.engineer import build_features, FEATURE_COLS
import pandas as pd
import numpy as np

ticker = "AAPL"

# Fetch raw data
df_raw = fetch_stock_data(ticker, "2018-01-01", "2023-12-31")
print("Raw rows:", len(df_raw))

# Build features
df_features = build_features(df_raw)

print("Feature rows:", len(df_features))
print("Dropped rows:", len(df_raw) - len(df_features))

print("\n--- FIRST 5 ROWS ---")
print(df_features.head())

print("\n--- LAST 5 ROWS ---")
print(df_features.tail())

# =============================
# UNIT CHECK 1 — No NaNs
# =============================
assert df_features.isna().sum().sum() == 0, "NaNs detected in features!"

# =============================
# UNIT CHECK 2 — Correct feature count
# =============================
assert len(FEATURE_COLS) == 11, "Unexpected number of features!"

# =============================
# UNIT CHECK 3 — Date strictly increasing
# =============================
assert df_features["Date"].is_monotonic_increasing, "Dates not sorted properly!"

# =============================
# UNIT CHECK 4 — No future leakage in features
# =============================
# Pick random sample rows and verify that
# features only depend on current and past data

sample_indices = np.random.choice(df_features.index[20:-5], size=5, replace=False)

for idx in sample_indices:
    row = df_features.loc[idx]
    
    # Ensure MA5 equals rolling mean from raw data manually computed
    raw_subset = df_raw[df_raw["Date"] <= row["Date"]].tail(5)
    manual_ma5 = raw_subset["Close"].mean()
    
    assert abs(manual_ma5 - row["MA5"]) < 1e-6, "Future leakage detected in MA5!"

print("\nFeature leakage check passed (MA5 validated).")

# =============================
# UNIT CHECK 5 — Target correctness
# =============================
for idx in sample_indices:
    current_close = df_raw.loc[df_raw["Date"] == df_features.loc[idx, "Date"], "Close"].values[0]
    
    # Next day's close
    raw_index = df_raw.index[df_raw["Date"] == df_features.loc[idx, "Date"]][0]
    next_close = df_raw.loc[raw_index + 1, "Close"]
    
    expected_target = int(next_close > current_close)
    
    assert expected_target == df_features.loc[idx, "Target"], "Target misalignment!"

print("Target alignment verified.")

print("\nAll unit checks passed successfully.")