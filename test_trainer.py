import pandas as pd
import numpy as np

from data.fetcher import fetch_stock_data
from features.engineer import build_features, FEATURE_COLS
from models.trainer import walk_forward_train
from models.evaluator import evaluate_model, sharpe_ratio

from strategy.backtest import (
    long_only_strategy,
    buy_and_hold,
    cumulative_returns,
    final_return,
    max_drawdown
)


# ==========================
# Load Data
# ==========================

ticker = "AAPL"

df_raw = fetch_stock_data(ticker, "2015-01-01", "2023-12-31")
df_features = build_features(df_raw)

results = walk_forward_train(df_features, FEATURE_COLS, initial_train_years=3)

y_true = results["y_true"]
lr_preds = results["lr_preds"]
xgb_preds = results["xgb_preds"]
daily_returns = results["daily_returns"]

assert len(y_true) == len(lr_preds) == len(xgb_preds) == len(daily_returns)

print("\nWalk-forward validation successful.")
print("Total Predictions:", len(y_true))


# ==========================
# Backtest Strategies
# ==========================

lr_strategy_returns = long_only_strategy(lr_preds, daily_returns)
xgb_strategy_returns = long_only_strategy(xgb_preds, daily_returns)
bh_strategy_returns = buy_and_hold(daily_returns)

# Final returns using backtest module
lr_final = final_return(lr_strategy_returns)
xgb_final = final_return(xgb_strategy_returns)
bh_final = final_return(bh_strategy_returns)

# Sharpe ratios
lr_sharpe = sharpe_ratio(lr_strategy_returns)
xgb_sharpe = sharpe_ratio(xgb_strategy_returns)
bh_sharpe = sharpe_ratio(bh_strategy_returns)

# Max drawdown
lr_dd = max_drawdown(lr_strategy_returns)
xgb_dd = max_drawdown(xgb_strategy_returns)
bh_dd = max_drawdown(bh_strategy_returns)


# ==========================
# Evaluate Classification Metrics
# ==========================

lr_eval = evaluate_model(y_true, lr_preds, daily_returns)
xgb_eval = evaluate_model(y_true, xgb_preds, daily_returns)


# ==========================
# Build Metrics Table
# ==========================

metrics = pd.DataFrame([
    {
        "Model": "Logistic Regression",
        "Accuracy": round(lr_eval["classification"]["accuracy"], 4),
        "F1": round(lr_eval["classification"]["f1"], 4),
        "Sharpe": round(lr_sharpe, 4),
        "Max_Drawdown": round(lr_dd, 4),
        "Final_Return": round(lr_final, 4)
    },
    {
        "Model": "XGBoost",
        "Accuracy": round(xgb_eval["classification"]["accuracy"], 4),
        "F1": round(xgb_eval["classification"]["f1"], 4),
        "Sharpe": round(xgb_sharpe, 4),
        "Max_Drawdown": round(xgb_dd, 4),
        "Final_Return": round(xgb_final, 4)
    },
    {
        "Model": "Buy & Hold",
        "Accuracy": None,
        "F1": None,
        "Sharpe": round(bh_sharpe, 4),
        "Max_Drawdown": round(bh_dd, 4),
        "Final_Return": round(bh_final, 4)
    }
])

print("\n================ MODEL PERFORMANCE ================\n")
print(metrics.to_string(index=False))


# ==========================
# Cumulative Logic Sanity Test
# ==========================

print("\n================ CUMULATIVE SANITY TEST ================")

test_returns = np.array([0.1, -0.1])
test_curve = cumulative_returns(test_returns)

print("Expected: [1.1, 0.99]")
print("Actual:  ", np.round(test_curve, 4))

assert np.isclose(test_curve[-1], 0.99, atol=1e-6)

print("Cumulative return calculation verified.")


# ==========================
# Final Sanity Checks
# ==========================

print("\n================ NUMERICAL SANITY CHECKS ================")

assert 0 <= lr_eval["classification"]["accuracy"] <= 1
assert 0 <= xgb_eval["classification"]["accuracy"] <= 1
assert bh_final > 0

print("All metrics within reasonable bounds.")