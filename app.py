import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go

from data.fetcher import fetch_stock_data
from features.engineer import build_features, FEATURE_COLS
from models.trainer import walk_forward_train
from models.evaluator import evaluate_model
from strategy.backtest import (
    long_only_strategy,
    buy_and_hold,
    cumulative_returns
)

from sklearn.metrics import confusion_matrix, roc_curve, auc

# =====================================
# Page Config
# =====================================

st.set_page_config(page_title="Stock ML Dashboard", layout="wide")
st.title("Stock Market ML Research Dashboard")

# =====================================
# Sidebar
# =====================================

st.sidebar.header("Configuration")

ticker = st.sidebar.text_input("Ticker", value="AAPL")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2015-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2023-12-31"))
initial_years = st.sidebar.slider("Initial Train Years", 2, 5, 3)

run_button = st.sidebar.button("Run Model")


# =====================================
# Tabs
# =====================================

tabs = st.tabs(["Overview", "Charts", "Model Results", "Strategy"])


# =====================================
# Run Pipeline
# =====================================

if run_button:

    df_raw = fetch_stock_data(ticker, str(start_date), str(end_date))
    df_features = build_features(df_raw)

    results = walk_forward_train(df_features, FEATURE_COLS, initial_train_years=initial_years)

    y_true = np.array(results["y_true"])
    lr_preds = np.array(results["lr_preds"])
    xgb_preds = np.array(results["xgb_preds"])
    daily_returns = np.array(results["daily_returns"])

    lr_eval = evaluate_model(y_true, lr_preds, daily_returns)
    xgb_eval = evaluate_model(y_true, xgb_preds, daily_returns)

    # Strategy returns
    lr_strategy = long_only_strategy(lr_preds, daily_returns)
    bh_strategy = buy_and_hold(daily_returns)

    lr_equity = cumulative_returns(lr_strategy)
    bh_equity = cumulative_returns(bh_strategy)

    # =====================================
    # Overview Tab
    # =====================================

    with tabs[0]:

        st.subheader("Basic Dataset Info")

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Predictions", len(y_true))
        col2.metric("LR Accuracy", round(lr_eval["classification"]["accuracy"], 4))
        col3.metric("XGB Accuracy", round(xgb_eval["classification"]["accuracy"], 4))

        st.write("Final Returns")
        st.write({
            "Logistic Regression": round(lr_eval["final_cumulative_return"], 3),
            "XGBoost": round(xgb_eval["final_cumulative_return"], 3)
        })


    # =====================================
    # Charts Tab
    # =====================================

    with tabs[1]:

        st.subheader("Price Chart")

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df_features.index,
            y=df_features["Close"],
            mode="lines",
            name="Close Price"
        ))

        fig.update_layout(title=f"{ticker} Price Chart")

        st.plotly_chart(fig, use_container_width=True)


    # =====================================
    # Model Results Tab
    # =====================================

    with tabs[2]:

        st.subheader("Confusion Matrix (Logistic Regression)")

        cm = confusion_matrix(y_true, lr_preds)

        cm_fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=["Pred Down", "Pred Up"],
            y=["Actual Down", "Actual Up"]
        ))

        st.plotly_chart(cm_fig, use_container_width=True)

        st.subheader("ROC Curve (Logistic Regression)")

        # Fake probabilities if not returning them yet
        # Replace with real proba output later
        fpr, tpr, _ = roc_curve(y_true, lr_preds)
        roc_auc = auc(fpr, tpr)

        roc_fig = go.Figure()
        roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"AUC={roc_auc:.2f}"))
        roc_fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Random"))

        roc_fig.update_layout(title="ROC Curve")

        st.plotly_chart(roc_fig, use_container_width=True)


    # =====================================
    # Strategy Tab
    # =====================================

    with tabs[3]:

        st.subheader("Equity Curve")

        eq_fig = go.Figure()

        eq_fig.add_trace(go.Scatter(
            y=lr_equity,
            mode="lines",
            name="Logistic Strategy"
        ))

        eq_fig.add_trace(go.Scatter(
            y=bh_equity,
            mode="lines",
            name="Buy & Hold"
        ))

        eq_fig.update_layout(title="Strategy Equity Curve")

        st.plotly_chart(eq_fig, use_container_width=True)