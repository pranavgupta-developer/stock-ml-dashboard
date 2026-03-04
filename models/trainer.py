"""
trainer.py

Implements walk-forward validation for stock trend classification.
"""

import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


def walk_forward_train(df: pd.DataFrame, feature_cols: list, initial_train_years: int = 3) -> dict:
    """
    Perform walk-forward validation.

    Parameters
    ----------
    df : DataFrame with features and Target
    feature_cols : list of feature column names
    initial_train_years : number of years used for initial training

    Returns
    -------
    dict containing predictions, probabilities, test dates, and aligned returns
    """

    df = df.copy()

    # Ensure datetime
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")

    # Extract year for walk-forward splitting
    df["Year"] = df["Date"].dt.year
    years = sorted(df["Year"].unique())

    if len(years) <= initial_train_years:
        raise ValueError("Not enough years for walk-forward validation.")

    # Storage
    all_dates = []
    all_y_true = []

    lr_preds = []
    lr_proba = []

    xgb_preds = []
    xgb_proba = []

    close_prices = []
    daily_returns = []

    # Walk-forward loop
    for i in range(initial_train_years, len(years)):

        train_years = years[:i]
        test_year = years[i]

        train_df = df[df["Year"].isin(train_years)]
        test_df = df[df["Year"] == test_year]

        if test_df.empty:
            continue

        X_train = train_df[feature_cols]
        y_train = train_df["Target"]

        X_test = test_df[feature_cols]
        y_test = test_df["Target"]

        # Scale (fit ONLY on train → avoids leakage)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Logistic Regression
        lr_model = LogisticRegression(max_iter=1000)
        lr_model.fit(X_train_scaled, y_train)

        lr_preds_fold = lr_model.predict(X_test_scaled)
        lr_proba_fold = lr_model.predict_proba(X_test_scaled)[:, 1]

        # XGBoost
        xgb_model = XGBClassifier(
            n_estimators=100,
            eval_metric="logloss"
        )

        xgb_model.fit(X_train_scaled, y_train)

        xgb_preds_fold = xgb_model.predict(X_test_scaled)
        xgb_proba_fold = xgb_model.predict_proba(X_test_scaled)[:, 1]

        # Store results (strict alignment)
        all_dates.extend(test_df["Date"].tolist())
        all_y_true.extend(y_test.tolist())

        lr_preds.extend(lr_preds_fold.tolist())
        lr_proba.extend(lr_proba_fold.tolist())

        xgb_preds.extend(xgb_preds_fold.tolist())
        xgb_proba.extend(xgb_proba_fold.tolist())

        close_prices.extend(test_df["Close"].tolist())
        daily_returns.extend(test_df["Daily_Return"].tolist())

        print(f"Completed fold: Train={train_years} Test={test_year}")

    return {
        "dates": all_dates,
        "y_true": all_y_true,
        "lr_preds": lr_preds,
        "lr_proba": lr_proba,
        "xgb_preds": xgb_preds,
        "xgb_proba": xgb_proba,
        "close_prices": close_prices,
        "daily_returns": daily_returns
    }