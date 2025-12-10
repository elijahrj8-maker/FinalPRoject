from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX


def compute_forward_return(df: pd.DataFrame, horizon_months: int = 60, col: str = "sp500") -> pd.Series:
    future = df[col].shift(-horizon_months)
    forward_ret = (future - df[col]) / df[col]
    return forward_ret / (horizon_months / 12)


def regression_valuation_vs_return(df: pd.DataFrame, horizon: int = 60):
    df = df.copy()
    df["fwd_ann_return"] = compute_forward_return(df, horizon_months=horizon)
    reg_df = df.dropna(subset=["fwd_ann_return", "cape"])
    X = sm.add_constant(reg_df[["cape"]])
    model = sm.OLS(reg_df["fwd_ann_return"], X).fit()
    return model


def crash_vs_normal_test(df: pd.DataFrame, crash_mask: pd.Series):
    crash_returns = df.loc[crash_mask, "sp500_return"].dropna()
    normal_returns = df.loc[~crash_mask, "sp500_return"].dropna()
    ttest = sm.stats.ttest_ind(crash_returns, normal_returns, alternative="two-sided")
    return {
        "crash_mean": crash_returns.mean(),
        "normal_mean": normal_returns.mean(),
        "t_stat": ttest[0],
        "p_value": ttest[1],
    }


def volatility_regimes(df: pd.DataFrame, peak_date: pd.Timestamp, window: int = 24) -> Dict[str, float]:
    pre = df.loc[(df.index >= peak_date - pd.DateOffset(months=window)) & (df.index < peak_date), "sp500_return"]
    post = df.loc[(df.index > peak_date) & (df.index <= peak_date + pd.DateOffset(months=window)), "sp500_return"]
    return {"pre_vol": pre.std(), "post_vol": post.std()}


def fit_arima_baseline(df: pd.DataFrame, order: Tuple[int, int, int] = (1, 0, 1)):
    series = df["sp500_return"].dropna()
    model = ARIMA(series, order=order)
    res = model.fit()
    return res


def fit_arimax(df: pd.DataFrame, order: Tuple[int, int, int] = (1, 0, 1)):
    series = df["sp500_return"].dropna()
    exog = df.loc[series.index, ["cape", "bubble_flag"]].fillna(method="ffill")
    model = SARIMAX(series, order=order, exog=exog)
    res = model.fit(disp=False)
    return res


def walk_forward_accuracy(df: pd.DataFrame, order: Tuple[int, int, int] = (1, 0, 1), horizon: int = 1) -> Dict[str, float]:
    series = df["sp500_return"].dropna()
    exog_full = df.loc[series.index, ["cape", "bubble_flag"]].fillna(method="ffill")
    split = int(len(series) * 0.7)
    train_y, test_y = series.iloc[:split], series.iloc[split:]
    train_x, test_x = exog_full.iloc[:split], exog_full.iloc[split:]

    model = SARIMAX(train_y, order=order, exog=train_x)
    res = model.fit(disp=False)
    forecast = res.get_forecast(steps=len(test_y), exog=test_x)
    preds = forecast.predicted_mean
    directional = np.sign(preds) == np.sign(test_y)
    mae = (preds - test_y).abs().mean()
    return {"directional_accuracy": directional.mean(), "mae": mae}
