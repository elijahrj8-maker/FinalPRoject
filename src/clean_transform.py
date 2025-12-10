from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .data_fetch import build_merged_dataset, ensure_data_dir


BUBBLE_WINDOWS: Dict[str, Tuple[pd.Timestamp, pd.Timestamp]] = {
    "nifty_fifty": (pd.Timestamp("1968-01-01"), pd.Timestamp("1975-12-31")),
    "dot_com": (pd.Timestamp("1995-01-01"), pd.Timestamp("2002-12-31")),
    "housing": (pd.Timestamp("1998-01-01"), pd.Timestamp("2010-12-31")),
}


@dataclass
class FeatureConfig:
    momentum_window: int = 24
    high_momentum_threshold: float = 0.25
    valuation_quantile: float = 0.9


def compute_drawdown(series: pd.Series) -> pd.Series:
    cumulative_max = series.cummax()
    drawdown = (series - cumulative_max) / cumulative_max
    return drawdown


def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    df["sp500_return"] = df["sp500"].pct_change()
    if "NASDAQCOM" in df.columns:
        df["nasdaq_return"] = df["NASDAQCOM"].pct_change()
    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["sp500_return", "nasdaq_return", "cape"]:
        if col in df.columns:
            df[f"{col}_roll6"] = df[col].rolling(6).mean()
            df[f"{col}_roll12"] = df[col].rolling(12).mean()
    return df


def add_trailing_performance(df: pd.DataFrame, months: int = 24) -> pd.DataFrame:
    df[f"sp500_trailing_{months}"] = (1 + df["sp500_return"]).rolling(months).apply(np.prod, raw=True) - 1
    if "nasdaq_return" in df:
        df[f"nasdaq_trailing_{months}"] = (1 + df["nasdaq_return"]).rolling(months).apply(np.prod, raw=True) - 1
    return df


def add_drawdowns(df: pd.DataFrame) -> pd.DataFrame:
    df["sp500_drawdown"] = compute_drawdown(df["sp500"].dropna()).reindex(df.index)
    if "NASDAQCOM" in df.columns:
        df["nasdaq_drawdown"] = compute_drawdown(df["NASDAQCOM"].dropna()).reindex(df.index)
    if "case_shiller" in df.columns:
        df["housing_drawdown"] = compute_drawdown(df["case_shiller"].dropna()).reindex(df.index)
    return df


def add_housing_overvaluation(df: pd.DataFrame) -> pd.DataFrame:
    if "case_shiller" not in df:
        return df
    yoy = df["case_shiller"].pct_change(12)
    zscore = (yoy - yoy.mean()) / yoy.std(ddof=0)
    df["cs_yoy"] = yoy
    df["cs_yoy_z"] = zscore
    df["housing_overvaluation"] = zscore > 1.5
    return df


def add_bubble_flags(df: pd.DataFrame, config: FeatureConfig = FeatureConfig()) -> pd.DataFrame:
    df["high_valuation"] = df["cape"] > df["cape"].quantile(config.valuation_quantile)
    df["high_momentum"] = df[f"sp500_trailing_{config.momentum_window}"] > config.high_momentum_threshold
    df["bubble_flag"] = df[["high_valuation", "high_momentum"]].any(axis=1)
    df = add_housing_overvaluation(df)
    df["bubble_window"] = None
    for name, (start, end) in BUBBLE_WINDOWS.items():
        df.loc[(df.index >= start) & (df.index <= end), "bubble_window"] = name
    return df


def prepare_features(force: bool = False, config: FeatureConfig = FeatureConfig()) -> pd.DataFrame:
    df = build_merged_dataset(force=force)
    df = add_returns(df)
    df = add_trailing_performance(df, months=config.momentum_window)
    df = add_rolling_features(df)
    df = add_drawdowns(df)
    df = add_bubble_flags(df, config=config)
    df = df.dropna(how="all")
    df.index.name = "date"
    merged_path = ensure_data_dir() / "merged_monthly.csv"
    df.to_csv(merged_path)
    return df


def compute_bubble_summary(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for name, (start, end) in BUBBLE_WINDOWS.items():
        window = df.loc[(df.index >= start) & (df.index <= end)]
        if window.empty:
            continue
        peak_idx = window["sp500"].idxmax()
        floor = window.loc[peak_idx:]["sp500"].idxmin()
        runup_12m = window.loc[:peak_idx, "sp500_return"].tail(12).add(1).prod() - 1
        runup_24m = window.loc[:peak_idx, "sp500_return"].tail(24).add(1).prod() - 1
        peak_val = window.loc[peak_idx, "cape"] if "cape" in window else np.nan
        max_draw = window.loc[peak_idx:, "sp500_drawdown"].min()
        months_to_trough = (floor.to_period("M") - peak_idx.to_period("M")).n
        records.append({
            "bubble": name,
            "peak_date": peak_idx.date(),
            "peak_valuation": peak_val,
            "runup_12m": runup_12m,
            "runup_24m": runup_24m,
            "max_drawdown": max_draw,
            "months_to_trough": months_to_trough,
        })
    return pd.DataFrame(records)
