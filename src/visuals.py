from __future__ import annotations

from typing import Dict, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .clean_transform import BUBBLE_WINDOWS

sns.set_style("whitegrid")


def shade_bubbles(ax, windows: Dict[str, Tuple[pd.Timestamp, pd.Timestamp]] = BUBBLE_WINDOWS):
    for name, (start, end) in windows.items():
        ax.axvspan(start, end, color="gray", alpha=0.15, label=name if name not in ax.get_legend_handles_labels()[1] else None)


def plot_index_with_bubbles(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df.index, df["sp500"], label="S&P 500")
    shade_bubbles(ax)
    ax.set_title("S&P 500 with bubble windows")
    ax.set_ylabel("Index level")
    ax.legend()
    ax.xaxis.set_major_locator(mdates.YearLocator(base=5))
    fig.tight_layout()
    return fig


def plot_valuation(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df.index, df["cape"], color="darkred", label="CAPE")
    shade_bubbles(ax)
    ax.set_title("Valuation metric over time (CAPE)")
    ax.set_ylabel("CAPE")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_dotcom_comparison(df: pd.DataFrame) -> plt.Figure:
    base_date = pd.Timestamp("1995-01-31")
    sub = df.loc[df.index >= base_date].copy()
    sub["sp500_norm"] = sub["sp500"] / sub["sp500"].iloc[0] * 100
    sub["nasdaq_norm"] = sub["NASDAQCOM"] / sub["NASDAQCOM"].iloc[0] * 100

    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    axes[0].plot(sub.index, sub["sp500_norm"], label="S&P 500")
    axes[0].plot(sub.index, sub["nasdaq_norm"], label="NASDAQ")
    shade_bubbles(axes[0])
    axes[0].set_title("Dot-Com comparison: normalized indices")
    axes[0].legend()

    axes[1].plot(sub.index, sub["nasdaq_drawdown"], label="NASDAQ drawdown", color="purple")
    axes[1].plot(sub.index, sub["sp500_drawdown"], label="S&P drawdown", color="black")
    shade_bubbles(axes[1])
    axes[1].set_title("Drawdowns during Dot-Com era")
    axes[1].legend()

    for ax in axes:
        ax.xaxis.set_major_locator(mdates.YearLocator(base=2))
    fig.tight_layout()
    return fig


def plot_housing(df: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    ax1, ax2 = axes
    ax1.plot(df.index, df["case_shiller"], color="teal", label="Case-Shiller")
    ax1.plot(df.index, df["sp500"], color="gray", alpha=0.7, label="S&P 500")
    shade_bubbles(ax1)
    ax1.set_title("Housing vs Equity levels")
    ax1.legend()

    ax2.plot(df.index, df["cs_yoy_z"], color="orange", label="Case-Shiller YoY z-score")
    ax2.axhline(1.5, color="red", linestyle="--", label="Overvaluation threshold")
    shade_bubbles(ax2)
    ax2.set_title("Housing heat: YoY growth z-score")
    ax2.legend()

    fig.tight_layout()
    return fig


def plot_volatility(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 5))
    rolling_std = df["sp500_return"].rolling(12).std()
    ax.plot(df.index, rolling_std, label="Rolling 12M vol")
    shade_bubbles(ax)
    ax.set_title("Volatility around bubbles")
    ax.set_ylabel("Std dev of monthly return")
    ax.legend()
    fig.tight_layout()
    return fig
