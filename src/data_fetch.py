import io
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from pandas_datareader import data as pdr

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

SHILLER_URL = "https://www.econ.yale.edu/~shiller/data/ie_data.xlsx"


def ensure_data_dir() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR


def _decimal_year_to_period(dec_year: float) -> pd.Period:
    year = int(np.floor(dec_year))
    month_float = (dec_year - year) * 12
    month = int(np.floor(month_float)) + 1
    month = min(max(month, 1), 12)
    return pd.Period(freq="M", year=year, month=month)


def download_shiller_data(force: bool = False) -> pd.DataFrame:
    ensure_data_dir()
    cache_path = DATA_DIR / "shiller_data.csv"
    if cache_path.exists() and not force:
        return pd.read_csv(cache_path, parse_dates=["date"])

    df = pd.read_excel(SHILLER_URL, sheet_name="Data", skiprows=7)
    df = df.rename(columns={"Date": "decimal_date", "P": "sp500", "CAPE": "cape"})
    df = df.loc[df["decimal_date"].notnull(), ["decimal_date", "sp500", "cape"]]
    df["date"] = df["decimal_date"].apply(lambda x: _decimal_year_to_period(float(x)).to_timestamp("M"))
    df = df.drop(columns=["decimal_date"])
    df = df.set_index("date").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    df.to_csv(cache_path)
    return df.reset_index()


def fetch_fred_series(series: str, start: str = "1900-01-01", freq: str = "M", force: bool = False) -> pd.Series:
    ensure_data_dir()
    cache_path = DATA_DIR / f"{series}.csv"
    if cache_path.exists() and not force:
        ser = pd.read_csv(cache_path, parse_dates=["DATE"], index_col="DATE")[series]
    else:
        ser = pdr.DataReader(series, "fred", start=start)
        ser.to_csv(cache_path)
        ser = ser[series]
    ser = ser.resample(freq).last()
    ser.name = series
    return ser


def fetch_market_series(force: bool = False) -> pd.DataFrame:
    shiller = download_shiller_data(force=force).set_index("date")
    sp500_level = shiller["sp500"]
    cape = shiller["cape"]

    nasdaq = fetch_fred_series("NASDAQCOM", start="1970-01-01", force=force)
    sp500_alt = fetch_fred_series("SP500", start="1950-01-01", force=force)

    df = pd.concat([sp500_level.rename("sp500_shiller"), sp500_alt.rename("sp500_fred"), nasdaq, cape], axis=1)
    df.index = pd.to_datetime(df.index)
    df["sp500"] = df["sp500_shiller"].combine_first(df["sp500_fred"])
    return df


def fetch_housing_series(force: bool = False) -> pd.DataFrame:
    case_shiller = fetch_fred_series("CSUSHPINSA", start="1987-01-01", force=force)
    mortgage_debt = fetch_fred_series("HDTGPDUSQ163N", start="1980-01-01", freq="Q", force=force)
    mortgage_debt = mortgage_debt.resample("M").ffill()
    homeownership = fetch_fred_series("RHORUSQ156N", start="1980-01-01", freq="Q", force=force)
    homeownership = homeownership.resample("M").ffill()

    df = pd.concat([
        case_shiller.rename("case_shiller"),
        mortgage_debt.rename("mortgage_gdp"),
        homeownership.rename("homeownership_rate"),
    ], axis=1)
    df.index = pd.to_datetime(df.index)
    return df


def build_merged_dataset(force: bool = False) -> pd.DataFrame:
    market = fetch_market_series(force=force)
    housing = fetch_housing_series(force=force)
    merged = pd.concat([market, housing], axis=1)
    merged = merged.sort_index()
    merged = merged[merged.index.notnull()]
    merged = merged.resample("M").last()
    merged_path = ensure_data_dir() / "merged_monthly_raw.csv"
    merged.to_csv(merged_path)
    return merged


def load_cached_merged(path: Optional[Path] = None) -> pd.DataFrame:
    if path is None:
        path = DATA_DIR / "merged_monthly.csv"
    if path.exists():
        return pd.read_csv(path, parse_dates=["date"], index_col="date")
    return build_merged_dataset(force=False)
