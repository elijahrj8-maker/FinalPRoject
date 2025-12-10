# Bubble Analysis Project

Reproducible notebook and helper modules for analyzing three historical bubbles: Nifty Fifty (1968-1975), Dot-Com (1995-2002), and Housing (1998-2010). The project pulls public data, engineers features, visualizes patterns, and runs simple statistical models.

## Project structure
- `notebooks/bubbles_analysis.ipynb`: Main report notebook with narrative, figures, and outputs.
- `src/data_fetch.py`: Functions to download and cache data from public APIs and Shiller data.
- `src/clean_transform.py`: Cleaning, feature engineering, and merged dataset creation.
- `src/visuals.py`: Plot helpers with bubble shading.
- `src/models.py`: Regression and time series utilities (ARIMA/ARIMAX and evaluation).
- `data/`: Cached CSV files including `merged_monthly.csv` output.
- `requirements.txt`: Python dependencies.

## Quickstart
1. Optional: create and activate a virtual environment.
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
2. Install dependencies.
   ```bash
   pip install -r requirements.txt
   ```
3. Pull and cache the data, then build the merged dataset. This is a fast sanity check to ensure the APIs are reachable and the cache folder exists.
   ```bash
   python - <<'PY'
   from src.data_fetch import build_merged_dataset
   merged = build_merged_dataset()
   print(merged.tail())
   print("Saved to data/merged_monthly_raw.csv")
   PY
   ```
4. Run the full notebook non-interactively (recommended for reproducibility).
   ```bash
   jupyter nbconvert --to notebook --execute notebooks/bubbles_analysis.ipynb --output notebooks/bubbles_analysis.ipynb
   ```
   You can also open it interactively with `jupyter lab` or `jupyter notebook` and run all cells.

If you have a FRED API key, set `FRED_API_KEY` in your environment before running. The `pandas_datareader` calls will pick it up automatically, but most public series should work without it.

## Data sources
- Shiller S&P 500 data and CAPE: `https://www.econ.yale.edu/~shiller/data/ie_data.xlsx`.
- FRED series pulled via `pandas_datareader`:
  - NASDAQ Composite (`NASDAQCOM`).
  - S&P 500 monthly close (`SP500`) as a level proxy.
  - Case-Shiller Home Price Index (`CSUSHPINSA`).
  - Mortgage debt to GDP (`HDTGPDUSQ163N`).
  - Homeownership rate (`RHORUSQ156N`).

If trailing P/E is unavailable, the notebook uses CAPE as the valuation metric and labels it clearly.

## Running notes
- All series are coerced to monthly frequency and merged on a calendar month end index.
- Downloaded data are cached in `data/` to avoid repeated network calls.
- The merged dataset is saved to `data/merged_monthly.csv` after the cleaning step.

## License
This project uses only public data sources and is provided for analytical purposes.
