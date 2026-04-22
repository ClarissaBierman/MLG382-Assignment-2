"""
data_fetcher.py
Fetches historical OHLCV data, VIX sentiment index, and company fundamentals
from Yahoo Finance using yfinance.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Core data fetchers
# ─────────────────────────────────────────────────────────────────────────────

def fetch_stock_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Download OHLCV data for a single ticker symbol.

    Parameters
    ----------
    ticker     : e.g. 'AAPL'
    start_date : 'YYYY-MM-DD'
    end_date   : 'YYYY-MM-DD'

    Returns
    -------
    DataFrame with columns: Open, High, Low, Close, Volume, Adj Close
    Index: DatetimeIndex
    """
    try:
        df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
        if df.empty:
            raise ValueError(f"No data returned for ticker '{ticker}'.")
        # Flatten multi-level column index that newer yfinance versions produce
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)
        return df
    except Exception as e:
        raise RuntimeError(f"fetch_stock_data failed: {e}")


def fetch_vix_data(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Download the CBOE Volatility Index (VIX) as a market-fear / sentiment proxy.
    """
    try:
        vix = yf.download("^VIX", start=start_date, end=end_date, progress=False, auto_adjust=True)
        if isinstance(vix.columns, pd.MultiIndex):
            vix.columns = vix.columns.get_level_values(0)
        vix = vix[["Close"]].rename(columns={"Close": "VIX"})
        return vix
    except Exception:
        return pd.DataFrame()   # graceful fallback


def fetch_sp500_data(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Download S&P 500 (^GSPC) as a broad market benchmark.
    """
    try:
        sp500 = yf.download("^GSPC", start=start_date, end=end_date, progress=False, auto_adjust=True)
        if isinstance(sp500.columns, pd.MultiIndex):
            sp500.columns = sp500.columns.get_level_values(0)
        sp500 = sp500[["Close"]].rename(columns={"Close": "SP500"})
        return sp500
    except Exception:
        return pd.DataFrame()

# ─────────────────────────────────────────────────────────────────────────────
# Company fundamentals
# ─────────────────────────────────────────────────────────────────────────────

def get_company_info(ticker: str) -> dict:
    """
    Return a dictionary of key fundamental metrics for display purposes.
    Falls back to empty dict on failure (fundamentals are not used in model training).
    """
    defaults = {
        "longName": ticker,
        "sector": "N/A",
        "industry": "N/A",
        "marketCap": None,
        "trailingPE": None,
        "forwardPE": None,
        "priceToBook": None,
        "dividendYield": None,
        "beta": None,
        "fiftyTwoWeekHigh": None,
        "fiftyTwoWeekLow": None,
        "averageVolume": None,
        "currentPrice": None,
        "targetMeanPrice": None,
        "recommendationKey": "N/A",
        "shortRatio": None,
        "trailingEps": None,
        "revenueGrowth": None,
    }
    try:
        info = yf.Ticker(ticker).info
        merged = {k: info.get(k, v) for k, v in defaults.items()}
        return merged
    except Exception:
        return defaults

# ─────────────────────────────────────────────────────────────────────────────
# Composite loader used by the Dash app
# ─────────────────────────────────────────────────────────────────────────────

def load_full_dataset(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch stock OHLCV, merge with VIX and S&P 500 returns, and return a
    single clean DataFrame ready for feature engineering.
    """
    df = fetch_stock_data(ticker, start_date, end_date)

    # VIX sentiment
    vix = fetch_vix_data(start_date, end_date)
    if not vix.empty:
        df = df.join(vix, how="left")
        df["VIX"].fillna(method="ffill", inplace=True)
    else:
        df["VIX"] = 20.0   # historical average when unavailable

    # S&P 500 daily return as a broad market feature
    sp500 = fetch_sp500_data(start_date, end_date)
    if not sp500.empty:
        sp500["SP500_Return"] = sp500["SP500"].pct_change()
        df = df.join(sp500[["SP500_Return"]], how="left")
        df["SP500_Return"].fillna(0, inplace=True)
    else:
        df["SP500_Return"] = 0.0

    df.dropna(how="all", inplace=True)
    return df


def load_static_dataset() -> pd.DataFrame:
    """
    Load the static CSV snapshot from the ml_ready directory.
    This is the pre-processed dataset created by clean_stock_data.py.
    """
    path = "data/ml_ready/AAPL_training_data.csv"
    df = pd.read_csv(path, parse_dates=["Date"])
    df.set_index("Date", inplace=True)
    # Only keep the OHLCV columns (remove MA5, MA20, Target as they'll be regenerated)
    core_cols = ["Open", "High", "Low", "Close", "Volume"]
    if all(col in df.columns for col in core_cols):
        df = df[core_cols]
    return df

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def format_market_cap(value) -> str:
    if value is None:
        return "N/A"
    value = float(value)
    if value >= 1e12:
        return f"${value/1e12:.2f}T"
    elif value >= 1e9:
        return f"${value/1e9:.2f}B"
    elif value >= 1e6:
        return f"${value/1e6:.2f}M"
    return f"${value:,.0f}"


def format_percentage(value) -> str:
    if value is None:
        return "N/A"
    return f"{float(value)*100:.2f}%"


def get_default_date_range():
    end = datetime.today()
    start = end - timedelta(days=5 * 365)   # 5 years default
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
