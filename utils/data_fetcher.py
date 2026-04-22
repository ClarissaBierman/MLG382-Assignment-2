"""
data_fetcher.py
Fetches historical OHLCV data, VIX sentiment index, and company fundamentals
from Yahoo Finance using yfinance.
 
OPTIMIZED: VIX, S&P 500, and stock data are now fetched concurrently
using ThreadPoolExecutor, cutting fetch time roughly in half.
"""
 
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings("ignore")
 
# ─────────────────────────────────────────────────────────────────────────────
# Core data fetchers
# ─────────────────────────────────────────────────────────────────────────────
 
def fetch_stock_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    try:
        df = yf.download(ticker, start=start_date, end=end_date,
                         progress=False, auto_adjust=True)
        if df.empty:
            raise ValueError(f"No data returned for ticker '{ticker}'.")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)
        return df
    except Exception as e:
        raise RuntimeError(f"fetch_stock_data failed: {e}")
 
 
def fetch_vix_data(start_date: str, end_date: str) -> pd.DataFrame:
    try:
        vix = yf.download("^VIX", start=start_date, end=end_date,
                          progress=False, auto_adjust=True)
        if isinstance(vix.columns, pd.MultiIndex):
            vix.columns = vix.columns.get_level_values(0)
        return vix[["Close"]].rename(columns={"Close": "VIX"})
    except Exception:
        return pd.DataFrame()
 
 
def fetch_sp500_data(start_date: str, end_date: str) -> pd.DataFrame:
    try:
        sp500 = yf.download("^GSPC", start=start_date, end=end_date,
                             progress=False, auto_adjust=True)
        if isinstance(sp500.columns, pd.MultiIndex):
            sp500.columns = sp500.columns.get_level_values(0)
        return sp500[["Close"]].rename(columns={"Close": "SP500"})
    except Exception:
        return pd.DataFrame()
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Company fundamentals
# ─────────────────────────────────────────────────────────────────────────────
 
def get_company_info(ticker: str) -> dict:
    defaults = {
        "longName": ticker, "sector": "N/A", "industry": "N/A",
        "marketCap": None, "trailingPE": None, "forwardPE": None,
        "priceToBook": None, "dividendYield": None, "beta": None,
        "fiftyTwoWeekHigh": None, "fiftyTwoWeekLow": None,
        "averageVolume": None, "currentPrice": None,
        "targetMeanPrice": None, "recommendationKey": "N/A",
        "shortRatio": None, "trailingEps": None, "revenueGrowth": None,
    }
    try:
        info   = yf.Ticker(ticker).info
        merged = {k: info.get(k, v) for k, v in defaults.items()}
        return merged
    except Exception:
        return defaults
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Composite loader — OPTIMIZED with concurrent fetching
# ─────────────────────────────────────────────────────────────────────────────
 
def load_full_dataset(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch stock OHLCV, VIX, and S&P 500 concurrently, then merge.
 
    Previously these were 3 sequential yfinance calls. Running them in
    parallel via ThreadPoolExecutor saves ~5-10 seconds on most connections
    since yfinance downloads are I/O-bound (network wait time).
    """
    with ThreadPoolExecutor(max_workers=3) as ex:
        f_stock = ex.submit(fetch_stock_data, ticker, start_date, end_date)
        f_vix   = ex.submit(fetch_vix_data,   start_date, end_date)
        f_sp500 = ex.submit(fetch_sp500_data,  start_date, end_date)
 
        df    = f_stock.result()   # raises immediately if fetch failed
        vix   = f_vix.result()
        sp500 = f_sp500.result()
 
    # VIX sentiment
    if not vix.empty:
        df = df.join(vix, how="left")
        df["VIX"] = df["VIX"].ffill()
    else:
        df["VIX"] = 20.0
 
    # S&P 500 daily return
    if not sp500.empty:
        sp500["SP500_Return"] = sp500["SP500"].pct_change()
        df = df.join(sp500[["SP500_Return"]], how="left")
        df["SP500_Return"] = df["SP500_Return"].fillna(0)
    else:
        df["SP500_Return"] = 0.0
 
    df.dropna(how="all", inplace=True)
    return df
 
 
def load_static_dataset() -> pd.DataFrame:
    path = "data/ml_ready/AAPL_training_data.csv"
    df   = pd.read_csv(path, parse_dates=["Date"])
    df.set_index("Date", inplace=True)
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
    end   = datetime.today()
    start = end - timedelta(days=5 * 365)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")