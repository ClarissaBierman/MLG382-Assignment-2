"""
feature_engineering.py  — CRISP-DM Phase 3: Data Preparation
══════════════════════════════════════════════════════════════
"""

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — Moving Averages
# ─────────────────────────────────────────────────────────────────────────────

def _sma(series: pd.Series, window: int) -> pd.Series:
    """Simple Moving Average over `window` periods."""
    return series.rolling(window=window, min_periods=1).mean()

def _ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential Moving Average with given span (adjust=False)."""
    return series.ewm(span=span, adjust=False, min_periods=1).mean()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — Momentum Indicators
# ─────────────────────────────────────────────────────────────────────────────

def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """
    Relative Strength Index.
    RSI = 100 - (100 / (1 + RS))  where RS = avg_gain / avg_loss.
    Values > 70 = overbought, < 30 = oversold.
    """
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()
    
    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def _macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """
    Moving Average Convergence Divergence.
    Returns: (macd_line, signal_line, histogram)
      macd_line   = EMA(fast) - EMA(slow)
      signal_line = EMA(macd_line, span=signal)
      histogram   = macd_line - signal_line
    """
    ema_fast = _ema(series, fast)
    ema_slow = _ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def _bollinger_bands(series: pd.Series, window: int = 20, num_std: float = 2.0):
    """
    Bollinger Bands.
    Returns: (upper, mid, lower)
      mid   = SMA(window)
      upper = mid + num_std * rolling_std(window)
      lower = mid - num_std * rolling_std(window)
    """
    mid = _sma(series, window)
    std = series.rolling(window=window, min_periods=1).std()
    upper = mid + (num_std * std)
    lower = mid - (num_std * std)
    return upper, mid, lower


def _atr(high: pd.Series, low: pd.Series, close: pd.Series,
         window: int = 14) -> pd.Series:
    """
    Average True Range.
    True Range = max(H-L, |H-C_prev|, |L-C_prev|)
    ATR = rolling mean of TR over `window`.
    """
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=window, min_periods=1).mean()
    return atr


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    On-Balance Volume.
    Add volume when price closes up, subtract when closes down. Cumsum.
    """
    direction = np.sign(close.diff())
    obv = (direction * volume).fillna(0).cumsum()
    return obv


def _stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
                k_window: int = 14, d_window: int = 3):
    """
    Stochastic Oscillator.
    %K = 100 * (close - lowest_low(k_window)) / (highest_high - lowest_low)
    %D = SMA(%K, d_window)
    Returns: (%K, %D)
    """
    lowest_low = low.rolling(window=k_window, min_periods=1).min()
    highest_high = high.rolling(window=k_window, min_periods=1).max()
    
    k = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-9)
    d = _sma(k, d_window)
    return k, d

def _williams_r(high: pd.Series, low: pd.Series, close: pd.Series,
                window: int = 14) -> pd.Series:
    """
    Williams %R.
    = -100 * (highest_high - close) / (highest_high - lowest_low)
    Range: -100 (oversold) to 0 (overbought).
    """
    highest_high = high.rolling(window=window, min_periods=1).max()
    lowest_low = low.rolling(window=window, min_periods=1).min()
    
    williams = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-9)
    return williams



def _cci(high: pd.Series, low: pd.Series, close: pd.Series,
         window: int = 20) -> pd.Series:
    """
    Commodity Channel Index.
    Typical Price = (H + L + C) / 3
    CCI = (TP - SMA(TP, window)) / (0.015 * mean_abs_deviation)
    """
    typical_price = (high + low + close) / 3
    sma_tp = _sma(typical_price, window)
    mean_dev = (typical_price - sma_tp).abs().rolling(window=window, min_periods=1).mean()
    
    cci = (typical_price - sma_tp) / (0.015 * mean_dev + 1e-9)
    return cci

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — Full feature builder
# ─────────────────────────────────────────────────────────────────────────────

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add ~40 technical indicator columns to an OHLCV DataFrame.

    Required input columns:  Open, High, Low, Close, Volume
    Returns a new DataFrame (does not modify the original).

    Expected output columns (must match exactly for app.py to work):
        SMA_5, SMA_10, SMA_20, SMA_50, SMA_100, SMA_200
        EMA_5, EMA_10, EMA_20, EMA_50, EMA_100, EMA_200
        Price_vs_SMA5, Price_vs_SMA20, Price_vs_SMA50, Price_vs_SMA200
            (formula: close / SMA - 1)
        GoldenCross  (int: 1 if SMA_50 > SMA_200 else 0)
        RSI_14, RSI_7
        MACD, MACD_Signal, MACD_Hist
        Stoch_K, Stoch_D
        Williams_R
        CCI
        ROC_1, ROC_5, ROC_10, ROC_20   (close.pct_change(n) * 100)
        BB_Upper, BB_Mid, BB_Lower
        BB_Width  = (upper - lower) / mid
        BB_Pct    = (close - lower) / (upper - lower)
        ATR_14
        ATR_Norm  = ATR_14 / close
        HV_21     = 21-day rolling std of log returns * sqrt(252)
        OBV
        Volume_SMA20
        Volume_Ratio = Volume / Volume_SMA20
        HL_Range     = (High - Low) / Close
        OC_Range     = (Close - Open) / Open
        Upper_Shadow = (High - max(Close,Open)) / Close
        Lower_Shadow = (min(Close,Open) - Low) / Close
        Daily_Return = Close.pct_change()
    """
    df = df.copy()
    c, h, l, o, v = df["Close"], df["High"], df["Low"], df["Open"], df["Volume"]

    # Moving Averages
    df["SMA_5"] = _sma(c, 5)
    df["SMA_10"] = _sma(c, 10)
    df["SMA_20"] = _sma(c, 20)
    df["SMA_50"] = _sma(c, 50)
    df["SMA_100"] = _sma(c, 100)
    df["SMA_200"] = _sma(c, 200)
    
    df["EMA_5"] = _ema(c, 5)
    df["EMA_10"] = _ema(c, 10)
    df["EMA_20"] = _ema(c, 20)
    df["EMA_50"] = _ema(c, 50)
    df["EMA_100"] = _ema(c, 100)
    df["EMA_200"] = _ema(c, 200)
    
    # Price vs Moving Averages
    df["Price_vs_SMA5"] = (c / df["SMA_5"]) - 1
    df["Price_vs_SMA20"] = (c / df["SMA_20"]) - 1
    df["Price_vs_SMA50"] = (c / df["SMA_50"]) - 1
    df["Price_vs_SMA200"] = (c / df["SMA_200"]) - 1
    
    # Golden Cross indicator
    df["GoldenCross"] = (df["SMA_50"] > df["SMA_200"]).astype(int)
    
    # RSI
    df["RSI_14"] = _rsi(c, 14)
    df["RSI_7"] = _rsi(c, 7)
    
    # MACD
    macd, signal, hist = _macd(c)
    df["MACD"] = macd
    df["MACD_Signal"] = signal
    df["MACD_Hist"] = hist
    
    # Stochastic
    stoch_k, stoch_d = _stochastic(h, l, c)
    df["Stoch_K"] = stoch_k
    df["Stoch_D"] = stoch_d
    
    # Williams %R
    df["Williams_R"] = _williams_r(h, l, c)
    
    # CCI
    df["CCI"] = _cci(h, l, c)
    
    # Rate of Change
    df["ROC_1"] = c.pct_change(1) * 100
    df["ROC_5"] = c.pct_change(5) * 100
    df["ROC_10"] = c.pct_change(10) * 100
    df["ROC_20"] = c.pct_change(20) * 100
    
    # Bollinger Bands
    bb_upper, bb_mid, bb_lower = _bollinger_bands(c)
    df["BB_Upper"] = bb_upper
    df["BB_Mid"] = bb_mid
    df["BB_Lower"] = bb_lower
    df["BB_Width"] = (bb_upper - bb_lower) / (bb_mid + 1e-9)
    df["BB_Pct"] = (c - bb_lower) / (bb_upper - bb_lower + 1e-9)
    
    # ATR
    df["ATR_14"] = _atr(h, l, c, 14)
    df["ATR_Norm"] = df["ATR_14"] / (c + 1e-9)
    
    # Historical Volatility
    log_returns = np.log(c / c.shift(1))
    df["HV_21"] = log_returns.rolling(window=21, min_periods=1).std() * np.sqrt(252) * 100
    
    # On-Balance Volume
    df["OBV"] = _obv(c, v)
    
    # Volume indicators
    df["Volume_SMA20"] = _sma(v, 20)
    df["Volume_Ratio"] = v / (df["Volume_SMA20"] + 1e-9)
    
    # Price Action indicators
    df["HL_Range"] = (h - l) / (c + 1e-9)
    df["OC_Range"] = (c - o) / (o + 1e-9)
    
    # Candlestick shadows
    df["Upper_Shadow"] = (h - pd.concat([c, o], axis=1).max(axis=1)) / (c + 1e-9)
    df["Lower_Shadow"] = (pd.concat([c, o], axis=1).min(axis=1) - l) / (c + 1e-9)
    
    # Daily return
    df["Daily_Return"] = c.pct_change()
    
    return df

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — Lag & time features
# ─────────────────────────────────────────────────────────────────────────────

def add_lag_features(df: pd.DataFrame,
                     target_col: str = "Close",
                     lags: list = None) -> pd.DataFrame:
    """Add lagged values of target_col."""
    if lags is None:
        lags = [1, 2, 3, 5, 10, 20]
    df = df.copy()
    for lag in lags:
        df[f"{target_col}_Lag{lag}"] = df[target_col].shift(lag)
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cyclically encode day-of-week, month, quarter."""
    df = df.copy()
    idx = df.index
    df["DayOfWeek"]  = idx.dayofweek
    df["Month"]      = idx.month
    df["Quarter"]    = idx.quarter
    df["DayOfYear"]  = idx.dayofyear
    df["DOW_sin"]    = np.sin(2 * np.pi * df["DayOfWeek"] / 5)
    df["DOW_cos"]    = np.cos(2 * np.pi * df["DayOfWeek"] / 5)
    df["Month_sin"]  = np.sin(2 * np.pi * df["Month"] / 12)
    df["Month_cos"]  = np.cos(2 * np.pi * df["Month"] / 12)
    df["WeekOfYear"] = idx.isocalendar().week.values.astype(int)
    return df

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — Master pipeline
# ─────────────────────────────────────────────────────────────────────────────

def prepare_features(df: pd.DataFrame,
                     target_col: str = "Close",
                     forecast_horizon: int = 1) -> tuple:
    """
    Full pipeline: indicators → lags → time features → target → dropna.
    Returns (X: DataFrame, y: Series, feature_names: list).
    """
    df = add_technical_indicators(df)
    df = add_lag_features(df, target_col)
    df = add_time_features(df)
    df["Target"] = df[target_col].shift(-forecast_horizon)
    df.dropna(inplace=True)

    exclude = {"Open", "High", "Low", "Close", "Volume", "Target",
               "Adj Close", "Dividends", "Stock Splits"}
    feature_cols = [c for c in df.columns if c not in exclude]
    return df[feature_cols], df["Target"], feature_cols

def get_feature_groups() -> dict:
    return {
        "Moving Averages": ["SMA_", "EMA_"],
        "Momentum":        ["RSI", "MACD", "Stoch", "Williams", "CCI", "ROC"],
        "Volatility":      ["BB_", "ATR", "HV"],
        "Volume":          ["OBV", "Volume"],
        "Price Action":    ["HL_Range", "OC_Range", "Shadow", "Return"],
        "Lag Features":    ["_Lag"],
        "Time Features":   ["DOW", "Month", "Quarter", "Week", "Day"],
        "Market Context":  ["VIX", "SP500"],
    }
