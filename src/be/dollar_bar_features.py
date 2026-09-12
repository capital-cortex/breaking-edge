from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Sequence

EPS = 1e-12

# -----------------------------
# Helpers (no leakage)
# -----------------------------
def _safe_div(a, b, eps: float = EPS):
    return a / (b + eps)

def _safe_div_time(a, duration_s, eps_time: float = 1e-3):
    # duration_s can be 0 for very fast bars; use a millisecond floor to avoid explosions.
    return a / (duration_s + eps_time)

def _log1p_clip(x):
    return np.log1p(np.clip(x, 0, None))

def _rolling_mean(s: pd.Series, window: int, strict_past: bool = True) -> pd.Series:
    out = s.rolling(window=window, min_periods=window).mean()
    return out.shift(1) if strict_past else out

def _rolling_std(s: pd.Series, window: int, strict_past: bool = True) -> pd.Series:
    out = s.rolling(window=window, min_periods=window).std()
    return out.shift(1) if strict_past else out

def _rolling_zscore(s: pd.Series, window: int, strict_past: bool = True) -> pd.Series:
    mu = _rolling_mean(s, window, strict_past=strict_past)
    sd = _rolling_std(s, window, strict_past=strict_past)
    return (s - mu) / (sd + EPS)

def _ema(s: pd.Series, span: int, strict_past: bool = True) -> pd.Series:
    out = s.ewm(span=span, adjust=False, min_periods=span).mean()
    return out.shift(1) if strict_past else out

# -----------------------------
# Base series (price, returns)
# -----------------------------
def calculate_log_return(df: pd.DataFrame, source: str = "price_close") -> pd.Series:
    s = df[source].astype("float64")
    return pd.Series(np.log(s).diff(), df.index)

def calculate_hl_range_pct(df: pd.DataFrame) -> pd.Series:
    high = df["price_high"].astype("float64")
    low = df["price_low"].astype("float64")
    op = df["price_open"].astype("float64")
    return _safe_div(high - low, op)

def calculate_close_position_in_range(df: pd.DataFrame) -> pd.Series:
    c = df["price_close"].astype("float64")
    l = df["price_low"].astype("float64")
    h = df["price_high"].astype("float64")
    return _safe_div(c - l, h - l)

def calculate_wick_ratios(df: pd.DataFrame) -> Dict[str, pd.Series]:
    o = df["price_open"].astype("float64")
    h = df["price_high"].astype("float64")
    l = df["price_low"].astype("float64")
    c = df["price_close"].astype("float64")

    rng = (h - l) + EPS
    upper_wick = h - np.maximum(o, c)
    lower_wick = np.minimum(o, c) - l
    body = (c - o).abs()

    return {
        "upper_wick_ratio": upper_wick / rng,
        "lower_wick_ratio": lower_wick / rng,
        "body_ratio": body / rng,
    }

# -----------------------------
# Activity / speed features
# -----------------------------
def calculate_duration_s(df: pd.DataFrame) -> pd.Series:
    return (df["time_close"] - df.index).dt.total_seconds()

def calculate_total_dollars(df: pd.DataFrame) -> pd.Series:
    return (df["volume_quote_buy"] + df["volume_quote_sell"]).astype("float64")

def calculate_trades_per_sec(df: pd.DataFrame) -> pd.Series:
    dur = calculate_duration_s(df) if not "duration_s" in df.columns else df["duration_s"]
    return _safe_div_time(df["trades_abs"], dur.astype("float64"))

def calculate_dollar_per_sec(df: pd.DataFrame) -> pd.Series:
    dur = calculate_duration_s(df) if not "duration_s" in df.columns else df["duration_s"]
    return _safe_div_time(calculate_total_dollars(df), dur.astype("float64"))

def calculate_duration_log(df: pd.DataFrame) -> pd.Series:
    dur = calculate_duration_s(df) if not "duration_s" in df.columns else df["duration_s"]
    return _log1p_clip(dur.astype("float64"))

def calculate_speed_index(df: pd.DataFrame) -> pd.Series:
    tps = calculate_trades_per_sec(df)
    dps = calculate_dollar_per_sec(df)
    return pd.Series(np.log1p(tps) + np.log1p(dps), df.index)

# -----------------------------
# Orderflow features
# -----------------------------

# TODO: impelemnt dbv.resample_trades_volume_bars() for trades_buy/sell, uncomment in add_features()
def calculate_trade_imbalance(df: pd.DataFrame) -> pd.Series:
    tb = df["trades_buy"].astype("float64")
    ts = df["trades_sell"].astype("float64")
    return _safe_div(tb - ts, tb + ts)

def calculate_dollar_imbalance(df: pd.DataFrame) -> pd.Series:
    vb = df["volume_quote_buy"].astype("float64")
    vs = df["volume_quote_sell"].astype("float64")
    return _safe_div(vb - vs, vb + vs)

def calculate_imbalance(df: pd.DataFrame) -> pd.Series:
    vb = df["volume_buy"].astype("float64")
    vs = df["volume_sell"].astype("float64")
    return _safe_div(vb - vs, vb + vs)

def calculate_imbalance_ema(df: pd.DataFrame, span: int = 50, strict_past: bool = True) -> pd.Series:
    imb = calculate_imbalance(df) if not "imbalance" in df.columns else df["imbalance"]
    return _ema(imb, span=span, strict_past=strict_past)

# -----------------------------
# VWAP microstructure
# -----------------------------
def calculate_vwap_dist(df: pd.DataFrame) -> pd.Series:
    c = df["price_close"].astype("float64")
    v = df["vwap"].astype("float64")
    return _safe_div(c - v, v)

def calculate_vwap_dist_ema(df: pd.DataFrame, span: int = 50, strict_past: bool = True) -> pd.Series:
    return _ema(calculate_vwap_dist(df), span=span, strict_past=strict_past)

# -----------------------------
# Activity-adjusted volatility
# -----------------------------
def calculate_abs_ret_per_sec(df: pd.DataFrame) -> pd.Series:
    r = calculate_log_return(df)
    dur = calculate_duration_s(df) if not "duration_s" in df.columns else df["duration_s"]
    return _safe_div_time(r, dur.astype("float64")).abs()

# -----------------------------
# Regime features (past-only rolling)
# -----------------------------
def calculate_regime_pack(
    df: pd.DataFrame,
    windows: Sequence[int],
    strict_past: bool = True,
) -> Dict[str, pd.Series]:
    out: Dict[str, pd.Series] = {}

    r = calculate_log_return(df)
    absr = r.abs()
    dur = calculate_duration_s(df) if not "duration_s" in df.columns else df["duration_s"].astype("float64")
    imb = calculate_imbalance (df) if not "imbalance"  in df.columns else df["imbalance" ].astype("float64")
    dps = calculate_dollar_per_sec(df)

    for w in windows:
        out[f"z_absret_{w}"] = _rolling_zscore(absr, w, strict_past=strict_past)
        out[f"z_duration_{w}"] = _rolling_zscore(dur, w, strict_past=strict_past)
        out[f"z_imbalance_{w}"] = _rolling_zscore(imb, w, strict_past=strict_past)
        out[f"z_dps_{w}"] = _rolling_zscore(dps, w, strict_past=strict_past)
        out[f"roll_std_ret_{w}"] = _rolling_std(r, w, strict_past=strict_past)

    return out

# -----------------------------
# Meta feature
# -----------------------------
def calculate_auction_quality(df: pd.DataFrame, strict_past: bool = True, span: int = 100) -> pd.Series:
    speed = calculate_speed_index(df)
    impact = calculate_hl_range_pct(df) + calculate_abs_ret_per_sec(df)
    quality_raw = -_safe_div(impact, (speed + EPS))
    return _ema(quality_raw, span=span, strict_past=strict_past)

# -----------------------------
# Main orchestrator
# -----------------------------
def add_features(
    df: pd.DataFrame,
    strict_past: bool = True,
    regime_windows: Sequence[int] = (50, 100, 200, 500),
    add_auction_quality: bool = True,
) -> pd.DataFrame:
    out = df.copy()

    out["duration_s"] = calculate_duration_s(out)
    out["imbalance" ] = calculate_imbalance(out)

    # Price/shape
    out["log_ret"] = calculate_log_return(out)
    out["hl_range_pct"] = calculate_hl_range_pct(out)
    out["close_pos_range"] = calculate_close_position_in_range(out)
    for k, s in calculate_wick_ratios(out).items():
        out[k] = s

    # Speed/liquidity
    out["duration_log"] = calculate_duration_log(out)
    out["dollar_per_sec"] = calculate_dollar_per_sec(out)
    out["speed_index"] = calculate_speed_index(out)

    # Orderflow
    out["dollar_imbalance"] = calculate_dollar_imbalance(out)
    # out["trade_imbalance"] = calculate_trade_imbalance(out)
    out["imbalance_ema_50"] = calculate_imbalance_ema(out, span=50, strict_past=strict_past)

    # VWAP microstructure
    out["vwap_dist"] = calculate_vwap_dist(out)
    out["vwap_dist_ema_50"] = calculate_vwap_dist_ema(out, span=50, strict_past=strict_past)

    # Noise/risk
    out["abs_ret_per_sec"] = calculate_abs_ret_per_sec(out)

    # Regime pack
    if regime_windows:
        reg = calculate_regime_pack(out, windows=regime_windows, strict_past=strict_past)
        for k, s in reg.items():
            out[k] = s

    # Meta (always computed)
    if add_auction_quality:
        out["auction_quality_ema100"] = calculate_auction_quality(
            out, strict_past=strict_past, span=100
        )

    out.replace([np.inf, -np.inf], np.nan, inplace=True)
    return out
