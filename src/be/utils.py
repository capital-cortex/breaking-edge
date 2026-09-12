import time
import numpy as np
import pandas as pd
from typing import cast

def invert_symbol(df: pd.DataFrame) -> pd.DataFrame:
    buy_cols = [col for col in df.columns if "_buy" in col]
    bid_cols = [col for col in df.columns if "_bid" in col]
    inv_cols = [col for col in df.columns if col.startswith("price_")]
    
    df_inv = df.copy()
    
    for col in buy_cols:
        df_inv = df_inv.rename(columns={
            col                         : col.replace("_buy", "_sell"),
            col.replace("_buy", "_sell"): col
        })
    for col in bid_cols:
        df_inv = df_inv.rename(columns={
            col                        : col.replace("_bid", "_ask"),
            col.replace("_bid", "_ask"): col
        })
    df_inv[inv_cols] = 1.0 / df_inv[inv_cols]
    df_inv = df_inv.rename(columns={
        "price_low" : "price_high",
        "price_high": "price_low"
    })
    
    return df_inv

def adjust_change_if_negative(change: float) -> float:
    if change < 0.0:
        return change / (1.0 - change)
    return change

def get_numbers(string: str) -> int | float:
    s = "".join(c for c in string if c.isdigit() or c == ".")
    return float(s) if "." in s else int(s)

def get_letters(string: str) -> str:
    return "".join([char for char in string if char.isalpha()])

def interval_to_freq(interval: str) -> str:
    return interval.replace("mo", "MS").replace("w", "W").replace("d", "D").replace("m", "min")

def interval_to_timedelta(interval: str) -> pd.Timedelta:
    """Approximates Xmo with X*30D"""
    return pd.Timedelta(interval if not get_letters(interval) == "mo" else f"{get_numbers(interval) * 30}D")

def interval_to_dateoffset(interval: str) -> pd.DateOffset:
    value = get_numbers(interval)
    unit  = get_letters(interval)
    assert isinstance(value, int), "Interval value must be whole number."
    if   unit == "m":
        return pd.DateOffset(minutes = value)
    elif unit == "h":
        return pd.DateOffset(hours   = value)
    elif unit == "d":
        return pd.DateOffset(days    = value)
    elif unit == "w":
        return pd.DateOffset(weeks   = value)
    elif unit == "mo":
        return pd.DateOffset(months  = value)
    else:
        raise ValueError(f"Unsupported interval: {interval}")

def get_sorted_intervals(intervals: list[str], reverse: bool = False) -> list[str]:
    timedeltas = [interval_to_timedelta(itvl) for itvl in intervals]
    sorted_intervals, _ = zip(*sorted(zip(intervals, timedeltas), key=lambda x: x[1], reverse=reverse))
    return list(sorted_intervals)

def get_largest_interval_upto(interval: str, intervals: list[str]) -> str:
    sorted_intervals = get_sorted_intervals(intervals, reverse=True)
    try:
        return next(itvl for itvl in sorted_intervals if interval_to_timedelta(itvl) <= interval_to_timedelta(interval))
    except StopIteration:
        raise ValueError(f"No largest interval found upto '{interval}'")

def get_bar_seconds(df: pd.DataFrame | pd.Series) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
    time_diffs  = df.index.to_series().diff(-1).dt.total_seconds()
    bar_seconds = time_diffs.ffill().to_numpy()
    return cast(np.ndarray[tuple[int], np.dtype[np.float64]], bar_seconds)

def fillna(df: pd.DataFrame) -> pd.DataFrame:
    df["price_close"] = df.price_close.ffill()
    df["time_close" ] = df.time_close.ffill()
    filled_idxs = df.price_open.isna()
    df.loc[filled_idxs, ["price_open", "price_high", "price_low"]] = np.array([df[filled_idxs].price_close.values] * 3).T
    df = df.fillna(0)
    return df

def get_time_factor(t: float | int) -> float:
    current_time = time.time()  # Current time in seconds since epoch
    unit_factors = [1e-0, 1e-3, 1e-6, 1e-9]  # seconds, milliseconds, microseconds, nanoseconds
    differences = [(abs(current_time - t * factor), factor) for factor in unit_factors]
    _, time_factor = min(differences, key=lambda x: x[0])
    return time_factor
