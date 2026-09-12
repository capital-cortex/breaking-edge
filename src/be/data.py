import os as _os
import pandas as _pd
import typing as _typing
from . import utils as _ut
from .yahoo import DataYahooFinance
from .binance import DataBinanceVision

__all__ = ["DataYahooFinance", "DataBinanceVision", "get_data_klines_agg"]

def get_data_klines_agg(
        symbol        : str             ,
        interval      : str  = "1d"     ,
        timestamp_bgn : str  = "1970-01",
        timestamp_end : str  = "2170-01",
        fill_closed   : bool = False    ,
) -> _pd.DataFrame:
    """
    fill_closed: only effects DataYahooFinance
    """
    if _os.getenv("be_dbv"):
        dbv = DataBinanceVision(
            interval      = ""           ,
            timestamp_bgn = timestamp_bgn,
            timestamp_end = timestamp_end,
        )
        if symbol in dbv.get_symbols():
            # resampling from klines with less or equal interval, regardless of unit (market always open)
            dbv_intervals = _os.listdir(dbv.get_dir(symbol))
            dbv.interval  = _ut.get_largest_interval_upto(interval, dbv_intervals)
            return dbv.get_data_klines_agg(symbol, interval)
    
    dyf = DataYahooFinance(
        interval      = ""           ,
        timestamp_bgn = timestamp_bgn,
        timestamp_end = timestamp_end,
        fill_closed   = fill_closed  ,
    )
    # resampling only from klines with less or equal interval and same unit (same trading hours)
    unit          = _ut.get_letters(interval)
    dyf_intervals = _typing.get_args(dyf.Intervals)
    dyf_intervals = [di for di in dyf_intervals if _ut.get_letters(di) == unit]
    dyf.interval  = _ut.get_largest_interval_upto(interval, dyf_intervals)
    return dyf.get_data_klines_agg(symbol, interval)
