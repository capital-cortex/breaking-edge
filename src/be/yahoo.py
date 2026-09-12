import pandas as pd
import yfinance as yf # type: ignore
from . import utils

from typing import Literal

class DataYahooFinance:
    
    Intervals = Literal["1m" , "2m", "5m", "15m", "30m", "60m",
                        "1h" ,
                        "1d" , "5d",
                        "1w" ,
                        "1mo", "3mo", ""] # "90m" and "4h" will mess up time_close
    COLUMN_RENAMES = {
        "Open"   : "price_open" ,
        "High"   : "price_high" ,
        "Low"    : "price_low"  ,
        "Close"  : "price_close",
        "Volume" : "volume_abs" ,
    }
    COLUMN_AGGRULES = {
        "price_open"  : "first",
        "price_high"  : "max"  ,
        "price_low"   : "min"  ,
        "price_close" : "last" ,
        "volume_abs"  : "sum"  ,
        "time_close"  : "last" ,
    }
    
    def __init__(
            self                                 ,
            interval      : Intervals = "1d"     ,
            timestamp_bgn : str       = "1970-01",
            timestamp_end : str       = "2170-01",
            fill_closed   : bool      = False    ,
    ) -> None:
        self.interval      = interval    
        self.timestamp_bgn = timestamp_bgn
        self.timestamp_end = timestamp_end
        self.fill_closed   = fill_closed
    
    def get_data_klines(self, symbol: str) -> pd.DataFrame:
        timestamp_bgn = f"{pd.to_datetime(self.timestamp_bgn):%F}"
        timestamp_end = f"{pd.to_datetime(self.timestamp_end):%F}"
        if timestamp_bgn == "1970-01-01": # zero timestamp can cause yfinance to return empty df
            timestamp_bgn = "1970-01-02"
        now  = pd.Timestamp.now("utc").tz_localize(None).floor("D")
        unit = utils.get_letters(self.interval)
        if   unit == "m" and (now - pd.to_datetime(timestamp_bgn)).days >=  60:
            timestamp_bgn = f"{(now - pd.Timedelta( 60 - 1, "D")):%F}"
        elif unit == "h" and (now - pd.to_datetime(timestamp_bgn)).days >= 730:
            timestamp_bgn = f"{(now - pd.Timedelta(730 - 1, "D")):%F}"
        df = yf.download( # type: ignore
            tickers           = symbol                          ,
            start             = timestamp_bgn                   ,
            end               = timestamp_end                   ,
            interval          = self.interval.replace("w", "wk"),
            multi_level_index = False                           ,
            progress          = False                           ,
        )
        assert df is not None, "Did not receive df from yfinance."
        df_index = pd.DatetimeIndex(df.index).tz_localize(None)
        df.index = df_index
        df["time_close"] = df.index + utils.interval_to_dateoffset(self.interval)
        day_bgn_time = df_index.time.min()
        day_bgn_idx = next(i for i, t in enumerate(df_index.time) if t == day_bgn_time)
        df = df[day_bgn_idx:]
        df.index = df.index.rename("time")
        df = df.rename(columns=self.COLUMN_RENAMES)
        if not self.fill_closed:
            return df
        df = self._klines_resample(df, self.interval)
        return df[self.COLUMN_AGGRULES.keys()]
    
    def get_data_klines_agg(self, symbol: str, interval: str) -> pd.DataFrame:
        df = self.get_data_klines(symbol)
        return self._klines_resample(df, interval)
    
    def _klines_resample(self, df: pd.DataFrame, interval: str) -> pd.DataFrame:
        subdaily = utils.get_letters(interval) in ["m", "h"]
        if subdaily:
            offset = df.index[0] - df[:1].resample(utils.interval_to_freq(interval), label="left").asfreq().index[0]
        else:
            offset = None
        resampled_df = df.resample(utils.interval_to_freq(interval), label="left", offset=offset).agg(self.COLUMN_AGGRULES) # type: ignore
        if self.fill_closed:
            resampled_df = utils.fillna(df)
            resampled_df["time_close"] = resampled_df.index + utils.interval_to_dateoffset(self.interval)
            return resampled_df
        resampled_df = resampled_df.dropna()
        if not subdaily:
            resampled_df["time_close"] = pd.Timestamp(resampled_df.time_close).ceil("1D") # type: ignore
        return resampled_df