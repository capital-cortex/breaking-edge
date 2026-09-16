#%%
import os
import time
import duckdb
import requests
import numpy as np
import pandas as pd
from tqdm import tqdm
from zipfile import ZipFile
from dotenv import load_dotenv
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from typing import Callable, Literal, Any

import sys
if "" in sys.path:
    sys.path.remove("")
    sys.path.append("")
from binance.client import Client # type: ignore

from be import utils
from be.time import PlatformTime, Timestamper

#%% Data Binance Vision
load_dotenv()
class DataBinanceVision:
    URL_F                   = "https://data.binance.vision/data/{market_type}/{futures_type}/{period}/{data_type}/{symbol}/{interval}/{symbol}-{specifier}-{date}.zip"
    BOOKDEPTH_TIMESTAMP_MIN = "2023-01-01"
    FUTURES_TIMESTAMP_MIN   = "2019-09-08"
    TRADES_COLUMNS          = ["id", "price", "volume", "volume_quote", "time", "is_sell", "is_best_match"]
    TRADES_CLEAN_COLUMNS    = [      "price", "volume", "volume_quote"]
    KLINES_AGGRULES         = {
        "time"              : "first",
        "price_open"        : "first",
        "price_high"        : "max"  ,
        "price_low"         : "min"  ,
        "price_close"       : "last" ,
        "volume_abs"        : "sum"  ,
        "time_close"        : "last" ,
        "volume_quote_abs"  : "sum"  ,
        "trades_abs"        : "sum"  ,
        "volume_buy"        : "sum"  ,
        "volume_quote_buy"  : "sum"  ,
        "ignore"            : "sum"  ,
    }
    KLINES_CLEAN_AGGRULES   = {
        "price_open"        : "first",
        "price_high"        : "max"  ,
        "price_low"         : "min"  ,
        "price_close"       : "last" ,
        "volume_abs"        : "sum"  ,
        "volume_buy"        : "sum"  ,
        "volume_sell"       : "sum"  ,
        "time_close"        : "last" ,
        "volume_quote_abs"  : "sum"  ,
        "volume_quote_buy"  : "sum"  ,
        "volume_quote_sell" : "sum"  ,
        "trades_abs"        : "sum"  ,
    }
    KLINES_COLUMNS          = [*KLINES_AGGRULES.keys()]
    KLINES_CLEAN_COLUMNS    = [*KLINES_CLEAN_AGGRULES.keys()]
    KLINES_LIVE_FILENAME_F  = "{symbol}-{interval}-live.csv"
    KLINESRDY_FILENAME_F    = ".klinesrdy{interval}"
    DATETIME_MIN            = "1970-01"
    DATETIME_MAX            = "2170-01"
    
    # TODO: input validation
    # TODO: disabled symbols _ -> ~
    # TODO: rename file_type -> source
    # TODO: add resample_data(df, by) function
    # TODO: rename klines_resample() -> _resample_data_by_time()
    # TODO: rename klines_resample_volume_bars() -> _resample_data_by_volume()
    # TODO: rename get_data_klines_agg() -> get_data_agg(symbol, by)
    # TODO: remove intern functions with _ (like _migrate_data_klines())
    # TODO: add db_only mode (delete zips)? Track downloaded data? Make get_symbols() work?
    def __init__(
            self,
            market_type   : Literal["futures"  , "spot"             ] = "spot"   ,
            futures_type  : Literal["um"       , "cm"     ,""       ] = "um"     ,
            period        : Literal["daily"    , "monthly", "live"  ] = "monthly",
            data_type     : Literal["bookDepth", "klines" , "trades"] = "klines" ,
            interval      : Literal["1s", "1m", "3m", "5m", "15m", "30m",
                                    "1h", "2h", "4h", "6h",  "8h", "12h",
                                    "1d", "3d", "1w", "1mo", ""     ] = "1h"     ,
            timestamp_bgn : str                                       = "1970-01",
            timestamp_end : str                                       = "2170-01",
    ) -> None:
        if period      == "live"      : timestamp_bgn = self.DATETIME_MIN; timestamp_end = self.DATETIME_MAX
        if data_type   == "bookDepth" : market_type   = "futures"; period = "daily"
        if data_type   != "klines"    : interval      = ""
        if market_type == "spot"      : futures_type  = ""
        self.market_type   = market_type 
        self.futures_type  = futures_type
        self.period        = period      
        self.data_type     = data_type   
        self.interval      = interval    
        self.timestamp_bgn = timestamp_bgn
        self.timestamp_end = timestamp_end
        self.path = os.getenv("be_dbv") or ""
        assert self.path, "Environment variable 'be_dbv' does not exist."
        self.db_path          = os.path.join(self.path, "data_binance_vision.duckdb")
        self.klinesrdy_path_f = os.path.join(self.path, "spot", "live", "klines", self.KLINESRDY_FILENAME_F)
    
    def get_url(self, symbol: str, date: str) -> str:
        specifier = self.interval if self.data_type == "klines" else self.data_type
        return "https://" + self.URL_F.format(
            market_type  = self.market_type ,
            futures_type = self.futures_type,
            period       = self.period      ,
            data_type    = self.data_type   ,
            interval     = self.interval    ,
            symbol       = symbol           ,
            specifier    = specifier        ,
            date         = date             ,
        ).replace("https://", "").replace("//", "/")
    
    def get_table_name(self) -> str:
        return "_".join([self.market_type, self.futures_type, self.data_type, self.interval]).replace("__", "_")
    
    def get_path(self, symbol: str, date: str) -> str:
        relative_path = self.get_url(symbol, date).split("/data/")[-1]
        return os.path.normpath(os.path.join(self.path, relative_path))
    
    def get_dir(self, symbol: str) -> str:
        return os.path.dirname(self.get_path(symbol, ""))
    
    def get_symbols(self) -> list[str]:
        head, tail = os.path.split(self.get_dir(""))
        symbol_dir = os.path.join(head, tail.replace(self.interval, ""))
        symbols = os.listdir(symbol_dir)
        return [s for s in symbols if not s.startswith("_") and not s.startswith(".")]
    
    def get_klinesrdy_mtime(self) -> float:
        return os.path.getmtime(self.klinesrdy_path_f.format(interval=self.interval))
    
    @staticmethod
    def _clean_zip(full_path: str) -> None:
        with ZipFile(full_path, "r") as zip:
            if len(zip.namelist()) == 1:
                return # zip ok
        old_path = full_path + ".tmp"
        name = os.path.basename(full_path).split(".")[0]
        os.rename(full_path, old_path)
        with ZipFile(old_path, "r") as zin:
            with ZipFile(full_path, "w") as zout:
                for item in zin.infolist():
                    if item.filename.startswith(name):
                        zout.writestr(item, zin.read(item.filename))
        os.remove(old_path)
    
    def download_data(self, symbol: str, migrate: bool = True, delay: float = 0.5) -> bool:
        # TODO: support list of symbols (move independant code to top out of symbols for-loop, migrate after for-loop)
        symbol_dir = self.get_dir(symbol)
        os.makedirs(symbol_dir, exist_ok=True)
        match self.period:
            case "daily":
                freq, format = "D" , "%Y-%m-%d"
            case "monthly":
                freq, format = "MS", "%Y-%m"
            case _:
                raise ValueError(f"Invalid 'period': Must be one of 'daily', 'monthly'. Got: '{self.period}'.")
        # get symbol timestamp_min
        if self.market_type == "spot":
            time.sleep(delay)
            params = {
                "symbol"    : symbol,
                "interval"  : "1M"  ,
                "startTime" : "0"   ,
                "limit"     : "1"   ,
            }
            try:
                response = requests.get("https://api.binance.com/api/v3/klines", params=params)
            except Exception as e:
                print(f"Failed to download all data for '{symbol}':")
                print(e)
                return False
            if response.status_code != 200:
                print(f"Failed to download all data for '{symbol}':")
                print(f"Response ({response.status_code}): {response.json()}")
                return False
            first_time : int = response.json()[0][0]
            timestamp_min = pd.to_datetime(first_time, unit="ms").strftime(format)
        # self.market_type == "futures"
        elif self.data_type == "bookDepth":
            timestamp_min = pd.to_datetime(self.BOOKDEPTH_TIMESTAMP_MIN)
        else:
            timestamp_min = pd.to_datetime(self.FUTURES_TIMESTAMP_MIN)
        timestamp_max = pd.Timestamp.now("utc").tz_localize(None).strftime(format)
        timestamp_bgn = max(pd.to_datetime(self.timestamp_bgn), pd.to_datetime(timestamp_min))
        timestamp_end = min(pd.to_datetime(self.timestamp_end), pd.to_datetime(timestamp_max))
        pd_date_range = pd.date_range(timestamp_bgn, timestamp_end, freq=freq, inclusive="left")
        dates = [date for pd_date in pd_date_range if not os.path.exists(self.get_path(symbol, date := pd_date.strftime(format)))]
        newest_filename      = max(filenames) if (filenames := os.listdir(symbol_dir)) else self.DATETIME_MAX
        newest_file_date     = "-".join(part for part in newest_filename.split(".")[0].split("-") if part.isnumeric())
        newest_file_datetime = pd.to_datetime(newest_file_date)
        desc_url = "/".join(self.get_url(symbol, "").split("/data/")[-1].split("/")[:-1])
        first_file_found = False
        for date in tqdm(dates, desc=f"Downloading .../{desc_url}", unit="file"):
            time.sleep(delay)
            url  = self.get_url (symbol, date)
            path = self.get_path(symbol, date)
            try:
                response = requests.get(url, stream=True)
            except Exception as e:
                print(f"Failed to download {"" if first_file_found else "all "}data for '{symbol}':")
                print(e)
                return False
            if response.status_code == 404:
                if not first_file_found and pd.to_datetime(date) < newest_file_datetime:
                    continue
                print(f"Failed to download {"" if first_file_found else "all "}data for '{symbol}':")
                print(f"Response (404): File '{os.path.basename(path)}' does not exist.")
                return False
            if response.status_code != 200:
                print(f"Failed to download {"" if first_file_found else "all "}data for '{symbol}':")
                print(f"Response ({response.status_code}): {response.json()}")
                return False
            first_file_found = True
            with open(path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
            self._clean_zip(path)
        if self.data_type == "klines" and migrate and dates:
            timestamp_tmp = self.timestamp_bgn
            self.timestamp_bgn = dates[0]
            self.migrate_data(symbol)
            self.timestamp_bgn = timestamp_tmp
        return True
    
    def get_filenames(self, data_dir: str) -> list[str]:
        filenames = os.listdir(data_dir)
        for i in reversed(range(len(filenames))):
            timestamp_file = pd.to_datetime("-".join(n for n in filenames[i].split(".")[0].split("-") if n.isnumeric()))
            if (not filenames[i].endswith(".zip")
                or pd.isna(timestamp_file)
                or timestamp_file <  pd.to_datetime(self.timestamp_bgn)
                or timestamp_file >= pd.to_datetime(self.timestamp_end)
            ):
                filenames.pop(i)
        file_timestamps = [pd.to_datetime("-".join(part for part in filename.split(".")[0].split("-") if part.isnumeric())) for filename in filenames]
        match self.period:
            case "daily":
                timestamps = [file_timestamps[0] + pd.DateOffset(days  =i) for i in range(len(file_timestamps))]
            case "monthly":
                timestamps = [file_timestamps[0] + pd.DateOffset(months=i) for i in range(len(file_timestamps))]
            case _:
                timestamps = file_timestamps
        assert file_timestamps == timestamps, f"Gaps in '{data_dir}' detected. Use download_data method to update."
        return filenames
    
    @staticmethod
    def _clean_time(df: pd.DataFrame) -> pd.DataFrame:
        time_columns      = [c for c in df.columns if "time" in c]
        df[time_columns] *= [utils.get_time_factor(df.iloc[0][col]) for col in time_columns]
        return df
    
    def _get_columns(self) -> list[str] | None:
        match self.data_type:
            case "bookDepth" : return None
            case "klines"    : return self.KLINES_COLUMNS
            case "trades"    : return self.TRADES_COLUMNS
            case _           : raise ValueError(f"Invalid 'data_type': Must be one of 'trades', 'klines', 'bookDepth'. Got: '{self.data_type}'.")
    
    def _read_csv(self, symbol: str) -> pd.DataFrame:
        data_dir = self.get_dir(symbol)
        filenames = self.get_filenames(data_dir) if self.period != "live" else os.listdir(data_dir)
        assert filenames, f"No files found in '{data_dir}' from '{self.timestamp_bgn}' to '{self.timestamp_end}'."
        columns = self._get_columns()
        dfs = [pd.read_csv(os.path.join(data_dir, filename), names=columns) for filename in filenames]
        dfs = [self._clean_time(df) for df in dfs]
        df  = pd.concat(dfs)
        df.index = pd.to_datetime(df.time, unit="s")
        return df
    
    def connect_db(self, mode: Literal["r", "w"] = "r", retry_seconds: int | float = 1) -> duckdb.DuckDBPyConnection:
        first_try = True
        while True:
            try:
                con = duckdb.connect(self.db_path, read_only=mode == "r")
                con.execute("SET preserve_insertion_order = false;")
                if not first_try:
                    print()
                return con
            except Exception as e:
                if first_try:
                    first_try = False
                    print(e)
                    print(f"Retrying every {retry_seconds} seconds", end="")
                print(".", end="", flush=True)
                time.sleep(retry_seconds)
    
    def _read_db(self, symbol: str) -> pd.DataFrame:
        con = self.connect_db()
        query = f"""
            SELECT * FROM {self.get_table_name()}
            WHERE symbol = '{symbol}'
              AND time  >= '{pd.to_datetime(self.timestamp_bgn)}'
              AND time  <  '{pd.to_datetime(self.timestamp_end)}'
            ORDER BY time
        """
        df = con.query(query).to_df()
        con.close()
        df.index = df.time
        df = df[self.KLINES_CLEAN_COLUMNS]
        return df
    
    @staticmethod
    def trades_clean_columns(df: pd.DataFrame, columns: list[str] = TRADES_CLEAN_COLUMNS) -> pd.DataFrame:
        df["volume"      ] = df.volume       * (-df.is_sell.astype(float) * 2.0 + 1.0)
        df["volume_quote"] = df.volume_quote * (-df.is_sell.astype(float) * 2.0 + 1.0)
        return df[columns]
    
    @staticmethod
    def klines_clean_columns(df: pd.DataFrame, columns: list[str] = KLINES_CLEAN_COLUMNS) -> pd.DataFrame:
        df["volume_sell"      ] = df.volume_abs       - df.volume_buy
        df["volume_quote_sell"] = df.volume_quote_abs - df.volume_quote_buy
        return df[columns]
    
    def get_data(self, symbol: str, file_type: Literal["csv", "db"] = "db", errors: Literal["raise", "empty"] = "raise") -> pd.DataFrame:
        match self.data_type:
            case "bookDepth" : raise NotImplementedError()
            case "klines"    : return self.get_data_klines(symbol, file_type, errors)
            case "trades"    : return self.get_data_trades(symbol,            errors)
            case _           : raise ValueError(f"Invalid 'data_type': Must be one of 'trades', 'klines', 'bookDepth'. Got: '{self.data_type}'.")
    
    def get_data_trades(self, symbol: str, errors: Literal["raise", "empty"] = "raise") -> pd.DataFrame:
        # TODO: file_type == "db": NotImplemented
        try:
            df = self._read_csv(symbol)
            assert df.index.is_monotonic_increasing  , "Timestamps are not monotonic increasing."
            assert (df.id.diff().dropna() == 1).all(),  "Trade IDs are not monotonic increasing."
        except:
            match errors:
                case "raise": raise
                case "empty": return pd.DataFrame()
        return self.trades_clean_columns(df)
    
    def get_data_klines(self, symbol: str, file_type: Literal["csv", "db"] = "db", errors: Literal["raise", "empty"] = "raise") -> pd.DataFrame:
        assert file_type in ["csv", "db"], f"Invalid arg: 'file_type' must be one of 'csv', 'db'. Got: '{file_type}'."
        if file_type == "db" and self.period != "live":
            df = self._read_db(symbol)
            if not (np.diff(utils.get_bar_seconds(df)) == 0.0).all():
                print(f"Index for symbol '{symbol}' from '{self.timestamp_bgn}' to '{self.timestamp_end}' is not evenly spaced.")
            if df.empty and errors == "raise":
                raise AssertionError(f"No data in db for symbol '{symbol}' from '{self.timestamp_bgn}' to '{self.timestamp_end}'.")
            return df
        try:
            df = self._read_csv(symbol)
            orig_len = len(df)
            df = df.resample(utils.interval_to_freq(self.interval), label="left").asfreq()
            if (filled_bars := len(df) - orig_len) != 0:
                print(f"Filled {filled_bars}/{len(df)} bar{"" if filled_bars == 1 else "s"} for symbol '{symbol}' from '{self.timestamp_bgn}' to '{self.timestamp_end}'.")
            df = utils.fillna(df)
            df["time_close"] = df.index + utils.interval_to_dateoffset(self.interval)
        except:
            match errors:
                case "raise": raise
                case "empty": return pd.DataFrame()
        return self.klines_clean_columns(df)
    
    def get_data_klines_agg(self, symbol: str, interval: str, file_type: Literal["csv", "db"] = "db", errors: Literal["raise", "empty"] = "raise") -> pd.DataFrame:
        df = self.get_data_klines(symbol, file_type, errors=errors)
        if df.empty:
            return df
        return self.klines_resample(df, interval)
    
    def klines_resample(self, df: pd.DataFrame, interval: str) -> pd.DataFrame:
        return df.resample(utils.interval_to_freq(interval), label="left").agg(self.KLINES_CLEAN_AGGRULES) # type: ignore
    
    def get_data_klines_vbars(self, symbol: str, volume_threshold: float, drop_last: bool = True) -> pd.DataFrame:
        df = self.get_data_klines(symbol)
        return self.klines_resample_volume_bars(df, volume_threshold, "asset", drop_last)
    
    def get_data_klines_vqbars(self, symbol: str, volume_threshold: float, drop_last: bool = True) -> pd.DataFrame:
        df = self.get_data_klines(symbol)
        return self.klines_resample_volume_bars(df, volume_threshold, "quote", drop_last)
    
    def klines_resample_volume_bars(self, df: pd.DataFrame, volume_threshold: float, volume_type: Literal["asset", "quote"], drop_last: bool = True) -> pd.DataFrame:
        match volume_type:
            case "asset": volume_col = "volume_abs"
            case "quote": volume_col = "volume_quote_abs"
            case _      : raise ValueError("`volume_type` must be 'asset' or 'quote'")
        vol = df[volume_col]
        vol = vol.to_numpy(dtype=np.float64)
        cum_vol = np.cumsum(vol)
        n_full_bars = int(cum_vol[-1] / volume_threshold)
        if n_full_bars > 0:
            thresholds     = np.arange(1, n_full_bars + 1, dtype=np.float64) * volume_threshold
            bar_ends       = np.searchsorted(cum_vol, thresholds, side="left")
            bar_ends       = np.clip(bar_ends, 0, len(df) - 1)
            bar_ends       = np.unique(bar_ends)
            bar_starts     = np.empty_like(bar_ends)
            bar_starts[0]  = 0
            bar_starts[1:] = bar_ends[:-1] + 1
        else:
            bar_ends   = np.array([], dtype=np.intp)
            bar_starts = np.array([], dtype=np.intp)
        last = bar_ends[-1] if len(bar_ends) else -1
        if last < len(df) - 1:
            bar_starts = np.append(bar_starts, last    + 1)
            bar_ends   = np.append(bar_ends  , len(df) - 1)
        idx = bar_starts.tolist()
        
        p_open  = df["price_open"      ].to_numpy(np.float64)
        p_high  = df["price_high"      ].to_numpy(np.float64)
        p_low   = df["price_low"       ].to_numpy(np.float64)
        p_close = df["price_close"     ].to_numpy(np.float64)
        v_abs   = df["volume_abs"      ].to_numpy(np.float64)
        vq_abs  = df["volume_quote_abs"].to_numpy(np.float64)
        t_abs   = df["trades_abs"      ].to_numpy(np.float64)
        v_buy   = df["volume_buy"      ].to_numpy(np.float64)
        vq_buy  = df["volume_quote_buy"].to_numpy(np.float64)
        
        t_abs_sum  = np.add.reduceat(t_abs , idx)
        v_abs_sum  = np.add.reduceat(v_abs , idx)
        v_buy_sum  = np.add.reduceat(v_buy , idx)
        vq_abs_sum = np.add.reduceat(vq_abs, idx)
        vq_buy_sum = np.add.reduceat(vq_buy, idx)
        highs  = np.maximum.reduceat(p_high, idx)
        lows   = np.minimum.reduceat(p_low , idx)
        opens  = p_open [bar_starts]
        closes = p_close[bar_ends  ]
        time_open  = df.index[bar_starts]
        time_close = df["time_close"].to_numpy()[bar_ends]
        vwap = np.where(v_abs_sum > 0, vq_abs_sum / v_abs_sum, opens)
        resampled_df = pd.DataFrame(
            data = {
                "price_open"        : opens                  ,
                "price_high"        : highs                  ,
                "price_low"         : lows                   ,
                "price_close"       : closes                 ,
                "volume_abs"        : v_abs_sum              ,
                "volume_buy"        : v_buy_sum              ,
                "volume_sell"       : v_abs_sum - v_buy_sum  ,
                "time_close"        : time_close             ,
                "volume_quote_abs"  : vq_abs_sum             ,
                "volume_quote_buy"  : vq_buy_sum             ,
                "volume_quote_sell" : vq_abs_sum - vq_buy_sum,
                "trades_abs"        : t_abs_sum              ,
                "vwap"              : vwap                   ,
            },
            index = time_open
        )
        if drop_last and resampled_df.iloc[-1][volume_col] != volume_threshold:
            return resampled_df[:-1]
        return resampled_df
    
    def migrate_data(self, symbols: list[str] | str | None = None) -> None:
        symbols = symbols or self.get_symbols()
        if isinstance(symbols, str):
            symbols = [symbols]
        match self.data_type:
            case "bookDepth" : raise NotImplementedError()
            case "klines"    : self._migrate_data_klines(symbols)
            case "trades"    : raise NotImplementedError()
            case _           : raise ValueError(f"Invalid 'data_type': Must be one of 'trades', 'klines', 'bookDepth'. Got: '{self.data_type}'.")
    
    def _migrate_data_klines(self, symbols: list[str]) -> None:
        con = self.connect_db("w")
        table_name = self.get_table_name()
        TABLE_SCHEMA = """
            symbol            VARCHAR  ,
            time              TIMESTAMP,
            price_open        DOUBLE   ,
            price_high        DOUBLE   ,
            price_low         DOUBLE   ,
            price_close       DOUBLE   ,
            volume_abs        DOUBLE   ,
            volume_buy        DOUBLE   ,
            volume_sell       DOUBLE   ,
            time_close        TIMESTAMP,
            volume_quote_abs  DOUBLE   ,
            volume_quote_buy  DOUBLE   ,
            volume_quote_sell DOUBLE   ,
            trades_abs        BIGINT   ,
            PRIMARY KEY (symbol, time)
        """
        con.execute(f"CREATE TABLE IF NOT EXISTS {table_name} ({TABLE_SCHEMA});")
        iter_symbols = symbols if len(symbols) == 1 else tqdm(symbols, desc="Migrating symbols to duckdb")
        for symbol in iter_symbols:
            df = self.get_data_klines(symbol, file_type="csv", errors="empty")
            if df.empty:
                continue
            df["time"  ] = df.index
            df["symbol"] = symbol
            df = df[["symbol", "time", *self.KLINES_CLEAN_COLUMNS]]
            con.execute(f"INSERT INTO {table_name} SELECT * FROM df ON CONFLICT (symbol, time) DO NOTHING")
        con.execute(f"""
            CREATE TABLE {table_name}_ordered ({TABLE_SCHEMA});
            INSERT INTO {table_name}_ordered SELECT * FROM {table_name}
            ORDER BY symbol, time;
            DROP TABLE {table_name};
            ALTER TABLE {table_name}_ordered RENAME TO {table_name};
        """)
        con.close()

#%% Binance Api Helper
class BinanceApiHelper():
    
    API_LIMIT     =  1000   # trades, klines
    REQUEST_FREQ  =     3.0 # Hz
    RECV_WINDOW   = 60000   # in ms
    PRECISION_USD =     2   # digits after point
    
    client                   : Client
    exchange_info            : dict[str, Any]
    time_factor              : float
    last_request_time        : float
    timestamp_offset_seconds : float
    
    def __init__(
            self                          ,
            api_key    : str | None = None,
            api_secret : str | None = None,
    ) -> None:
        if not api_key or not api_secret:
            print("Running with public binance api client.")
        else:
            print(f"Running with binance api key.")
        while True:
            try:
                self.client            = Client(api_key, api_secret)
                self.exchange_info     = self.client_get_exchange_info()
                self.time_factor       = utils.get_time_factor(self.client_get_server_time())
                self.last_request_time = time.time()
                assert self.set_timestamp_offset_seconds()
                break
            except Exception as e:
                print("An error occurred while initializing BinanceApiHelper:")
                print(e)
                print("Retrying in 60 seconds.")
                time.sleep(60)
    
    # Pylance type: ignore Functions
    def client_get_exchange_info(self) -> dict[str, Any]:
        return self.client.get_exchange_info() # type: ignore
    def client_get_server_time(self) -> int:
        return self.client.get_server_time()["serverTime"] # type: ignore
    def client_get_historical_trades(self, **params: Any) -> list[dict[str, Any]]:
        return self.client.get_historical_trades(**params) # type: ignore
    def client_get_klines(self, **params: Any) -> list[list[Any]]:
        return self.client.get_klines(**params) # type: ignore
    def client_create_order(self, **params: Any) -> dict[str, Any]:
        return self.client.create_order(**params) # type: ignore
    def client_create_oco_order(self, **params: Any) -> dict[str, Any]:
        return self.client.create_oco_order(**params) # type: ignore
    def client_get_open_orders(self, **params: Any) -> list[dict[str, Any]]:
        return self.client.get_open_orders(**params) # type: ignore
    def client_get_order(self, **params: Any) -> dict[str, Any]:
        return self.client.get_order(**params) # type: ignore
    def client_get_account(self, **params: Any) -> dict[str, Any]:
        return self.client.get_account(**params) # type: ignore
    
    # Time Functions
    def set_timestamp_offset_seconds(self) -> bool:
        try:
            self.timestamp_offset_seconds = self.client_get_server_time() * self.time_factor - time.time()
            self.client.timestamp_offset  = int(self.timestamp_offset_seconds / self.time_factor)
        except Exception as e:
            print(f"Failed to update timestamp offset:")
            print(e)
            return False
        return True
    
    def get_binance_datetime(self) -> pd.Timestamp:
        return pd.to_datetime(time.time() + self.timestamp_offset_seconds, unit="s")
    
    # Binance Klines Data Handlers
    @retry( # ~2:30 min if all tries fail
            wait         = wait_exponential(multiplier=1, min=2, max=30),
            stop         = stop_after_attempt(10)                       ,
            retry        = retry_if_exception_type(Exception)           ,
            reraise      = True                                         ,
            before_sleep = lambda s: print(f"Retrying ({s.attempt_number}/9)...")                           ,
    )
    def get_klines_df(
            self                               ,
            symbol     : str                   ,
            interval   : str                   ,
            limit      : int        = API_LIMIT,
            start_time : int | None = None     ,
            end_time   : int | None = None     ,
    ) -> pd.DataFrame:
        try:
            raw_klines = self.client_get_klines(symbol=symbol, interval=interval, limit=limit, startTime=start_time, endTime=end_time)
        except Exception as e:
            print(f"Failed to get '{interval}' klines for '{symbol}':")
            print(e)
            raise e
        return pd.DataFrame(raw_klines, columns=DataBinanceVision.KLINES_COLUMNS).apply(pd.to_numeric)
    
    # Binance Trades Data Handlers
    @staticmethod
    def _dicts_to_df(dicts: list[dict[str, Any]]) -> pd.DataFrame:
        new_data : list[dict[str, Any]] = [
            {
                "id"           :       t["id"          ] ,
                "price"        : float(t["price"       ]),
                "qty"          : float(t["qty"         ]),
                "time"         :       t["time"        ] ,
                "isBuyerMaker" :       t["isBuyerMaker"] ,
            }
            for t in dicts
        ]
        # Convert the list to a DataFrame
        new_rows_df = pd.DataFrame(new_data).set_index("id")
        return new_rows_df
    
    def get_historical_trades_save(
            self                                    ,
            symbol       : str                      ,
            limit        : int        = API_LIMIT   ,
            from_id      : int | None = None        ,
            request_freq : float      = REQUEST_FREQ,
            tries        : int        = 45*2        ,
            delay        : int        = 30          ,
    ) -> list[dict[str, Any]]:
        trades  = []
        attempt = 0
        for attempt in range(1, tries + 1):
            time_to_wait = self.last_request_time + 1 / request_freq - time.time()
            if time_to_wait > 0:
                time.sleep(time_to_wait)
            try:
                self.last_request_time = time.time()
                trades = self.client_get_historical_trades(symbol=symbol, limit=limit, fromId=from_id)
                valid_data = True
                trade_id = from_id if from_id else trades[0]["id"]
                for trade in trades:
                    valid_data &= isinstance(              trade["id"          ] , int  ) \
                              and isinstance(pd.to_numeric(trade["price"       ]), float) \
                              and isinstance(pd.to_numeric(trade["qty"         ]), float) \
                              and isinstance(pd.to_numeric(trade["quoteQty"    ]), float) \
                              and isinstance(              trade["time"        ] , int  ) \
                              and isinstance(              trade["isBuyerMaker"] , bool ) \
                              and isinstance(              trade["isBestMatch" ] , bool ) \
                              and len(trade) == 7 \
                              and trade["id"] == trade_id
                    trade_id += 1
                assert valid_data, "Corrupted data received from Binance API"
                break
            except Exception as e:
                print(f"An error occurred while fetching historical trades on attempt {attempt}/{tries}:")
                print(e)
                if attempt < tries:
                    time.sleep(delay)  # Wait before retrying
                    continue
                print("Raising Exception. Goodbye :(")
                raise  # Re-raise if all tries fail
        if attempt > 1:
            print(f"Succeeded fetching historical trades on attempt {attempt}/{tries} :)")
        return trades
    
    def preload_trades(
            self                                      ,
            symbol        : str                       ,
            interval      : str                       ,
            history_bars  : int                       ,
            df_index_last : pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        if df_index_last is None:
            epoch_time = self.get_binance_datetime().timestamp() - pd.Timedelta(interval).total_seconds() * history_bars
            oldest_time = pd.to_datetime(epoch_time, unit="s").floor(interval).timestamp() / self.time_factor
        else:
            oldest_time = (df_index_last.timestamp() + pd.Timedelta(interval).total_seconds()) / self.time_factor
        
        columns = ["id", "price", "qty", "time", "isBuyerMaker"]
        dtypes = {
            "id"           : "uint64" ,
            "price"        : "float32",
            "qty"          : "float32",
            "time"         : "uint64" ,
            "isBuyerMaker" : "bool"   ,
        }
        trade_df = pd.DataFrame({col: pd.Series(dtype=dtypes[col]) for col in columns})
        trade_df.set_index("id", inplace=True)
        
        newest_trade, = self.get_historical_trades_save(symbol=symbol, limit=1)
        newest_id     = newest_trade["id"]
        newest_time   = newest_trade["time"]
        
        progress_bar = tqdm(total=round((newest_time - oldest_time) * self.time_factor, 1), desc=f"Preloading trades: {symbol}, Progress", unit="s")
        
        # load backwards
        trade_time = newest_time
        trade_id   = newest_id
        while trade_time >= oldest_time:
            trades_list = self.get_historical_trades_save(symbol=symbol, from_id=trade_id)
            if len(trade_df) == 0:
                trade_df = self._dicts_to_df(trades_list)
            else:
                trade_df = pd.concat([self._dicts_to_df(trades_list), trade_df])
            trade_id  -= self.API_LIMIT
            trade_time = trade_df.iloc[0]["time"]
            
            progress = min(newest_time - trade_time, newest_time - oldest_time) * self.time_factor
            progress_bar.n = round(progress, 1)
            progress_bar.refresh() # type: ignore
        
        # load forwards
        trade_id = trade_df.index[-1] + 1
        while True:
            trades_list = self.get_historical_trades_save(symbol=symbol, from_id=trade_id)
            if not trades_list:
                break
            trade_df = pd.concat([trade_df, self._dicts_to_df(trades_list)])
            trade_id += self.API_LIMIT
        
        progress_bar.close()
        return trade_df
    
    def update_trades(
            self                                ,
            symbol        : str                 ,
            interval      : str                 ,
            df_index_last : pd.Timestamp        ,
            trade_df      : pd.DataFrame        ,
            request_freq  : float = REQUEST_FREQ,
            warn          : bool  = True        ,
    ) -> pd.DataFrame:       
        # append new trades
        trade_id = trade_df.index[-1] + 1
        while True:
            trades_list = self.get_historical_trades_save(symbol=symbol, limit=self.API_LIMIT, from_id=trade_id, request_freq=request_freq)
            if not trades_list:
                break
            trade_df = pd.concat([trade_df, self._dicts_to_df(trades_list)])
            trade_id = trades_list[-1]["id"] + 1
            
            # if warn: print out when irregularities occur
            if warn:
                irregular_ids = not all(np.diff(trade_df.index) == 1)
                if irregular_ids:
                    print(self.get_binance_datetime())
                    print(f"Warning: Found irregularities in trade_df index for {symbol}")
            
            if len(trades_list) < self.API_LIMIT:
                break
        
        # remove old trades
        last_bar_epoch_time = df_index_last.timestamp() + pd.Timedelta(interval).total_seconds()
        last_bar_idx        = trade_df["time"].searchsorted(last_bar_epoch_time / self.time_factor, side="left")
        trade_df            = trade_df.iloc[last_bar_idx - 1 :]
        
        return trade_df
    
    def resample(self, trade_df: pd.DataFrame, interval: str) -> pd.DataFrame:
        trade_df_copy = trade_df.copy()
        trade_df_copy.set_index("time", inplace=True)
        trade_df_copy.index = pd.to_datetime(trade_df_copy.index * self.time_factor, unit="s")
        
        trade_df_copy["qty_buy" ] = trade_df_copy["qty"].where(~trade_df_copy["isBuyerMaker"], 0)
        trade_df_copy["qty_sell"] = trade_df_copy["qty"].where( trade_df_copy["isBuyerMaker"], 0)
        
        resampled_df = trade_df_copy.resample(interval).apply({
            "price"    : ["first", "max", "min", "last"],  # Open, High, Low, Close
            "qty_buy"  : "sum",
            "qty_sell" : "sum",
        })
        
        # Rename columns for clarity
        resampled_df.columns = ["price_open", "price_high", "price_low", "price_close", "volume_buy", "volume_sell"]
        
        return resampled_df
    
    def update_df(self, df: pd.DataFrame, trade_df: pd.DataFrame, interval: str, history_bars: int) -> pd.DataFrame:
        # seconds since epoche of newly completed bar
        this_bar_epoch_time = self.get_binance_datetime().floor(interval).timestamp()
        last_bar_epoch_time = pd.DatetimeIndex(df.index)[-1].timestamp() + pd.Timedelta(interval).total_seconds()
        n_new_bars = int((this_bar_epoch_time - last_bar_epoch_time) / pd.Timedelta(interval).total_seconds())
        
        # indexes for newly completed bar
        this_bar_idx = trade_df["time"].searchsorted(this_bar_epoch_time / self.time_factor, side="left")
        last_bar_idx = trade_df["time"].searchsorted(last_bar_epoch_time / self.time_factor, side="left")
        
        # add newly completed bar to df, drop older bars
        new_df = self.resample(trade_df[last_bar_idx : this_bar_idx], interval)
        if not len(new_df):
            for ibar in range(n_new_bars):
                bar_time = pd.to_datetime(last_bar_epoch_time + pd.Timedelta(interval).total_seconds() * ibar, unit="s")
                new_df.loc[bar_time] = [0.0] * df.shape[1]
                new_df.loc[bar_time, [col for col in df.columns if "price" in col]] = df.iloc[-1]["price_close"]
        
        updated_df = pd.concat([df, new_df])
        updated_df = updated_df.iloc[-history_bars:]
        
        return updated_df
    
    # Trading Functions
    def get_quantity_precision(self, symbol: str) -> int:
        s = next(s for s in self.exchange_info["symbols"] if s["symbol"    ] == symbol    )
        f = next(f for f in s["filters"]                  if f["filterType"] == "LOT_SIZE")
        step_size = float(f["stepSize"])
        return int(round(-np.log10(step_size)))
        
    
    def get_price_precision(self, symbol: str) -> int:
        s = next(s for s in self.exchange_info["symbols"] if s["symbol"    ] == symbol        )
        f = next(f for f in s["filters"]                  if f["filterType"] == "PRICE_FILTER")
        tick_size = float(f["tickSize"])
        return int(round(-np.log10(tick_size)))
    
    def execute_with_time_error_retry(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if getattr(e, "code", None) == -1021:
                print("A timestamp offset error occurred:")
                print(e)
                print("Updating timestamp offset and retrying.")
                assert self.set_timestamp_offset_seconds()
                result = func(*args, **kwargs)
                print("Retry succeeded :)")
                return result
            raise e
    
    def create_entry_order(
            self                                  ,
            symbol      : str                     ,
            side        : Literal["BUY", "SELL"]  ,
            qty_asset   : str | None = None       ,
            qty_usd     : str | None = None       ,
            recv_window : int        = RECV_WINDOW,
    ) -> dict[str, Any]:
        return self.execute_with_time_error_retry(
            self.client_create_order   ,
            symbol        = symbol     ,
            side          = side       ,
            type          = "MARKET"   ,
            quantity      = qty_asset  ,
            quoteOrderQty = qty_usd    ,
            recvWindow    = recv_window,
        )
    
    def create_unentry_order(
            self                                ,
            symbol      : str                   ,
            side        : Literal["BUY", "SELL"],
            qty_asset   : str                   ,
            recv_window : int = RECV_WINDOW     ,
    ) -> dict[str, Any]:
        precision   = self.get_quantity_precision(symbol)
        qty_asset_f = round(float(qty_asset), precision)
        qty_asset   = f"{qty_asset_f:.{precision}f}"
        return self.execute_with_time_error_retry(
            self.client_create_order,
            symbol     = symbol     ,
            side       = side       ,
            type       = "MARKET"   ,
            quantity   = qty_asset  ,
            recvWindow = recv_window,
        )
    
    def create_exit_order(
            self                                ,
            symbol      : str                   ,
            side        : Literal["BUY", "SELL"],
            qty_asset   : str                   ,
            sl_price    : str                   ,
            tp_price    : str                   ,
            recv_window : int = RECV_WINDOW     ,
    ) -> dict[str, Any]:
        precision   = self.get_quantity_precision(symbol)
        qty_asset_f = round(float(qty_asset), precision)
        qty_asset   = f"{qty_asset_f:.{precision}f}"
        
        if side == "SELL": # long
            above_type  = "TAKE_PROFIT"
            above_price = tp_price
            below_type  = "STOP_LOSS"
            below_price = sl_price
        else: # short
            above_type  = "STOP_LOSS"
            above_price = sl_price
            below_type  = "TAKE_PROFIT"
            below_price = tp_price
        
        return self.execute_with_time_error_retry(
            self.client_create_oco_order ,
            symbol          = symbol     ,
            side            = side       , # "SELL" for long, "BUY" for short
            quantity        = qty_asset  ,
            aboveType       = above_type , # "TAKE_PROFIT" for long, "STOP_LOSS" for short
            aboveStopPrice  = above_price,
            belowType       = below_type , # "TAKE_PROFIT" for short, "STOP_LOSS" for long
            belowStopPrice  = below_price,
            recvWindow      = recv_window,
        )
    
    def enter_trade_save(
            self                                  ,
            symbol        : str                   ,
            side          : Literal["BUY", "SELL"],
            qty_asset     : float | None          ,
            qty_usd       : float | None          ,
            sl_price      : str                   ,
            tp_price      : str                   ,
            recv_window   : int = RECV_WINDOW     ,
            precision_usd : int = PRECISION_USD   ,
            revert_tries  : int = 30              ,
            delay         : int = 30              ,
    ) -> tuple[dict[str, Any], dict[str, Any]] | None:
        reverse_side = "BUY" if side == "SELL" else "SELL"
        precision = self.get_quantity_precision(symbol)
        
        qty_usd_str   = None
        qty_asset_str = None
        if qty_asset is not None:
            qty_asset     = np.floor(float(qty_asset) * 10 ** precision) * 10 ** -precision
            qty_asset_str = f"{qty_asset:.{precision}f}"
        if qty_usd is not None:
            qty_usd     = np.floor(float(qty_usd) * 10 ** precision_usd) * 10 ** -precision_usd
            qty_usd_str = f"{qty_usd:.{precision_usd}f}"
        
        try:
            entry_order = self.create_entry_order(symbol, side, qty_asset_str, qty_usd_str, recv_window)
        except Exception as e:
            print("An error occurred while placing the entry order:")
            print(e)
            return None
        
        # get received amount in assets qty
        executed_qty_asset_str = entry_order["executedQty"]
        
        exit_oco_order = {}
        try:
            exit_oco_order = self.create_exit_order(symbol, reverse_side, executed_qty_asset_str, sl_price, tp_price, recv_window)
        except Exception as e:
            print("An error occurred while placing the exit order:")
            print(e)
            for attempt in range(1, revert_tries + 1):
                try:
                    self.create_unentry_order(symbol, reverse_side, executed_qty_asset_str, recv_window)
                    print("Reverted entry order")
                    return None
                except Exception as e:
                    print(f"An error occurred while reverting the entry on attempt {attempt}/{revert_tries}:")
                    print(e)
                    if attempt < revert_tries:
                        time.sleep(delay)
                        continue
                    print("Raising Exception. Goodbye :(")
                    raise  # Re-raise if all tries fail
        return entry_order, exit_oco_order
    
    def get_open_orders(self, symbol: str, recv_window: int = RECV_WINDOW) -> list[dict[str, Any]]:
        return self.execute_with_time_error_retry(
            self.client_get_open_orders,
            symbol     = symbol        ,
            recvWindow = recv_window   ,
        )
    
    def get_order_save(
            self                           ,
            symbol      : str              ,
            order_id    : int              ,
            recv_window : int = RECV_WINDOW,
            tries       : int = 45*2       ,
            delay       : int = 30         ,
    ) -> dict[str, Any]:
        order : dict[str, Any] = {}
        attempt = 0
        for attempt in range(1, tries + 1):
            try:
                order = self.execute_with_time_error_retry(
                    self.client_get_order   ,
                    symbol     = symbol     ,
                    orderId    = order_id   ,
                    recvWindow = recv_window,
                )
                break
            except Exception as e:
                print(f"An error occurred while fetching order on attempt {attempt}/{tries}:")
                print(e)
                if attempt < tries:
                    time.sleep(delay)  # Wait before retrying
                else:
                    print("Raising Exception. Goodbye :(")
                    raise  # Re-raise if all tries fail
        if attempt > 1:
            print(f"Succeeded fetching order on attempt {attempt}/{tries} :)")
        return order
    
    def get_account_balance(self, asset: str, recv_window: int = RECV_WINDOW) -> dict[str, float]:
        account_info = self.execute_with_time_error_retry(
            self.client_get_account ,
            recvWindow = recv_window,
        )
        balances = account_info["balances"]
        for balance in balances:
            if asset == balance["asset"]:
                return {"free": float(balance["free"]), "locked": float(balance["locked"])}
        raise ValueError(f"Asset {asset} not in balances of client.")
    
    # Function to log trades
    def log_trade( 
            self                     ,
            log_path : str           ,
            order    : dict[str, Any],
            trade_id : int           ,
            symbol   : str           ,
            sl_price : str = "NA"    ,
            tp_price : str = "NA"    ,
    ) -> None:
        is_entry         = "STOP_LOSS" not in order["type"] and "TAKE_PROFIT" not in order["type"]
        trade_time : int = order["transactTime" if is_entry else "updateTime"]
        datetime         = str(pd.to_datetime(trade_time * self.time_factor, unit="s").round("ms"))
        if len(datetime) != 26:
            datetime += "." + "0" * (26 - 1 - len(datetime))
        fill : dict[str, Any] = {
            "time"      : datetime                    ,
            "order_id"  : order["orderId"]            ,
            "trade_id"  : trade_id                    ,
            "symbol"    : symbol                      ,
            "side"      : order["side"]               ,
            "qty"       : order["executedQty"]        ,
            "quote_qty" : order["cummulativeQuoteQty"],
            "sl_price"  : sl_price                    ,
            "tp_price"  : tp_price                    ,
        }
        line = ",".join(str(v) for v in fill.values())
        print(line.replace(",", ", "))
        with open(log_path, "a") as f:
            f.write(f"{line}\n")

#%% Kliner
class Kliner():
    
    #platform_time   : PlatformTime
    timestamper      : Timestamper
    dbv              : DataBinanceVision
    bah              : BinanceApiHelper
    data_path_f      : str
    interval_symbols : dict[str, list[str]]
    history_seconds  : float
    timer_seconds    : float
    log_path         : str
    intervals        : list[str]
    symbols          : list[str]
    symbol_intervals : dict[str, list[str]]
    symbol_klines    : dict[str, pd.DataFrame]
    
    def __init__(
            self                                   ,
            interval_symbols : dict[str, list[str]],
            history_seconds  : float               ,
            timer_seconds    : float               ,
            log_path         : str | None = None   ,
    ) -> None:
        self.platform_time = PlatformTime()
        self.timestamper   = Timestamper(log_path=log_path)
        self.timestamper.update()
        self.dbv = DataBinanceVision(period="live", interval="")
        self.bah = BinanceApiHelper()
        self.data_path_f = os.path.join(self.dbv.get_dir("{symbol}"), "{interval}", DataBinanceVision.KLINES_LIVE_FILENAME_F)
        self.interval_symbols = interval_symbols
        self.history_seconds  = history_seconds
        self.timer_seconds    = timer_seconds
        self.intervals = utils.get_sorted_intervals(list(self.interval_symbols.keys()))
        self.symbols   = list({symbol for interval in self.intervals for symbol in self.interval_symbols[interval]})
        self.symbol_intervals = {
            symbol: [interval for interval in self.intervals if symbol in self.interval_symbols[interval]]
            for symbol in self.symbols
        }
        for symbol in self.symbols:
            for interval in self.symbol_intervals[symbol]:
                data_path = self.data_path_f.format(symbol=symbol, interval=interval)
                data_dir  = os.path.dirname(self.data_path_f.format(symbol=symbol, interval=interval))
                if not os.path.exists(data_dir):
                    os.makedirs(data_dir)
        self.symbol_klines = {
            symbol: 
                pd.read_csv(data_path, names=DataBinanceVision.KLINES_COLUMNS)
                if os.path.exists(data_path := self.data_path_f.format(symbol=symbol, interval=self.symbol_intervals[symbol][0]))
                else pd.DataFrame(columns=DataBinanceVision.KLINES_COLUMNS)
            for symbol in self.symbols
        }
        self.on_timer_next_time = time.time()
    
    def on_timer(self) -> None:
        t = time.time()
        if t < self.on_timer_next_time:
            return
        self.bah.set_timestamp_offset_seconds()
        self.on_timer_next_time = t + self.timer_seconds
    
    def klines_update(self, symbol: str, interval: str, bar_time: float) -> bool:
        klines = self.symbol_klines[symbol]
        interval_seconds = pd.Timedelta(interval).total_seconds()
        history_bars = int(self.history_seconds / interval_seconds)
        while True:
            limit = int((bar_time - klines.iloc[-1].time * self.bah.time_factor) / interval_seconds - 1) if len(klines) else history_bars
            start_time = int((bar_time - interval_seconds * limit) / self.bah.time_factor)
            limit = min(limit, self.bah.API_LIMIT)
            try:
                new_klines = self.bah.get_klines_df(symbol, interval, limit, start_time)
                klines = pd.concat([klines, new_klines]) if len(klines) else new_klines
                assert (klines.time.diff().dropna() * self.bah.time_factor == interval_seconds).all(), "Irregular time column in klines df."
                klines = klines[-history_bars:]
                if klines.iloc[-1].time * self.bah.time_factor == bar_time - interval_seconds:
                    break
            except AssertionError as e:
                print(f"Failed to update '{interval}' klines for '{symbol}':")
                print(e)
                print("Resetting klines df.")
                self.symbol_klines[symbol] = pd.DataFrame(columns=DataBinanceVision.KLINES_COLUMNS)
                return False
            except Exception as e:
                print(f"Failed to update '{interval}' klines for '{symbol}':")
                print(e)
                return False
        self.symbol_klines[symbol] = klines
        return True
    
    def files_update(self, datetime: pd.Timestamp, bar_time: float) -> None:
        for interval in self.intervals:
            if bar_time != datetime.floor(interval).timestamp():
                continue
            failed_symbols : list[str] = []
            for symbol in self.interval_symbols[interval]:
                if self.symbol_intervals[symbol][0] == interval:
                    # update klines
                    if not self.klines_update(symbol, interval, bar_time):
                        failed_symbols.append(symbol)
                        continue
                    self.symbol_klines[symbol].to_csv(self.data_path_f.format(symbol=symbol, interval=interval), header=False, index=False)
                else:
                    # resample from klines
                    klines = self.symbol_klines[symbol]
                    if not len(klines):
                        failed_symbols.append(symbol)
                        continue
                    klines.index = pd.to_datetime(klines.time * self.bah.time_factor, unit="s")
                    klines = klines.resample(utils.interval_to_freq(interval), label="left").agg(self.dbv.KLINES_AGGRULES) # type: ignore
                    if klines.iloc[-1].time * self.bah.time_factor == bar_time:
                        klines = klines[:-1] # drop open bar from lower interval agg (normally there is none)
                    if klines.iloc[-1].time * self.bah.time_factor != bar_time - pd.Timedelta(interval).total_seconds():
                        failed_symbols.append(symbol)
                        continue
                    klines.to_csv(self.data_path_f.format(symbol=symbol, interval=interval), header=False, index=False)
            if failed_symbols:
                print(f"Did not update '{interval}' klines. Failed symbols: {str(failed_symbols)[1:-1]}.")
                continue
            with open(self.dbv.klinesrdy_path_f.format(interval=interval), "w"):
                pass
            print(f"Updated '{interval}' klines for {str(self.interval_symbols[interval])[1:-1]}.")
    
    def run(self) -> None:
        last_bar_time = self.bah.get_binance_datetime().floor(self.intervals[0]).timestamp()
        while True:
            if self.platform_time.was_synced():
                self.bah.set_timestamp_offset_seconds()
            datetime = self.bah.get_binance_datetime()
            bar_time = datetime.floor(self.intervals[0]).timestamp()
            if bar_time == last_bar_time:
                time.sleep(0.1)
                self.timestamper.update(self.bah.get_binance_datetime())
                self.on_timer()
                continue
            last_bar_time = bar_time
            
            self.timestamper.update(self.bah.get_binance_datetime())
            self.files_update(datetime, bar_time)
