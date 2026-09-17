#%%
# pyright: strict
# pandas ships no py.typed stub package here (pandas-stubs not installed), so its own
# inline types are incomplete under strict mode -- not something this file can fix.
# pyright: reportMissingTypeStubs=false
import os
import time
import re
import duckdb
import requests
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from typing import Literal, Any, TypedDict, Required, cast
from urllib.parse import quote

#%% Data Alpaca Vision
load_dotenv()

class AlpacaDataError(RuntimeError):
    pass


class AlpacaBar(TypedDict):
    """One raw bar as returned by Alpaca's /v2/stocks/{symbol}/bars endpoint."""
    t  : str   # RFC3339 timestamp
    o  : float
    h  : float
    l  : float
    c  : float
    v  : float
    n  : float
    vw : float


class AlpacaBarsPage(TypedDict):
    bars            : list[AlpacaBar] | None
    next_page_token : str | None
    symbol          : str


class AlpacaCalendarDay(TypedDict):
    date            : str
    open            : str
    close           : str
    session_open    : str
    session_close   : str
    settlement_date : str


class AlpacaClock(TypedDict):
    timestamp  : str
    is_open    : bool
    next_open  : str
    next_close : str


class AlpacaAccount(TypedDict):
    cash            : str
    buying_power    : str
    portfolio_value : str


class AlpacaAsset(TypedDict, total=False):
    symbol       : str
    fractionable : bool
    close_price  : str


class AlpacaPosition(TypedDict, total=False):
    symbol        : str
    current_price : str


class AlpacaOrder(TypedDict, total=False):
    id                : Required[str]
    side              : Required[str]
    submitted_at      : Required[str]
    type              : str
    status            : str
    filled_at         : str | None
    filled_qty        : str
    filled_avg_price  : str | None


class DataAlpacaMarkets:
    """Download, cache and query US stock bars from Alpaca's Market Data API.

    Mirrors ``DataBinanceVision``'s data/interval handling exactly: ``interval`` is a
    constructor-only setting (never passed again to ``download_data``/``get_data_klines``)
    and determines the NATIVE provider bars downloaded, the csv path, the duckdb table
    identity and what ``get_data_klines`` returns -- ``interval="30m"`` (the default)
    downloads and stores Alpaca's actual native 30-minute bars, not some other interval
    resampled into 30m. ``download_data`` fetches bars in bulk (contiguous missing/stale
    months merged into as few paginated requests as possible) and splits them back into
    the same per-month local csv files, ``migrate_data`` loads those files into duckdb,
    and ``get_data_klines`` reads from the duckdb cache (or the csv files directly)
    instead of hitting the API on every call.

    Contract (verified against the live API, see audit notes in git history):
      * Source      : Alpaca Market Data API v2. ``interval`` (constructor-only, default
                       '30m' -- chosen as the canonical research base because 30 minutes
                       divides the 09:30-16:00 RTH session exactly, with no leftover
                       partial bucket, unlike native calendar-hour-aligned '1h' bars)
                       selects the NATIVE provider bar via ``INTERVAL_TO_TIMEFRAME``:
                       '1m'->1Min, '5m'->5Min, '15m'->15Min, '30m'->30Min, '1h'->1Hour,
                       '1d'->1Day (every entry verified live against the bars endpoint;
                       unsupported values raise ValueError -- fail closed, no invented
                       timeframe strings). Native '1h' remains fully supported and
                       explicitly selectable; it is simply no longer the default. ``feed
                       ='sip'`` = consolidated tape, all US exchanges, 15-minute delay on
                       the free Basic plan. ``feed='iex'`` = IEX exchange only (~2-3% of
                       volume), no delay. Default: 'sip'.
      * Adjustment  : ``adjustment='raw'`` (the default) returns UNADJUSTED historical
                       prints -- a stock split or reverse split shows up as a sudden
                       price/volume jump on the effective date (verified: NVDA's
                       2024-06-10 10-for-1 split, GE's 2021-08-02 1-for-8 reverse split).
                       Pass ``adjustment='split'`` (or 'all' to also fold in dividends)
                       for a continuous, split-adjusted series with proportionally
                       scaled volume -- required for any return/backtest calculation
                       spanning a split date.
      * Session     : Regular trading hours only by default (``regular_hours_only=True``),
                       09:30 to the day's official close, per symbol-agnostic close time
                       fetched from Alpaca's own ``/v2/calendar`` endpoint (handles
                       holidays and early-close/half days, e.g. 13:00 on the day before
                       July 4th -- verified against real 2024-07-03 volume data).
                       ``regular_hours_only=False`` includes pre-/post-market prints.
                       For native ``interval='1d'`` bars this flag has NO effect: Alpaca's
                       daily bar already represents the whole official trading day (its
                       't' timestamp is a calendar-day label, not a session open time),
                       so the minute-of-day RTH filter is skipped entirely for daily data.
      * Timezone    : all timestamps are stored/returned as timezone-naive UTC (project
                       convention, matching ``DataBinanceVision``). Session/DST logic
                       is computed internally in America/New_York wall-clock time and
                       converted back to UTC before being stored or returned.
      * Bar labeling: ``time`` is the bar's open (left-labeled), ``time_close`` is
                       open + ``interval``. ``get_data_klines`` returns exactly the
                       NATIVE provider bars for the configured ``interval`` -- e.g.
                       ``DataAlpacaMarkets(interval='1h').get_data_klines(...)`` is Alpaca's own
                       native 1-hour bar, calendar-hour aligned (e.g. 09:00-10:00, mixing
                       pre-market with regular session -- NOT session-anchored). This is
                       exactly why '30m' (not '1h') is the canonical default: 30 minutes
                       divides RTH cleanly, so no native bucket straddles the 09:30 open.
                       ``get_data_klines_agg(symbol, interval, alignment=...)`` resamples
                       the native source into a COARSER interval using one of two
                       EXPLICIT, deliberately different aggregation semantics (never a
                       hidden default the caller has to guess):
                         - ``alignment='session'`` (the default): for sub-daily targets,
                           groups by NY-local trading date FIRST and resamples each
                           session independently before concatenating, so a bucket can
                           never merge data across two sessions (Friday's close with
                           Monday's open, or an early-close day's last partial bar with
                           the next session) no matter how coarse the target is -- this
                           is true by construction, not merely because the overnight/
                           weekend gap happens to be larger than the bucket. Bins are
                           anchored to a FIXED 09:30 session open, not to midnight and
                           not to whatever the first available bar of a session happens
                           to be (holds even for a late-starting session: IPO, halt, a
                           query window starting mid-day) -- buckets stay 09:30-10:30,
                           10:30-11:30, ..., and the session's last bucket is intentionally
                           SHORTER when the session doesn't divide evenly (a 6.5h RTH day
                           can't be cut into whole hours -- verified: 30m->'1h' on a
                           normal day yields 09:30,...,15:30; verified on an
                           early-close day it correctly stops at the session's last
                           partial bucket, e.g. 12:30, and creates nothing after the
                           actual close). Stays aligned across DST transitions (verified
                           for both 2024 changes). For a daily-or-coarser target, 'session'
                           behaves identically to 'calendar' (see below) -- there is no
                           meaningful "session-anchored day/week/month" beyond calendar
                           trading-period boundaries.
                         - ``alignment='calendar'``: plain calendar-period resampling (no
                           09:30 anchor) -- for daily/weekly/monthly interday research,
                           grouped by the NY-local trading date so a bar is never
                           misattributed across the UTC/NY date boundary. No weekend or
                           holiday bars are ever fabricated (there is nothing to
                           aggregate on those days, so no row is produced).
                       Native ``interval='1h'`` bars and
                       ``DataAlpacaMarkets(interval='30m').get_data_klines_agg(symbol, '1h',
                       alignment='session')`` are DELIBERATELY DIFFERENT semantics for the
                       same nominal interval and are not required to (and do not) produce
                       identical bucket boundaries -- do not conflate native coarser bars
                       with this class's own session-aligned aggregation of a finer source.
                       Aggregating to a FINER interval than the native source, or to a
                       target that isn't an exact multiple of the native interval, is
                       rejected (ValueError, fail closed) -- e.g. native '30m' can never
                       produce '15m', and can't produce '20m' either.
      * Caching     : a cached month csv file always holds the FULL calendar month's
                       data, regardless of any particular instance's own timestamp_bgn/
                       timestamp_end -- those are output filters applied at read time
                       (``_read_csv``/``_read_db``), never fetch-clipping bounds. This
                       matters because two ``DataAlpacaMarkets`` instances with different
                       windows share the same on-disk cache: a file is only ever treated
                       as permanently complete once its calendar month has fully elapsed
                       in real time (independent of who happens to be asking), so a
                       narrow instance (e.g. a mid-month ``timestamp_bgn``) can never
                       write a partial file that a later, wider instance then silently
                       trusts as complete -- a real data-loss bug that was found,
                       reproduced and fixed (verified live for both directions: partial-
                       then-full and full-then-partial requests sharing one cache). An
                       in-progress (current) month is always re-fetched and overwritten
                       so it gets completed as more days become available -- it is never
                       silently frozen on whatever partial data was saved first (verified
                       live). Bulk downloads (see below) are written per month atomically
                       (temp file + rename). Cache identity (csv directory + duckdb table
                       name) includes ``interval``, ``feed``, ``adjustment`` and
                       ``regular_hours_only``, so e.g. native '1h' vs '30m', 'raw' vs
                       'split', or RTH-only vs extended-hours data can never silently
                       share the same cached files/table (verified live).
      * Performance : contiguous missing/stale months are merged into as few paginated
                       ``_request_bars`` calls as possible instead of one request per
                       month (A/B-verified: byte-identical output either way). In
                       practice Alpaca's server caps the actual page size well below the
                       requested ``limit`` for wide date ranges (observed ~200-350 bars/
                       page regardless of ``limit=10000``), so the request-count
                       reduction for a full 2016-to-now single-symbol history is modest
                       (measured: ~129 monthly requests -> ~103 paginated requests for
                       native '1h'; similar page count for '30m' despite ~2x the rows,
                       since the API's per-page cap -- not the data volume -- dominates).
                       Most of the realized wall-clock speedup (measured ~83s -> ~27s per
                       symbol) comes from applying the inter-request delay once per
                       download block instead of once per month, not from dramatically
                       bigger pages.
      * Missing data: bars are never fabricated. A halt or a day with zero prints simply
                       produces no row (or no bar at all) -- there is no forward-fill,
                       no synthetic zero-volume bar (verified against GME 2021-01-28).
      * OHLCV       : open=first, high=max, low=min, close=last, volume=sum, trades=sum
                       (standard). ``vwap`` is NOT a naive average: it is recomputed as
                       sum(sub_bar_vwap * sub_bar_volume) / sum(sub_bar_volume), i.e. a
                       volume-weighted average of the underlying bars' VWAPs.
      * Symbols     : queried literally by ticker string, no identity resolution. Alpaca
                       resolves a renamed security's FULL history (pre- and post-rename)
                       under its current ticker (verified: querying 'META' for dates
                       before the 2022-06-09 FB->META rename returns the same data as
                       'FB' did back then). The reverse is NOT safe: an old ticker can
                       be reused later by a totally unrelated security (verified: 'FB'
                       is now a ProShares ETF, unrelated to Meta) -- so this class will
                       silently return whatever security currently holds that ticker
                       for a given date range. Ticker-identity/rename mapping is
                       deliberately NOT handled here; it belongs in the caller (see the
                       historical universe builder).
    """

    Intervals = Literal["1m", "5m", "15m", "30m", "1h", "1d"]
    Feed      = Literal["sip", "iex"]
    Errors    = Literal["raise", "empty"]
    Alignment = Literal["session", "calendar"]

    URL_F                 = "https://data.alpaca.markets/v2/stocks/{symbol}/bars"
    CALENDAR_URL          = "https://api.alpaca.markets/v2/calendar"
    DATETIME_MIN          = "2016-01-01"
    DATETIME_MAX          = "2170-01"
    # library interval -> Alpaca's native bar timeframe string. Only entries verified
    # live against the bars endpoint (200 OK, real data) -- no invented timeframe strings.
    INTERVAL_TO_TIMEFRAME = {
        "1m" : "1Min" ,
        "5m" : "5Min" ,
        "15m": "15Min",
        "30m": "30Min",
        "1h" : "1Hour",
        "1d" : "1Day" ,
    }
    SIP_DELAY             = pd.Timedelta(minutes=15)
    MARKET_TIMEZONE       = "America/New_York"
    MARKET_OPEN_MINUTE    = 9 * 60 + 30
    PAGE_LIMIT            = 10_000
    ADJUSTMENTS           = {"raw", "split", "dividend", "spin-off", "all"}
    API_COLUMNS_REQUIRED  = ["t", "o", "h", "l", "c", "v", "n", "vw"]
    COLUMN_RENAMES        = {
        "o"  : "price_open" ,
        "h"  : "price_high" ,
        "l"  : "price_low"  ,
        "c"  : "price_close",
        "v"  : "volume_abs" ,
        "n"  : "trades"     ,
        "vw" : "vwap"       ,
    }
    KLINES_AGGRULES       = {
        "time"        : "first",
        "price_open"  : "first",
        "price_high"  : "max"  ,
        "price_low"   : "min"  ,
        "price_close" : "last" ,
        "volume_abs"  : "sum"  ,
        "time_close"  : "last" ,
        "trades"      : "sum"  ,
    }
    KLINES_CLEAN_AGGRULES = { # vwap excluded: computed separately as a volume-weighted average
        "price_open"  : "first",
        "price_high"  : "max"  ,
        "price_low"   : "min"  ,
        "price_close" : "last" ,
        "volume_abs"  : "sum"  ,
        "time_close"  : "last" ,
        "trades"      : "sum"  ,
    }
    KLINES_COLUMNS       = [*KLINES_AGGRULES.keys(), "vwap"]
    KLINES_CLEAN_COLUMNS = [*KLINES_CLEAN_AGGRULES.keys(), "vwap"]

    def __init__(
            self,
            api_key            : str                                ,
            api_secret         : str                                ,
            interval           : Intervals               = "30m"    ,
            timestamp_bgn      : str                     = "1970-01",
            timestamp_end      : str                     = "2170-01",
            feed               : Feed                    = "sip"    ,
            adjustment         : str                     = "split"  ,
            regular_hours_only : bool                    =  True    ,
            timeout            : float                   = 30.0     ,
            max_retries        : int                     = 3        ,
            session            : requests.Session | None = None     ,
    ) -> None:
        if interval not in self.INTERVAL_TO_TIMEFRAME:
            raise ValueError(f"Unsupported Alpaca interval: '{interval}'. Must be one of {sorted(self.INTERVAL_TO_TIMEFRAME)}.")
        if feed not in {"sip", "iex"}:
            raise ValueError(f"Invalid 'feed': Must be one of 'sip', 'iex'. Got: '{feed}'.")
        self._validate_adjustment(adjustment)
        if timeout <= 0:
            raise ValueError("'timeout' must be greater than zero.")
        if max_retries < 0:
            raise ValueError("'max_retries' must be zero or greater.")
        
        self.api_key            = api_key
        self.api_secret         = api_secret
        self.interval           = interval
        self.timestamp_bgn      = timestamp_bgn
        self.timestamp_end      = timestamp_end
        self.feed               = feed
        self.adjustment         = adjustment
        self.regular_hours_only = regular_hours_only
        self.timeout            = timeout
        self.max_retries        = max_retries
        self.session            = session or requests.Session()
        self._calendar_cache : dict[tuple[str, str], pd.DataFrame] = {}

        self.path = os.getenv("be_dam") or ""
        assert self.path, "Environment variable 'be_dam' does not exist."
        self.db_path = os.path.join(self.path, "data_alpaca_vision.duckdb")

    @classmethod
    def _validate_adjustment(cls, adjustment: str) -> None:
        adjustments = adjustment.split(",")
        if not adjustments or any(value not in cls.ADJUSTMENTS for value in adjustments):
            raise ValueError(f"Invalid Alpaca adjustment: '{adjustment}'.")
        if len(adjustments) > 1 and ({"raw", "all"} & set(adjustments)):
            raise ValueError("Alpaca adjustments 'raw' and 'all' cannot be combined with other values.")

    @staticmethod
    def _now_utc() -> pd.Timestamp:
        return pd.Timestamp.now("UTC")

    @classmethod
    def _empty_df(cls) -> pd.DataFrame:
        return pd.DataFrame(columns=cls.KLINES_CLEAN_COLUMNS, index=pd.DatetimeIndex([], name="time"))

    @classmethod
    def _parse_interval(cls, interval: str) -> tuple[int, str]:
        match = re.fullmatch(r"([1-9][0-9]*)(mo|m|h|d|w)", interval)
        if match is None:
            raise ValueError(f"Unsupported interval: '{interval}'.")
        return int(match.group(1)), match.group(2)

    @classmethod
    def _interval_timedelta(cls, interval: str) -> pd.Timedelta:
        value, unit = cls._parse_interval(interval)
        if unit == "m": return pd.Timedelta(minutes=value)
        if unit == "h": return pd.Timedelta(hours=value)
        if unit == "d": return pd.Timedelta(days=value)
        if unit == "w": return pd.Timedelta(weeks=value)
        return pd.Timedelta(days=30 * value) # "mo": approximate, resampling itself uses calendar-correct "MS"

    def get_timeframe(self) -> str:
        """Alpaca's native bar timeframe for the CURRENTLY configured ``self.interval`` --
        always computed fresh, never cached, so changing ``self.interval`` after
        construction can never leave this (or anything derived from it) stale."""
        if self.interval not in self.INTERVAL_TO_TIMEFRAME:
            raise ValueError(f"Unsupported Alpaca interval: '{self.interval}'. Must be one of {sorted(self.INTERVAL_TO_TIMEFRAME)}.")
        return self.INTERVAL_TO_TIMEFRAME[self.interval]

    def get_interval_timedelta(self) -> pd.Timedelta:
        """Duration of one native bar for the CURRENTLY configured ``self.interval``."""
        self.get_timeframe() # validates self.interval is still a supported native interval
        return self._interval_timedelta(self.interval)

    @classmethod
    def _resample_frequency(cls, interval: str) -> str:
        value, unit = cls._parse_interval(interval)
        if unit == "m":
            return f"{value}min"
        if unit == "h":
            return f"{value}h"
        if unit == "d":
            return f"{value}D"
        if unit == "w":
            return f"{value}W-MON"
        return f"{value}MS"

    def _to_utc(self, timestamp: str | pd.Timestamp) -> pd.Timestamp:
        value = pd.Timestamp(timestamp)
        if value.tzinfo is None:
            value = value.tz_localize("UTC")
        return value.tz_convert("UTC")

    @staticmethod
    def _to_rfc3339(timestamp: pd.Timestamp) -> str:
        return timestamp.isoformat().replace("+00:00", "Z")

    def _headers(self) -> dict[str, str]:
        if not self.api_key or not self.api_secret:
            raise ValueError(
                "Alpaca credentials are required. Pass 'api_key'/'api_secret' or set "
                
            )
        return {
            "APCA-API-KEY-ID"     : self.api_key   ,
            "APCA-API-SECRET-KEY" : self.api_secret,
        }

    @staticmethod
    def _retry_delay(response: requests.Response | None, attempt: int) -> float:
        if response is not None:
            retry_after = response.headers.get("Retry-After")
            if retry_after is not None:
                try:
                    return min(max(float(retry_after), 0.0), 60.0)
                except ValueError:
                    pass
            rate_limit_reset = response.headers.get("X-RateLimit-Reset")
            if rate_limit_reset is not None:
                try:
                    return min(max(float(rate_limit_reset) - time.time(), 0.0), 60.0)
                except ValueError:
                    pass
        return min(2.0 ** attempt, 30.0)

    def _request_json(self, url: str, params: dict[str, Any]) -> dict[str, Any]:
        retry_statuses = {429, 500, 502, 503, 504}
        response: requests.Response | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.get(
                    url,
                    headers=self._headers(),
                    params=params,
                    timeout=self.timeout,
                )
            except requests.RequestException:
                if attempt == self.max_retries:
                    raise
                time.sleep(self._retry_delay(None, attempt))
                continue

            if response.status_code == 200:
                try:
                    data: Any = response.json()
                except ValueError as error:
                    raise AlpacaDataError("Alpaca returned invalid JSON.") from error
                if not isinstance(data, dict):
                    raise AlpacaDataError("Alpaca returned an unexpected response shape.")
                return cast("dict[str, Any]", data)

            if response.status_code in retry_statuses and attempt < self.max_retries:
                time.sleep(self._retry_delay(response, attempt))
                continue

            message: Any = response.text
            try:
                error_data: Any = response.json()
                if isinstance(error_data, dict):
                    message = cast("dict[str, Any]", error_data).get("message", response.text)
            except ValueError:
                pass
            raise AlpacaDataError(f"Alpaca request failed ({response.status_code}): {str(message)[:500]}")

        raise AlpacaDataError("Alpaca request failed without a response.")

    def _request_bars(self, symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> list[AlpacaBar]:
        url = self.URL_F.format(symbol=quote(symbol, safe=""))
        params: dict[str, Any] = {
            "timeframe"  : self.get_timeframe()      ,
            "start"      : self._to_rfc3339(start)  ,
            "end"        : self._to_rfc3339(end)    ,
            "feed"       : self.feed                ,
            "adjustment" : self.adjustment          ,
            "limit"      : self.PAGE_LIMIT          ,
            "sort"       : "asc"                    ,
        }
        bars: list[AlpacaBar] = []
        seen_tokens: set[str] = set()
        while True:
            data = self._request_json(url, params)
            page_bars: Any = data.get("bars") or [] # Alpaca returns 'bars': null (not []) when nothing is in range
            if not isinstance(page_bars, list) or not all(isinstance(bar, dict) for bar in cast("list[Any]", page_bars)):
                raise AlpacaDataError("Alpaca returned an unexpected 'bars' payload.")
            bars.extend(cast("list[AlpacaBar]", page_bars))

            page_token = data.get("next_page_token")
            if page_token is None:
                return bars
            if not isinstance(page_token, str) or not page_token or page_token in seen_tokens:
                raise AlpacaDataError("Alpaca returned an invalid or repeated pagination token.")
            seen_tokens.add(page_token)
            params["page_token"] = page_token

    def _get_calendar(self, date_bgn: str, date_end: str) -> pd.DataFrame:
        """Official per-day regular-session close time (e.g. '13:00' on early-close days) from
        Alpaca's own trading calendar. Dates absent from the result are non-trading days
        (weekends/holidays). Open is always '09:30' for equities, so only close is tracked."""
        cache_key = (date_bgn, date_end)
        if cache_key in self._calendar_cache:
            return self._calendar_cache[cache_key]
        params = {"start": date_bgn, "end": date_end}
        response: requests.Response | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.get(self.CALENDAR_URL, headers=self._headers(), params=params, timeout=self.timeout)
            except requests.RequestException:
                if attempt == self.max_retries:
                    raise
                time.sleep(self._retry_delay(None, attempt))
                continue
            if response.status_code == 200:
                break
            if response.status_code in {429, 500, 502, 503, 504} and attempt < self.max_retries:
                time.sleep(self._retry_delay(response, attempt))
                continue
            raise AlpacaDataError(f"Alpaca calendar request failed ({response.status_code}): {response.text[:500]}")
        else:
            raise AlpacaDataError("Alpaca calendar request failed without a response.")

        rows_raw: Any = response.json()
        if not isinstance(rows_raw, list):
            raise AlpacaDataError("Alpaca returned an unexpected 'calendar' payload.")
        rows = cast("list[AlpacaCalendarDay]", rows_raw)
        calendar: pd.DataFrame = pd.DataFrame(rows, columns=["date", "close"])
        date_col = calendar["date"]
        close_col = calendar["close"]
        calendar["date"] = pd.to_datetime(date_col).dt.date
        calendar["close_minute"] = close_col.str.slice(0, 2).astype(int) * 60 + close_col.str.slice(3, 5).astype(int)
        calendar = calendar.set_index("date")[["close_minute"]]
        self._calendar_cache[cache_key] = calendar
        return calendar

    def _bars_to_df(self, bars: list[AlpacaBar]) -> pd.DataFrame:
        if not bars:
            return pd.DataFrame(columns=self.KLINES_COLUMNS)
        df: pd.DataFrame = pd.DataFrame(cast("list[dict[str, Any]]", bars))
        missing = [column for column in self.API_COLUMNS_REQUIRED if column not in df.columns]
        if missing:
            raise AlpacaDataError(f"Alpaca bars are missing required columns: {missing}")
        index = pd.DatetimeIndex(pd.to_datetime(df["t"], utc=True, errors="coerce"))
        if index.hasnans:
            raise AlpacaDataError("Alpaca bars contain invalid timestamps.")
        df.index = index
        df = df.sort_index()
        if not cast("pd.DatetimeIndex", df.index).is_unique:
            raise AlpacaDataError("Alpaca bars contain duplicate timestamps.")

        numeric_columns = ["o", "h", "l", "c", "v", "n", "vw"]
        df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors="coerce") # type: ignore[reportUnknownMemberType]
        if df[numeric_columns].isna().to_numpy().any():
            raise AlpacaDataError("Alpaca bars contain invalid numeric values.")
        if (
            (df[["o", "h", "l", "c", "vw"]] <= 0).to_numpy().any()
            or (df[["v", "n"]] < 0).to_numpy().any()
            or (df["h"] < df[["o", "c"]].max(axis=1)).to_numpy().any()
            or (df["l"] > df[["o", "c"]].min(axis=1)).to_numpy().any()
            or (df["h"] < df["l"]).to_numpy().any()
        ):
            raise AlpacaDataError("Alpaca bars contain inconsistent prices, volume, or trade counts.")

        _, source_unit = self._parse_interval(self.interval)
        # regular_hours_only is a minute-of-day concept; it doesn't apply to native daily
        # bars, whose single 't' timestamp is a calendar-day label (Alpaca's daily bar
        # already represents the official RTH-conventioned OHLCV for the whole day, and
        # there is no separate "extended-hours daily bar" to filter against)
        if self.regular_hours_only and source_unit != "d":
            di = cast("pd.DatetimeIndex", df.index)
            market_index = di.tz_convert(self.MARKET_TIMEZONE)
            calendar = self._get_calendar(f"{market_index.min():%Y-%m-%d}", f"{market_index.max():%Y-%m-%d}")
            close_minutes = calendar["close_minute"].reindex(market_index.date).to_numpy(dtype=float) # NaN on non-trading days (weekends/holidays)
            interval_minutes = self.get_interval_timedelta().total_seconds() / 60
            bar_open_minutes  = market_index.hour * 60 + market_index.minute
            bar_close_minutes = bar_open_minutes + interval_minutes
            # keep a native bar if its [open, open+interval) window overlaps the regular
            # session at all, not only if it *starts* exactly inside it. For interval='1h'
            # the native 09:00-10:00 bucket starts before the 09:30 open but still holds
            # real regular-session trading -- requiring bar_open >= 09:30 would silently
            # drop that entire bucket instead of keeping it as Alpaca reports it (this only
            # matters for '1h': 1m/5m/15m/30m bars are always clock-aligned exactly on 09:30)
            df = df[
                (bar_close_minutes > self.MARKET_OPEN_MINUTE)
                & (bar_open_minutes <  close_minutes) # NaN comparisons are False, so non-trading days are dropped too
            ]
        if df.empty:
            return pd.DataFrame(columns=self.KLINES_COLUMNS)

        di = cast("pd.DatetimeIndex", df.index)
        output_index = di.tz_localize(None) # project convention: timestamps are stored tz-naive UTC
        df.index = output_index.rename("time")
        df = df.rename(columns=self.COLUMN_RENAMES)
        df["time"      ] = df.index
        df["time_close"] = cast("pd.DatetimeIndex", df.index) + self.get_interval_timedelta()
        return df[self.KLINES_COLUMNS]

    def _cache_tag(self) -> str:
        # adjustment and regular_hours_only change the actual data content (split-adjusted
        # vs raw prints, RTH-only vs extended hours) -- they must be part of the cache
        # identity, or two differently-configured instances would silently read/overwrite
        # each other's csv files and duckdb rows as if they were the same data
        adjustment_tag = self.adjustment.replace(",", "-")
        hours_tag = "rth" if self.regular_hours_only else "eh"
        return f"{adjustment_tag}_{hours_tag}"

    def get_table_name(self) -> str:
        return f"stocks_{self.feed}_{self._cache_tag()}_bars_{self.interval}".replace("-", "_")

    def get_dir(self, symbol: str) -> str:
        # mirrors DataBinanceVision's {market_type}/{period}/{data_type}/{symbol}/{interval} layout:
        # "stocks" stands in for market_type, "feed" for futures_type, "monthly" is the only period downloaded/migrated in bulk (daily is reserved for live/update fill-ins, like DBV)
        return os.path.join(self.path, "stocks", self.feed, self._cache_tag(), "monthly", "bars", symbol, self.interval)

    def get_path(self, symbol: str, date: str) -> str:
        return os.path.join(self.get_dir(symbol), f"{symbol}-{self.interval}-{date}.csv")

    def get_symbols(self) -> list[str]:
        symbol_dir = os.path.join(self.path, "stocks", self.feed, self._cache_tag(), "monthly", "bars")
        if not os.path.isdir(symbol_dir):
            return []
        return [s for s in os.listdir(symbol_dir) if not s.startswith("_") and not s.startswith(".")]

    def download_data(self, symbol: str, migrate: bool = True, delay: float = 0.3) -> bool:
        symbol = symbol.strip().upper()
        symbol_dir = self.get_dir(symbol)
        os.makedirs(symbol_dir, exist_ok=True)

        now = self._now_utc()
        available_until: pd.Timestamp = now - self.SIP_DELAY if self.feed == "sip" else now
        timestamp_bgn = max(self._to_utc(self.timestamp_bgn), self._to_utc(self.DATETIME_MIN))
        timestamp_end = min(self._to_utc(self.timestamp_end), available_until)
        if timestamp_bgn >= timestamp_end:
            return True

        # floor to the month start so a mid-month timestamp_bgn still yields a label for its
        # own (partial) first month, instead of pd.date_range skipping straight to next month
        range_bgn = timestamp_bgn.tz_localize(None).replace(day=1)
        pd_date_range = pd.date_range(range_bgn, timestamp_end.tz_localize(None), freq="MS", inclusive="left")
        dates: list[str] = []
        for pd_date in pd_date_range:
            date = pd_date.strftime("%Y-%m")
            # A cached month file always represents the FULL calendar month (see the
            # unclamped block_bgn/block_end below) -- never just whatever slice a
            # particular caller's own timestamp_bgn/timestamp_end happened to want. So
            # "complete" depends only on whether the calendar month has fully elapsed in
            # real time (available_until), never on this instance's own window: two
            # DataAlpacaMarkets instances with different timestamp_bgn/timestamp_end must always
            # agree on whether a given month's cache file is trustworthy. Without this, a
            # narrow instance (e.g. a mid-month timestamp_bgn) could write a partial file
            # that a later, wider instance would then silently trust as complete -- a real
            # data-loss bug that was found and must not return.
            month_is_complete = pd_date + pd.DateOffset(months=1) <= available_until.tz_localize(None)
            if month_is_complete and os.path.exists(self.get_path(symbol, date)):
                continue
            dates.append(date)

        # group consecutive missing/stale months into contiguous download blocks, so a
        # multi-year gap becomes one (internally paginated) request instead of one
        # request per month -- the persisted layout stays exactly monthly either way
        blocks: list[list[str]] = []
        for date in dates:
            if blocks and pd.Timestamp(date) == pd.Timestamp(blocks[-1][-1]) + pd.DateOffset(months=1):
                blocks[-1].append(date)
            else:
                blocks.append([date])

        for block in tqdm(blocks, desc=f"Downloading .../{symbol}/{self.interval}", unit="block"):
            time.sleep(delay)
            # fetch the full calendar range for this block -- clamped only by the hard
            # provider-data-availability bounds (DATETIME_MIN, available_until), NOT by
            # this instance's own timestamp_bgn/timestamp_end, so every write is complete
            # and safely reusable by any other instance regardless of its own window;
            # get_data_klines()/_read_csv already filters output rows to this instance's
            # actual requested range, so nothing extra ever leaks out
            block_bgn = max(self._to_utc(block[0]), self._to_utc(self.DATETIME_MIN))
            block_end = min(self._to_utc(block[-1]) + pd.DateOffset(months=1), available_until)
            try:
                bars = self._request_bars(symbol, block_bgn, block_end)
                df   = self._bars_to_df(bars)
            except Exception as e:
                print(f"Failed to download data for '{symbol}':")
                print(e)
                return False
            if df.empty:
                continue
            # split the block back into the same per-month csv files the old month-by-month
            # download produced, written atomically (tmp file + rename) so an interrupted
            # write can never leave a file that looks complete but isn't
            month_labels = df["time"].dt.strftime("%Y-%m")
            for month_label, month_df in df.groupby(month_labels):
                if month_label not in block:
                    continue # defensive: shouldn't happen, block bounds match the request range
                final_path = self.get_path(symbol, month_label)
                tmp_path = final_path + ".tmp"
                month_df.to_csv(tmp_path, header=False, index=False)
                os.replace(tmp_path, final_path)
        if migrate and dates:
            self.migrate_data(symbol)
        return True

    def get_filenames(self, data_dir: str, symbol: str) -> list[str]:
        if not os.path.isdir(data_dir):
            return []
        prefix = f"{symbol}-{self.interval}-"
        filenames: list[str] = []
        for filename in os.listdir(data_dir):
            if not (filename.startswith(prefix) and filename.endswith(".csv")):
                continue
            timestamp_file = pd.to_datetime(filename[len(prefix):-4], errors="coerce")
            if pd.isna(timestamp_file):
                continue
            # compare the file's month END against timestamp_bgn (not its start label) -- a
            # month file starting before timestamp_bgn can still hold rows on/after it (e.g. a
            # mid-month timestamp_bgn), and _read_csv trims to the exact bgn/end afterwards
            if timestamp_file + pd.DateOffset(months=1) <= pd.to_datetime(self.timestamp_bgn) or timestamp_file >= pd.to_datetime(self.timestamp_end):
                continue
            filenames.append(filename)
        return sorted(filenames)

    def _read_csv(self, symbol: str) -> pd.DataFrame:
        data_dir = self.get_dir(symbol)
        filenames = self.get_filenames(data_dir, symbol)
        assert filenames, f"No files found in '{data_dir}' from '{self.timestamp_bgn}' to '{self.timestamp_end}'."
        dfs: list[pd.DataFrame] = [pd.read_csv(os.path.join(data_dir, filename), names=self.KLINES_COLUMNS, parse_dates=["time", "time_close"]) for filename in filenames]
        df: pd.DataFrame = pd.concat(dfs)
        df.index = pd.DatetimeIndex(df["time"], name="time")
        di = cast("pd.DatetimeIndex", df.index)
        df = df[(di >= pd.to_datetime(self.timestamp_bgn)) & (di < pd.to_datetime(self.timestamp_end))]
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
        df: pd.DataFrame = con.query(query).to_df()
        con.close()
        df.index = df["time"]
        df = df[self.KLINES_CLEAN_COLUMNS]
        return df

    def get_data(self, symbol: str, file_type: Literal["csv", "db"] = "db", errors: Errors = "raise") -> pd.DataFrame:
        return self.get_data_klines(symbol, file_type, errors)

    def get_data_klines(self, symbol: str, file_type: Literal["csv", "db"] = "db", errors: Errors = "raise") -> pd.DataFrame:
        assert file_type in ["csv", "db"], f"Invalid arg: 'file_type' must be one of 'csv', 'db'. Got: '{file_type}'."
        symbol = symbol.strip().upper()
        if file_type == "db":
            try:
                df = self._read_db(symbol)
            except Exception:
                match errors:
                    case "raise": raise
                    case "empty": return self._empty_df()
            if df.empty and errors == "raise":
                raise AssertionError(f"No data in db for symbol '{symbol}' from '{self.timestamp_bgn}' to '{self.timestamp_end}'.")
            return df
        try:
            df = self._read_csv(symbol)
        except Exception:
            match errors:
                case "raise": raise
                case "empty": return self._empty_df()
        return df[self.KLINES_CLEAN_COLUMNS]

    def get_data_klines_agg(
            self                                  ,
            symbol    : str                       ,
            interval  : str                       ,
            file_type : Literal["csv", "db"] = "db",
            errors    : Errors               = "raise" ,
            alignment : Alignment             = "session",
    ) -> pd.DataFrame:
        if interval == self.interval:
            return self.get_data_klines(symbol, file_type, errors)
        df = self.get_data_klines(symbol, file_type, errors)
        if df.empty:
            return df
        return self.klines_resample(df, interval, alignment)

    def _resample_block(self, source: pd.DataFrame, frequency: str, anchor: bool) -> pd.DataFrame:
        """Resample one already-NY-local-indexed block (a single trading session, or the
        whole series for calendar alignment). ``anchor`` fixes sub-daily bins to 09:30."""
        kwargs: dict[str, Any] = {"rule": frequency, "label": "left", "closed": "left"}
        if anchor:
            # align bin edges to the fixed 09:30 session open, not midnight and not
            # whatever the first available data point happens to be (an IPO, halt or a
            # query window starting later in the day must not shift the whole bucket
            # grid, e.g. to 10:00-11:00 -- buckets stay 09:30-10:30, 10:30-11:30, ...)
            anchor_ts = pd.Timestamp("2000-01-03 09:30:00") # any date; only the time-of-day matters
            anchor_series = pd.Series(dtype="float64", index=pd.DatetimeIndex([anchor_ts]))
            anchor_edge_index = cast(pd.DatetimeIndex, anchor_series.resample(frequency, label="left").asfreq().index) # type: ignore[reportUnknownMemberType]
            kwargs["offset"] = anchor_ts - anchor_edge_index[0]
        result: pd.DataFrame = source.resample(**kwargs).agg(self.KLINES_CLEAN_AGGRULES) # type: ignore[reportUnknownMemberType,reportUnknownVariableType,reportCallIssue]

        valid_vwap_volume = source["volume_abs"].where(source["vwap"].notna())
        weighted_vwap = source["vwap"] * valid_vwap_volume
        vwap_numerator: pd.Series = weighted_vwap.resample(**kwargs).sum(min_count=1) # type: ignore[reportUnknownMemberType]
        vwap_denominator: pd.Series = valid_vwap_volume.resample(**kwargs).sum(min_count=1) # type: ignore[reportUnknownMemberType]
        result["vwap"] = vwap_numerator / vwap_denominator

        return result.dropna(subset=["price_open", "price_high", "price_low", "price_close"])

    def klines_resample(self, df: pd.DataFrame, interval: str, alignment: Alignment = "session") -> pd.DataFrame:
        if alignment not in ("session", "calendar"):
            raise ValueError(f"Invalid 'alignment': must be one of 'session', 'calendar'. Got: '{alignment}'.")
        target_timedelta = self._interval_timedelta(interval)
        native_timedelta = self.get_interval_timedelta()
        # never fabricate finer data out of a coarser native source (e.g. native '1h'
        # can't produce '15m') -- fail closed instead of silently returning garbage
        if target_timedelta < native_timedelta:
            raise ValueError(f"Cannot aggregate native '{self.interval}' data down to a finer interval '{interval}'.")
        if target_timedelta % native_timedelta != pd.Timedelta(0):
            raise ValueError(f"Target interval '{interval}' must be an exact multiple of the native '{self.interval}' interval.")

        frequency = self._resample_frequency(interval)
        missing = [column for column in self.KLINES_CLEAN_COLUMNS if column not in df.columns]
        if missing:
            raise ValueError(f"Kline data is missing required columns: {missing}")
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("Kline index must be a pandas DatetimeIndex.")
        if not df.index.is_monotonic_increasing or not df.index.is_unique:
            raise ValueError("Kline timestamps must be sorted and unique.")
        if df.empty:
            return self._empty_df()

        source = df.copy()
        # group in America/New_York wall-clock time so subdaily bins align to the 09:30
        # session open and day boundaries match trading days consistently across DST
        # (index is stored tz-naive UTC; localize before converting)
        di = cast(pd.DatetimeIndex, source.index)
        source_tz = di if di.tz is not None else di.tz_localize("UTC")
        source.index = source_tz.tz_convert(self.MARKET_TIMEZONE).tz_localize(None)
        di = cast(pd.DatetimeIndex, source.index)

        source_value, source_unit = self._parse_interval(self.interval)
        if source_unit == "m":
            aligned_ok = bool(((di.minute % source_value == 0) & (di.second == 0) & (di.microsecond == 0)).all())
        elif source_unit == "h":
            aligned_ok = bool(((di.minute == 0) & (di.second == 0) & (di.microsecond == 0)).all())
        else: # "d" (or coarser): no intraday alignment to check
            aligned_ok = True
        if not aligned_ok:
            raise ValueError(f"Kline timestamps must be aligned to the native '{self.interval}' interval.")

        _, target_unit = self._parse_interval(interval)
        result: pd.DataFrame
        if alignment == "session" and target_unit in {"m", "h"}:
            # Group by NY-local trading date FIRST, then resample independently within
            # each group. This makes "never merge across sessions" true by construction
            # (not just because the overnight/weekend gap happens to be bigger than the
            # target bucket) -- a Friday's last partial bar can never absorb Monday's
            # first bar, and an early-close day's last partial bar can never absorb data
            # from the following session, no matter how coarse the target interval is.
            parts = [self._resample_block(day_df, frequency, anchor=True) for _, day_df in source.groupby(di.date)]
            result = pd.concat(parts) if parts else self._empty_df().set_index(pd.DatetimeIndex([], name="time"))
        else:
            # 'calendar' alignment, or a daily-or-coarser target under 'session' (where
            # the session/calendar distinction is moot -- both just group by NY-local
            # trading day/week/month, with no 09:30 anchor to apply)
            result = self._resample_block(source, frequency, anchor=False)

        # bin-open timestamps are currently NY wall-clock; convert back to the project's UTC storage convention
        rdi = cast(pd.DatetimeIndex, result.index)
        result.index = rdi.tz_localize(self.MARKET_TIMEZONE).tz_convert("UTC").tz_localize(None)
        result.index = cast(pd.DatetimeIndex, result.index).rename("time")
        return result[self.KLINES_CLEAN_COLUMNS]

    def migrate_data(self, symbols: list[str] | str | None = None) -> None:
        symbols = symbols or self.get_symbols()
        if isinstance(symbols, str):
            symbols = [symbols]
        con = self.connect_db("w")
        table_name = self.get_table_name()
        TABLE_SCHEMA = """
            symbol      VARCHAR  ,
            time        TIMESTAMP,
            price_open  DOUBLE   ,
            price_high  DOUBLE   ,
            price_low   DOUBLE   ,
            price_close DOUBLE   ,
            volume_abs  DOUBLE   ,
            time_close  TIMESTAMP,
            trades      BIGINT   ,
            vwap        DOUBLE   ,
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

#%% Alpaca Api Helper
class AlpacaApiHelper():

    TRADING_URL       = "https://api.alpaca.markets"
    TRADING_URL_PAPER = "https://paper-api.alpaca.markets"
    REQUEST_FREQ      =    3.0 # Hz
    PRECISION_USD     =    2   # digits after point

    def __init__(
            self                                        ,
            api_key     : str                           ,
            api_secret  : str                           ,
            paper       : bool                    = True,
            timeout     : float                   = 30.0,
            max_retries : int                     = 3   ,
            session     : requests.Session | None = None,
    ) -> None:
        self.api_key     = api_key    
        self.api_secret  = api_secret 
        assert self.api_key and self.api_secret, (
            "Alpaca trading requires credentials. Pass 'api_key'/'api_secret' or set "
            "'APCA_API_KEY_ID'/'APCA_API_SECRET_KEY'."
        )
        print(f"Running with alpaca api key ({'paper' if paper else 'live'} trading).")
        self.base_url        = self.TRADING_URL_PAPER if paper else self.TRADING_URL
        self.timeout         = timeout
        self.max_retries     = max_retries
        self.session         = session or requests.Session()
        self.last_request_time = time.time()
        while True:
            try:
                self.account = self.client_get_account()
                self.clock   = self.client_get_clock()
                break
            except Exception as e:
                print("An error occurred while initializing AlpacaApiHelper:")
                print(e)
                print("Retrying in 60 seconds.")
                time.sleep(60)

    def _headers(self) -> dict[str, str]:
        return {
            "APCA-API-KEY-ID"     : self.api_key   ,
            "APCA-API-SECRET-KEY" : self.api_secret,
        }

    def _request(self, method: str, path: str, params: dict[str, Any] | None = None, json_body: dict[str, Any] | None = None) -> Any:
        retry_statuses = {429, 500, 502, 503, 504}
        url = f"{self.base_url}{path}"
        response: requests.Response | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.request(method, url, headers=self._headers(), params=params, json=json_body, timeout=self.timeout)
            except requests.RequestException:
                if attempt == self.max_retries:
                    raise
                time.sleep(min(2.0 ** attempt, 30.0))
                continue

            if response.status_code in (200, 204):
                return response.json() if response.content else {}
            if response.status_code in retry_statuses and attempt < self.max_retries:
                time.sleep(min(2.0 ** attempt, 30.0))
                continue

            try:
                message = response.json().get("message", response.text)
            except ValueError:
                message = response.text
            raise AlpacaDataError(f"Alpaca request failed ({response.status_code}): {str(message)[:500]}")

        raise AlpacaDataError("Alpaca request failed without a response.")

    # Client Functions
    def client_get_account(self) -> AlpacaAccount:
        return cast(AlpacaAccount, self._request("GET", "/v2/account"))
    def client_get_clock(self) -> AlpacaClock:
        return cast(AlpacaClock, self._request("GET", "/v2/clock"))
    def client_get_asset(self, symbol: str) -> AlpacaAsset:
        return cast(AlpacaAsset, self._request("GET", f"/v2/assets/{quote(symbol, safe='')}"))
    def client_create_order(self, **params: Any) -> AlpacaOrder:
        return cast(AlpacaOrder, self._request("POST", "/v2/orders", json_body=params))
    def client_get_open_orders(self, **params: Any) -> list[AlpacaOrder]:
        return cast("list[AlpacaOrder]", self._request("GET", "/v2/orders", params=params))
    def client_get_order(self, order_id: str) -> AlpacaOrder:
        return cast(AlpacaOrder, self._request("GET", f"/v2/orders/{order_id}"))
    def client_cancel_order(self, order_id: str) -> None:
        self._request("DELETE", f"/v2/orders/{order_id}")
    def client_get_position(self, symbol: str) -> AlpacaPosition:
        return cast(AlpacaPosition, self._request("GET", f"/v2/positions/{quote(symbol, safe='')}"))

    # Time Functions
    def get_alpaca_datetime(self) -> pd.Timestamp:
        clock = self.client_get_clock()
        return pd.to_datetime(clock["timestamp"]).tz_localize(None)

    # Trading Functions
    def get_quantity_precision(self, symbol: str) -> int:
        asset = self.client_get_asset(symbol)
        return 9 if asset.get("fractionable") else 0

    def get_price_precision(self, symbol: str) -> int:
        return self.PRECISION_USD

    def create_entry_order(
            self                                ,
            symbol    : str                     ,
            side      : Literal["buy", "sell"]  ,
            qty_asset : str | None = None       ,
            qty_usd   : str | None = None       ,
    ) -> AlpacaOrder:
        params: dict[str, Any] = {"symbol": symbol, "side": side, "type": "market", "time_in_force": "day"}
        if qty_asset is not None:
            params["qty"] = qty_asset
        if qty_usd is not None:
            params["notional"] = qty_usd
        return self.client_create_order(**params)

    def create_exit_order(
            self                              ,
            symbol    : str                   ,
            side      : Literal["buy", "sell"],
            qty_asset : str                   ,
    ) -> AlpacaOrder:
        return self.client_create_order(symbol=symbol, side=side, type="market", time_in_force="day", qty=qty_asset)

    def create_bracket_order(
            self                              ,
            symbol    : str                   ,
            side      : Literal["buy", "sell"],
            qty_asset : str                   ,
            sl_price  : str                   ,
            tp_price  : str                   ,
    ) -> AlpacaOrder:
        return self.client_create_order(
            symbol        = symbol                          ,
            side          = side                             ,
            type          = "market"                         ,
            time_in_force = "day"                            ,
            qty           = qty_asset                        ,
            order_class   = "bracket"                        ,
            take_profit   = {"limit_price": tp_price}        ,
            stop_loss     = {"stop_price" : sl_price}        ,
        )

    def enter_trade_save(
            self                                  ,
            symbol        : str                   ,
            side          : Literal["buy", "sell"],
            qty_asset     : float | None          ,
            qty_usd       : float | None          ,
            sl_price      : str                   ,
            tp_price      : str                   ,
            precision_usd : int = PRECISION_USD   ,
    ) -> AlpacaOrder | None:
        precision = self.get_quantity_precision(symbol)

        qty_asset_str = None
        if qty_asset is not None:
            qty_asset_rounded = np.floor(float(qty_asset) * 10 ** precision) * 10 ** -precision
            qty_asset_str = f"{qty_asset_rounded:.{precision}f}"
        elif qty_usd is not None:
            # bracket orders require a share quantity: convert the notional amount using the current ask
            qty_usd_rounded = np.floor(float(qty_usd) * 10 ** precision_usd) * 10 ** -precision_usd
            ask_price  = float(self.client_get_position(symbol).get("current_price") or 0) or float(self.client_get_asset(symbol).get("close_price") or 0)
            assert ask_price > 0, f"Could not determine a reference price for '{symbol}' to size the order."
            qty_asset_rounded = np.floor(qty_usd_rounded / ask_price * 10 ** precision) * 10 ** -precision
            qty_asset_str = f"{qty_asset_rounded:.{precision}f}"
        assert qty_asset_str is not None, "Either 'qty_asset' or 'qty_usd' must be given."

        try:
            return self.create_bracket_order(symbol, side, qty_asset_str, sl_price, tp_price)
        except Exception as e:
            print("An error occurred while placing the bracket order:")
            print(e)
            return None

    def get_open_orders(self, symbol: str | None = None) -> list[AlpacaOrder]:
        params: dict[str, Any] = {"status": "open"}
        if symbol is not None:
            params["symbols"] = symbol
        return self.client_get_open_orders(**params)

    def get_order_save(
            self                     ,
            order_id : str           ,
            tries    : int = 45 * 2  ,
            delay    : int = 30      ,
    ) -> AlpacaOrder:
        order : AlpacaOrder | None = None
        attempt = 0
        for attempt in range(1, tries + 1):
            try:
                order = self.client_get_order(order_id)
                break
            except Exception as e:
                print(f"An error occurred while fetching order on attempt {attempt}/{tries}:")
                print(e)
                if attempt < tries:
                    time.sleep(delay)
                else:
                    print("Raising Exception. Goodbye :(")
                    raise
        if attempt > 1:
            print(f"Succeeded fetching order on attempt {attempt}/{tries} :)")
        assert order is not None
        return order

    def get_account_balance(self) -> dict[str, float]:
        account = self.client_get_account()
        return {
            "cash"            : float(account["cash"           ]),
            "buying_power"    : float(account["buying_power"   ]),
            "portfolio_value" : float(account["portfolio_value"]),
        }

    # Function to log trades
    def log_trade(
            self                     ,
            log_path : str           ,
            order    : AlpacaOrder   ,
            trade_id : int           ,
            symbol   : str           ,
            sl_price : str = "NA"    ,
            tp_price : str = "NA"    ,
    ) -> None:
        datetime = str(pd.to_datetime(order.get("filled_at") or order["submitted_at"]).round("ms"))
        if len(datetime) != 26:
            datetime += "." + "0" * (26 - 1 - len(datetime))
        filled_qty   = float(order.get("filled_qty"       ) or 0.0)
        filled_price = float(order.get("filled_avg_price" ) or 0.0)
        fill : dict[str, Any] = {
            "time"      : datetime               ,
            "order_id"  : order["id"    ]        ,
            "trade_id"  : trade_id                ,
            "symbol"    : symbol                  ,
            "side"      : order["side"  ]        ,
            "qty"       : filled_qty              ,
            "quote_qty" : filled_qty * filled_price,
            "sl_price"  : sl_price                ,
            "tp_price"  : tp_price                ,
        }
        line = ",".join(str(v) for v in fill.values())
        print(line.replace(",", ", "))
        with open(log_path, "a") as f:
            f.write(f"{line}\n")
