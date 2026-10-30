from typing import TypedDict, Required

class Bar(TypedDict):
    """One raw bar as returned by Alpaca's /v2/stocks/{symbol}/bars endpoint."""
    t  : str   # RFC3339 timestamp
    o  : float
    h  : float
    l  : float
    c  : float
    v  : float
    n  : float
    vw : float

class BarsPage(TypedDict):
    bars            : list[Bar] | None
    next_page_token : str | None
    symbol          : str

class CalendarDay(TypedDict):
    date            : str
    open            : str
    close           : str
    session_open    : str
    session_close   : str
    settlement_date : str

class Clock(TypedDict):
    timestamp  : str
    is_open    : bool
    next_open  : str
    next_close : str

class Account(TypedDict):
    cash            : str
    buying_power    : str
    portfolio_value : str

class Asset(TypedDict, total=False):
    symbol       : str
    fractionable : bool
    close_price  : str

class Position(TypedDict, total=False):
    symbol        : str
    current_price : str

class Order(TypedDict, total=False):
    id                : Required[str]
    side              : Required[str]
    submitted_at      : Required[str]
    type              : str
    status            : str
    filled_at         : str | None
    filled_qty        : str
    filled_avg_price  : str | None
