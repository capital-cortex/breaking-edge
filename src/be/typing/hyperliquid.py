from typing import TypedDict, List, Literal

type Side          = Literal["A", "B"]
type Dir           = Literal["Open Long", "Open Short", "Close Long", "Close Short"]
type Status        = Literal["ok", "err"]
type PortfolioType = Literal["day", "week", "month", "allTime", "perpDay", "perpWeek", "perpMonth", "perpAllTime"]

class OpenOrder(TypedDict):
    coin      : str
    side      : Side
    limitPx   : str
    sz        : str
    oid       : int
    timestamp : int
    origSz    : str
    reduceOnly: bool

class MarginSummary(TypedDict):
    accountValue    : str
    totalNtlPos     : str
    totalRawUsd     : str
    totalMarginUsed : str

class LeverageInfo(TypedDict):
    type  : Literal["cross", "isolated"]
    value : int

class CumFunding(TypedDict):
    allTime     : str
    sinceOpen   : str
    sinceChange : str

class Position(TypedDict):
    coin           : str
    szi            : str
    leverage       : LeverageInfo
    entryPx        : str
    positionValue  : str
    unrealizedPnl  : str
    returnOnEquity : str
    liquidationPx  : str | None
    marginUsed     : str
    maxLeverage    : int
    cumFunding     : CumFunding

class AssetPosition(TypedDict):
    type     : Literal["oneWay", "hedged"]
    position : Position

class UserState(TypedDict):
    marginSummary              : MarginSummary
    crossMarginSummary         : MarginSummary
    crossMaintenanceMarginUsed : str
    withdrawable               : str
    assetPositions             : List[AssetPosition]
    time                       : int

class Filled(TypedDict):
    totalSz : str
    avgPx   : str
    oid     : int

class Resting(TypedDict):
    oid : int

class StatusFilled(TypedDict):
    filled : Filled

class StatusResting(TypedDict):
    resting : Resting

class StatusError(TypedDict):
    error : str

class Data(TypedDict):
    statuses: list[StatusFilled | StatusResting | StatusError | Literal["success"]]

class Response(TypedDict):
    type : Literal["order", "cancel"]
    data : Data

class OrderResult(TypedDict):
    status   : Status
    response : Response

class LeverageResult(TypedDict):
    status   : Status
    response : dict[Literal["type"], Literal["default"]] | str

class Fill(TypedDict):
    coin          : str
    px            : str
    sz            : str
    side          : Side
    time          : int
    startPosition : str
    dir           : Dir
    closedPnl     : str
    hash          : str
    oid           : int
    crossed       : bool
    fee           : str
    tid           : int
    feeToken      : str
    twapId        : int | None
 
class Delta(TypedDict):
    type        : Literal["funding"]
    coin        : str
    usdc        : str
    szi         : str
    fundingRate : str
    nSamples    : int | None

class UserFunding(TypedDict):
    time  : int
    hash  : str
    delta : Delta

class Funding(TypedDict):
    coin        : str
    fundingRate : str
    premium     : str
    time        : int

class PortfolioHistory(TypedDict):
    accountValueHistory : list[tuple[int, str]]
    pnlHistory          : list[tuple[int, str]]
    vlm                 : str
