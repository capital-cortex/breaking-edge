import os
import sys
import time
import numpy as np
import pandas as pd

from eth_account import Account
from eth_account.signers.local import LocalAccount

from hyperliquid.exchange      import Exchange  # type: ignore
from hyperliquid.utils.signing import OrderType # type: ignore
from hyperliquid.utils         import constants # type: ignore
from hyperliquid.info          import Info      # type: ignore

from typing import TypedDict, List, Literal, Any, get_args

from .ansi import Rainbow

Side          = Literal["A", "B"]
Dir           = Literal["Open Long", "Open Short", "Close Long", "Close Short"]
Status        = Literal["ok", "err"]
PortfolioType = Literal["day", "week", "month", "allTime", "perpDay", "perpWeek", "perpMonth", "perpAllTime"]

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

class HyperliquidApiHelper():
    
    DECIMALS_PERP = 6
    DECIMALS_SPOT = 8
    
    eth_account : LocalAccount
    address     : str
    info        : Info
    exchange    : Exchange
    user_state  : UserState       | None
    open_orders : list[OpenOrder] | None
    
    def __init__(
            self                                                          ,
            private_key     : str                                         ,
            account_address : str       | None = None                     ,
            base_url        : str              = constants.MAINNET_API_URL,
            skip_ws         : bool             = True                     ,
            perp_dexs       : list[str] | None = None                     ,
    ) -> None:
        account = Account.from_key(private_key)
        address = account_address or account.address
        print(f"Running with account address.")
        if address != account.address:
            print(f"Running with agent address.")
        self.eth_account = account
        self.address     = address
        while True:
            try:
                self.info        = Info(base_url, skip_ws, perp_dexs=perp_dexs)
                self.exchange    = Exchange(self.eth_account, base_url, account_address=self.address, perp_dexs=perp_dexs)
                self.user_state  = self.info_user_state (self.address)
                self.open_orders = self.info_open_orders(self.address)
                break
            except Exception as e:
                print("Failed to initialize HyperliquidApiHelper:")
                print(e)
                print("Retrying in 60 seconds.")
                time.sleep(60)
    
    # Pylance type: ignore Functions:
    def info_user_state(self, address: str, dex: str = "") -> UserState:
        return self.info.user_state(address, dex) # type: ignore
    def info_open_orders(self, address: str, dex: str = "") -> list[OpenOrder]:
        return self.info.open_orders(address, dex) # type: ignore
    def info_all_mids(self) -> dict[str, str]:
        return self.info.all_mids() # type: ignore
    def info_asset_to_sz_decimals(self, asset: int) -> int:
        return self.info.asset_to_sz_decimals[asset] # type: ignore
    def exchange_market_open(self, coin: str, is_buy: bool, sz: float, px: float | None = None, slippage: float = Exchange.DEFAULT_SLIPPAGE) -> OrderResult:
        return self.exchange.market_open(coin, is_buy, sz, px, slippage) # type: ignore
    def exchange_market_close(self, coin: str, sz: float | None = None, px: float | None = None, slippage: float = Exchange.DEFAULT_SLIPPAGE) -> OrderResult:
        return self.exchange.market_close(coin, sz, px, slippage) # type: ignore
    def exchange_order(self, coin: str, is_buy: bool, sz: float, px: float, order_type: OrderType, reduce_only: bool = False) -> OrderResult:
        return self.exchange.order(coin, is_buy, sz, px, order_type, reduce_only) # type: ignore
    def exchange_update_leverage(self, leverage: int, coin: str) -> LeverageResult:
        return self.exchange.update_leverage(leverage, coin) # type: ignore
    def exchange_cancel(self, coin: str, oid: int) -> OrderResult:
        return self.exchange.cancel(coin, oid) # type: ignore
    def info_user_fills_by_time(self, address: str, start_time: int) -> list[Fill]:
        return self.info.user_fills_by_time(address, start_time) # type: ignore
    def info_user_funding_history(self, address: str, start_time: int, end_time: int | None = None) -> list[UserFunding]:
        return self.info.user_funding_history(address, start_time, end_time) # type: ignore
    def info_funding_history(self, coin: str, start_time: int, end_time: int | None = None) -> list[Funding]:
        return self.info.funding_history(coin, start_time, end_time) # type: ignore
    def info_portfolio(self, address: str) -> list[tuple[PortfolioType, PortfolioHistory]]:
        return self.info.portfolio(address) # type: ignore
    
    def update(self) -> bool:
        try:
            self.user_state  = self.info_user_state( self.address)
            self.open_orders = self.info_open_orders(self.address)
            return True
        except Exception as e:
            print("Failed to update: user state, open orders:")
            print(e)
            self.user_state  = None
            self.open_orders = None
            return False
    
    def get_mid_price(self, coin: str) -> float | None:
        try:
            return float(self.info_all_mids()[coin])
        except Exception as e:
            print(f"Failed to get mid price for '{coin}':")
            print(e)
            return None
    
    def format_sz(self, sz: float, coin: str) -> float:
        sz_decimals = int(self.info_asset_to_sz_decimals(self.info.name_to_asset(coin)))
        return round(sz, sz_decimals)
    
    def format_px(self, px: float, coin: str, is_perp: bool = True) -> float | int:
        sz_decimals = int(self.info_asset_to_sz_decimals(self.info.name_to_asset(coin)))
        decimals_max = self.DECIMALS_PERP if is_perp else self.DECIMALS_SPOT
        decimals_max = decimals_max - sz_decimals
        exponent = int(np.floor(np.log10(px)))
        decimals_sig = 5 - (exponent + 1) # exponent + 1 for digits
        decimals_sig = max(0, decimals_sig)
        decimals = min(decimals_max, decimals_sig)
        return round(px, decimals) if decimals else int(0.5 + px)
    
    @staticmethod
    def get_filled_avg_price(order_result: OrderResult, idx: int = 0) -> float | None:
        try:
            status = order_result["response"]["data"]["statuses"][idx]
            assert isinstance(status, dict) and "filled" in status
            price = float(status["filled"]["avgPx"])
        except Exception as e:
            print(f"Failed to get filled avg price from order result: {order_result}")
            print(e)
            return None
        return price
    
    def position_market_open(
            self,
            coin     : str                                     ,
            is_buy   : bool                                    ,
            sz       : float                                   ,
            px       : float | None = None                     ,
            slippage : float        = Exchange.DEFAULT_SLIPPAGE,
    ) -> OrderResult | None:
        sz = self.format_sz(sz, coin)
        if px is not None:
            px = self.format_px(px, coin)
        print(f"Trying to Market {"Buy" if is_buy else "Sell"} {sz} '{coin}'.")
        try:
            order_result = self.exchange_market_open(coin, is_buy, sz, px, slippage)
            if order_result["status"] != "ok":
                print(f"Failed: {order_result}")
                return None
            status, = order_result["response"]["data"]["statuses"]
            if not "filled" in status:
                print(f"Failed: {order_result}")
                return None
        except Exception as e:
            print("Failed:")
            print(e)
            return None
        print(f"Succeeded: {order_result}")
        return order_result
    
    def position_market_close(
            self                                                  ,
            coin     : str                                     ,
            sz       : float | None = None                     ,
            px       : float | None = None                     ,
            slippage : float        = Exchange.DEFAULT_SLIPPAGE,
    ) -> OrderResult | None:
        if sz is not None:
            sz = self.format_sz(sz, coin)
        if px is not None:
            px = self.format_px(px, coin)
        print(f"Trying to Market Close {"All" if sz is None else sz} '{coin}'.")
        try:
            order_result = self.exchange_market_close(coin, sz, px, slippage)
            if order_result["status"] != "ok":
                print(f"Failed: {order_result}")
                return None
            status, = order_result["response"]["data"]["statuses"]
            if not "filled" in status:
                print(f"Failed: {order_result}")
                return None
        except Exception as e:
            print("Failed:")
            print(e)
            return None
        print(f"Succeeded: {order_result}")
        return order_result
    
    def order_stop_sl(
            self                                        ,
            coin     : str                              ,
            is_buy   : bool                             ,
            sz       : float                            ,
            sl_px    : float                            ,
            slippage : float = Exchange.DEFAULT_SLIPPAGE,
    ) -> OrderResult | None:
        limit_px = ((1.0 if is_buy else -1.0) * slippage + 1.0) * sl_px
        sz       = self.format_sz(sz, coin)
        sl_px    = self.format_px(sl_px, coin)
        limit_px = self.format_px(limit_px, coin)
        stop_order_type = OrderType({"trigger": {"triggerPx": sl_px, "isMarket": True, "tpsl": "sl"}})
        print(f"Trying to place SL {"Buy" if is_buy else "Sell"} {sz} '{coin}' at {sl_px}.")
        try:
            stop_result = self.exchange_order(coin, is_buy, sz, limit_px, stop_order_type, reduce_only=True)
            if stop_result["status"] != "ok":
                print(f"Failed: {stop_result}")
                return None
            status, = stop_result["response"]["data"]["statuses"]
            if not "resting" in status:
                print(f"Failed: {stop_result}")
                return None
        except Exception as e:
            print("Failed:")
            print(e)
            return None
        print(f"Succeeded: {stop_result}")
        return stop_result
    
    def order_stop_tp(
            self                                        ,
            coin     : str                              ,
            is_buy   : bool                             ,
            sz       : float                            ,
            tp_px    : float                            ,
            slippage : float = Exchange.DEFAULT_SLIPPAGE,
    ) -> OrderResult | None:
        limit_px = ((1.0 if is_buy else -1.0) * slippage + 1.0) * tp_px
        sz       = self.format_sz(sz, coin)
        tp_px    = self.format_px(tp_px, coin)
        limit_px = self.format_px(limit_px, coin)
        stop_order_type = OrderType({"trigger": {"triggerPx": tp_px, "isMarket": True, "tpsl": "tp"}})
        print(f"Trying to place TP {"Buy" if is_buy else "Sell"} {sz} '{coin}' at {tp_px}.")
        try:
            stop_result = self.exchange_order(coin, is_buy, sz, limit_px, stop_order_type, reduce_only=True)
            if stop_result["status"] != "ok":
                print(f"Failed: {stop_result}")
                return None
            status, = stop_result["response"]["data"]["statuses"]
            if not "resting" in status:
                print(f"Failed: {stop_result}")
                return None
        except Exception as e:
            print("Failed:")
            print(e)
            return None
        print(f"Succeeded: {stop_result}")
        return stop_result
    
    def set_leverage(self, leverage: int, coin: str | None = None) -> int | None:
        def _update_leverage(leverage: int, coin: str) -> int:
            result = self.exchange_update_leverage(leverage, coin)
            if not result["status"] == "ok":
                print(f"Failed to set leverage to {leverage} for '{coin}':")
                print(result)
                return 0
            print(f"Set leverage to {leverage} for '{coin}'.")
            return 1
        set_cnt = 0
        try:
            if coin is not None:
                return _update_leverage(leverage, coin)
            assert self.user_state is not None, "user state is None."
            open_positions = self.user_state["assetPositions"]
            for position in open_positions:
                posi = position["position"]
                lvrg = min(leverage, posi["maxLeverage"])
                coin = posi["coin"]
                if posi["leverage"]["value"] == lvrg:
                    continue
                set_cnt += _update_leverage(lvrg, coin)
        except Exception as e:
            print(f"Failed to set leverage to {leverage} {f"for '{coin}'" if coin else ""}:")
            print(e)
            return None
        return set_cnt
    
    def get_open_positions(self, coin: str) -> list[AssetPosition] | None:
        try:
            assert self.user_state is not None, "user state is None."
            open_positions = self.user_state["assetPositions"]
            open_positions = [p for p in open_positions if p["position"]["coin"] == coin]
        except Exception as e:
            print(f"Failed to get open positions for '{coin}':")
            print(e)
            return None
        return open_positions
    
    def get_open_position_side(self, coin: str) -> int | None:
        open_positions = self.get_open_positions(coin)
        try:
            assert open_positions is not None, "open positions is None."
            size = sum(float(p["position"]["szi"]) for p in open_positions)
        except Exception as e:
            print(f"Failed to get open position side for '{coin}':")
            print(e)
            return None
        return int(np.sign(size))
    """
    def get_open_position_cnt(self, coin: str) -> int | None:
        open_positions = self.get_open_positions(coin)
        if open_positions is None:
            return None
        return len(open_positions)
    """
    def get_open_orders(self, coin: str) -> list[OpenOrder] | None:
        try:
            assert self.open_orders is not None, "open orders is None"
            open_orders = [o for o in self.open_orders if o["coin"] == coin]
        except Exception as e:
            print(f"Failed to get open orders for '{coin}':")
            print(e)
            return None
        return open_orders
    """
    def get_open_order_cnt(self, coin: str) -> int | None:
        open_orders = self.get_open_orders(coin)
        if open_orders is None:
            return None
        return len(open_orders)
    """
    def order_cancel(self, order: OpenOrder) -> OrderResult | None:
        print(f"Trying to cancel order: {order}")
        try:
            cancel_result = self.exchange_cancel(order["coin"], order["oid"])
            if cancel_result["status"] != "ok":
                print(f"Failed: {cancel_result}")
                return None
            status, = cancel_result["response"]["data"]["statuses"]
            if not "success" in status:
                print(f"Failed: {cancel_result}")
                return None
        except Exception as e:
            print("Failed:")
            print(e)
            return None
        print(f"Succeeded: {cancel_result}")
        return cancel_result
    """
    def get_last_fill(self, coin: str) -> Fill | None:
        try:
            user_fills: list[Fill] = self.info.user_fills(self.address)
            last_fill = next(f for f in user_fills if f["coin"] == coin)
        except Exception as e:
            print(f"Failed to get last fill for '{coin}':")
            print(e)
            return None
        return last_fill
    """
    def get_fills_by_time(self, start_time: int) -> list[Fill] | None:
        try:
            return self.info_user_fills_by_time(self.address, start_time)
        except Exception as e:
            print("Failed to get fills by time:")
            print(e)
            return None
    
    def get_fundings_by_time(self, start_time: int, end_time: int | None = None) -> list[UserFunding] | None:
        try:
            return self.info_user_funding_history(self.address, start_time, end_time)
        except Exception as e:
            print("Failed to get user funding history by time:")
            print(e)
            return None
    
    def get_funding_df(
            self                           ,
            coin          : str            ,
            timestamp_bgn : str            ,
            timestamp_end : str = "2170-01",
    ) -> pd.DataFrame:
        funding_hist: list[Funding] = []
        t     = 1000 * int(    pd.to_datetime(timestamp_bgn).timestamp())
        t_end = 1000 * int(min(pd.to_datetime(timestamp_end).timestamp(), pd.Timestamp.now("utc").timestamp()))
        while True:
            rows = self.info_funding_history(coin, t, t_end)
            if not len(rows):
                break
            t = rows[-1]["time"] + 1
            if t >= t_end:
                break
            funding_hist.extend(rows)
        df = pd.DataFrame(funding_hist, columns=list(Funding.__annotations__.keys()))
        df.index = pd.to_datetime(df.time, unit="ms")
        df = df.drop(columns=["time", "coin"]).astype(float)
        return df
    
    def get_portfolio_df(self, portfolio_type: PortfolioType) -> pd.DataFrame:
        assert portfolio_type in (types := get_args(PortfolioType)), f"Portfolio type must be one of {types}"
        portfolios = self.info_portfolio(self.address)
        history = next(p for p in portfolios if p[0] == portfolio_type)[1]
        val_df = pd.DataFrame(history["accountValueHistory"], columns=["time", "accountValueHistory"])
        pnl_df = pd.DataFrame(history["pnlHistory"         ], columns=["time", "pnlHistory"         ])
        df = val_df.merge(pnl_df, on="time")
        df.index = pd.to_datetime(df.time, unit="ms")
        df = df.drop(columns="time").astype(float)
        return df

class HyperliquidLogger():

    COLUMNS : list[str]
    path    : str
    df      : pd.DataFrame
    
    def __init__(self, log_path: str) -> None:
        self.create(log_path, overwrite=False)
        self.path = log_path
        self.df   = pd.read_csv(log_path)
    
    def create(self, log_path: str, overwrite: bool = False) -> bool:
        if os.path.exists(log_path) and not overwrite:
            return False
        print("Creating new log file.")
        with open(log_path, "w") as f:
            line = ",".join(c for c in self.COLUMNS)
            f.write(f"{line}\n")
        return True
    
    def get_last_time(self) -> int:
        return int(self.df.iloc[-1].time) if len(self.df) else -1

class HyperliquidFillsLogger(HyperliquidLogger):
    
    COLUMNS = [*Fill.__annotations__]
    
    def update(self, new_fills: list[Fill]) -> bool:
        if len(new_fills) == 0:
            return False
        print("Logging new user fills:")
        with open(self.path, "a") as f:
            for fill in new_fills:
                row: list[Any] = [fill[col] for col in self.COLUMNS]
                line = ",".join(str(v) for v in row)
                print(str(row)[1:-1])
                f.write(f"{line}\n")
                self.df.loc[len(self.df), self.COLUMNS] = [
                    n if pd.notna(n := pd.to_numeric(v, errors="coerce")) else v for v in row # type: ignore
                ]
        return True
    
    def get_df_agg(self, round_decimals: int = 6) -> pd.DataFrame:
        groups = self.df.groupby("time")
        aggrules = {col: "first" for col in self.COLUMNS}
        aggrules["sz"       ] = "sum"
        aggrules["closedPnl"] = "sum"
        aggrules["fee"      ] = "sum"
        df_agg = groups.agg(aggrules)
        for _, group in groups:
            df_agg.loc[group.time, "px"] = (group.px * group.sz).sum() / group.sz.sum()
        float_columns = df_agg.columns[df_agg.dtypes == float]
        df_agg[float_columns] = df_agg[float_columns].round(round_decimals)
        df_agg.index = pd.to_datetime(df_agg.index, unit="ms")
        return df_agg

class HyperliquidFundingsLogger(HyperliquidLogger):
    
    COLUMNS = [*[*UserFunding.__annotations__][:-1], *Delta.__annotations__]
    
    def update(self, new_fundings: list[UserFunding], verbose : bool = False) -> bool:
        if len(new_fundings) == 0:
            return False
        print("Logging new user fundings:")
        funding_usdc = 0.0
        with open(self.path, "a") as f:
            for funding in new_fundings:
                funding_usdc += float(funding["delta"]["usdc"])
                row: list[Any] = [funding["time"], funding["hash"], *funding["delta"].values()]
                line = ",".join(str(v) for v in row)
                if verbose:
                    print(str(row)[1:-1])
                f.write(f"{line}\n")
                self.df.loc[len(self.df), self.COLUMNS] = [
                    n if pd.notna(n := pd.to_numeric(v, errors="coerce")) else v for v in row # type: ignore
                ]
        if not verbose:
            print(f"... ({len(new_fundings)}) ...")
        sys.stdout.write("Funding total: " + Rainbow.paint_value(f"{funding_usdc:.2f}") + " $.\n")
        return True
    
    def get_df(self) -> pd.DataFrame:
        df = self.df.copy()
        df.index = pd.to_datetime(df["time"], unit="ms")
        return df
