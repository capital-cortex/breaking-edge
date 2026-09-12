# %%
import numpy as np
import pandas as pd
from numba import njit
import matplotlib.dates as mdates
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection

from . import utils

from typing import Callable, Literal, Any, cast

#%%
def optimal_strategy(
        prices  : pd.Series   ,
        fee_pct : float =  0.1, # %/(one-way)
        swap_ppa: float = 20.0, # %/a
) -> list[int]:
    """
    fee_pct [%/(one-way)], swap_ppa [%/a]
    """
    bar_seconds = utils.get_bar_seconds(prices)
    fee         = fee_pct  * 0.01
    swap        = swap_ppa * 0.01 / (3600.0 * 24.0 * 365.2422)
    n           = len(prices)
    dp          = [{-1: -np.inf, 0: -np.inf, 1: -np.inf} for _ in range(n)]
    prev        = [{-1:  0     , 0:  0     , 1:  0     } for _ in range(n)]
    dp[0][0]    = 0.0 
    for t in range(n - 1):
        price_change = (prices.iloc[t + 1] - prices.iloc[t]) / prices.iloc[t]
        for s in range(-1, 2):
            if dp[t][s] == -np.inf:
                continue
            curr_val = dp[t][s]
            for next_s in range(-1, 2):
                val = curr_val
                if next_s != 0:
                    val += next_s * price_change - swap * bar_seconds[t]
                val -= abs(next_s - s) * fee
                if val > dp[t + 1][next_s]:
                    dp  [t + 1][next_s] = val
                    prev[t + 1][next_s] = s
    state = max(range(-1, 2), key=lambda s: dp[n - 1][s])
    strategy = [0] * n
    for t in reversed(range(1, n)):
        strategy[t - 1] = state
        state = prev[t][state]
    strategy[-1] = 0
    return strategy

def generate_soft_targets(
        prices   : pd.Series          ,
        runs     : int          = 50  ,
        fee_pct  : float        =  0.1,
        swap_ppa : float        = 20.0,
        noise_std: float | None = None,
        seed     : int   | None = None,
) -> np.ndarray:
    np.random.seed(seed)
    n = len(prices)
    counts = np.zeros((n, 3))
    log_prices = np.log(prices)
    if noise_std == None:
        noise_std = np.std((prices[1:] - prices[:-1]) / prices[:-1])
    for _ in range(runs):
        noise = np.random.normal(0, noise_std, size=n)
        noisy_prices = pd.Series(np.exp(log_prices + noise), index=prices.index)
        strategy = optimal_strategy(noisy_prices, fee_pct, swap_ppa)
        for t, action in enumerate(strategy):
            counts[t, action + 1] += 1
    probabilities = counts / runs
    return probabilities

"""
def optimal_strategy_c(
        src_path : str  ,
        fee      : float,
        swap     : float,
        noise    : float,
        runs     : int  ,
) -> dict:
    dst_path = f".strategy.{time.perf_counter_ns()}.csv"
    orig_dir = os.getcwd()
    os.chdir(os.path.dirname(__file__))
    argv = [dst_path, src_path, fee, swap, noise, runs]
    argv = [str(arg) for arg in argv]
    result = subprocess.run(["./bin/optimal_strategy.exe"] + argv, capture_output=True, text=True)
    assert result.returncode == 0, f"C error (retcode {hex(result.returncode)})\nStdout: {result.stdout}"
    df = pd.read_csv(dst_path, index_col=None, header=None)
    os.remove(dst_path)
    os.chdir(orig_dir)
    return {"strategy": df[0].values, "probs": df[[1, 2, 3]].values}
"""

#%%
def visualize_strategy(
        prices   : pd.Series,
        strategy : pd.Series,
) -> None:
    index = cast(np.typing.NDArray[Any], mdates.date2num(prices.index)) # type: ignore
    x = np.array(index)
    strategy_arr = np.array(strategy.values, dtype="float")
    assert len(prices) == len(strategy), "Prices and strategy must have same length"
    assert not np.isnan(strategy).any(), "Strategy contains nans"
    strategy_abs_max = np.max(np.abs(strategy_arr))
    idx_bgn = 0
    lines : list[np.ndarray[tuple[int, ...], Any]] = []
    cs : list[list[float]]= []
    for i in range(len(prices) - 1):
        if strategy_arr[i] == strategy_arr[i + 1] and i + 1 != len(prices) - 1:
            continue
        color: list[float] = [
           -strategy_arr[i] / strategy_abs_max if strategy_arr[i] < 0 else 0.0,
            strategy_arr[i] / strategy_abs_max if strategy_arr[i] > 0 else 0.0,
            0.0
        ]
        lines.append(np.array([x[idx_bgn : i + 1 + 1], prices[idx_bgn : i + 1 + 1]]).T)
        cs.append(color)
        idx_bgn = i + 1
    lc = LineCollection(lines, colors=cs)
    ax = plt.gca()
    ax.add_collection(lc)
    ax.autoscale_view()
    ax.set_yscale('log') # type: ignore
    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.AutoDateFormatter(locator))
    plt.title("Trading Strategy Visualization") # type: ignore
    plt.xlabel("Time") # type: ignore
    plt.ylabel("Price") # type: ignore
    plt.grid(True, "both") # type: ignore

def visualize_probs(
        prices : pd.Series ,
        probs  : np.ndarray[tuple[int, int, int], Any],
) -> None:
    x = prices.index
    p = prices.values
    assert len(prices) == len(probs), "Prices and strategy must have same length"
    n = len(p)
    for t in range(n - 1):
        color : list[float] = [
            np.clip(probs[t][0] * (1 - probs[t][1]), 0.0, 1.0),
            np.clip(probs[t][2] * (1 - probs[t][1]), 0.0, 1.0),
            0.0
        ]
        plt.semilogy( # type: ignore
            [x[t], x[t + 1]],
            [p[t], p[t + 1]],
            color=color,
        )
    plt.title("Trading Strategy Visualization") # type: ignore
    plt.xlabel("Time") # type: ignore
    plt.ylabel("Price") # type: ignore
    plt.grid() # type: ignore

def visualize_evaluation(
        prices          : pd.Series | pd.DataFrame            ,
        strategy        : pd.Series                           ,
        fee_pct         : float                     =      0.1, # %/(one-way)
        swap_long_ppa   : float                     =     22.0, # %/a
        swap_short_ppa  : float                     =     12.0, # %/a
        compound_pct    : float                     =      0.0, # %
        sl_pct          : float | None              =     None, # %
        tp_pct          : float | None              =     None, # %
        adjust_neg_sltp : bool                      =     True,
        reentry         : Literal["change", "next"] = "change",
) -> pd.DataFrame:
    """
    fee_pct [%/(one-way)], swap_*_ppa [%/a]
    """
    pnls = evaluate_strategy(prices, strategy, fee_pct, swap_long_ppa, swap_short_ppa, compound_pct, sl_pct, tp_pct, adjust_neg_sltp, reentry)
    line, = plt.plot(pnls.equity, linewidth=0.5, label="_nolegend_") # type: ignore
    plt.plot(pnls.rpnl, linewidth=1.0, color=line.get_color()) # type: ignore
    plt.title("PnL Visualization") # type: ignore
    plt.xlabel("Time") # type: ignore
    plt.ylabel("PnL") # type: ignore
    plt.grid() # type: ignore
    return pnls

class SchmittTrigger3Level:
    def __init__(
            self,
            neg_thresholds : tuple[float, float],
            pos_thresholds : tuple[float, float],
    ) -> None:
        self.n_outer, self.n_inner = neg_thresholds
        self.p_inner, self.p_outer = pos_thresholds
        self.state = 0

    def process(self, val: float) -> int:
        if self.state ==  1:
            if val >= self.p_outer: return self.state
            if val < self.n_outer: self.state = -1; return self.state
            if val < self.p_inner: self.state =  0; return self.state
        if self.state ==  0:
            if val > self.p_outer: self.state =  1; return self.state
            if val < self.n_outer: self.state = -1; return self.state
        if self.state == -1:
            if val <= self.n_outer: return self.state
            if val > self.p_outer: self.state =  1; return self.state
            if val > self.n_inner: self.state =  0; return self.state
        return self.state

def get_strategy_schmitt(
        signal          : pd.Series,
        threshold_lower : float,
        threshold_upper : float,
) -> pd.Series:
    strategy = signal.values
    assert strategy.ndim == 1, "Input must be 1D"
    trigger = SchmittTrigger3Level((-threshold_upper, -threshold_lower), (threshold_lower, threshold_upper))
    strategy = [trigger.process(v) for v in strategy]
    return pd.Series(strategy, index=signal.index).rename("strategy")

def get_strategy_probs(
        probs : pd.DataFrame,
        p     : float,
) -> pd.Series:
    strategy = probs.values
    assert strategy.ndim == 2 and strategy.shape[1] == 3, "Input must be 2D with shape (n, 3)"
    strategy = np.where(strategy > p, np.inf, -np.inf)
    allninfs = np.where(strategy == -np.inf, True, False).all(axis=1)
    strategy = strategy.argmax(axis=1) - 1.0
    strategy[allninfs] = np.nan
    strategy = pd.Series(strategy).ffill().fillna(0).rename("strategy")
    strategy.index = probs.index
    return strategy

def sharpe_ratio(equity: pd.Series) -> float:
    equity = equity.dropna()
    if len(equity) < 3:
        return np.nan
    equity = equity - equity.iloc[0]
    assert isinstance(equity.index, pd.DatetimeIndex), "equity.index must be of type DatetimeIndex."
    returns = equity.diff().dropna()
    returns_std = returns.std()
    if returns_std == 0.0:
        return np.nan
    interval_seconds = equity.index.to_series().diff().mean().total_seconds()
    periods_per_year = (365.2422 * 86400.0) / interval_seconds
    return (returns.mean() / returns_std) * float(np.sqrt(periods_per_year))

def calmar_ratio(equity: pd.Series) -> float:
    equity = equity.dropna()
    if len(equity) < 2:
        return np.nan
    equity = equity - equity.iloc[0]
    assert isinstance(equity.index, pd.DatetimeIndex), "equity.index must be of type DatetimeIndex."
    interval_seconds = equity.index.to_series().diff().mean().total_seconds()
    max_dd = (equity.cummax() - equity).max()
    if max_dd == 0.0:
        return np.nan
    years = interval_seconds / (365.2422 * 86400.0) * len(equity)
    if years == 0.0:
        return np.nan
    annual_return = equity.iloc[-1] / years
    return annual_return / max_dd

def robust_score(func: Callable[[pd.Series], float], equities: pd.DataFrame, penalty: float = 0.5) -> float:
    scores = equities.apply(func).dropna()
    score_median = scores.median()
    score_std = scores.std(ddof=0)
    return score_median - penalty * score_std

@njit(cache=True, boundscheck=False)
def _evaluate_strategy_core(
        closes          : np.ndarray[tuple[int], np.dtype[np.float64]],
        highs           : np.ndarray[tuple[int], np.dtype[np.float64]],
        lows            : np.ndarray[tuple[int], np.dtype[np.float64]],
        strategy        : np.ndarray[tuple[int], np.dtype[np.float64]],
        bar_seconds     : np.ndarray[tuple[int], np.dtype[np.float64]],
        fee             : float                                       ,
        swap_long       : float                                       ,
        swap_short      : float                                       ,
        compound        : float                                       ,
        sl              : float                                       ,
        tp              : float                                       ,
        adjust_neg_sltp : bool                                        ,
        reentry_mode    : int                                         ,
) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
    n = len(closes)
    out = np.zeros((n, 2), dtype=np.float64) # [rpnl, equity]
    
    realized_pnl  = 0.0
    current_qty   = 0.0
    price_entry   = np.nan
    ignore_signal = np.nan
    
    # Track the active signal to prevent continuous micro-rebalancing
    current_signal = 0.0 
    
    for t in range(n):
        # 1. Charge continuous swap for held position
        if current_qty != 0.0:
            swap          = swap_long if current_qty > 0 else swap_short
            realized_pnl -= abs(current_qty) * swap * bar_seconds[t - 1]
        
        s_raw = strategy[t]
        
        # 2. Re-entry Block Logic
        if not np.isnan(ignore_signal):
            if np.sign(s_raw) == np.sign(ignore_signal) and s_raw != 0:
                s_target = 0.0
            else:
                ignore_signal = np.nan
                s_target = s_raw
        else:
            s_target = s_raw
        
        # 3. SL / TP Check
        hit_sl = False
        hit_tp = False
        adj_sl = (1.0 + sl) if adjust_neg_sltp and current_qty > 0 else 1.0
        adj_tp = (1.0 + tp) if adjust_neg_sltp and current_qty < 0 else 1.0
        if current_qty != 0: 
            if (
                not np.isnan(sl) and (
                    current_qty > 0 and  lows[t] <= (1.0 - sl / adj_sl) * price_entry
                 or current_qty < 0 and highs[t] >= (1.0 + sl / adj_sl) * price_entry
                )
            ):
                hit_sl = True
            elif (
                not np.isnan(tp) and (
                    current_qty > 0 and highs[t] >= (1.0 + tp / adj_tp) * price_entry
                 or current_qty < 0 and  lows[t] <= (1.0 - tp / adj_tp) * price_entry
                )
            ):
                hit_tp = True
        
        if hit_sl or hit_tp:
            s_target = 0.0 
            if reentry_mode == 0: # "change"
                ignore_signal = s_raw
        
        # 4. Determine Execution Price
        exec_price = closes[t]
        if hit_sl:
            exec_price = price_entry * (1.0 - np.sign(current_qty) * sl / adj_sl)
        elif hit_tp:
            exec_price = price_entry * (1.0 + np.sign(current_qty) * tp / adj_tp)
        
        # 5. Position Sizing & Delta Execution
        if s_target != current_signal:
            base_size      = max(realized_pnl * compound + 1.0, 0.0)
            target_qty     = base_size * s_target
            current_signal = s_target  # Lock in the new state
        else:
            # Maintain current quantity to bypass swap-induced micro adjustments
            target_qty = current_qty 
        
        qty_delta = target_qty - current_qty
        
        if qty_delta != 0:
            if current_qty != 0:
                if np.sign(target_qty) == np.sign(current_qty):
                    if abs(target_qty) < abs(current_qty):
                        closed_qty = current_qty - target_qty
                    else:
                        closed_qty = 0.0
                else:
                    closed_qty = current_qty
                
                if closed_qty != 0:
                    exec_ret      = (exec_price - price_entry) / price_entry
                    realized_pnl += abs(closed_qty) * np.sign(current_qty) * exec_ret
            
            if target_qty != 0:
                if np.sign(target_qty) != np.sign(current_qty):
                    price_entry = exec_price
                else:
                    if abs(target_qty) > abs(current_qty):
                        added_qty   = target_qty - current_qty
                        price_entry = (abs(current_qty) * price_entry + abs(added_qty) * exec_price) / abs(target_qty)
            else:
                price_entry = np.nan
            
            realized_pnl -= abs(qty_delta) * fee
            current_qty   = target_qty
        
        # 6. Mark-to-Market Equity Curve
        unrealized_pnl = 0.0
        if current_qty != 0:
            unrealized_ret = (closes[t] - price_entry) / price_entry
            unrealized_pnl = abs(current_qty) * np.sign(current_qty) * unrealized_ret
        
        out[t, 0] = realized_pnl
        out[t, 1] = realized_pnl + unrealized_pnl
    
    return out

def evaluate_strategy(
        prices          : pd.Series | pd.DataFrame            ,
        strategy        : pd.Series                           ,
        fee_pct         : float                     =      0.1, # %/(one-way)
        swap_long_ppa   : float                     =     22.0, # %/a
        swap_short_ppa  : float                     =     12.0, # %/a
        compound_pct    : float                     =      0.0, # %
        sl_pct          : float | None              =     None, # %
        tp_pct          : float | None              =     None, # %
        adjust_neg_sltp : bool                      =     True,
        reentry         : Literal["change", "next"] = "change",
) -> pd.DataFrame:
    """
    fee_pct [%/(one-way)], swap_*_ppa [%/a]
    Evaluates a trading strategy with position sizing, fees, swaps, and SL/TP logic.
    Node: SL alwaays has priority over TP if both are hit in the same bar.
    """
    assert (prices.index == strategy.index).all(), "Invalid data: Prices and strategy must have the same index."
    if isinstance(prices, pd.DataFrame):
        assert "price_close" in prices.columns, "Invalid arg: 'prices' must be Series or DataFrame and contain 'price_close' column."
        closes = prices.price_close
    else:
        closes = prices
    if sl_pct is not None or tp_pct is not None:
        assert isinstance(prices, pd.DataFrame), "Invalid arg type: 'prices' must be of type DataFrame in SL/TP mode."
        assert "price_high" in prices.columns  , "Invalid arg: 'prices' must contain 'price_high' column in SL/TP mode."
        assert "price_low"  in prices.columns  , "Invalid arg: 'prices' must contain 'price_low' column in SL/TP mode."
        highs = prices.price_high
        lows  = prices.price_low
    else:
        highs = None
        lows  = None
    
    if swap_long_ppa or swap_short_ppa:
        assert pd.api.types.is_datetime64_any_dtype(closes.index), "Invalid arg: 'prices.index' must be of type DatetimeIndex for swap calculations."

    bar_seconds = utils.get_bar_seconds(closes)
    
    closes_arr   =   closes.to_numpy(dtype=np.float64, copy=True)
    highs_arr    =    highs.to_numpy(dtype=np.float64, copy=True) if highs is not None else np.zeros(len(closes))
    lows_arr     =     lows.to_numpy(dtype=np.float64, copy=True) if  lows is not None else np.zeros(len(closes))
    strategy_arr = strategy.to_numpy(dtype=np.float64, copy=True)
    assert len(closes_arr) == len(strategy_arr), "Invalid data: Prices and strategy must have same length."
    
    fee = fee_pct * 0.01
    swap_long, swap_short = [s * 0.01 / (3600.0 * 24.0 * 365.2422) for s in [swap_long_ppa, swap_short_ppa]]
    compound = compound_pct * 0.01
    sl = sl_pct * 0.01 if sl_pct is not None else np.nan
    tp = tp_pct * 0.01 if tp_pct is not None else np.nan
    reentry_mode = 0 if reentry == "change" else 1
    
    results = _evaluate_strategy_core(
        closes_arr     ,
        highs_arr      ,
        lows_arr       ,
        strategy_arr   ,
        bar_seconds    ,
        fee            ,
        swap_long      ,
        swap_short     ,
        compound       ,
        sl             ,
        tp             ,
        adjust_neg_sltp,
        reentry_mode   ,
    )

    rpnl_arr   = results[:, 0]
    equity_arr = results[:, 1]
        
    return pd.DataFrame({
            "equity" : equity_arr,
            "rpnl"   : rpnl_arr,
            "upnl"   : equity_arr - rpnl_arr,
        }, index=prices.index
    )
