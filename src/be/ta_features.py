import pandas as pd
import numpy as np
import ta
import os
from tqdm import tqdm
from joblib import Parallel, delayed
from typing import Literal

def calculate_sma(data, window=14, source='price_close'):
    return data[source].rolling(window=window).mean()

def calculate_ema(data, window=14, source='price_close'):
    return data[source].ewm(span=window, adjust=False).mean()

def calculate_rsi(data, window=14, source='price_close'):
    return ta.momentum.RSIIndicator(data[source], window=window).rsi()

def calculate_macd(data, source='price_close'):
    macd = ta.trend.MACD(data[source])
    return macd.macd_diff()

def calculate_bollinger_bands(data, window=20, std_dev=2, source='price_close'):
    sma = data[source].rolling(window=window).mean()
    rstd = data[source].rolling(window=window).std()
    upper_band = sma + std_dev * rstd
    lower_band = sma - std_dev * rstd
    return upper_band, lower_band

def calculate_adl(data):
    return ta.volume.AccDistIndexIndicator(high=data['price_high'], low=data['price_low'], close=data['price_close'], volume=data['volume_buy'] + data['volume_sell'], fillna=True).acc_dist_index()

def calculate_cho(data):
    adl = calculate_adl(data)
    return ta.trend.ema_indicator(adl, 3) - ta.trend.ema_indicator(adl, 10)

def calculate_volatility(data, window=14, source='price_close'):
    return data[source].pct_change().rolling(window=window).std() * np.sqrt(window)

def calculate_price_change(data, periods=1, source='price_close'):
    return data[source].pct_change(periods=periods)

def calculate_rate_of_change(data, window=14, source='price_close'):
    return data[source].pct_change(periods=window)

def calculate_momentum(data, window=14, source='price_close'):
    return data[source] - data[source].shift(window)

# Additional Indicators

def calculate_stochastic_oscillator(data, window=14):
    return ta.momentum.StochasticOscillator(data['price_high'], data['price_low'], data['price_close'], window=window, fillna=True).stoch()

def calculate_williams_r(data, window=14):
    return ta.momentum.WilliamsRIndicator(data['price_high'], data['price_low'], data['price_close'], lbp=window, fillna=True).williams_r()

def calculate_awesome_oscillator(data):
    return ta.momentum.AwesomeOscillatorIndicator(data['price_high'], data['price_low'], fillna=True).awesome_oscillator()

def calculate_money_flow_index(data, window=14):
    return ta.volume.MFIIndicator(high=data['price_high'], low=data['price_low'], close=data['price_close'], volume=data['volume_buy']+data['volume_sell'], window=window, fillna=True).money_flow_index()

def calculate_abs(data, source):
    return data[f'{source}_buy'] + data[f'{source}_sell']

# TODO: Remove lookahead
# def calculate_herding_sheeps(df):
#     zvolume = scipy.stats.zscore(df['volume_buy'] + df['volume_sell'])
#     ztrades = scipy.stats.zscore(df['trades_buy'] + df['trades_sell'])
#     volumech = (df['volume_buy'] + df['volume_sell']).pct_change().fillna(0)
#     volumech[volumech == np.inf] = 0
#     tradesch = (df['trades_buy'] + df['trades_sell']).pct_change().fillna(0)
#     tradesch[tradesch == np.inf] = 0
#     zvolumech = scipy.stats.zscore(volumech)
#     ztradesch = scipy.stats.zscore(tradesch)
#     herding_indicator = zvolume + ztrades + zvolumech + ztradesch
#     return herding_indicator

def calculate_average_true_range(df, window=14):
    return ta.volatility.average_true_range(df['price_high'], df['price_low'], df['price_close'], window=window)

def calculate_pr_to_atr(df):
    return (df['price_high'] - df['price_low']) / calculate_average_true_range(df)

def calculate_distance_from_high_20(df,window=20):
    return df["price_close"] / df["price_high"].rolling(window=window).max()

def calculate_distance_from_low_20(df, window=20):
    return df['price_close'] / df['price_low'].rolling(window=window).min()

def calculate_macd_diff(df):
    return ta.trend.macd_diff(df['price_close'])
    
def calculate_macd_hist_chg(df):
    return ta.trend.macd(df['price_close']).diff()

def calculate_fourier_time(df:pd.DataFrame, func:Literal["cos","sin"], period:Literal["4y","1m","1d"], t0=pd.Timestamp("2018-01-01")) -> pd.DataFrame:
    fn = None
    if func == "cos":
        fn = np.cos
    if func == "sin":
        fn = np.sin
    omega = None
    if period == "4y":
        omega = 2.0 * np.pi / (3600.0 * 24.0 * 365.2422 * 4.0)
    if period == "1m":
        omega = 2.0 * np.pi / (3600.0 * 24.0 * 365.2422      ) * 12.0
    if period == "1d":
        omega = 2.0 * np.pi / (3600.0 * 24.0)
    
    index : pd.DatetimeIndex = df.index
    fourier_time_data = []
    for timestamp in index:
        t = (timestamp - t0).total_seconds()
        fourier_time_data.append({
            f"{func}_{period}": fn(omega * t),
        })
    return pd.DataFrame(fourier_time_data, index=index)
    
# Heldental

def calculate_relative_volume(df, days=7, start_hour=0, verbose=False):
    # TODO
    if verbose:
        print('* relative_volume calculation is slow. Consider precalculating and saving df *')
    """
    Berechnet das Relative Volumen für ein DataFrame mit Zeitstempel-Index.
    
    :param df: Pandas DataFrame mit pd.Timestamp als Index und einer Spalte für das Handelsvolumen.
    :param days: Die Anzahl der vergangenen Handelstage, um den Durchschnitt zu berechnen.
    :return: Pandas Series mit dem Relativen Volumen.
    """
    
    # Erstelle eine Liste, um das kumulierte Volumen und das relative Volumen zu speichern
    cumulative_volume = (df['volume_buy'] + df['volume_sell']).cumsum()
    relative_volume = np.zeros(len(df))
    historical_volumes = np.zeros(days)
    
    sample_timedelta = df.index[1] - df.index[0]
    
    for i, t in enumerate(df.index):
        hist_len = 0
        # Berechnung des durchschnittlichen kumulierten Volumens der letzten 'days' Handelstage
        for day in range(1, days + 1):
            t0 = (t - pd.Timedelta(days=day)).replace(hour=start_hour, minute=0, second=0) - sample_timedelta
            if t0 > df.index[0]:
                historical_volumes[hist_len] = cumulative_volume[t - pd.Timedelta(days=day)] - cumulative_volume[t0]
                hist_len += 1
                
        # Durchschnittliches Volumen der letzten Tage
        if hist_len == days:
            average_volume = sum(historical_volumes[0:hist_len]) / hist_len
        else:
            average_volume = np.nan
        
        
        # Berechnung des relativen Volumens
        t0 = t.replace(hour=0, minute=0, second=0) - sample_timedelta
        if t0 < df.index[0]:
            cumulative_volume0 = 0
        else:
            cumulative_volume0 = cumulative_volume[t0]
        if average_volume != 0:
            rel_volume = (cumulative_volume[t] - cumulative_volume0) / average_volume
        else:
            rel_volume = 0
        
        # Füge das relative Volumen zur Liste hinzu
        relative_volume[i] = rel_volume
    
    # Rückgabe der berechneten Series für das relative Volumen
    return pd.Series(relative_volume, index=df.index, name='relative_volume')

def calculate_vwap(df, volume_profile_levels=10, start_hour=0):
    vwap = np.zeros(len(df))
    init = True
    t0 = pd.to_datetime(f'{start_hour}:00:00').time()
    for i, t in enumerate(df.index):
        if init and t.time() == t0:
            init = False
        if init and t.time() != t0:
            vwap[i] = np.nan
            continue
        
        if t.time() == t0:
            pv = 0
            v = 0
        
        p_level_range = (df.at[t, 'price_high'] - df.at[t, 'price_low']) / volume_profile_levels
        p = np.linspace(df.at[t, 'price_low'] + 0.5 * p_level_range, df.at[t, 'price_high'] - 0.5 * p_level_range, volume_profile_levels)

        for l in range(volume_profile_levels):
            pv += (df.at[t, f'volume_buy_{l}'] + df.at[t, f'volume_sell_{l}']) * p[l]
            v  += df.at[t, f'volume_buy_{l}'] + df.at[t, f'volume_sell_{l}']
        if v == 0:
            vwap[i] = p.mean()
        else:
            vwap[i] = pv / v
        
        
    vwap = pd.Series(vwap, index=df.index, name='vwap')
    return vwap

def calculate_zprice_vwap(df, volume_profile_levels=10, start_hour=0, days=7):
    vwap = calculate_vwap(df, volume_profile_levels=volume_profile_levels, start_hour=start_hour)
    window = round(days * 86400 / (df.index[1] - df.index[0]).total_seconds())
    zscore = (df['price_close'] - vwap.rolling(window=window).mean()) / vwap.rolling(window=window).std()
    return pd.Series(zscore, index=df.index, name='zprice_vwap')


def calculate_blocks_total(df, blockchain_path):
    filenames = os.listdir(blockchain_path)
    csv_filenames = []
    for filename in filenames:
        if ".csv" in filename:
            csv_filenames.append(filename)
    dtype_mapping = {
        8: 'string',
        9: 'string',
    }
    chain_data = pd.read_csv(os.path.join(blockchain_path, csv_filenames[-1]), dtype=dtype_mapping)
    chain_data['time'] = pd.to_datetime(chain_data['time'])
    chain_data.set_index('time', inplace=True)
    timedelta = df.index[1] - df.index[0]
    return chain_data['id'].resample(timedelta).agg({'id': 'last'}).ffill()

def _polyfit_process_window(i, closes, x, df_index, n):
    if x[0] + i < 0 or x[-1] + i >= len(closes):
        return df_index[i], [np.nan] * (n + 2)
    
    y = closes[x + i]
    coefs = np.polyfit(x, y, n)
    fitted_values = np.polyval(coefs, x)
    error = np.sum(((fitted_values - y) / y)**2)
    
    coefs[-1] -= y.mean()
    coefs /= y.mean()
    
    return df_index[i], [*coefs, error]

def calculate_polyfit_df(df, n, x, n_jobs=-1, progress_bar=True):
    assert max(x) <= 0, "Values in 'x' must be <= 0"
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    
    columns = [*[f"coef_{n - i}" for i in range(n + 1)], "error"]
    new_df = pd.DataFrame(np.ones((len(df), n + 2)) * np.nan, columns=columns, index=df.index)
    closes = df['price_close'].values
    
    parallel_range = tqdm(range(len(df)), "Polyfit") if progress_bar else range(len(df))
    
    results = Parallel(n_jobs=n_jobs)(
        delayed(_polyfit_process_window)(i, closes, x, df.index, n) 
        for i in parallel_range
    )

    for idx, values, in results:
        new_df.loc[idx, [col for col in new_df]] = values

    return new_df

def _pivot_features_row(row, pivots, pivot_cnt, fluctuation, bonus_pivot_cnt=2):
    pivots[-1] = row["price_close"]

    # update forming pivot
    if (   pivots[-2] <= pivots[-3] and pivots[-1] < pivots[-2]
        or pivots[-2] >= pivots[-3] and pivots[-1] > pivots[-2]):

        pivots[-2] = pivots[-1]

    if (len(pivots) >= pivot_cnt + bonus_pivot_cnt): # +2 because initial pivots are not used in pattern
        pattern = pivots[-pivot_cnt:]
        # calculate features
        features = [
            *[(pattern[    5] - pattern[    4]) / (pattern[i + 1] - pattern[i    ]) for i in range(0,  4)],
            *[(pattern[    5] - pattern[    2]) / (pattern[j - 3] - pattern[j - 4]) for j in range(4,  6)],
            *[(pattern[k - 2] - pattern[k - 3]) / (pattern[k - 4] - pattern[k - 5]) for k in range(6,  8)],
            *[(pattern[h - 6] - pattern[h - 7]) / (pattern[h - 7] - pattern[h - 8]) for h in range(8, 11)]
        ]
    else:
        features = [np.nan] * 11

    if (   pivots[-1] > (1 + fluctuation) * pivots[-2]
        or pivots[-1] < (1 - fluctuation) * pivots[-2]):
        
        pivots.append(pivots[-1]) # new forming pivot

    return features

def calculate_pivot_features(df: pd.DataFrame, fluctuation: float) -> pd.DataFrame:
    # fluctuation = 10 * df.price_close.pct_change().std()
    pivot_cnt = 6
    features : list[list[float]] = []
    pivots   = [df.iloc[0]["price_close"] for _ in range(3)] # [initial pivot, forming pivot, current price]
    for _, row in df.iterrows():
        features_row = _pivot_features_row(row, pivots, pivot_cnt, fluctuation)
        features.append(features_row)

    return pd.DataFrame(features, columns=["pivot_feature_" + str(i) for i in range(11)], index=df.index)

def calculate_candlestick_features(df: pd.DataFrame) -> pd.DataFrame:
    o, h, l, c = df['price_open'], df['price_high'], df['price_low'], df['price_close']

    # Current candle metrics
    body = (c - o).abs()
    rng = (h - l) + 1e-9
    upper_ratio = (h - c.clip(o, c)) / rng
    lower_ratio = (o.clip(o, c) - l) / rng
    body_ratio = body / rng
    direction_up = c > o
    direction_down = c < o

    # Previous candle metrics (shifted by 1)
    o_1, h_1, l_1, c_1 = o.shift(1), h.shift(1), l.shift(1), c.shift(1)
    body_1 = (c_1 - o_1).abs()
    rng_1 = (h_1 - l_1) + 1e-9
    direction_up_1 = c_1 > o_1
    direction_down_1 = c_1 < o_1

    # Helper for body midpoint of previous candle
    prev_body_mid = (o_1 + c_1) / 2.0

    # Single-candle patterns
    doji = (body_ratio < 0.1)
    marubozu = (upper_ratio < 0.05) & (lower_ratio < 0.05)

    # Two-candle patterns
    hammer          = direction_down_1 & (lower_ratio > 0.6) & (upper_ratio < 0.1) & (body_ratio < 0.25)
    inverted_hammer = direction_down_1 & (upper_ratio > 0.6) & (lower_ratio < 0.1) & (body_ratio < 0.25)
    shooting_star   = direction_up_1   & (upper_ratio > 0.6) & (lower_ratio < 0.1) & (body_ratio < 0.25)
    hanging_man     = direction_up_1   & (lower_ratio > 0.6) & (upper_ratio < 0.1) & (body_ratio < 0.25)
    bullish_engulfing = (direction_down_1 & direction_up & (o <= c_1) & (c >= o_1))
    bearish_engulfing = (direction_up_1 & direction_down & (o >= c_1) & (c <= o_1))
    bullish_harami = (direction_down_1 & direction_up & (o >= o_1) & (c <= c_1))
    bearish_harami = (direction_up_1 & direction_down & (o <= o_1) & (c >= c_1))
    piercing_line = (direction_down_1 & direction_up & (c > prev_body_mid) & (o < c_1))
    dark_cloud_cover = (direction_up_1 & direction_down & (c < prev_body_mid) & (o > c_1))

    # Tweezer Top/Bottom (similar highs/lows with opposite directions)
    tweezer_top = (direction_up_1 & direction_down & (h.sub(h_1).abs() / (h_1 + 1e-9) < 0.002))
    tweezer_bottom = (direction_down_1 & direction_up & (l.sub(l_1).abs() / (l_1 + 1e-9) < 0.002))

    # Three-candle patterns (use shifted 2 where needed)
    o_2, h_2, l_2, c_2 = o.shift(2), h.shift(2), l.shift(2), c.shift(2)
    direction_up_2 = c_2 > o_2
    direction_down_2 = c_2 < o_2

    small_candle_1 = ((h_1 - l_1) / (h_2 - l_2 + 1e-9) < 0.6) & ((h_1 - l_1) / (h - l + 1e-9) < 0.6)
    morning_star = (direction_down_2 & small_candle_1 & direction_up)
    evening_star = (direction_up_2 & small_candle_1 & direction_down)

    candlestick_df = pd.DataFrame(index=df.index, data={
        "hammer_star"       : hammer.astype("int8") - shooting_star.astype("int8"),
        "inverted_man"      : inverted_hammer.astype("int8") - hanging_man.astype("int8"),
        "engulfing"         : bullish_engulfing.astype("int8") - bearish_engulfing.astype("int8"),
        "star"              : morning_star.astype("int8") - evening_star.astype("int8"),
        "harami"            : bullish_harami.astype("int8") - bearish_harami.astype("int8"),
        "piercing_darkcloud": piercing_line.astype("int8") - dark_cloud_cover.astype("int8"),
        "tweezer"           : tweezer_bottom.astype("int8") - tweezer_top.astype("int8"),
        "doji"              : doji.astype("int8"),
        "marubozu"          : marubozu.astype("int8")
    })

    return candlestick_df

def load_market_cap(dir: str, symbol: str) -> pd.DataFrame:
    """
    Load the latest market cap data from CSV file.
    """
    filenames = [n for n in os.listdir(dir) if n.startswith(symbol)]
    filepaths = [os.path.join(dir, n) for n in filenames]
    mtimes    = [os.path.getmtime(p) for p in filepaths]
    filepath  = filepaths[np.argmax(mtimes)]
    market_cap_df = pd.read_csv(filepath, parse_dates=['timestamp'], index_col='timestamp')
    return market_cap_df["market_cap"]

def calculate_ichimoku_df(df,
                         conversion_periods=9,
                         base_periods=26,
                         span_b_periods=52,
                         displacement=26):

    high = df["price_high"]
    low = df["price_low"]
    # close = df["price_close"]

    def donchian(length):
        return (high.rolling(length).max() + low.rolling(length).min()) / 2

    # Tenkan & Kijun
    tenkan = donchian(conversion_periods)
    kijun = donchian(base_periods)

    # Senkou A & B (TradingView-Shift)
    senkou_a = ((tenkan + kijun) / 2).shift(displacement - 1)
    senkou_b = donchian(span_b_periods).shift(displacement - 1)

    # Chikou Span
    # chikou = close.shift(-displacement + 1)

    ichi = pd.DataFrame({
        "tenkan": tenkan,
        "kijun": kijun,
        "senkou_a": senkou_a,
        "senkou_b": senkou_b,
        # "chikou": chikou
    }, index=df.index)

    return ichi.reindex(df.index)

def calculate_volume_profile(df, window, bins, volume_source='volume_abs', price_source='price_close'):
    window_size = window
    """
    Calculate the volume profile for a given window size.
    
    Parameters:
    - df: DataFrame with columns 'price_high', 'price_low', 'price_close', 'volume_buy', 'volume_sell'
    - window_size: The size of the window (in number of rows) to calculate the volume profile over
    - bins: Number of bins to segment the price range into
    
    Returns:
    - A DataFrame containing the volume profile for each window
    """
    
    window = df.iloc[-window_size:]
    high = window[price_source].max()
    low = window[price_source].min()
    
    # Create bins within the price range
    price_bins = np.linspace(low, high, bins+1)
    
    # Initialize volume profile for the current window
    volume_profile = np.zeros(bins)
    
    if volume_source == 'volume_abs':
        volumes = window['volume_buy'] + window['volume_sell']
    else:
        volumes = window[volume_source]
    
    for i in range(window_size):
        row = window.iloc[i]
        # Determine the bin for the closing price
        bin_index = np.digitize(row[price_source], price_bins) - 1
        # Safeguard for the closing price being equal to the high price
        bin_index = min(bin_index, bins - 1)
        # Aggregate buy and sell volumes
        volume_profile[bin_index] += volumes[i]
    
    return volume_profile, price_bins
