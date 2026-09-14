import warnings

import numpy as np
import pandas as pd
import talib
from scipy.signal import find_peaks
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning

def compute_sqn(
    df: pd.DataFrame,
    window: int,
    price_col: str = "close",
    use_log_returns: bool = True,
    min_periods: int | None = None,
    ddof: int = 1,
    clip: float | None = 10.0,
) -> pd.Series:
    """
    Rolling SQN on a price series.

    SQN = sqrt(N) * mean(returns) / std(returns)

    Parameters
    ----------
    df : pd.DataFrame
        Must contain `price_col`.
    window : int
        Rolling window length in bars (e.g., 30, 90).
    price_col : str
        Price column name (default "close").
    use_log_returns : bool
        True -> log returns, False -> simple returns.
    min_periods : int | None
        Rolling min periods. Default: window (strict).
        If you want earlier values, set e.g. int(window * 0.7).
    ddof : int
        Std degrees of freedom (1 = sample std, 0 = population).
    clip : float | None
        Optional absolute cap on SQN to avoid blowups when std ~ 0.
        Set None to disable.

    Returns
    -------
    pd.Series
        SQN series aligned to df index.
    """
    if window <= 1:
        raise ValueError("window must be > 1")

    if price_col not in df.columns:
        raise KeyError(f"Missing column: {price_col}")

    price = pd.to_numeric(df[price_col], errors="coerce")

    if use_log_returns:
        r = np.log(price / price.shift(1))
    else:
        r = price.pct_change()

    mp = window if (min_periods is None) else int(min_periods)

    mu = r.rolling(window=window, min_periods=mp).mean()
    sd = r.rolling(window=window, min_periods=mp).std(ddof=ddof)

    sqn = (np.sqrt(window) * mu) / sd
    sqn = sqn.replace([np.inf, -np.inf], np.nan)

    if clip is not None:
        sqn = sqn.clip(lower=-abs(clip), upper=abs(clip))

    return sqn.fillna(0.0)


def rolling_percentile(
    series: pd.Series,
    window: int = 252,
    min_periods: int | None = None,
    neutral: float = 0.5,
) -> pd.Series:
    """
    Rolling percentile rank of each value vs its trailing window.

    Output is the fraction of the trailing `window` observations that are
    <= the current value, in [0, 1] (pandas rolling rank with pct=True).

    Values that cannot be computed (NaN inputs / insufficient history) are
    filled with `neutral` (default 0.5) so downstream gates never break on
    warmup.
    """
    if window <= 1:
        raise ValueError("window must be > 1")

    s = pd.to_numeric(series, errors="coerce")
    mp = int(window) if (min_periods is None) else int(min_periods)
    pct = s.rolling(window=int(window), min_periods=mp).rank(pct=True)
    return pct.fillna(neutral)


def classify_sqn_regime(
    df: pd.DataFrame,
    sqn_fast_period: int = 30,
    sqn_slow_period: int = 90,
    slope_lookback: int = 5,
    # thresholds (tune later, these are sensible crypto defaults)
    bear_slow: float = -0.5,
    weak_bear_slow: float = -0.2,
    weak_bull_slow: float = 0.3,
    strong_bull_slow: float = 1.2,
    fast_floor: float = 0.0,
    fast_strong: float = 1.2,
    rollover_slope: float = -0.2,
    # rolling-percentile regime mode (Rec 5). Default off for backward compat.
    use_percentile: bool = False,
    percentile_window: int = 252,
    percentile_min_periods: int = 90,
    pct_strong_bear: float = 0.05,
    pct_weak_bear: float = 0.25,
    pct_neutral: float = 0.50,
    pct_weak_bull: float = 0.80,
) -> pd.Series:
    """
    Classify timeframe regime into:
      - strong_bear
      - weak_bear
      - neutral
      - weak_bull
      - strong_bull

    Uses:
      - slow SQN = structural (sqn_90 default)
      - fast SQN = responsive (sqn_30 default)
      - fast slope  = transition / rollover hint

    use_percentile=True buckets the slow/fast SQN by their rolling percentile
    rank (trailing `percentile_window` bars) instead of the fixed absolute
    thresholds, so regime labels adapt to the market's own recent history.

    Returns
    -------
    pd.Series[str] aligned to daily.index
    """

    sqn_fast_series = compute_sqn(df, sqn_fast_period)
    sqn_slow_series = compute_sqn(df, sqn_slow_period)

    sqn_fast = sqn_fast_series.astype(float)
    sqn_slow = sqn_slow_series.astype(float)
    slope = sqn_fast.diff(slope_lookback)

    regime = pd.Series("neutral", index=df.index, dtype="object")

    if use_percentile:
        slow_pct = rolling_percentile(
            sqn_slow,
            window=percentile_window,
            min_periods=percentile_min_periods,
        )
        fast_pct = rolling_percentile(
            sqn_fast,
            window=percentile_window,
            min_periods=percentile_min_periods,
        )
        # ---- Bears (assign first; later bull assignments can overwrite) ----
        strong_bear = slow_pct <= pct_strong_bear
        weak_bear = (slow_pct > pct_strong_bear) & (slow_pct <= pct_weak_bear)

        regime[weak_bear] = "weak_bear"
        regime[strong_bear] = "strong_bear"

        # Bulls
        strong_bull = (
            (slow_pct > pct_weak_bull) &
            (fast_pct > pct_weak_bull) &
            (slope > rollover_slope)
        )
        weak_bull = (
            (slow_pct > pct_neutral) & (slow_pct <= pct_weak_bull) &
            (fast_pct > pct_neutral) &
            (slope > rollover_slope)
        )

        regime[weak_bull] = "weak_bull"
        regime[strong_bull] = "strong_bull"
    else:
        # ---- Bears (assign first; later bull assignments can overwrite) ----
        strong_bear = sqn_slow <= bear_slow
        weak_bear = (sqn_slow > bear_slow) & (sqn_slow <= weak_bear_slow)

        regime[weak_bear] = "weak_bear"
        regime[strong_bear] = "strong_bear"

        # Bulls
        strong_bull = (
            (sqn_slow >= strong_bull_slow) &
            (sqn_fast >= fast_strong) &
            (slope > rollover_slope)
        )

        weak_bull = (
            (sqn_slow >= weak_bull_slow) & (sqn_slow < strong_bull_slow) &
            (sqn_fast >= fast_floor) & (sqn_fast < fast_strong) &
            (slope > rollover_slope)
        )

        regime[weak_bull] = "weak_bull"
        regime[strong_bull] = "strong_bull"

    out = pd.DataFrame(
        {
            f"sqn_{sqn_fast_period}": sqn_fast_series,
            f"sqn_{sqn_slow_period}": sqn_slow_series,
            "regime": regime,
        },
        index=df.index,
    )
    return out


    
def attach_sqn_to_dataset(dict_of_pairs, btc_pair: str = "BTCUSDT"):

    for pair in dict_of_pairs:

        
        for frame in dict_of_pairs[pair]['dict_of_frames']:

            if dict_of_pairs[pair]['dict_of_frames'][frame].empty or ("open_time" not in dict_of_pairs[pair]['dict_of_frames'][frame].columns):
                print(pair,frame)
                break
            
            loop_df = dict_of_pairs[pair]['dict_of_frames'][frame]
            res = classify_sqn_regime(loop_df)  # returns df with sqn_30, sqn_90, regime

            loop_df["sqn_30"] = res["sqn_30"].astype("float64")
            loop_df["sqn_90"] = res["sqn_90"].astype("float64")
            loop_df["regime"] = res["regime"].astype("string")

    # Rec 3: attach relative strength vs BTC (daily features asof-merged onto
    # the execution frames). Runs here so the RS columns exist before
    # get_common_aux_columns blends them into rank_score.
    dict_of_pairs = attach_relative_strength_to_dataset(
        dict_of_pairs=dict_of_pairs,
        btc_pair=btc_pair,
    )

    # Rec 5: attach rolling SQN percentiles (daily) so the gates and the
    # exposure multiplier can use adaptive percentiles instead of fixed
    # absolute sqn thresholds.
    dict_of_pairs = add_sqn_percentiles_to_dataset(
        dict_of_pairs=dict_of_pairs,
        window=252,
    )

    # BTC regime strength (rolling sqn percentile) asof-merged onto the
    # execution frames, so the RS boost can scale with the BTC regime.
    dict_of_pairs = attach_btc_regime_strength(
        dict_of_pairs=dict_of_pairs,
        btc_pair=btc_pair,
    )

    return dict_of_pairs


def add_sqn_percentiles_to_dataset(
    dict_of_pairs: dict,
    window: int = 252,
    min_periods: int | None = None,
) -> dict:
    """
    Add rolling SQN percentiles on each pair's 1daily frame (Rec 5).

    Computes ``sqn_30_pct`` / ``sqn_90_pct`` = rolling percentile rank of the
    existing ``sqn_30`` / ``sqn_90`` columns vs the trailing `window` days.
    These feed the adaptive gates and the exposure multiplier, replacing the
    fixed absolute sqn thresholds.

    Returns the same dict with the percentile columns added to the 1daily
    frames (frames without sqn columns are skipped).
    """
    for pair in dict_of_pairs:
        frames = dict_of_pairs[pair].get("dict_of_frames", {})
        daily = frames.get("1daily")
        if daily is None or daily.empty:
            continue
        if "sqn_30" not in daily.columns or "sqn_90" not in daily.columns:
            continue

        daily["sqn_30_pct"] = rolling_percentile(
            daily["sqn_30"], window=window, min_periods=min_periods
        ).to_numpy()
        daily["sqn_90_pct"] = rolling_percentile(
            daily["sqn_90"], window=window, min_periods=min_periods
        ).to_numpy()

    return dict_of_pairs


def attach_btc_regime_strength(
    dict_of_pairs: dict,
    btc_pair: str = "BTCUSDT",
) -> dict:
    """
    As-of merge BTC's rolling SQN percentile onto every pair's execution frames
    as ``btc_sqn_90_pct`` (BTC regime strength).

    This is the BTC-regime-strength signal used by the regime-conditional RS
    boost (and potentially other scoring) to scale up in strong regimes and
    down in weak ones. Runs in STEP 1 so the column exists before
    ``get_common_aux_columns`` blends it into rank_score.
    """
    btc_daily = dict_of_pairs.get(btc_pair, {}).get("dict_of_frames", {}).get("1daily")
    if btc_daily is None or btc_daily.empty or "sqn_90_pct" not in btc_daily.columns:
        return dict_of_pairs

    btc_pct = (
        btc_daily[["close_time", "sqn_90_pct"]]
        .dropna(subset=["close_time"])
        .sort_values("close_time")
        .rename(columns={"sqn_90_pct": "btc_sqn_90_pct"})
    )
    if btc_pct.empty:
        return dict_of_pairs

    for pair in dict_of_pairs:
        frames = dict_of_pairs[pair].get("dict_of_frames", {})
        for frame_name, frame_df in frames.items():
            if frame_name == "1daily":
                continue
            if frame_df is None or frame_df.empty or "close_time" not in frame_df.columns:
                continue
            target = frame_df.drop(columns=[c for c in frame_df.columns if c == "btc_sqn_90_pct"])
            frames[frame_name] = pd.merge_asof(
                target.sort_values("close_time"),
                btc_pct,
                left_on="close_time",
                right_on="close_time",
                direction="backward",
            )
    return dict_of_pairs


def compute_relative_strength_daily(
    pair_daily: pd.DataFrame,
    btc_daily: pd.DataFrame,
    ema_period: int = 50,
    momentum_days: int = 60,
    sqn_period: int = 90,
    time_col: str = "close_time",
) -> pd.DataFrame:
    """
    Relative strength of a token vs BTC on the daily frame (Rec 3, BTC only).

    ratio = token_close / BTC_close

    Attaches:
      - rs_ratio          : the ratio series itself
      - rs_ratio_ema50    : 50-day EMA of the ratio
      - rs_ratio_mom60    : 60-day ratio momentum (ratio / ratio.shift(60) - 1)
      - rs_sqn_90         : rolling SQN of the ratio
      - rs_ok             : ratio above its EMA AND positive momentum AND ratio SQN > 0

    Returns a copy of pair_daily with the RS columns added. Missing values are
    left as NaN/False so downstream code can treat them as neutral.
    """
    out = pair_daily.sort_values(time_col).copy()
    if time_col not in out.columns or "close" not in out.columns:
        return out
    if time_col not in btc_daily.columns or "close" not in btc_daily.columns:
        return out

    btc = btc_daily.sort_values(time_col).copy()

    aligned = pd.merge_asof(
        out[[time_col, "close"]].sort_values(time_col),
        btc[[time_col, "close"]].sort_values(time_col).rename(columns={"close": "btc_close"}),
        left_on=time_col,
        right_on=time_col,
        direction="backward",
    )

    pair_close = pd.to_numeric(aligned["close"], errors="coerce")
    btc_close = pd.to_numeric(aligned["btc_close"], errors="coerce")
    ratio_values = (pair_close / btc_close.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    ratio = pd.Series(ratio_values.to_numpy(), index=out.index)

    ratio_ema = talib.EMA(ratio.to_numpy(), timeperiod=max(int(ema_period), 2))
    ratio_mom = ratio / ratio.shift(max(int(momentum_days), 1)) - 1.0

    ratio_df = pd.DataFrame({"close": ratio.to_numpy()}, index=out.index)
    ratio_sqn = compute_sqn(ratio_df, window=max(int(sqn_period), 2), price_col="close")

    out["rs_ratio"] = ratio.to_numpy()
    out["rs_ratio_ema50"] = ratio_ema
    out["rs_ratio_mom60"] = ratio_mom.to_numpy()
    out["rs_sqn_90"] = ratio_sqn.to_numpy()
    out["rs_ok"] = (
        (out["rs_ratio"] > out["rs_ratio_ema50"])
        & (out["rs_ratio_mom60"] > 0.0)
        & (out["rs_sqn_90"] > 0.0)
    ).fillna(False)

    return out


RS_COLUMNS = [
    "rs_ratio",
    "rs_ratio_ema50",
    "rs_ratio_mom60",
    "rs_sqn_90",
    "rs_ok",
]


def attach_relative_strength_to_dataset(
    dict_of_pairs: dict,
    btc_pair: str = "BTCUSDT",
    ema_period: int = 50,
    momentum_days: int = 60,
    sqn_period: int = 90,
) -> dict:
    """
    Compute relative strength vs BTC on each pair's 1daily frame and
    asof-merge the RS columns onto the pair's execution frames (4H/1H).

    The RS features are computed on the daily scale (50-day EMA, 60-day
    momentum, ratio SQN) and carried down to the execution frame, mirroring how
    the daily gating layer is attached. Pure additive: pairs without daily data
    are skipped and existing columns are left untouched.
    """
    btc_frames = dict_of_pairs.get(btc_pair, {}).get("dict_of_frames", {})
    btc_daily = btc_frames.get("1daily")
    if btc_daily is None or btc_daily.empty or "close" not in btc_daily.columns:
        return dict_of_pairs

    for pair in dict_of_pairs:
        frames = dict_of_pairs[pair].get("dict_of_frames", {})
        daily = frames.get("1daily")
        if daily is None or daily.empty or "close" not in daily.columns:
            continue

        daily_rs = compute_relative_strength_daily(
            pair_daily=daily,
            btc_daily=btc_daily,
            ema_period=ema_period,
            momentum_days=momentum_days,
            sqn_period=sqn_period,
        )
        frames["1daily"] = daily_rs

        rs_cols = [c for c in RS_COLUMNS if c in daily_rs.columns]
        if not rs_cols:
            continue

        for frame_name, frame_df in frames.items():
            if frame_name == "1daily":
                continue
            if frame_df is None or frame_df.empty or "close_time" not in frame_df.columns:
                continue
            frames[frame_name] = pd.merge_asof(
                frame_df.sort_values("close_time"),
                daily_rs[["close_time", *rs_cols]].sort_values("close_time"),
                left_on="close_time",
                right_on="close_time",
                direction="backward",
            )

    return dict_of_pairs


def get_interval_of_frame(df):
    difference = df.open_time.iloc[1] - df.open_time.iloc[0]
    
    days = difference.days
    hours = difference.seconds // 3600
    minutes = difference.seconds // 60
    
    if days != 0:
        if days // 7 > 0:
            return str(days//7)+"weekly"
        else:
            return str(days)+"daily"
    
    elif hours != 0:
        return str(hours)+"hourly"
    
    else:
        return str(minutes)+"min"
    
def prepare_df(df):
    
    frame = df.copy()
    
    look_back_dict = {
        "1hourly" : 24*2,
        "4hourly" : 6*14,
        "1daily": 30,
        "3daily": 10*6,
        "1weekly": 52,
    }
    
    frame_interval = get_interval_of_frame(frame)
    
    if frame_interval in look_back_dict:
        look_back = look_back_dict[frame_interval]
    else:
        look_back = 24
    
    if len(frame) < look_back:
        look_back = len(frame)


    frame["ema7"] = talib.EMA(frame.close, 7)
    frame["ema7_high"] = talib.EMA(frame.high, 7)
    frame["ema25"] = talib.EMA(frame.close, 25)
    frame["ema50"] = talib.EMA(frame.close, 50)
    frame["ma25"] = talib.MA(frame.close, 25)
    frame["ma50"] = talib.MA(frame.close, 50)
    frame["ema100"] = talib.EMA(frame.close, 100)
    frame["ema200"] = talib.EMA(frame.close, 200)
    frame["ma100"] = talib.MA(frame.close, 100)
    frame["ma200"] = talib.MA(frame.close, 200)
    frame["rsi"] = talib.RSI(frame["close"], 6)
    frame["max"] = talib.MAX(frame.close, look_back).shift(1)
    frame["min"] = talib.MIN(frame.close, look_back).shift(1)
    frame[["boll_upper", "boll_mid", "boll_lower"]] = pd.DataFrame(talib.BBANDS(frame.close, 21)).T
    frame["short_spread"] = talib.MA(frame.high-frame.low, look_back)
    frame["volatility"] = frame["short_spread"] / frame["close"]

    return frame


def add_hh_ll_columns(frame_df):
    close_highs = find_peaks(frame_df["close"], distance=6, width = 5)[0]
    close_times_at_close_highs = frame_df.iloc[close_highs]["close_time"]
    shifted_close_highs_closes = frame_df.iloc[close_highs][["close", "rsi"]].shift(1)

    shifted_peak_frame = pd.concat([close_times_at_close_highs, shifted_close_highs_closes], axis=1)
    shifted_peak_frame.rename(columns={"close": "prev_high", "rsi": "prev_high_rsi"}, inplace=True)

    frame_df = pd.merge_asof(
        frame_df.sort_values("close_time"),
        shifted_peak_frame.sort_values("close_time"),
        left_on="close_time",
        right_on="close_time",
        direction="backward",
        suffixes=[f"", f""]
    )

    frame_df.loc[close_highs, "HH"] = frame_df.loc[close_highs]["close"] > frame_df.loc[close_highs]["prev_high"]
    frame_df["HH"] = frame_df["HH"].ffill()

    # higher_rsi mirrors HH but on RSI: at each confirmed swing high it is True
    # when the CURRENT high's RSI is higher than the PREVIOUS swing high's RSI
    # (momentum-confirming - RSI makes a higher high alongside price, i.e. no
    # bearish divergence). Persists via ffill until the next swing high, exactly
    # like HH. The very first swing high has no prior RSI -> False.
    frame_df.loc[close_highs, "higher_rsi"] = (
        frame_df.loc[close_highs]["rsi"] > frame_df.loc[close_highs]["prev_high_rsi"]
    )
    frame_df["higher_rsi"] = frame_df["higher_rsi"].ffill()

    close_lows = find_peaks(-frame_df["close"], distance=6, width = 5)[0]
    close_times_at_close_lows = frame_df.iloc[close_lows]["close_time"]

    shifted_close_lows_closes = frame_df.iloc[close_lows][["close", "rsi"]].shift(1)

    shifted_low_frame = pd.concat([close_times_at_close_lows, shifted_close_lows_closes], axis=1)
    shifted_low_frame.rename(columns={"close": "prev_low", "rsi": "prev_low_rsi"}, inplace=True)

    frame_df = pd.merge_asof(
        frame_df.sort_values("close_time"),
        shifted_low_frame.sort_values("close_time"),
        left_on="close_time",
        right_on="close_time",
        direction="backward",
        suffixes=[f"", f""]
    )

    frame_df.loc[close_lows, "LL"] = frame_df.loc[close_lows]["close"] < frame_df.loc[close_lows]["prev_low"]
    frame_df["LL"] = frame_df["LL"].ffill()

    # lower_rsi mirrors LL but on RSI: at each confirmed swing low it is True
    # when the CURRENT low's RSI is LOWER than the PREVIOUS swing low's RSI
    # (momentum-weakening - RSI makes a lower low alongside price). Persists via
    # ffill until the next swing low, exactly like LL. The very first swing low
    # has no prior RSI -> False.
    frame_df.loc[close_lows, "lower_rsi"] = (
        frame_df.loc[close_lows]["rsi"] < frame_df.loc[close_lows]["prev_low_rsi"]
    )
    frame_df["lower_rsi"] = frame_df["lower_rsi"].ffill()

    return frame_df


# =============================================================================
# Causal swing-based trend lines (upper resistance / lower support)
# -----------------------------------------------------------------------------
# Draws trend lines the way a human would at a given moment, with NO look-ahead,
# so the values are identical offline (backtest) and on a live feed:
#   * Swing highs/lows are found with find_peaks(..., distance=d). A swing at
#     index i is only treated as KNOWN once the full `distance` window to its
#     right has closed (index <= t - d), so the line never uses a bar that is
#     still forming (no repainting).
#   * Upper line = line through the two most recent confirmed swing HIGHS such
#     that it (a) is not pierced by any of the most recent `window_peaks` swing
#     highs by more than `margin * short_spread` (i.e. it loosely contains the
#     recent peaks), and (b) has not been broken by a close printing above it
#     since its earlier anchor (tolerance `break_margin * short_spread`).
#     Candidates are scanned from the most recent pair backward (the newest peak
#     is dropped first if it is a fresh outlier); the first valid one is used,
#     else NaN.
#   * Lower line = mirror through swing LOWS (pierced from below / broken by a
#     close printing below it).
#   * Both lines are recomputed as new swings confirm (rolling), so a broken
#     line is naturally replaced by the next valid one (or NaN until one exists).
#
# Adds four columns to a single OHLC frame:
#     <prefix>upper_trend  : upper resistance line value at the row
#     <prefix>peak_count   : how many of the most recent swing highs the upper
#                            line contains within margin*short_spread (touches it)
#     <prefix>lower_trend  : lower support line value at the row
#     <prefix>trough_count : how many of the most recent swing lows the lower
#                            line contains within margin*short_spread
#   (all NaN where no valid line)
# so it can be applied per pair/per timeframe exactly like prepare_df.
# =============================================================================

def _rolling_spread(high, low, window=7):
    h = pd.Series(high)
    l = pd.Series(low)
    return (h - l).rolling(window, min_periods=1).mean().to_numpy()


def _trend_line(x1, y1, x2, y2, x):
    if x2 == x1:
        return np.full(np.shape(x), np.nan, dtype=float)
    slope = (y2 - y1) / (x2 - x1)
    return slope * (np.asarray(x, dtype=float) - x1) + y1


def _derive_upper_trend(peaks, high, close, spread, t, window_peaks, margin, break_margin):
    """Best valid upper line ``(p2, p1)`` at row ``t``, else None."""
    for k in range(len(peaks) - 1, 0, -1):
        if (len(peaks) - 1 - k) >= window_peaks:
            break  # only the most recent `window_peaks` anchor pairs
        p2, p1 = peaks[k - 1], peaks[k]
        y2, y1 = high[p2], high[p1]
        if not (np.isfinite(y2) and np.isfinite(y1)):
            continue
        # (a) no recent peak pierces above the line by more than margin*spread
        recent = peaks[max(0, len(peaks) - window_peaks):]
        pierced = False
        for p in recent:
            sp = spread[p] if np.isfinite(spread[p]) else 0.0
            if high[p] - _trend_line(p2, y2, p1, y1, p) > margin * sp:
                pierced = True
                break
        if pierced:
            continue
        # (b) no close above the line since the earlier anchor
        xs = np.arange(p2, t + 1)
        line = _trend_line(p2, y2, p1, y1, xs)
        spx = np.where(np.isfinite(spread[xs]), spread[xs], 0.0)
        if np.any(close[xs] > line + break_margin * spx):
            continue
        return (p2, p1)
    return None


def _derive_lower_trend(troughs, low, close, spread, t, window_peaks, margin, break_margin):
    """Best valid lower line ``(t2, t1)`` at row ``t``, else None."""
    for k in range(len(troughs) - 1, 0, -1):
        if (len(troughs) - 1 - k) >= window_peaks:
            break
        t2, t1 = troughs[k - 1], troughs[k]
        y2, y1 = low[t2], low[t1]
        if not (np.isfinite(y2) and np.isfinite(y1)):
            continue
        recent = troughs[max(0, len(troughs) - window_peaks):]
        pierced = False
        for p in recent:
            sp = spread[p] if np.isfinite(spread[p]) else 0.0
            if low[p] - _trend_line(t2, y2, t1, y1, p) < -margin * sp:
                pierced = True
                break
        if pierced:
            continue
        xs = np.arange(t2, t + 1)
        line = _trend_line(t2, y2, t1, y1, xs)
        spx = np.where(np.isfinite(spread[xs]), spread[xs], 0.0)
        if np.any(close[xs] < line - break_margin * spx):
            continue
        return (t2, t1)
    return None


def attach_trend_lines(
    frame_df,
    *,
    distance: int = 3,
    window_peaks: int = 4,
    margin: float = 1.0,
    break_margin: float = 0.0,
    spread_col: str = "short_spread",
    prefix: str = "",
    inplace: bool = False,
):
    """Add ``<prefix>upper_trend``/``<prefix>lower_trend`` and the matching
    ``<prefix>peak_count``/``<prefix>trough_count`` columns (how many recent
    swing highs/lows the active line touches, within margin*short_spread).

    Parameters
    ----------
    frame_df : pd.DataFrame
        OHLC frame (requires ``high``, ``low``, ``close``). A ``short_spread``
        column is used for the relative margins if present, otherwise a rolling
        mean of (high - low) is computed.
    distance : int
        Minimum bars between swing points (find_peaks distance). Also the
        confirmation lag: a swing is only known after its full right window
        closed (no look-ahead).
    window_peaks : int
        How many of the most recent swing points the line must loosely contain.
    margin : float
        Pierce tolerance in short_spread units.
    break_margin : float
        Close-vs-line break tolerance in short_spread units.
    prefix : str
        Optional column-name prefix.
    inplace : bool
        Add the columns to ``frame_df`` itself instead of a copy.
    """
    out = frame_df if inplace else frame_df.copy()
    n = len(out)
    high = pd.to_numeric(out["high"], errors="coerce").to_numpy(dtype=float)
    low = pd.to_numeric(out["low"], errors="coerce").to_numpy(dtype=float)
    close = pd.to_numeric(out["close"], errors="coerce").to_numpy(dtype=float)
    if spread_col in out.columns:
        spread = pd.to_numeric(out[spread_col], errors="coerce").to_numpy(dtype=float)
    else:
        spread = _rolling_spread(high, low, window=max(distance, 3))
    spread = np.where(np.isfinite(spread) & (spread > 0), spread, np.nan)

    all_peaks = find_peaks(high, distance=distance)[0]
    all_troughs = find_peaks(-low, distance=distance)[0]
    peak_set = set(all_peaks.tolist())
    trough_set = set(all_troughs.tolist())

    peaks = []
    troughs = []
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    peak_count = np.full(n, np.nan)
    trough_count = np.full(n, np.nan)

    up_pair = None
    lo_pair = None

    for t in range(n):
        idx = t - distance
        new_peak = idx >= 0 and idx in peak_set
        new_trough = idx >= 0 and idx in trough_set
        if new_peak:
            peaks.append(idx)
        if new_trough:
            troughs.append(idx)

        # ---- upper ----
        # A newly confirmed swing re-anchors the line for THIS row.
        if new_peak:
            up_pair = _derive_upper_trend(
                peaks, high, close, spread, t, window_peaks, margin, break_margin
            )
        # Value at t from the active line. If the close breaks it at t, row t keeps
        # the value of the line it just broke (so a breakout test `close > value`
        # can fire at t) — the new state applies from t+1 onward.
        broken_up = False
        if up_pair is not None:
            p2, p1 = up_pair
            upper[t] = _trend_line(p2, high[p2], p1, high[p1], t)
            # count how many of the most recent swing highs touch the line
            recent = peaks[max(0, len(peaks) - window_peaks):]
            cnt = 0
            for p in recent:
                sp = spread[p] if np.isfinite(spread[p]) else 0.0
                line_p = _trend_line(p2, high[p2], p1, high[p1], p)
                if np.isfinite(line_p) and abs(high[p] - line_p) <= margin * sp:
                    cnt += 1
            peak_count[t] = cnt
            sp_t = spread[t] if np.isfinite(spread[t]) else 0.0
            broken_up = close[t] > upper[t] + break_margin * sp_t
        if broken_up:
            up_pair = _derive_upper_trend(
                peaks, high, close, spread, t, window_peaks, margin, break_margin
            )

        # ---- lower ----
        if new_trough:
            lo_pair = _derive_lower_trend(
                troughs, low, close, spread, t, window_peaks, margin, break_margin
            )
        broken_lo = False
        if lo_pair is not None:
            t2, t1 = lo_pair
            lower[t] = _trend_line(t2, low[t2], t1, low[t1], t)
            # count how many of the most recent swing lows touch the line
            recent = troughs[max(0, len(troughs) - window_peaks):]
            cnt = 0
            for p in recent:
                sp = spread[p] if np.isfinite(spread[p]) else 0.0
                line_p = _trend_line(t2, low[t2], t1, low[t1], p)
                if np.isfinite(line_p) and abs(low[p] - line_p) <= margin * sp:
                    cnt += 1
            trough_count[t] = cnt
            sp_t = spread[t] if np.isfinite(spread[t]) else 0.0
            broken_lo = close[t] < lower[t] - break_margin * sp_t
        if broken_lo:
            lo_pair = _derive_lower_trend(
                troughs, low, close, spread, t, window_peaks, margin, break_margin
            )

    out[f"{prefix}upper_trend"] = upper
    out[f"{prefix}peak_count"] = peak_count
    out[f"{prefix}lower_trend"] = lower
    out[f"{prefix}trough_count"] = trough_count
    return out


def attach_trend_lines_to_dataset(dict_of_pairs, *, frames=None, **kwargs):
    """Apply ``attach_trend_lines`` to every (pair, frame) — like ``prepare_df``.

    ``frames`` optionally restricts which frames get the columns. Extra keyword
    arguments are forwarded to ``attach_trend_lines``.
    """
    for pdata in dict_of_pairs.values():
        frames_dict = pdata.get("dict_of_frames", {})
        for frame_name, df in frames_dict.items():
            if frames is not None and frame_name not in frames:
                continue
            if df is None or not isinstance(df, pd.DataFrame) or df.empty:
                continue
            if not {"high", "low", "close"}.issubset(df.columns):
                continue
            frames_dict[frame_name] = attach_trend_lines(df, **kwargs)
    return dict_of_pairs


def get_common_aux_columns(frame_df, rs_boost_weight: float = 0.5, rs_boost_regime_conditional: bool = True):
    frame_df = frame_df.copy()
    close = pd.to_numeric(frame_df["close"], errors="coerce")
    high = pd.to_numeric(frame_df["high"], errors="coerce")
    low = pd.to_numeric(frame_df["low"], errors="coerce")
    qav = pd.to_numeric(frame_df["quote_asset_volume"], errors="coerce")

    stop_price_series = frame_df[["open", "close"]].min(axis=1) - 1.25*frame_df["short_spread"]

    vol_window = 7
    equity_proxy = 10_000.0
    max_positions = 5
    fee_bps = 10.0

    rolling_mean_move = ((high / low) - 1).rolling(vol_window, min_periods=vol_window).mean()
    rolling_mean_volume = qav.rolling(vol_window, min_periods=vol_window).mean()

    denom_vol = rolling_mean_move.replace(0, np.nan)
    qvpuc = (rolling_mean_volume / denom_vol).fillna(0.0)



    stop_ratio = (1.0 - stop_price_series / close).clip(lower=1e-6).fillna(1e-6)

    approx_position_cash = equity_proxy / max(max_positions, 1)
    slippage_pct_estimate = (approx_position_cash / qvpuc.replace(0, np.nan)).fillna(0.0)
    fee_pct = fee_bps / 10_000.0
    round_trip_cost = 2.0 * (slippage_pct_estimate + fee_pct)

    rank = (rolling_mean_move - round_trip_cost) / stop_ratio
    # rank = (frame_df['rolling_max'] - frame_df['rolling_min']) / close
    # expected_exit_price = (frame_df['ema7'] + 2.5*frame_df['short_spread'])
    # rank = (expected_exit_price - stop_price_series) / stop_ratio

    # Rec 3: blend relative strength vs BTC into the entry score so capital
    # shifts toward genuine outperformers (ratio > 50d EMA, 60d momentum > 0,
    # ratio SQN > 0). RS-ok pairs get a rank multiplier of
    # (1 + rs_boost_weight * regime_strength), where regime_strength is the
    # BTC rolling SQN percentile (btc_sqn_90_pct) when available, else 1.0.
    # This makes the RS boost strong in strong BTC regimes (expansion) and ~0
    # in weak regimes (slow market), per the A/B findings. Neutral when rs_ok
    # is absent.
    if "rs_ok" in frame_df.columns:
        rs_ok = pd.to_numeric(frame_df["rs_ok"], errors="coerce").fillna(False).astype(bool)
        regime_strength = 1.0
        if rs_boost_regime_conditional and "btc_sqn_90_pct" in frame_df.columns:
            regime_strength = pd.to_numeric(
                frame_df["btc_sqn_90_pct"], errors="coerce"
            ).fillna(0.5).clip(lower=0.0, upper=1.0).to_numpy(dtype=float)
        rs_boost = 1.0 + float(rs_boost_weight) * regime_strength * rs_ok.to_numpy()
        rank = rank * rs_boost

    aux_df = pd.DataFrame(index=frame_df.index)

    aux_df["quote_vol_per_unit_change"] = qvpuc.to_numpy()
    aux_df["rank_score"] = rank.fillna(0.0).to_numpy()

    for column in aux_df.columns:
        if column in frame_df.columns:
            frame_df = frame_df.drop(columns=[column])

    return pd.concat([frame_df, aux_df], axis=1)



def _nearest_levels_from_sorted(sorted_levels, price):
    if len(sorted_levels) == 0 or np.isnan(price):
        return np.nan, np.nan

    idx = np.searchsorted(sorted_levels, price, side="left")

    if idx <= 0:
        return np.nan, float(sorted_levels[0])
    if idx >= len(sorted_levels):
        return float(sorted_levels[-1]), np.nan

    return float(sorted_levels[idx - 1]), float(sorted_levels[idx])


def populate_historical_levels_fast(
    ohlc_df,
    levels_reactions_limit=12,
    lookback_bars=None,
    price_col="close",
    skip_rows=24
):
    """
    Fast, past-only level population.

    Adds the following columns:
    - levels: sorted list of clustered levels available up to that bar
    - nearest_support: closest level <= current price
    - nearest_resistance: closest level >= current price

    Notes
    -----
    - Uses only information available up to each bar index.
    - Re-clusters only when new extrema appear, which is much faster than
      re-running KMeans on every row.
    - If lookback_bars is provided, levels are built from extrema inside the
      rolling window ending at the current bar.
    """
    df = ohlc_df.copy().reset_index(drop=True)

    required_cols = {"high", "low", price_col}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    high_vals = pd.to_numeric(df["high"], errors="coerce").to_numpy(dtype=float)
    low_vals = pd.to_numeric(df["low"], errors="coerce").to_numpy(dtype=float)
    price_vals = pd.to_numeric(df[price_col], errors="coerce").to_numpy(dtype=float)

    n = len(df)
    if n == 0:
        df["levels"] = []
        df["nearest_support"] = []
        df["nearest_resistance"] = []
        return df

    peak_idx = find_peaks(high_vals)[0]
    trough_idx = find_peaks(-low_vals)[0]

    extrema_events = []
    extrema_events.extend((int(i), float(high_vals[i])) for i in peak_idx if not np.isnan(high_vals[i]))
    extrema_events.extend((int(i), float(low_vals[i])) for i in trough_idx if not np.isnan(low_vals[i]))
    extrema_events.sort(key=lambda x: x[0])

    ext_indices = []
    ext_prices = []
    event_ptr = 0
    cached_levels = np.array([], dtype=float)

    levels_out = [None] * n
    support_out = np.full(n, np.nan, dtype=float)
    resistance_out = np.full(n, np.nan, dtype=float)

    for t in range(n):
        if t % skip_rows != 0 and t > 0:
            levels_out[t] = levels_out[t - 1]
            support_out[t] = support_out[t - 1]
            resistance_out[t] = resistance_out[t - 1]
            continue

        changed = False
        while event_ptr < len(extrema_events) and extrema_events[event_ptr][0] <= t:
            idx_i, px_i = extrema_events[event_ptr]
            ext_indices.append(idx_i)
            ext_prices.append(px_i)
            event_ptr += 1
            changed = True

        if lookback_bars is not None:
            left = max(0, t - int(lookback_bars) + 1)
            while ext_indices and ext_indices[0] < left:
                ext_indices.pop(0)
                ext_prices.pop(0)
                changed = True
            window_slice = slice(left, t + 1)
        else:
            window_slice = slice(0, t + 1)

        if changed or t == 0:
            high_max = np.nanmax(high_vals[window_slice]) if t >= window_slice.start else np.nan
            low_min = np.nanmin(low_vals[window_slice]) if t >= window_slice.start else np.nan

            if len(ext_prices) == 0:
                base_levels = [high_max, low_min]
            else:
                extrema_arr = np.asarray(ext_prices, dtype=float).reshape(-1, 1)
                k = min(int(levels_reactions_limit), len(extrema_arr))
                if k <= 0:
                    base_levels = [high_max, low_min]
                else:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", ConvergenceWarning)
                        model = KMeans(n_clusters=k, random_state=0)
                        model.fit(extrema_arr)
                    centers = model.cluster_centers_.reshape(-1).tolist()
                    base_levels = centers + [high_max, low_min]

            clean_levels = np.array(sorted({float(x) for x in base_levels if pd.notna(x)}), dtype=float)
            cached_levels = clean_levels

        support, resistance = _nearest_levels_from_sorted(cached_levels, price_vals[t])
        support_out[t] = support
        resistance_out[t] = resistance
        levels_out[t] = cached_levels.tolist()

    df["levels"] = levels_out
    df["nearest_support"] = support_out
    df["nearest_resistance"] = resistance_out
    return df