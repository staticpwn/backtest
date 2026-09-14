from __future__ import annotations

import numpy as np
import pandas as pd
import talib
from helpers import *

# Aux columns emitted by all strategy functions.
# These are consumed by compile_backtest_payload (aux_columns argument)
# and by the backtest engine for sizing and candidate ranking.
STRATEGY_AUX_COLUMNS = [
    "stop_price",
    "take_profit_price",
    "quote_vol_per_unit_change",
    "rank_score",
    # trailing_stop_pct: per-entry PERCENT hard-stop ratchet (peak*(1-pct)), baked
    # per entry. The engine reads it via EngineConfig.trailing_stop_pct_col and
    # stores it on the position as trailing_ratchet_pct. This is NOT the
    # spread-multiple trailing exit (that lives in EngineConfig under
    # trailing_arm_spread_multiple / trailing_exit_spread_multiple).
    "trailing_stop_pct",
    "max_hold_bars",
    "vol_take_profit_threshold",
    "min_signal_hold_bars",
    "short_spread",   # engine: read as short_spread_arr inside _evaluate_and_close_position
]


def rebake_take_profit_levels(
    dict_of_pairs: dict,
    execution_frame: str,
    strategy_specs: list[dict[str, str]],
    fast_profit_by_entry_col: dict[str, float],
) -> dict:
    """
    Recompute take-profit levels (and the TP component of the exit column) for
    already-applied strategies, without re-running the full entry logic.

    This is the cheap way to sweep ``fast_profit_threshold``: the only thing a
    different threshold changes is

        take_profit_price = ema7_high + fast_profit_threshold * short_spread.shift(1)
        tp_exit           = high > take_profit_price

    Entries, stop prices, gating and every other aux column are left untouched,
    so the resulting payload is identical to re-running the ``apply_*`` strategy
    with the new threshold, minus the wasted recomputation.

    Parameters
    ----------
    dict_of_pairs : dict
        Post-STEP-3 dataset (entry/exit columns plus the
        ``{entry_col}__stop_price`` / ``{entry_col}__take_profit_price`` aux
        columns already present on the execution frame).
    execution_frame : str
        Execution frame key, e.g. "4hourly".
    strategy_specs : list[dict[str, str]]
        Strategy specs (each must contain ``entry_col`` / ``exit_col``).
    fast_profit_by_entry_col : dict[str, float]
        Maps each strategy's ``entry_col`` (which is also its aux prefix) to the
        new ``fast_profit_threshold`` to bake in. Strategies absent from this map
        are left untouched.

    Returns
    -------
    dict
        The same ``dict_of_pairs`` with take-profit levels and exit columns
        updated in place.
    """
    for spec in strategy_specs:
        entry_col = spec.get("entry_col")
        exit_col = spec.get("exit_col")
        fpt = fast_profit_by_entry_col.get(entry_col)
        if fpt is None:
            continue

        tp_col = f"{entry_col}__take_profit_price"
        stop_col = f"{entry_col}__stop_price"

        for pair, pdata in dict_of_pairs.items():
            df = pdata["dict_of_frames"].get(execution_frame)
            if df is None or df.empty:
                continue
            if tp_col not in df.columns or stop_col not in df.columns:
                continue

            df = df.copy()
            if "ema7_high" not in df.columns:
                df["ema7_high"] = talib.EMA(pd.to_numeric(df["high"], errors="coerce"), 7)

            high = pd.to_numeric(df["high"], errors="coerce")
            close = pd.to_numeric(df["close"], errors="coerce")
            ema7_high = pd.to_numeric(df["ema7_high"], errors="coerce")
            short_spread = pd.to_numeric(df["short_spread"], errors="coerce")
            stop_price = pd.to_numeric(df[stop_col], errors="coerce")

            take_profit = ema7_high + fpt * short_spread.shift(1)
            tp_exit = high > take_profit
            stop_exit = close < stop_price

            df[tp_col] = take_profit.to_numpy()
            df[exit_col] = (stop_exit | tp_exit).to_numpy()
            pdata["dict_of_frames"][execution_frame] = df

    return dict_of_pairs


def _attach_aux_columns(
    frame_df: pd.DataFrame,
    stop_price_series: pd.Series,
    take_profit_price_series: pd.Series,
    vol_window: int = 7,
    fee_bps: float = 10.0,
    equity_proxy: float = 10_000.0,
    max_positions: int = 5,
    trailing_stop_pct: float = 0.06,
    max_hold_bars: int = 36,
    vol_take_profit_multiple: float = 5.0,
    min_signal_hold_bars: int = 6,
    aux_prefix: str = "",
    rsi_slope: pd.Series = None,
) -> pd.DataFrame:
    """
    Attach sizing and ranking aux columns to an execution-frame DataFrame.

    Parameters
    ----------
    frame_df : pd.DataFrame
        Execution-frame DataFrame with at least: high, low, close, quote_asset_volume.
    stop_price_series : pd.Series
        Stop-loss price per bar (same index as frame_df).
    vol_window : int
        Rolling window for volume/move estimation (default 7 bars).
    fee_bps : float
        Fee in basis points per side (used in rank_score estimate).
    equity_proxy : float
        Proxy for current equity used only in rank_score slippage estimate.
        Actual equity is unknown at precompute time; this is a conservative proxy.
    max_positions : int
        Max simultaneous positions used in count-cap size proxy for rank_score.
    trailing_stop_pct : float
        Trailing stop distance as a fraction of peak price after entry.
    max_hold_bars : int
        Maximum holding period in candles before time-based exit.
    vol_take_profit_multiple : float
        Volatility multiple used to derive a profit-spike take-profit threshold.

    Returns
    -------
    pd.DataFrame
        frame_df with stop_price, quote_vol_per_unit_change, rank_score attached.
    """
    def _name(col: str) -> str:
        return f"{aux_prefix}__{col}" if aux_prefix else col

    frame_df = frame_df.copy()

    aux_df = pd.DataFrame(index=frame_df.index)
    aux_df[_name("stop_price")] = stop_price_series.fillna(0.0).to_numpy()
    aux_df[_name("take_profit_price")] = take_profit_price_series.fillna(0.0).to_numpy()
    aux_df[_name("rsi_slope")] = rsi_slope.fillna(0.0).to_numpy() if rsi_slope is not None else np.zeros(len(frame_df))
    aux_df[_name("trailing_stop_pct")] = float(max(trailing_stop_pct, 0.0))
    aux_df[_name("max_hold_bars")] = float(max(max_hold_bars, 0))
    aux_df[_name("vol_take_profit_threshold")] = float(max(vol_take_profit_multiple, 0.0))
    aux_df[_name("min_signal_hold_bars")] = float(max(min_signal_hold_bars, 0))

    cols_to_remove = [col for col in aux_df.columns if col in frame_df.columns]
    if cols_to_remove:
        frame_df = frame_df.drop(columns=cols_to_remove)

    return pd.concat([frame_df, aux_df], axis=1)


def apply_bullish_divergence_strategy(
    dict_of_pairs: dict,
    execution_frame: str,
    entry_col: str,
    exit_col: str,
    osc_col: str = "rsi",
    left_bars: int = 3,
    right_bars: int = 3,
    oversold_threshold: float = 35.0,
    max_wait_bars: int = 12,
    use_trend_filter: bool = False,
    trend_col: str = "ema50",
    trend_slope_col: str = "ema50_slope",
    fast_profit_threshold: float = 1.5,
    stop_lookback_bars: int = 10,
    vol_window: int = 7,
    fee_bps: float = 10.0,
    equity_proxy: float = 10_000.0,
    max_positions: int = 5,
    trailing_stop_pct: float = 0.06,
    max_hold_bars: int = 36,
    vol_take_profit_multiple: float = 5.0,
    min_signal_hold_bars: int = 6,
    aux_prefix: str = "",
) -> dict:
    """
    Bullish divergence reversal strategy.

    Entry:
    - confirmed bullish divergence on osc_col
    - optional trend filter on trend_col / trend_slope_col

    Exit:
    - stop below recent swing structure
    - take profit above short-term high structure
    """

    for pair in dict_of_pairs:
        frame_df = dict_of_pairs[pair]["dict_of_frames"].get(execution_frame)

        if frame_df is None or frame_df.empty:
            continue

        required_cols = {"close", "high", "low", "short_spread", osc_col}
        missing = [col for col in required_cols if col not in frame_df.columns]
        if missing:
            raise KeyError(f"{pair} missing required columns for bullish divergence: {missing}")

        if "ema7_high" not in frame_df.columns:
            frame_df["ema7_high"] = talib.EMA(pd.to_numeric(frame_df["high"], errors="coerce"), 7)

        if use_trend_filter:
            trend_ok = pd.Series(True, index=frame_df.index, dtype="bool")

            if trend_col in frame_df.columns:
                trend_ok &= pd.to_numeric(frame_df[trend_col], errors="coerce").notna()
                trend_ok &= pd.to_numeric(frame_df["close"], errors="coerce") > pd.to_numeric(frame_df[trend_col], errors="coerce")

            if trend_slope_col in frame_df.columns:
                trend_ok &= pd.to_numeric(frame_df[trend_slope_col], errors="coerce") > 0
        else:
            trend_ok = pd.Series(True, index=frame_df.index, dtype="bool")

        divergence_entry = confirmed_bullish_divergence_entry(
            frame_df,
            osc_col=osc_col,
            left_bars=left_bars,
            right_bars=right_bars,
            oversold_threshold=oversold_threshold,
            max_wait_bars=max_wait_bars,
        )

        frame_df[entry_col] = divergence_entry & trend_ok

        recent_low = pd.to_numeric(frame_df["low"], errors="coerce").rolling(
            stop_lookback_bars,
            min_periods=max(3, right_bars + 1),
        ).min()
        stop_price_series = recent_low - 1.25 * pd.to_numeric(frame_df["short_spread"], errors="coerce")

        take_profit_price_series = (
            pd.to_numeric(frame_df["ema7_high"], errors="coerce")
            + fast_profit_threshold * pd.to_numeric(frame_df["short_spread"], errors="coerce").shift(1)
        )

        stop_exit = pd.to_numeric(frame_df["close"], errors="coerce") < stop_price_series
        tp_exit = pd.to_numeric(frame_df["high"], errors="coerce") > take_profit_price_series

        frame_df[exit_col] = stop_exit | tp_exit

        frame_df = _attach_aux_columns(
            frame_df=frame_df,
            stop_price_series=stop_price_series,
            take_profit_price_series=take_profit_price_series,
            vol_window=vol_window,
            fee_bps=fee_bps,
            equity_proxy=equity_proxy,
            max_positions=max_positions,
            trailing_stop_pct=trailing_stop_pct,
            max_hold_bars=max_hold_bars,
            vol_take_profit_multiple=vol_take_profit_multiple,
            min_signal_hold_bars=min_signal_hold_bars,
            aux_prefix=aux_prefix,
        )

        dict_of_pairs[pair]["dict_of_frames"][execution_frame] = frame_df

    return dict_of_pairs

def apply_break_strategy(
    dict_of_pairs: dict,
    execution_frame: str,
    entry_col: str,
    exit_col: str,
    break_column: str,
    fast_profit_threshold: float = 1.5,
    vol_window: int = 7,
    fee_bps: float = 10.0,
    equity_proxy: float = 10_000.0,
    max_positions: int = 5,
    trailing_stop_pct: float = 0.06,
    max_hold_bars: int = 36,
    vol_take_profit_multiple: float = 5.0,
    min_signal_hold_bars: int = 6,
    aux_prefix: str = "",
    gain_ratio: float = 1.2,
    lower_volatility_threshold: float = 0.025,
    upper_volatility_threshold: float = 0.05,
    use_rsi_slope_exit : bool = False,
    use_target_slope : bool = False,
    candle_rel_size: float = None,
    candle_own_size: float = None,
    rsi_slope_entry_lever: float = 1,
    reject_rsi_div : bool = False,
) -> dict:
    """
    Sample strategy: break of a column (i.e., breaks above a moving average or a fixed previous level).

    Attaches entry/exit trigger columns and the shared aux columns
    (stop_price, quote_vol_per_unit_change, rank_score) required by
    the backtest engine for sizing and candidate ranking.

    Parameters
    ----------
    dict_of_pairs : dict
        Dataset in project format: {pair: {"dict_of_frames": {...}}}
    execution_frame : str
        Target execution frame key (e.g. '1hourly' or '4hourly').
    entry_col : str
        Output column name for entry trigger.
    exit_col : str
        Output column name for exit trigger.
    break_column : str
        Column name to be checked for breaks (e.g., a moving average or a fixed level).

    Returns
    -------
    dict
        Modified dict_of_pairs with strategy + aux columns attached.
    """

    for pair in dict_of_pairs:
        frame_df = dict_of_pairs[pair]["dict_of_frames"].get(execution_frame)

        if frame_df is None or frame_df.empty:
            continue

        volatile = (frame_df["volatility"] > lower_volatility_threshold) & (frame_df["volatility"] < upper_volatility_threshold)

        broken = frame_df[f"{break_column}_is_broken"]
       
        is_gainer = frame_df["rolling_gain"] > gain_ratio

        rsi_exit = (use_rsi_slope_exit and (frame_df["rsi_slope"] < 0)) if use_rsi_slope_exit else True

        candle_relative_structure = ((frame_df["close"] - frame_df[break_column]) / (frame_df["close"] - frame_df["open"])) > candle_rel_size if candle_rel_size is not None else True

        candle_own_structure = ((frame_df["close"] - frame_df["open"]) / frame_df["open"]) < candle_own_size if candle_own_size is not None else True


        # Engine expects stop_price to be a price level, not a boolean condition.

        stop_price_series = (frame_df[["open", "close"]].min(axis=1) - 1.25*frame_df["short_spread"])
        # stop_price_series = (frame_df[f"{break_column}"] - 2*frame_df["short_spread"])
        # stop_price_series = frame_df["rolling_min"] - 1.25*frame_df["short_spread"]
        stop_exit = (frame_df["close"] < stop_price_series) & rsi_exit

        # strong = frame_df["ema7_high"].shift(1) > frame_df["ema25"].shift(1)
        # eff_fpt = np.where(strong, fast_profit_threshold, 0.35 * fast_profit_threshold)
        # take_profit_price_series = frame_df["ema7_high"] + eff_fpt * frame_df.shift(1)["short_spread"]
        take_profit_price_series = frame_df["ema7_high"] + fast_profit_threshold * frame_df.shift(1)["short_spread"]
        tp_exit = (frame_df["high"] > take_profit_price_series)

        rsi_slope_sign = True
        if rsi_slope_entry_lever == 1:
            rsi_slope_sign = frame_df["rsi_slope"] > 0
        elif rsi_slope_entry_lever == -1:
            rsi_slope_sign = frame_df["rsi_slope"] < 0
        elif rsi_slope_entry_lever == 0:
            rsi_slope_sign = True

        div_safe = (frame_df["HH"] == True) & (frame_df["higher_rsi"] == True) if reject_rsi_div else True

        frame_df[entry_col] = (
            broken
            & volatile
            & rsi_slope_sign
            & is_gainer
            & candle_relative_structure
            & candle_own_structure
            & div_safe
        )

        frame_df[exit_col] =  stop_exit | tp_exit #fast_profit_exit |
        frame_df = _attach_aux_columns(
            frame_df,
            stop_price_series=stop_price_series,
            take_profit_price_series=take_profit_price_series,
            vol_window=vol_window,
            fee_bps=fee_bps,
            equity_proxy=equity_proxy,
            max_positions=max_positions,
            trailing_stop_pct=trailing_stop_pct,
            max_hold_bars=max_hold_bars,
            vol_take_profit_multiple=vol_take_profit_multiple,
            min_signal_hold_bars=min_signal_hold_bars,
            aux_prefix=aux_prefix,
            rsi_slope = frame_df["rsi_slope"],
        )

        dict_of_pairs[pair]["dict_of_frames"][execution_frame] = frame_df

    return dict_of_pairs

def apply_test_strategy(
    dict_of_pairs: dict,
    execution_frame: str,
    entry_col: str,
    exit_col: str,
    retest_column: str,
    fast_profit_threshold: float = 1.5,
    vol_window: int = 7,
    fee_bps: float = 10.0,
    equity_proxy: float = 10_000.0,
    max_positions: int = 5,
    trailing_stop_pct: float = 0.06,
    max_hold_bars: int = 36,
    vol_take_profit_multiple: float = 5.0,
    min_signal_hold_bars: int = 6,
    aux_prefix: str = "",
    lower_volatility_threshold : float = 0.025,
    upper_volatility_threshold : float = 0.08,
    gain_ratio : float = 1.0,
    use_rsi_slope_exit : bool = False,
    require_confirmation: bool = False,
) -> dict:
    """
    Sample strategy: retest of a column (i.e., price retests a moving average or a fixed previous level).

    Attaches entry/exit trigger columns and the shared aux columns
    (stop_price, quote_vol_per_unit_change, rank_score) required by
    the backtest engine for sizing and candidate ranking.

    Parameters
    ----------
    dict_of_pairs : dict
        Dataset in project format: {pair: {"dict_of_frames": {...}}}
    execution_frame : str
        Target execution frame key (e.g. '1hourly' or '4hourly').
    entry_col : str
        Output column name for entry trigger.
    exit_col : str
        Output column name for exit trigger.
    retest_column : str
        Column name to be retested (e.g., a moving average or a fixed level).

    Returns
    -------
    dict
        Modified dict_of_pairs with strategy + aux columns attached.
    """

    for pair in dict_of_pairs:
        frame_df = dict_of_pairs[pair]["dict_of_frames"].get(execution_frame)

        if frame_df is None or frame_df.empty:
            continue

        volatile = (frame_df["volatility"] > lower_volatility_threshold) & (frame_df["volatility"] < upper_volatility_threshold)

        is_gainer = frame_df["rolling_gain"] > gain_ratio



        rsi_exit = (use_rsi_slope_exit and (frame_df["rsi_slope"] < 0)) if use_rsi_slope_exit else True

        # Engine expects stop_price to be a price level, not a boolean condition.
        stop_price_series = frame_df[retest_column] - 1.25*frame_df["short_spread"]
        stop_exit = (frame_df["close"] < stop_price_series) & rsi_exit

        # strong = frame_df["ema7_high"].shift(1) > frame_df["ema25"].shift(1)
        # eff_fpt = np.where(strong, fast_profit_threshold, 0.35 * fast_profit_threshold)
        # take_profit_price_series = frame_df["ema7_high"] + eff_fpt * frame_df.shift(1)["short_spread"]
        take_profit_price_series = frame_df["ema7_high"] + fast_profit_threshold * frame_df.shift(1)["short_spread"]
        tp_exit = (frame_df["high"] > take_profit_price_series)

        green_candle = frame_df["close"] > frame_df["open"] if require_confirmation else True
        frame_df[entry_col] = (
            (frame_df[f"{retest_column}_tested"].shift(1) > 0) if require_confirmation else (frame_df[f"{retest_column}_tested"] > 0)
            # & (frame_df["allowed_to_trade"])
            & (frame_df["rsi_slope"] > 0)
            & (frame_df[f"{retest_column}_last_broken"] > 5)
            & is_gainer
            & volatile
            & green_candle
        )
        frame_df[exit_col] = stop_exit | tp_exit #fast_profit_exit | 
        frame_df = _attach_aux_columns(
            frame_df,
            stop_price_series=stop_price_series,
            take_profit_price_series=take_profit_price_series,
            vol_window=vol_window,
            fee_bps=fee_bps,
            equity_proxy=equity_proxy,
            max_positions=max_positions,
            trailing_stop_pct=trailing_stop_pct,
            max_hold_bars=max_hold_bars,
            vol_take_profit_multiple=vol_take_profit_multiple,
            min_signal_hold_bars=min_signal_hold_bars,
            aux_prefix=aux_prefix,
        )

        dict_of_pairs[pair]["dict_of_frames"][execution_frame] = frame_df

    return dict_of_pairs

def apply_strategy_mr_v1(
    dict_of_pairs: dict,
    execution_frame: str,
    entry_col: str,
    exit_col: str,
    vol_window: int = 7,
    fee_bps: float = 10.0,
    equity_proxy: float = 10_000.0,
    max_positions: int = 5,
    entry_zscore_threshold: float = -0.85,
    stop_buffer: float = 0.985,
    exit_zscore_threshold: float = 1.8,
    use_fast_mean_exit: bool = False,
    use_zscore_exit: bool = True,
    trailing_stop_pct: float = 0.06,
    max_hold_bars: int = 36,
    vol_take_profit_multiple: float = 5.0,
    min_signal_hold_bars: int = 6,
    aux_prefix: str = "",
    gain_ratio: float = 1.2,
    volatility_threshold: float = 0.025,
) -> dict:
    """
    Sample strategy: trend-filtered pullback mean reversion.

    Attaches entry/exit trigger columns and the shared aux columns
    (stop_price, quote_vol_per_unit_change, rank_score) required by
    the backtest engine for sizing and candidate ranking.

    Parameters
    ----------
    dict_of_pairs : dict
        Dataset in project format: {pair: {"dict_of_frames": {...}}}
    execution_frame : str
        Target execution frame key (e.g. '1hourly' or '4hourly').
    entry_col : str
        Output column name for entry trigger.
    exit_col : str
        Output column name for exit trigger.
    vol_window : int
        Rolling window for volume/move estimation.
    fee_bps : float
        Fee per side in basis points (for rank_score estimate).
    equity_proxy : float
        Approximate equity for rank_score slippage estimate.
    max_positions : int
        Max positions used in rank_score count-cap proxy.
    trailing_stop_pct : float
        Trailing stop distance used by the engine after entry.
    max_hold_bars : int
        Time-based exit horizon in candles.
    vol_take_profit_multiple : float
        Multiplier of rolling mean move used for profit-spike exits.

    Returns
    -------
    dict
        Modified dict_of_pairs with strategy + aux columns attached.
    """
    for pair in dict_of_pairs:
        frame_df = dict_of_pairs[pair]["dict_of_frames"].get(execution_frame)

        if frame_df is None or frame_df.empty:
            continue

        close = pd.to_numeric(frame_df["close"], errors="coerce")

        # Signal feature columns
        frame_df["mr_ema_fast_20"] = close.ewm(span=20, adjust=False).mean()
        frame_df["mr_ema_slow_50"] = close.ewm(span=50, adjust=False).mean()
        frame_df["mr_roll_mean_20"] = close.rolling(20, min_periods=20).mean()
        frame_df["mr_roll_std_20"] = close.rolling(20, min_periods=20).std(ddof=0)

        denom = frame_df["mr_roll_std_20"].replace(0, pd.NA)
        frame_df["mr_zscore"] = ((close - frame_df["mr_roll_mean_20"]) / denom).fillna(0.0)

        # if "rsi_slope" not in frame_df.columns:
        #     frame_df["rsi_slope"] = frame_df["rsi"].rolling(6).apply(get_simple_slope_2, raw=True)
        
        # Entry: gate + above slow EMA + pulled back below entry_zscore_threshold
        frame_df[entry_col] = (
            (close > frame_df["mr_ema_slow_50"])
            & (frame_df["mr_zscore"] < entry_zscore_threshold)
            # & frame_df["allowed_to_trade"]
            # & (frame_df["regime"] != "weak_bull")
        )

        # Exit conditions (conditionally combined):
        # - Trend break: always included
        # - Fast mean exit: included if use_fast_mean_exit is True
        # - Z-score exit: included if use_zscore_exit is True
        trend_break = close < frame_df["mr_ema_slow_50"] * stop_buffer
        fast_exit = use_fast_mean_exit and (close >= frame_df["mr_ema_fast_20"])
        zscore_exit = use_zscore_exit and (frame_df["mr_zscore"] > exit_zscore_threshold)
        
        frame_df[exit_col] = trend_break | fast_exit | zscore_exit

        # Stop: just below slow EMA using stop_buffer
        stop_price_series = frame_df["mr_ema_slow_50"] * stop_buffer
        
        take_profit_price_series = frame_df["mr_ema_fast_20"] + vol_take_profit_multiple * frame_df["mr_roll_std_20"]

        frame_df = _attach_aux_columns(
            frame_df,
            stop_price_series=stop_price_series,
            take_profit_price_series=take_profit_price_series,
            vol_window=vol_window,
            fee_bps=fee_bps,
            equity_proxy=equity_proxy,
            max_positions=max_positions,
            trailing_stop_pct=trailing_stop_pct,
            max_hold_bars=max_hold_bars,
            vol_take_profit_multiple=vol_take_profit_multiple,
            min_signal_hold_bars=min_signal_hold_bars,
            aux_prefix=aux_prefix,
        )

        dict_of_pairs[pair]["dict_of_frames"][execution_frame] = frame_df

    return dict_of_pairs

def _bars_since_last_change(series: pd.Series) -> np.ndarray:
    """Positional bars since the sticky indicator last CHANGED value.

    HH/LL are ffill-carried (sticky): once a higher high is confirmed, HH stays
    True until the next swing peak re-labels it. "Freshness" therefore means bars
    since the flag last transitioned, not since it was last True (which would be 0
    forever). ``np.inf`` if the series never changed. Causal.
    """
    s = series.fillna(False).astype(bool)
    changed = (s != s.shift(1)).fillna(False)
    pos = pd.Series(np.arange(len(s)), index=s.index)
    last_change = pos.where(changed).ffill()
    out = (pos - last_change).to_numpy(dtype=float)
    out[np.isnan(out)] = np.inf
    return out


def _indicator_to_bool(series: pd.Series, nan_value: bool) -> pd.Series:
    """Coerce a bool/float indicator column to bool, mapping NaN to ``nan_value``.

    add_hh_ll_columns leaves HH/LL as float (1.0/0.0/NaN, carried by ffill); tests
    may feed plain bool columns. Keeps dtype bool so unary ``~`` is always valid.
    """
    s = pd.to_numeric(series, errors="coerce")
    if s.dtype == bool:
        return s
    return s.fillna(float(nan_value)).astype(bool)


def apply_momentum_hh_hl_strategy(
    dict_of_pairs: dict,
    execution_frame: str,
    entry_col: str,
    exit_col: str,
    structure_confirm_bars: int = 5,
    require_plunge_confirmation: bool = True,
    plunge_window: int = 5,
    plunge_points: float = 10.0,
    plunge_ceiling: float | None = None,
    max_hh_age_bars: int | None = None,
    max_hl_age_bars: int | None = None,
    require_fresh_breakout: bool = False,
    gain_ratio: float = 1.15,
    lower_volatility_threshold: float = 0.01,
    upper_volatility_threshold: float = 0.06,
    stop_lookback_bars: int = 10,
    fast_profit_threshold: float = 3.0,
    vol_window: int = 7,
    fee_bps: float = 10.0,
    equity_proxy: float = 10_000.0,
    max_positions: int = 5,
    trailing_stop_pct: float = 0.06,
    max_hold_bars: int = 36,
    vol_take_profit_multiple: float = 5.0,
    min_signal_hold_bars: int = 6,
    aux_prefix: str = "",
    reject_rsi_div: bool = False,
) -> dict:
    """
    Momentum continuation on confirmed higher-high / higher-low structure after a
    short-term RSI plunge, with price still holding above the previous low.

    This is a pullback-within-uptrend continuation entry (buy strength after a
    shallow dip), not a mean-reversion fade. All signals come from columns already
    populated on the execution frame (rsi is period-6, so already "short term").

    Entry conditions (all causal):
    - higher high    : the most recent confirmed swing high is a higher high (HH).
    - higher low     : the most recent confirmed swing low is NOT a lower low (HL).
    - RSI plunge     : rsi fell by >= plunge_points over the last plunge_window bars.
    - hold structure : close has not closed below the previous confirmed swing low.
    - STEP 3 filters : volatility band + rolling_gain > gain_ratio (knobs, on by default).

    Timing modes (``require_plunge_confirmation``):
    - True  (momentum resume, default): the plunge must be confirmed by the PREVIOUS
      bar; entry fires after the pullback has already completed.
    - False (dip-buy / fade)          : the plunge is still in progress at the current
      bar; entry fades the dip while the higher-low structure stays intact.

    Look-ahead note: ``HH``/``LL``/``prev_low`` come from add_hh_ll_columns, which uses
    scipy find_peaks(width=5, distance=6) — a peak/trough flag at its bar is only
    confirmable ~``structure_confirm_bars`` bars later. Every structure-based condition
    is therefore shifted by ``structure_confirm_bars`` (default 5) so the backtest does
    not fire entries earlier than live-feasible.

    Exits:
    - stop  : close below ``low.rolling(stop_lookback_bars).min()`` - 1.25*short_spread
              (a close below the most recent low invalidates the higher-low thesis).
    - take profit : high above ``ema7_high + fast_profit_threshold*short_spread.shift(1)``.

    Returns
    -------
    dict
        Modified dict_of_pairs with entry/exit + shared aux columns attached.
    """
    required_cols = {
        "close", "high", "low", "rsi", "short_spread", "volatility",
        "rolling_gain", "rolling_max", "rolling_min", "ema7_high",
        "HH", "LL", "prev_low",
    }

    for pair in dict_of_pairs:
        frame_df = dict_of_pairs[pair]["dict_of_frames"].get(execution_frame)

        if frame_df is None or frame_df.empty:
            continue

        missing = [col for col in required_cols if col not in frame_df.columns]
        if missing:
            raise KeyError(f"{pair} missing required columns for momentum HH/HL: {missing}")

        close = pd.to_numeric(frame_df["close"], errors="coerce")
        high = pd.to_numeric(frame_df["high"], errors="coerce")
        low = pd.to_numeric(frame_df["low"], errors="coerce")
        rsi = pd.to_numeric(frame_df["rsi"], errors="coerce")
        short_spread = pd.to_numeric(frame_df["short_spread"], errors="coerce")
        ema7_high = pd.to_numeric(frame_df["ema7_high"], errors="coerce")

        # ---- RSI plunge (rsi is period-6, already short-term) ----
        # plunge_prior: plunge confirmed over the window ENDING at the previous bar
        # (entry only after the pullback has already happened -> momentum resume).
        plunge_prior = (rsi.shift(1) <= rsi.shift(1 + plunge_window) - plunge_points).fillna(False)
        # plunge_now: plunge still in progress at the current bar (dip-buy / fade).
        plunge_now = (rsi <= rsi.shift(plunge_window) - plunge_points).fillna(False)

        plunge = plunge_prior if require_plunge_confirmation else plunge_now

        if plunge_ceiling is not None:
            ref_rsi = rsi.shift(1) if require_plunge_confirmation else rsi
            plunge = plunge & (ref_rsi <= plunge_ceiling).fillna(False)

        # ---- Higher-high / higher-low / hold-above-low structure ----
        # Shifted by structure_confirm_bars to keep find_peaks-based flags causal.
        # HL uses nan_value=True so the no-structure (NaN) region negates to False:
        # no confirmed higher low until a trough has actually been confirmed.
        hh_ok = _indicator_to_bool(frame_df["HH"], nan_value=False).astype("float")
        hh_ok = hh_ok.shift(structure_confirm_bars).fillna(0.0).astype(bool)

        hl_ok = (~_indicator_to_bool(frame_df["LL"], nan_value=True)).astype("float")
        hl_ok = hl_ok.shift(structure_confirm_bars).fillna(0.0).astype(bool)

        prev_low = pd.to_numeric(frame_df["prev_low"], errors="coerce")
        hold_above_low = (close > prev_low.shift(structure_confirm_bars)).fillna(False)

        # Optional freshness caps: the current structure must have been (re)confirmed
        # recently, otherwise it is an ancient, already-broken uptrend, not momentum.
        if max_hh_age_bars is not None:
            hh_ok = hh_ok & pd.Series(
                _bars_since_last_change(hh_ok) <= max_hh_age_bars, index=hh_ok.index
            )
        if max_hl_age_bars is not None:
            hl_ok = hl_ok & pd.Series(
                _bars_since_last_change(hl_ok) <= max_hl_age_bars, index=hl_ok.index
            )

        # Optional fresh-breakout confirmation: close prints a new high of the
        # trailing rolling_max window -> momentum is resuming right now.
        if require_fresh_breakout:
            breakout = (close > pd.to_numeric(frame_df["rolling_max"], errors="coerce").shift(1)).fillna(False)
        else:
            breakout = pd.Series(True, index=frame_df.index)

        # ---- Standard STEP 3 filters (knobs) ----
        volatility = pd.to_numeric(frame_df["volatility"], errors="coerce")
        volatile = (volatility > lower_volatility_threshold) & (volatility < upper_volatility_threshold)
        is_gainer = pd.to_numeric(frame_df["rolling_gain"], errors="coerce") > gain_ratio

        div_safe = (frame_df["HH"] == True) & (frame_df["higher_rsi"] == True) if reject_rsi_div else True

        frame_df[entry_col] = (
            hh_ok
            & hl_ok
            & plunge
            & hold_above_low
            & breakout
            & volatile
            & is_gainer
            & div_safe
        )

        # ---- Stop: a close below the most recent low invalidates the higher-low thesis ----
        recent_low = low.rolling(stop_lookback_bars, min_periods=max(3, stop_lookback_bars // 2)).min()
        stop_price_series = recent_low - 1.25 * short_spread

        # stop_price_series = (frame_df[["open", "close"]].min(axis=1) - 1.25*frame_df["short_spread"])
        take_profit_price_series = ema7_high + fast_profit_threshold * short_spread.shift(1)

        stop_exit = close < stop_price_series
        tp_exit = high > take_profit_price_series
        frame_df[exit_col] = stop_exit | tp_exit

        frame_df = _attach_aux_columns(
            frame_df,
            stop_price_series=stop_price_series,
            take_profit_price_series=take_profit_price_series,
            vol_window=vol_window,
            fee_bps=fee_bps,
            equity_proxy=equity_proxy,
            max_positions=max_positions,
            trailing_stop_pct=trailing_stop_pct,
            max_hold_bars=max_hold_bars,
            vol_take_profit_multiple=vol_take_profit_multiple,
            min_signal_hold_bars=min_signal_hold_bars,
            aux_prefix=aux_prefix,
            rsi_slope=frame_df["rsi_slope"] if "rsi_slope" in frame_df.columns else None,
        )

        dict_of_pairs[pair]["dict_of_frames"][execution_frame] = frame_df

    return dict_of_pairs


def apply_three_green_candles_strategy(
    dict_of_pairs: dict,
    execution_frame: str,
    entry_col: str,
    exit_col: str,
    pattern_bars: int = 3,
    spread_multiple: float = 2.0,
    volume_col: str = "quote_asset_volume",
    rsi_slope_min: float = 0.0,
    use_rsi_slope: bool = True,
    require_rising_gaps: bool = True,
    stop_lookback_bars: int = 3,
    stop_spread_multiple: float = 1.25,
    fast_profit_threshold: float = 3.0,
    lower_volatility_threshold: float | None = None,
    upper_volatility_threshold: float | None = None,
    gain_ratio: float | None = None,
    vol_window: int = 7,
    fee_bps: float = 10.0,
    equity_proxy: float = 10_000.0,
    max_positions: int = 5,
    trailing_stop_pct: float = 0.06,
    max_hold_bars: int = 36,
    vol_take_profit_multiple: float = 5.0,
    min_signal_hold_bars: int = 6,
    aux_prefix: str = "",
) -> dict:
    """
    Three-green-candles thrust ("three white soldiers" shape, volume-confirmed,
    short-spread-bounded).

    The edge is a low-volatility continuation buy: three consecutive green
    candles that are still RISING progressively (both closes and volume), whose
    whole advance is small in short-spread units (so the move is a controlled
    push, not a blow-off already extended), and whose short-term RSI slope is
    positive (momentum still up at the trigger bar).

    Entry conditions, all evaluated at the close of the current bar (the third
    candle) and computed only from bars <= t, so nothing is look-ahead:

    1. three consecutive green candles : close > open for t, t-1, t-2.
    2. rising closes                   : close[t] > close[t-1] > close[t-2].
    3. growing difference              : (close[t] - close[t-1]) > (close[t-1] - close[t-2]).
    4. increasing quote volume         : qav[t] > qav[t-1] > qav[t-2].
    5. whole move bound                : close[t] - close[t-2] < spread_multiple * short_spread[t].
    6. positive RSI slope              : rsi_slope[t] > rsi_slope_min (default > 0).

    ``short_spread`` is the frame's rolling MA of (high - low); reading it at bar t
    is causal because bar t has already closed when the signal is evaluated. The
    condition keeps entries away from candles that have already travelled several
    spreads, which is what makes the tight stop below survivable.

    Optional STEP 3 gates (``lower_volatility_threshold`` /
    ``upper_volatility_threshold`` / ``gain_ratio``) are OFF by default (None) so
    the pattern itself is the edge; set them to use the same volatility band /
    rolling-gain filter the test_*/break_* strategies use.

    Exits (same convention as the other strategies, engine reads them per bar):
    - stop        : ``min(open, close) - stop_spread_multiple * short_spread`` on the
                    entry bar (1.25 spreads by default) - the break strategy's stop
                    geometry; the engine snapshots the level at entry. The
                    pattern-low variant (``low.rolling(stop_lookback_bars).min()``,
                    below candle 1's low) is kept commented out in the body.
    - take profit : high above ``ema7_high + fast_profit_threshold * short_spread.shift(1)``.

    NOTE on the take-profit interaction: because condition 5 caps the pattern's
    advance at ``spread_multiple`` (2 by default) spreads while the TP line is
    ``ema7_high + fpt * short_spread``, some signals fire with the TP already
    at/below the signal close. Both the backtest engine
    (``EngineConfig.skip_entries_with_tp_below_close``, currently True) and the
    live runtime (``EntryDecision.tp_below_entry``) treat that case explicitly -
    see those flags if this cohort needs to be taken or postponed instead.

    Parameters
    ----------
    dict_of_pairs : dict
        Dataset in project format: {pair: {"dict_of_frames": {...}}}.
    execution_frame : str
        Target execution frame key (e.g. '4hourly').
    entry_col / exit_col : str
        Output column names for the entry/exit triggers.
    pattern_bars : int
        Candles in the thrust (default 3). Conditions 2/4 are checked for every
        consecutive pair; condition 3 compares the LAST two gaps only.
    spread_multiple : float
        Condition 5 bound, in short-spread units.
    volume_col : str
        Volume column for condition 4 (falls back to 'volume' if absent).
    rsi_slope_min / use_rsi_slope : float / bool
        Condition 6 threshold and switch.
    require_rising_gaps : bool
        Condition 3 switch.
    stop_lookback_bars / stop_spread_multiple / fast_profit_threshold :
        Exit geometry (see above).
    lower_volatility_threshold / upper_volatility_threshold / gain_ratio :
        Optional extra gates, None = disabled.
    aux_prefix : str
        Prefix for the shared aux columns (normally the entry column name).

    Returns
    -------
    dict
        Modified dict_of_pairs with entry/exit + shared aux columns attached.
    """
    bars = int(max(pattern_bars, 2))

    required_cols = {"open", "high", "low", "close", "short_spread", "ema7_high"}
    if use_rsi_slope:
        required_cols.add("rsi_slope")

    for pair in dict_of_pairs:
        frame_df = dict_of_pairs[pair]["dict_of_frames"].get(execution_frame)

        if frame_df is None or frame_df.empty:
            continue

        missing = [col for col in sorted(required_cols) if col not in frame_df.columns]
        if missing:
            raise KeyError(f"{pair} missing required columns for three green candles: {missing}")

        vol_col = volume_col
        if vol_col not in frame_df.columns:
            if "volume" not in frame_df.columns:
                raise KeyError(
                    f"{pair} missing volume column for three green candles: "
                    f"tried '{volume_col}' and 'volume'"
                )
            vol_col = "volume"

        open_price = pd.to_numeric(frame_df["open"], errors="coerce")
        close = pd.to_numeric(frame_df["close"], errors="coerce")
        high = pd.to_numeric(frame_df["high"], errors="coerce")
        low = pd.to_numeric(frame_df["low"], errors="coerce")
        volume = pd.to_numeric(frame_df[vol_col], errors="coerce")
        short_spread = pd.to_numeric(frame_df["short_spread"], errors="coerce")
        ema7_high = pd.to_numeric(frame_df["ema7_high"], errors="coerce")

        # ---- 1/2/4: `bars` consecutive greens, strictly rising closes and volume.
        # Every shift is <= the current bar, so all three checks are known at the
        # close of bar t. `.fillna(False)` keeps the warm-up region excluded.
        green = pd.Series(True, index=frame_df.index)
        rising_close = pd.Series(True, index=frame_df.index)
        rising_volume = pd.Series(True, index=frame_df.index)
        for lag in range(bars):
            green &= (close.shift(lag) > open_price.shift(lag)).fillna(False)
            if lag:
                rising_close &= (close.shift(lag - 1) > close.shift(lag)).fillna(False)
                rising_volume &= (volume.shift(lag - 1) > volume.shift(lag)).fillna(False)

        # ---- 3: the last close-to-close gap must EXPAND ----  -> (c3-c2) > (c2-c1)
        gap_last = close - close.shift(1)
        gap_prev = close.shift(1) - close.shift(2)
        rising_gaps = (gap_last > gap_prev).fillna(False) if require_rising_gaps else True

        # ---- 5: the whole thrust stays under `spread_multiple` short spreads ----
        thrust = close - close.shift(bars - 1)
        bounded_thrust = (thrust < spread_multiple * short_spread).fillna(False)

        # ---- 6: positive short-term RSI slope (the rsi_slope column) ----
        if use_rsi_slope:
            rsi_slope = pd.to_numeric(frame_df["rsi_slope"], errors="coerce")
            slope_ok = (rsi_slope > rsi_slope_min).fillna(False)
        else:
            slope_ok = True

        # ---- optional STEP 3 style gates (disabled unless thresholds are given) ----
        if lower_volatility_threshold is not None and upper_volatility_threshold is not None:
            volatility = pd.to_numeric(frame_df["volatility"], errors="coerce")
            volatile = (volatility > lower_volatility_threshold) & (volatility < upper_volatility_threshold)
        else:
            volatile = True

        if gain_ratio is not None:
            is_gainer = pd.to_numeric(frame_df["rolling_gain"], errors="coerce") > gain_ratio
        else:
            is_gainer = True

        frame_df[entry_col] = (
            green
            & rising_close
            & rising_volume
            & rising_gaps
            & bounded_thrust
            & slope_ok
            & volatile
            & is_gainer
        ).to_numpy()

        # ---- stop: close back under the pattern's lowest low invalidates the thrust ----
        pattern_low = low.rolling(
            stop_lookback_bars, min_periods=max(2, stop_lookback_bars // 2)
        ).min()
        # stop_price_series = pattern_low - stop_spread_multiple * short_spread

        stop_price_series = (frame_df[["open", "close"]].min(axis=1) - stop_spread_multiple *frame_df["short_spread"])

        take_profit_price_series = ema7_high + fast_profit_threshold * short_spread.shift(1)

        stop_exit = close < stop_price_series
        tp_exit = high > take_profit_price_series
        frame_df[exit_col] = (stop_exit | tp_exit).to_numpy()

        frame_df = _attach_aux_columns(
            frame_df,
            stop_price_series=stop_price_series,
            take_profit_price_series=take_profit_price_series,
            vol_window=vol_window,
            fee_bps=fee_bps,
            equity_proxy=equity_proxy,
            max_positions=max_positions,
            trailing_stop_pct=trailing_stop_pct,
            max_hold_bars=max_hold_bars,
            vol_take_profit_multiple=vol_take_profit_multiple,
            min_signal_hold_bars=min_signal_hold_bars,
            aux_prefix=aux_prefix,
            rsi_slope=frame_df["rsi_slope"] if "rsi_slope" in frame_df.columns else None,
        )

        dict_of_pairs[pair]["dict_of_frames"][execution_frame] = frame_df

    return dict_of_pairs


def _bars_since_last_true(mask: pd.Series) -> np.ndarray:
    """Positional bars since ``mask`` was last True.

    0 on a bar where the mask is True, ``np.inf`` if it has never been True.
    Causal: only bars <= t contribute, so it is safe to use as an entry filter.
    """
    s = mask.fillna(False).astype(bool).to_numpy()
    pos = np.arange(len(s), dtype=float)
    last = np.where(s, pos, -np.inf)
    return pos - np.maximum.accumulate(last)


def apply_conviction_pullback_strategy(
    dict_of_pairs: dict,
    execution_frame: str,
    entry_col: str,
    exit_col: str,
    conviction_entry_cols: tuple[str, ...] | list[str] = (
        "entry_break_max",
        "entry_break_boll_upper",
        "entry_momentum_hh_hl",
    ),
    test_entry_cols: list[str] | None = None,
    entry_lookback_bars: int = 6,
    pullback_spread_multiple: float = 2.0,
    spread_reference: str = "current",
    stop_spread_multiple: float = 1.25,
    fast_profit_threshold: float = 3.0,
    lower_volatility_threshold: float | None = None,
    upper_volatility_threshold: float | None = None,
    gain_ratio: float | None = None,
    vol_window: int = 7,
    fee_bps: float = 10.0,
    equity_proxy: float = 10_000.0,
    max_positions: int = 5,
    trailing_stop_pct: float = 0.06,
    max_hold_bars: int = 36,
    vol_take_profit_multiple: float = 5.0,
    min_signal_hold_bars: int = 6,
    aux_prefix: str = "",
) -> dict:
    """
    Pullback test entry AFTER a conviction signal: buy the dip that follows a
    break / momentum trigger, while that trigger structure is still fresh.

    Composition strategy: it does NOT recompute the parent signals, it consumes
    their entry columns. The parents must already be applied on the same frame
    (that is what makes this cheap), by default:

        entry_break_max            <- apply_break_strategy(break_column="max", ...)
        entry_break_boll_upper     <- apply_break_strategy(break_column="boll_upper", ...)
        entry_momentum_hh_hl       <- apply_momentum_hh_hl_strategy(...)

    A missing parent column raises immediately (silently treating it as absent
    would quietly change the strategy).

    Entry conditions, all evaluated at the close of the current bar and computed
    only from bars <= t (no look-ahead):

    1. conviction : at least one of ``conviction_entry_cols`` fired within the last
                    ``entry_lookback_bars`` candles (default 6, current bar included;
                    firing on the current bar cannot pass condition 3 anyway, since
                    there the signal close IS the current close).
    2. test entry : at least one ``test_entry_cols`` column is True on the current
                    bar. Default None = every plain ``entry_test_*`` column present
                    on the frame (aux columns such as ``entry_test_x__stop_price``
                    are excluded).
    3. pullback   : close < (close of the most recent conviction bar) -
                    ``pullback_spread_multiple`` * short_spread. The spread is read
                    at the CURRENT bar (``spread_reference="current"``); pass
                    ``"signal"`` to measure with the conviction bar's own spread
                    instead (that one is carried forward from the signal bar).

    Optional STEP 3 style gates (``lower_volatility_threshold`` /
    ``upper_volatility_threshold`` / ``gain_ratio``) are OFF by default (None).

    Exits (identical convention to ``apply_break_strategy``):
    - stop        : ``min(open, close) - stop_spread_multiple * short_spread``
                    (default 1.25 = the break strategy's stop geometry). The engine
                    snapshots this level at entry; the bar's own open/close are
                    enough because the value is only used on the entry bar.
    - take profit : high above ``ema7_high + fast_profit_threshold * short_spread.shift(1)``.
    - exit column : ``(close < stop) | (high > tp)``.

    Parameters
    ----------
    dict_of_pairs : dict
        Dataset in project format: {pair: {"dict_of_frames": {...}}}.
    execution_frame : str
        Target execution frame key (e.g. '4hourly').
    entry_col / exit_col : str
        Output column names for the entry/exit triggers.
    conviction_entry_cols : sequence of str
        Parent entry columns, OR-ed together to form the conviction trigger.
    test_entry_cols : list of str | None
        Test-entry columns, OR-ed together. None = auto-detect ``entry_test_*``.
    entry_lookback_bars : int
        How many recent candles (including the current one) count as "fresh".
    pullback_spread_multiple : float
        Required distance below the conviction close, in short-spread units.
    spread_reference : {"current", "signal"}
        Which bar's ``short_spread`` measures the pullback (see condition 3).
    stop_spread_multiple / fast_profit_threshold : float
        Exit geometry (see above).
    lower_volatility_threshold / upper_volatility_threshold / gain_ratio :
        Optional extra gates, None = disabled.
    aux_prefix : str
        Prefix for the shared aux columns (normally the entry column name).

    Returns
    -------
    dict
        Modified dict_of_pairs with entry/exit + shared aux columns attached.
    """
    if spread_reference not in {"current", "signal"}:
        raise ValueError("spread_reference must be 'current' or 'signal'")

    conviction_cols = list(conviction_entry_cols or [])
    if not conviction_cols:
        raise ValueError("conviction_entry_cols must name at least one parent entry column")
    lookback = max(int(entry_lookback_bars), 1)

    for pair in dict_of_pairs:
        frame_df = dict_of_pairs[pair]["dict_of_frames"].get(execution_frame)

        if frame_df is None or frame_df.empty:
            continue

        missing = [col for col in conviction_cols if col not in frame_df.columns]
        if missing:
            raise KeyError(
                f"{pair} missing conviction entry column(s) for the conviction-pullback "
                f"strategy: {missing}. Apply the parent strategies on this frame first "
                "(apply_break_strategy(break_column='max', ...), "
                "apply_break_strategy(break_column='boll_upper', ...), "
                "apply_momentum_hh_hl_strategy(...)), or pass conviction_entry_cols with "
                "the columns you did apply."
            )

        # Plain trigger columns only: the aux columns of a strategy share its
        # prefix, e.g. 'entry_test_ma50__stop_price', and must not count as signals.
        test_cols = (
            # [col for col in frame_df.columns if col.startswith("entry_test_") and "__" not in col]
            [col for col in frame_df.columns if col.endswith("_tested") and "__" not in col]
            if test_entry_cols is None
            else list(test_entry_cols)
        )
        if not test_cols:
            raise KeyError(
                f"{pair} has no test-entry column for the conviction-pullback strategy: "
                "none matched '*_tested' and test_entry_cols was not given"
            )
        missing_tests = [col for col in test_cols if col not in frame_df.columns]
        if missing_tests:
            raise KeyError(f"{pair} missing test entry column(s): {missing_tests}")

        close = pd.to_numeric(frame_df["close"], errors="coerce")
        high = pd.to_numeric(frame_df["high"], errors="coerce")
        short_spread = pd.to_numeric(frame_df["short_spread"], errors="coerce")
        ema7_high = pd.to_numeric(frame_df["ema7_high"], errors="coerce")

        # ---- 1: conviction trigger, OR-ed over the parent entry columns ----
        conviction = pd.Series(False, index=frame_df.index)
        for col in conviction_cols:
            conviction |= _indicator_to_bool(frame_df[col], nan_value=False)

        recent_conviction = pd.Series(
            _bars_since_last_true(conviction) <= (lookback - 1),
            index=frame_df.index,
        )

        # Close of the most recent conviction bar, carried forward to this bar.
        conviction_close = close.where(conviction).ffill()

        # ---- 3: the price is now below that close by N short spreads ----
        if spread_reference == "signal":
            spread_ref = short_spread.where(conviction).ffill()
        else:
            spread_ref = short_spread
        pullback = (close < conviction_close - pullback_spread_multiple * spread_ref).fillna(False)

        # ---- 2: a test entry is firing right now ----
        test_entry = pd.Series(False, index=frame_df.index)
        for col in test_cols:
            test_entry |= _indicator_to_bool(frame_df[col], nan_value=False)

        # ---- optional STEP 3 style gates ----
        if lower_volatility_threshold is not None and upper_volatility_threshold is not None:
            volatility = pd.to_numeric(frame_df["volatility"], errors="coerce")
            volatile = (volatility > lower_volatility_threshold) & (volatility < upper_volatility_threshold)
        else:
            volatile = True

        if gain_ratio is not None:
            is_gainer = pd.to_numeric(frame_df["rolling_gain"], errors="coerce") > gain_ratio
        else:
            is_gainer = True

        frame_df[entry_col] = (
            recent_conviction & pullback & test_entry & volatile & is_gainer
        ).to_numpy()

        # ---- stop: the break strategy's geometry (min of open/close - 1.25 spreads) ----
        stop_price_series = frame_df[["open", "close"]].min(axis=1) - stop_spread_multiple * short_spread
        take_profit_price_series = ema7_high + fast_profit_threshold * short_spread.shift(1)

        stop_exit = close < stop_price_series
        tp_exit = high > take_profit_price_series
        frame_df[exit_col] = (stop_exit | tp_exit).to_numpy()

        frame_df = _attach_aux_columns(
            frame_df,
            stop_price_series=stop_price_series,
            take_profit_price_series=take_profit_price_series,
            vol_window=vol_window,
            fee_bps=fee_bps,
            equity_proxy=equity_proxy,
            max_positions=max_positions,
            trailing_stop_pct=trailing_stop_pct,
            max_hold_bars=max_hold_bars,
            vol_take_profit_multiple=vol_take_profit_multiple,
            min_signal_hold_bars=min_signal_hold_bars,
            aux_prefix=aux_prefix,
            rsi_slope=frame_df["rsi_slope"] if "rsi_slope" in frame_df.columns else None,
        )

        dict_of_pairs[pair]["dict_of_frames"][execution_frame] = frame_df

    return dict_of_pairs
