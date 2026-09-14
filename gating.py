import numpy as np
import pandas as pd
import talib


def add_btc_risk_on(btc_daily: pd.DataFrame,
                    slow_threshold: float = -0.2,
                    use_percentile: bool = True,
                    slow_pct_threshold: float = 0.25,
                    adaptive: bool = False,
                    turn_on_pct: float = 0.30,
                    turn_off_pct: float = 0.15,
                    turn_on_bars: int = 2,
                    turn_off_bars: int = 2) -> pd.DataFrame:
    """
    Add BTC global risk switch on BTC daily frame.

    use_percentile=True (default): risk-on when the rolling percentile of
    sqn_90 (``sqn_90_pct``, from add_sqn_percentiles_to_dataset) is above
    ``slow_pct_threshold``. Falls back to the fixed threshold if the percentile
    column is missing.

    adaptive=True: hysteresis state machine on ``sqn_90_pct`` - turns OFF only
    after the percentile stays below ``turn_off_pct`` for ``turn_off_bars``
    consecutive days, and turns back ON after it stays above ``turn_on_pct``
    for ``turn_on_bars`` days. This makes the gate "soft": brief dips inside a
    strong regime keep risk-on (so expansion trades aren't blocked during minor
    BTC pullbacks), while sustained weakness in a slow market still flips it
    off (filters chop).
    """
    out = btc_daily.sort_values("close_time").copy()
    if adaptive and "sqn_90_pct" in out.columns:
        pct_s = pd.to_numeric(out["sqn_90_pct"], errors="coerce").fillna(0.5)
        on_signal = pct_s > float(turn_on_pct)
        off_signal = pct_s < float(turn_off_pct)
        tb = max(int(turn_on_bars), 1)
        ob = max(int(turn_off_bars), 1)
        on_b = on_signal.rolling(tb, min_periods=tb).sum() >= tb
        off_b = off_signal.rolling(ob, min_periods=ob).sum() >= ob
        state = False
        risk = []
        for i in range(len(out)):
            if (not state) and bool(on_b.iloc[i]):
                state = True
            elif state and bool(off_b.iloc[i]):
                state = False
            risk.append(state)
        out["btc_risk_on"] = pd.Series(risk, index=out.index, dtype="bool")
    elif use_percentile and "sqn_90_pct" in out.columns:
        out["btc_risk_on"] = out["sqn_90_pct"] > slow_pct_threshold
    else:
        out["btc_risk_on"] = out["sqn_90"] > slow_threshold
    
    return out


def add_btc_exposure_multiplier(
    btc_daily: pd.DataFrame,
    sqn_col: str = "sqn_90",
    anchors: tuple[tuple[float, float], ...] = (
        (-0.5, 0.0),
        (-0.2, 0.50),
        (0.3, 0.75),
        (1.2, 1.0),
    ),
    use_percentile: bool = True,
    pct_anchors: tuple[tuple[float, float], ...] = (
        (0.05, 0.0),
        (0.25, 0.50),
        (0.50, 0.75),
        (0.80, 1.0),
    ),
    use_recovery_floor: bool = True,
    recovery_ma_period: int = 50,
    recovery_floor: float = 0.35,
) -> pd.DataFrame:
    """
    Convert BTC SQN into a continuous capital-allocation multiplier.

    use_percentile=True (default): interpolate over the rolling percentile of
    ``sqn_col`` (e.g. ``sqn_90_pct`` from add_sqn_percentiles_to_dataset) using
    ``pct_anchors``. Because the percentile is relative to BTC's own trailing
    252-day history, the multiplier is only ~0 in the worst ~5% of BTC's own
    history — it no longer stays flat for entire regimes (the fixed-anchor
    failure mode that hurt 2020-21 and 2022-24).
    use_percentile=False: legacy fixed ``anchors`` on the raw sqn value.

    The multiplier is used ONLY to scale the total capital deployed at entry
    (sizing). It never blocks a pair from being selected (BTC health is
    decoupled from pair selection).

    Adds column: ``btc_exposure_multiplier`` (float64, clamped to [0, 1]).
    """
    out = btc_daily.sort_values("close_time").copy()
    pct_col = f"{sqn_col}_pct"

    if use_percentile and pct_col in out.columns:
        series = pd.to_numeric(out[pct_col], errors="coerce").fillna(0.5).to_numpy(dtype=float)
        pts = sorted(pct_anchors, key=lambda p: float(p[0]))
    else:
        if sqn_col not in out.columns:
            raise KeyError(f"btc_daily must contain '{sqn_col}' (run classify_sqn_regime first)")
        series = pd.to_numeric(out[sqn_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        pts = sorted(anchors, key=lambda p: float(p[0]))

    xs = np.array([float(p[0]) for p in pts], dtype=float)
    ys = np.array([float(p[1]) for p in pts], dtype=float)

    multiplier = np.interp(series, xs, ys, left=float(ys[0]), right=float(ys[-1]))
    multiplier = np.clip(multiplier, 0.0, 1.0)

    # Recovery floor (Rec 5): even while the slow SQN percentile is still
    # catching up after a crash, once BTC price is back above its short/medium
    # MA the multiplier is floored at `recovery_floor` so the bot starts
    # participating in the recovery instead of staying flat. Inactive during
    # the crash itself (price below the MA) so drawdown protection is kept.
    if use_recovery_floor and "close" in out.columns:
        rp = max(int(recovery_ma_period), 2)
        close = pd.to_numeric(out["close"], errors="coerce")
        ma = close.rolling(rp, min_periods=rp).mean()
        recovered = (close > ma).fillna(False).to_numpy(dtype=float)
        multiplier = np.maximum(multiplier, float(recovery_floor) * recovered)

    out["btc_exposure_multiplier"] = multiplier
    return out


def add_btc_4h_drift(btc_4h: pd.DataFrame,
                     drift_bars: int = 12,
                     time_col: str = "close_time") -> pd.DataFrame:
    """
    Add BTC 4h drift: rolling SUM of per-bar close returns over ``drift_bars``
    (default 12 = 2 days on 4h bars). Column: ``btc_4h_drift`` (float64).

    The rolling sum of returns captures the PATH of the last N bars (not just
    the endpoint change), per the "rolling sum of bar returns" design.
    """
    out = btc_4h.sort_values(time_col).copy()
    close = pd.to_numeric(out["close"], errors="coerce")
    ret = close.pct_change(fill_method=None)
    out["btc_4h_drift"] = ret.rolling(max(int(drift_bars), 1)).sum()
    return out


def add_btc_drift_exposure_multiplier(
    btc_4h: pd.DataFrame,
    drift_bars: int = 12,
    turn_on_threshold: float = 0.01,
    turn_off_threshold: float = -0.01,
    turn_on_bars: int = 2,
    turn_off_bars: int = 2,
    floor: float = 0.4,
    time_col: str = "close_time",
) -> pd.DataFrame:
    """
    Convert BTC 4h drift into a lenient, hysteresis-softened exposure
    multiplier in [floor, 1.0].

    Adds columns:
    - ``btc_4h_drift`` : rolling sum of per-bar returns (if missing).
    - ``btc_4h_drift_ok`` : hysteresis boolean (mirror of add_btc_risk_on's
      adaptive state machine). Turns OFF only after ``turn_off_bars``
      consecutive bars with drift < ``turn_off_threshold``; turns ON after
      ``turn_on_bars`` consecutive bars with drift > ``turn_on_threshold``.
      Brief dips inside a strong regime keep risk-on ("does not gate too
      much"), while sustained weakness still scales exposure down.
    - ``btc_drift_exposure_multiplier`` : 1.0 when ``btc_4h_drift_ok`` else
      ``floor``. Never 0, so entries are sized down but never fully blocked.

    This is a SOFT exposure multiplier: it only scales entry sizing; it never
    blocks a pair from being selected.
    """
    out = btc_4h.sort_values(time_col).copy()
    if "btc_4h_drift" not in out.columns:
        out = add_btc_4h_drift(out, drift_bars=drift_bars, time_col=time_col)

    drift = pd.to_numeric(out["btc_4h_drift"], errors="coerce").fillna(0.0)
    on_signal = drift > float(turn_on_threshold)
    off_signal = drift < float(turn_off_threshold)
    tb = max(int(turn_on_bars), 1)
    ob = max(int(turn_off_bars), 1)
    on_b = on_signal.rolling(tb, min_periods=tb).sum() >= tb
    off_b = off_signal.rolling(ob, min_periods=ob).sum() >= ob

    state = False
    risk = []
    for i in range(len(out)):
        if (not state) and bool(on_b.iloc[i]):
            state = True
        elif state and bool(off_b.iloc[i]):
            state = False
        risk.append(state)

    out["btc_4h_drift_ok"] = pd.Series(risk, index=out.index, dtype="bool")
    out["btc_drift_exposure_multiplier"] = np.where(
        out["btc_4h_drift_ok"], 1.0, float(floor)
    ).astype("float64")
    return out


def add_pair_tradable(pair_daily: pd.DataFrame,
                      slow_floor: float = 0.3,
                      fast_floor: float = 0.0,
                      use_percentile: bool = True,
                      slow_pct_floor: float = 0.5,
                      fast_pct_floor: float = 0.4) -> pd.DataFrame:
    """
    Add pair tradability flag on pair daily frame.

    use_percentile=True (default): tradable when the rolling percentiles
    ``sqn_90_pct > slow_pct_floor`` AND ``sqn_30_pct > fast_pct_floor``
    (adaptive). Falls back to the fixed thresholds if the percentile columns
    are missing.
    """
    out = pair_daily.sort_values("close_time").copy()
    if use_percentile and {"sqn_30_pct", "sqn_90_pct"} <= set(out.columns):
        out["pair_tradable"] = (
            (out["sqn_90_pct"] > slow_pct_floor) &
            (out["sqn_30_pct"] > fast_pct_floor)
        )
    else:
        out["pair_tradable"] = (
            (out["sqn_90"] > slow_floor) &
            (out["sqn_30"] > fast_floor)
        )
    return out


def add_pair_tradable_v2(
    pair_daily: pd.DataFrame,
    slow_floor: float = 0.35,
    fast_floor: float = 0.15,
    slope_lookback: int = 5,
    min_adx: float = 18.0,
    adx_period: int = 14,
    vol_window: int = 30,
    vol_q_low: float = 0.25,
    vol_q_high: float = 0.85,
    turn_on_bars: int = 2,
    turn_off_bars: int = 2,
) -> pd.DataFrame:
    """
    Regime-aware pair tradability gate with hysteresis.

    Required columns:
    - sqn_90
    - sqn_30
    """
    out = pair_daily.sort_values("close_time").copy()

    if "sqn_90" not in out.columns or "sqn_30" not in out.columns:
        raise KeyError("pair_daily must contain 'sqn_90' and 'sqn_30'")

    sqn_90 = pd.to_numeric(out["sqn_90"], errors="coerce")
    sqn_30 = pd.to_numeric(out["sqn_30"], errors="coerce")
    sqn_slope = sqn_30.diff(max(int(slope_lookback), 1))

    if "adx" in out.columns:
        adx = pd.to_numeric(out["adx"], errors="coerce")
    elif all(c in out.columns for c in ["high", "low", "close"]):
        adx = pd.Series(
            talib.ADX(
                pd.to_numeric(out["high"], errors="coerce"),
                pd.to_numeric(out["low"], errors="coerce"),
                pd.to_numeric(out["close"], errors="coerce"),
                timeperiod=max(int(adx_period), 2),
            ),
            index=out.index,
            dtype="float64",
        )
    else:
        adx = pd.Series(25.0, index=out.index, dtype="float64")

    if "volatility" in out.columns:
        vol = pd.to_numeric(out["volatility"], errors="coerce")
    elif all(c in out.columns for c in ["high", "low", "close"]):
        high = pd.to_numeric(out["high"], errors="coerce")
        low = pd.to_numeric(out["low"], errors="coerce")
        close = pd.to_numeric(out["close"], errors="coerce")
        vol = ((high - low) / close).replace([float("inf"), float("-inf")], pd.NA)
    else:
        vol = pd.Series(0.0, index=out.index, dtype="float64")

    vol_window = max(int(vol_window), 5)
    min_periods = max(5, int(vol_window // 3))
    vol_low = vol.rolling(vol_window, min_periods=min_periods).quantile(float(vol_q_low))
    vol_high = vol.rolling(vol_window, min_periods=min_periods).quantile(float(vol_q_high))
    vol_ok = (vol >= vol_low) & (vol <= vol_high)

    trend_quality = (
        (sqn_90 > float(slow_floor))
        & (sqn_30 > float(fast_floor))
        & (sqn_slope > 0.0)
        & (adx >= float(min_adx))
    )

    raw_ok = (trend_quality & vol_ok).fillna(False)

    turn_on_bars = max(int(turn_on_bars), 1)
    turn_off_bars = max(int(turn_off_bars), 1)
    on_signal = raw_ok.rolling(turn_on_bars, min_periods=turn_on_bars).sum() >= turn_on_bars
    off_signal = (~raw_ok).rolling(turn_off_bars, min_periods=turn_off_bars).sum() >= turn_off_bars

    state = False
    tradable = []
    for i in range(len(out)):
        if (not state) and bool(on_signal.iloc[i]):
            state = True
        elif state and bool(off_signal.iloc[i]):
            state = False
        tradable.append(state)

    out["pair_trend_quality"] = trend_quality.fillna(False).astype(bool)
    out["pair_vol_ok"] = vol_ok.fillna(False).astype(bool)
    out["pair_tradable_raw"] = raw_ok.astype(bool)
    out["pair_tradable"] = pd.Series(tradable, index=out.index, dtype="bool")
    return out


def asof_merge_column(source_df: pd.DataFrame,
                      target_df: pd.DataFrame,
                      column: str,
                      source_time_col: str = "close_time",
                      target_time_col: str = "close_time",
                      fill_value=None,
                      dtype=None) -> pd.Series:

    merged = pd.merge_asof(
        target_df[[target_time_col]].sort_values(target_time_col),
        source_df[[source_time_col, column]].sort_values(source_time_col),
        left_on=target_time_col,
        right_on=source_time_col,
        direction="backward"
    )

    series = merged[column]

    if dtype == "bool":
        series = series.astype("boolean")   # pandas nullable bool
        if fill_value is not None:
            series = series.fillna(fill_value)
        return series.astype("bool")

    if dtype is not None:
        series = series.astype(dtype)
        if fill_value is not None:
            series = series.fillna(fill_value)
        return series

    if fill_value is not None:
        series = series.where(series.notna(), fill_value)

    return series


def attach_daily_gating_to_1h(
    btc_daily: pd.DataFrame,
    pair_daily: pd.DataFrame,
    pair_1h: pd.DataFrame,
) -> pd.DataFrame:
    """
    Attach only the daily gating layer to the 1H execution frame:
        - btc_risk_on
        - pair_tradable
        - allowed_daily
    """
    btc_daily = btc_daily.sort_values("close_time")
    pair_daily = pair_daily.sort_values("close_time")
    out = pair_1h.sort_values("close_time").copy()

    out["btc_risk_on"] = asof_merge_column(
        source_df=btc_daily,
        target_df=out,
        column="btc_risk_on",
        source_time_col="close_time",
        target_time_col="close_time",
        fill_value=False,
        dtype="bool"
    ).astype(bool)

    out["pair_tradable"] = asof_merge_column(
        source_df=pair_daily,
        target_df=out,
        column="pair_tradable",
        source_time_col="close_time",
        target_time_col="close_time",
        fill_value=False,
        dtype="bool"
    ).astype(bool)

    out["allowed_daily"] = out["btc_risk_on"] & out["pair_tradable"]

    if "btc_exposure_multiplier" in btc_daily.columns:
        out["btc_exposure_multiplier"] = asof_merge_column(
            source_df=btc_daily,
            target_df=out,
            column="btc_exposure_multiplier",
            source_time_col="close_time",
            target_time_col="close_time",
            fill_value=1.0,
            dtype="float64",
        )
    else:
        out["btc_exposure_multiplier"] = 1.0

    return out


def attach_daily_gating_to_execution_frame(
    btc_daily: pd.DataFrame,
    pair_daily: pd.DataFrame,
    pair_execution: pd.DataFrame,
    execution_time_col: str = "close_time",
    daily_time_col: str = "close_time",
) -> pd.DataFrame:
    """
    Attach daily gating to any execution frame (1H or 4H).
    Output columns:
        - btc_risk_on
        - pair_tradable
        - allowed_daily
    """
    btc_daily = btc_daily.sort_values(daily_time_col)
    pair_daily = pair_daily.sort_values(daily_time_col)
    out = pair_execution.sort_values(execution_time_col).copy()

    out["btc_risk_on"] = asof_merge_column(
        source_df=btc_daily,
        target_df=out,
        column="btc_risk_on",
        source_time_col=daily_time_col,
        target_time_col=execution_time_col,
        fill_value=False,
        dtype="bool"
    ).astype(bool)

    out["pair_tradable"] = asof_merge_column(
        source_df=pair_daily,
        target_df=out,
        column="pair_tradable",
        source_time_col=daily_time_col,
        target_time_col=execution_time_col,
        fill_value=False,
        dtype="bool"
    ).astype(bool)

    out["allowed_daily"] = out["btc_risk_on"] & out["pair_tradable"]

    if "btc_exposure_multiplier" in btc_daily.columns:
        out["btc_exposure_multiplier"] = asof_merge_column(
            source_df=btc_daily,
            target_df=out,
            column="btc_exposure_multiplier",
            source_time_col=daily_time_col,
            target_time_col=execution_time_col,
            fill_value=1.0,
            dtype="float64",
        )
    else:
        out["btc_exposure_multiplier"] = 1.0

    return out


def attach_btc_drift_to_execution_frame(
    btc_4h: pd.DataFrame,
    pair_execution: pd.DataFrame,
    execution_time_col: str = "close_time",
) -> pd.DataFrame:
    """
    As-of merge BTC 4h drift columns onto the pair's execution frame.

    Output columns (when present on ``btc_4h``):
        - btc_4h_drift
        - btc_4h_drift_ok
        - btc_drift_exposure_multiplier

    The multiplier is a BTC-only signal identical across pairs at the same bar
    close; merge_asof (backward) maps the latest completed BTC 4h bar onto each
    execution bar (works for 4h == 4h and 4h -> 1h). Missing bars default to
    neutral (multiplier 1.0, drift_ok False).
    """
    out = pair_execution.sort_values(execution_time_col).copy()
    for col in ("btc_4h_drift", "btc_4h_drift_ok", "btc_drift_exposure_multiplier"):
        if col not in btc_4h.columns:
            continue
        merged = pd.merge_asof(
            out[[execution_time_col]].sort_values(execution_time_col),
            btc_4h[[execution_time_col, col]].sort_values(execution_time_col),
            left_on=execution_time_col,
            right_on=execution_time_col,
            direction="backward",
        )
        series = merged[col]
        if col == "btc_4h_drift_ok":
            series = series.astype("boolean").fillna(False).astype(bool)
        else:
            fill = 1.0 if col == "btc_drift_exposure_multiplier" else 0.0
            series = series.astype("float64").fillna(fill)
        out[col] = series
    return out


def attach_daily_gating_to_dataset(dict_of_pairs: dict, btc_pair: str = "BTCUSDT") -> dict:
    btc_daily = dict_of_pairs[btc_pair]["dict_of_frames"]["1daily"]
    btc_daily = add_btc_risk_on(btc_daily)
    dict_of_pairs[btc_pair]["dict_of_frames"]["1daily"] = btc_daily

    for pair in dict_of_pairs:
        frames = dict_of_pairs[pair]["dict_of_frames"]

        if (
            "1daily" not in frames or
            "1hourly" not in frames or
            frames["1daily"].empty or
            frames["1hourly"].empty
        ):
            continue

        pair_daily = add_pair_tradable(frames["1daily"])
        pair_1h = attach_daily_gating_to_1h(
            btc_daily=btc_daily,
            pair_daily=pair_daily,
            pair_1h=frames["1hourly"]
        )

        dict_of_pairs[pair]["dict_of_frames"]["1daily"] = pair_daily
        dict_of_pairs[pair]["dict_of_frames"]["1hourly"] = pair_1h

    return dict_of_pairs




def add_4h_structure_ok(
    pair_4h: pd.DataFrame,
    ema_fast_period: int = 25,
    ema_mid_period: int = 50,
    ema_slow_period: int = 200,
    slope_lookback: int = 5,
) -> pd.DataFrame:
    """
    Add 4H structure columns and structure_ok flag.

    Structure is valid when:
    - close > ema_slow
    - ema_fast > ema_mid > ema_slow
    - ema_slow slope > 0

    Returns a new dataframe. Does not mutate input.
    """
    out = pair_4h.sort_values("close_time").copy()

    fast_col = f"ema_{ema_fast_period}"
    mid_col = f"ema_{ema_mid_period}"
    slow_col = f"ema_{ema_slow_period}"
    slope_col = f"{slow_col}_slope"

    out[fast_col] = talib.EMA(out["close"], timeperiod=ema_fast_period)
    out[mid_col] = talib.EMA(out["close"], timeperiod=ema_mid_period)
    out[slow_col] = talib.EMA(out["close"], timeperiod=ema_slow_period)

    out[slope_col] = out[slow_col].diff(slope_lookback)

    out["structure_ok"] = (
        (out["close"] > out[slow_col]) &
        (out[fast_col] > out[mid_col]) &
        (out[mid_col] > out[slow_col]) &
        (out[slope_col] > 0)
    ).astype(bool)

    return out


def compute_4h_structure_for_dataset(
    dict_of_pairs: dict,
    ema_fast_period: int = 25,
    ema_mid_period: int = 50,
    ema_slow_period: int = 200,
    slope_lookback: int = 5,
) -> dict:
    """
    Compute 4H structure columns and structure_ok
    for every eligible pair in the dataset.
    """
    for pair in dict_of_pairs:
        frames = dict_of_pairs[pair]["dict_of_frames"]

        if "4hourly" not in frames or frames["4hourly"].empty:
            print(f"Skipping {pair} - missing or empty 4hourly")
            continue

        frames["4hourly"] = add_4h_structure_ok(
            pair_4h=frames["4hourly"],
            ema_fast_period=ema_fast_period,
            ema_mid_period=ema_mid_period,
            ema_slow_period=ema_slow_period,
            slope_lookback=slope_lookback,
        )

    return dict_of_pairs


def attach_4h_structure_to_1h(
    pair_4h: pd.DataFrame,
    pair_1h: pd.DataFrame,
    structure_col: str = "structure_ok"
) -> pd.DataFrame:
    """
    Attach 4H structure flag onto 1H and build final gate.
    Requires pair_1h to already contain 'allowed_daily'.
    """
    if structure_col not in pair_4h.columns:
        raise KeyError(f"Missing column in pair_4h: {structure_col}")

    if "allowed_daily" not in pair_1h.columns:
        raise KeyError("pair_1h must already contain 'allowed_daily'")

    out = pair_1h.sort_values("close_time").copy()

    out[structure_col] = asof_merge_column(
        source_df=pair_4h.sort_values("close_time"),
        target_df=out,
        column=structure_col,
        source_time_col="close_time",
        target_time_col="close_time",
        fill_value=False,
        dtype="bool"
    )

    out["allowed_to_trade"] = out[structure_col] # out["allowed_daily"]# & out[structure_col] #

    return out


def attach_structure_to_execution_frame(
    pair_4h: pd.DataFrame,
    pair_execution: pd.DataFrame,
    execution_frame: str,
    structure_col: str = "structure_ok",
    time_col: str = "close_time",
) -> pd.DataFrame:
    """
    Attach 4H structure to selected execution frame.

    Rules:
    - execution_frame == "1hourly": merge 4H structure down to 1H.
    - execution_frame == "4hourly": use structure directly on 4H bars.

    Requires pair_execution to contain 'allowed_daily'.
    """
    if structure_col not in pair_4h.columns:
        raise KeyError(f"Missing column in pair_4h: {structure_col}")

    if "allowed_daily" not in pair_execution.columns:
        raise KeyError("pair_execution must already contain 'allowed_daily'")

    out = pair_execution.sort_values(time_col).copy()

    if execution_frame == "1hourly":
        out[structure_col] = asof_merge_column(
            source_df=pair_4h.sort_values(time_col),
            target_df=out,
            column=structure_col,
            source_time_col=time_col,
            target_time_col=time_col,
            fill_value=False,
            dtype="bool"
        ).astype(bool)
    elif execution_frame == "4hourly":
        local_4h = pair_4h.sort_values(time_col).copy()
        if structure_col not in out.columns:
            out = out.merge(
                local_4h[[time_col, structure_col]],
                on=time_col,
                how="left",
                suffixes=("", "_4h")
            )
        out[structure_col] = out[structure_col].fillna(False).astype(bool)
    else:
        raise ValueError("execution_frame must be one of: '1hourly', '4hourly'")

    out["allowed_to_trade"] = out["allowed_daily"] & out[structure_col]

    return out


def prepare_execution_frame_dataset(
    dict_of_pairs: dict,
    execution_frame: str = "1hourly",
    btc_pair: str = "BTCUSDT",
    structure_col: str = "structure_ok",
    ema_fast_period: int = 25,
    ema_mid_period: int = 50,
    ema_slow_period: int = 200,
    slope_lookback: int = 5,
    use_percentile: bool = True,
    adaptive_btc_risk: bool = False,
    use_recovery_floor: bool = True,
    recovery_ma_period: int = 50,
    recovery_floor: float = 0.35,
    use_btc_drift_multiplier: bool = False,
    btc_drift_bars: int = 12,
    btc_drift_floor: float = 0.4,
    btc_drift_turn_on_threshold: float = 0.01,
    btc_drift_turn_off_threshold: float = -0.01,
    btc_drift_turn_on_bars: int = 2,
    btc_drift_turn_off_bars: int = 2,
) -> dict:
    """
    Build a configurable tradable frame for backtesting.

    Daily gating remains on 1D in all modes.
    Structure always comes from 4H.

    Modes:
    - execution_frame='1hourly':
        daily gate (1D) + structure (4H asof) + trigger columns on 1H
    - execution_frame='4hourly':
        daily gate (1D asof) + structure (native 4H) + trigger columns on 4H

    use_percentile : bool
        True (default): btc_risk_on / pair_tradable / btc_exposure_multiplier
        use rolling 252-day SQN percentiles (adaptive). False: legacy fixed
        absolute SQN thresholds. Exposed so the notebook can A/B the two.

    adaptive_btc_risk : bool
        True: use the hysteresis state machine on the BTC sqn percentile so
        the risk gate is "soft" (relaxed during strong regimes). False (default):
        plain threshold gate. Exposed so the notebook can toggle it.

    use_btc_drift_multiplier : bool
        True: attach a BTC 4h-drift soft exposure multiplier (btc_drift_bars /
        btc_drift_floor / btc_drift_*_threshold / btc_drift_*_bars) onto every
        pair's execution frame as ``btc_drift_exposure_multiplier``. The engine
        scales entry sizing by it when EngineConfig.exposure_multiplier_col
        points at the column. Default False (no behavior change).
    """
    if execution_frame not in {"1hourly", "4hourly"}:
        raise ValueError("execution_frame must be one of: '1hourly', '4hourly'")

    if btc_pair not in dict_of_pairs:
        raise KeyError(f"BTC pair missing from dataset: {btc_pair}")

    btc_frames = dict_of_pairs[btc_pair]["dict_of_frames"]
    if "1daily" not in btc_frames or btc_frames["1daily"].empty:
        raise KeyError("BTC daily frame (1daily) is missing or empty")

    btc_daily = add_btc_risk_on(btc_frames["1daily"], use_percentile=use_percentile, adaptive=adaptive_btc_risk)
    btc_daily = add_btc_exposure_multiplier(
        btc_daily,
        use_percentile=use_percentile,
        use_recovery_floor=use_recovery_floor,
        recovery_ma_period=recovery_ma_period,
        recovery_floor=recovery_floor,
    )
    dict_of_pairs[btc_pair]["dict_of_frames"]["1daily"] = btc_daily

    # BTC 4h drift soft exposure multiplier (opt-in). BTC-only signal, computed
    # once on the BTC 4h frame and as-of merged onto each pair's execution frame.
    btc_4h = btc_frames.get("4hourly")
    if use_btc_drift_multiplier:
        if btc_4h is None or btc_4h.empty:
            raise KeyError("BTC 4h frame (4hourly) is required when use_btc_drift_multiplier=True")
        btc_4h = add_btc_drift_exposure_multiplier(
            btc_4h,
            drift_bars=btc_drift_bars,
            turn_on_threshold=btc_drift_turn_on_threshold,
            turn_off_threshold=btc_drift_turn_off_threshold,
            turn_on_bars=btc_drift_turn_on_bars,
            turn_off_bars=btc_drift_turn_off_bars,
            floor=btc_drift_floor,
        )
        btc_frames["4hourly"] = btc_4h

    for pair in dict_of_pairs:
        frames = dict_of_pairs[pair]["dict_of_frames"]

        if (
            "1daily" not in frames or
            execution_frame not in frames or
            "4hourly" not in frames or
            frames["1daily"].empty or
            frames[execution_frame].empty or
            frames["4hourly"].empty
        ):
            continue

        pair_daily = add_pair_tradable(frames["1daily"], use_percentile=use_percentile)

        execution_df = attach_daily_gating_to_execution_frame(
            btc_daily=btc_daily,
            pair_daily=pair_daily,
            pair_execution=frames[execution_frame],
            execution_time_col="close_time",
            daily_time_col="close_time",
        )

        if structure_col not in frames["4hourly"].columns:
            frames["4hourly"] = add_4h_structure_ok(
                pair_4h=frames["4hourly"],
                ema_fast_period=ema_fast_period,
                ema_mid_period=ema_mid_period,
                ema_slow_period=ema_slow_period,
                slope_lookback=slope_lookback,
            )

        execution_df = attach_structure_to_execution_frame(
            pair_4h=frames["4hourly"],
            pair_execution=execution_df,
            execution_frame=execution_frame,
            structure_col=structure_col,
            time_col="close_time",
        )

        if use_btc_drift_multiplier and btc_4h is not None and "btc_drift_exposure_multiplier" in btc_4h.columns:
            execution_df = attach_btc_drift_to_execution_frame(btc_4h, execution_df)

        frames["1daily"] = pair_daily
        frames[execution_frame] = execution_df

    return dict_of_pairs


def attach_4h_structure_to_dataset(
    dict_of_pairs: dict,
    structure_col: str = "structure_ok"
) -> dict:
    """
    Attach 4H structure flag from each pair's 4H frame
    onto its 1H execution frame.
    Requires:
    - pair 4H contains structure_col
    - pair 1H contains allowed_daily
    """
    for pair in dict_of_pairs:
        frames = dict_of_pairs[pair]["dict_of_frames"]

        if (
            "4hourly" not in frames or
            "1hourly" not in frames or
            frames["4hourly"].empty or
            frames["1hourly"].empty
        ):
            print(f"Skipping {pair} - missing or empty 4hourly/1hourly")
            continue

        if structure_col not in frames["4hourly"].columns:
            print(f"Skipping {pair} - missing {structure_col} in 4hourly")
            continue

        if "allowed_daily" not in frames["1hourly"].columns:
            print(f"Skipping {pair} - missing allowed_daily in 1hourly")
            continue

        frames["1hourly"] = attach_4h_structure_to_1h(
            pair_4h=frames["4hourly"],
            pair_1h=frames["1hourly"],
            structure_col=structure_col
        )

    return dict_of_pairs