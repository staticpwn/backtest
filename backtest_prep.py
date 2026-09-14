from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pandas as pd


def build_strategy_specs(
    entry_columns: list[str],
    exit_columns: list[str],
    mode: str = "pairwise",
) -> list[dict[str, str]]:
    """
    Build strategy specs from entry/exit trigger column names.

    mode='pairwise':
        entry[i] with exit[i] (lists must have equal length)

    mode='cartesian':
        all combinations of entries x exits
    """
    if not entry_columns:
        raise ValueError("entry_columns must not be empty")
    if not exit_columns:
        raise ValueError("exit_columns must not be empty")

    specs: list[dict[str, str]] = []

    if mode == "pairwise":
        if len(entry_columns) != len(exit_columns):
            raise ValueError("pairwise mode requires entry_columns and exit_columns to have same length")
        for entry_col, exit_col in zip(entry_columns, exit_columns):
            specs.append(
                {
                    "name": f"{entry_col}__{exit_col}",
                    "entry_col": entry_col,
                    "exit_col": exit_col,
                }
            )
    elif mode == "cartesian":
        for entry_col in entry_columns:
            for exit_col in exit_columns:
                specs.append(
                    {
                        "name": f"{entry_col}__{exit_col}",
                        "entry_col": entry_col,
                        "exit_col": exit_col,
                    }
                )
    else:
        raise ValueError("mode must be one of: 'pairwise', 'cartesian'")

    return specs


def _to_bool_series(df: pd.DataFrame, col: str, default: bool = False) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype="bool")

    series = df[col]

    if pd.api.types.is_bool_dtype(series):
        return series.fillna(default).astype(bool)

    if pd.api.types.is_numeric_dtype(series):
        values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64, copy=True)
        out = np.full(values.shape, default, dtype=np.bool_)
        valid_mask = ~np.isnan(values)
        out[valid_mask] = values[valid_mask] != 0.0
        return pd.Series(out, index=df.index, dtype="bool")

    values = series.to_numpy(dtype=object, copy=True)
    out = np.full(values.shape, default, dtype=np.bool_)

    true_tokens = {"1", "true", "t", "yes", "y", "on"}
    false_tokens = {"0", "false", "f", "no", "n", "off"}

    for i, v in enumerate(values):
        if pd.isna(v):
            out[i] = default
            continue

        if isinstance(v, (bool, np.bool_)):
            out[i] = bool(v)
            continue

        if isinstance(v, (int, float, np.integer, np.floating)):
            out[i] = bool(v)
            continue

        token = str(v).strip().lower()
        if token in true_tokens:
            out[i] = True
        elif token in false_tokens:
            out[i] = False
        else:
            out[i] = bool(v)

    return pd.Series(out, index=df.index, dtype="bool")


def _ensure_columns(df: pd.DataFrame, columns: list[str], fill_value: Any = np.nan) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            out[col] = fill_value
    return out


def _expected_bar_seconds(execution_frame: str) -> int:
    frame_to_seconds = {
        "1hourly": 3600,
        "4hourly": 4 * 3600,
        "1daily": 86400,
        "3daily": 3 * 86400,
        "1weekly": 7 * 86400,
    }
    if execution_frame not in frame_to_seconds:
        raise ValueError("execution_frame must be one of: '1hourly', '4hourly', '1daily', '3daily', '1weekly'")
    return frame_to_seconds[execution_frame]


def _sanitize_bar_cadence(
    frame_df: pd.DataFrame,
    time_col: str,
    execution_frame: str,
    min_spacing_ratio: float = 0.90,
) -> pd.DataFrame:
    """
    Remove rows that arrive too soon after the previous kept bar.

    This guards against malformed timestamps (for example, sub-hour bursts inside
    a 4H frame) that can distort mark-to-market equity while leaving trade logic
    seemingly reasonable.
    """
    if frame_df.empty or time_col not in frame_df.columns:
        return frame_df

    out = frame_df.copy()
    out[time_col] = pd.to_datetime(out[time_col], errors="coerce")
    out = out.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)
    if out.empty:
        return out

    min_spacing_seconds = _expected_bar_seconds(execution_frame) * float(min_spacing_ratio)

    keep_indices: list[int] = [0]
    last_kept_time = out.loc[0, time_col]

    for i in range(1, len(out)):
        cur_time = out.loc[i, time_col]
        dt_seconds = (cur_time - last_kept_time).total_seconds()
        if dt_seconds >= min_spacing_seconds:
            keep_indices.append(i)
            last_kept_time = cur_time

    return out.loc[keep_indices].reset_index(drop=True)


def sanitize_dataset_bar_cadence(
    dict_of_pairs: dict,
    frames: list[str] | None = None,
    min_spacing_ratio: float = 0.90,
    time_col: str = "close_time",
) -> dict:
    """
    Sanitize bar cadence on the RAW dataset, BEFORE prepare_df/indicators run.

    Applies ``_sanitize_bar_cadence`` to each frame of each pair (daily and
    execution frames), dropping bars that arrive too soon after the previous
    kept bar (malformed timestamps, sub-hour bursts, duplicate close_times).
    Doing this before ``prepare_df`` means indicators, levels and strategy
    signals are computed on clean bars instead of dirty frames — so the
    cadence fix affects both the timeline AND the signal history, not just the
    mark-to-market curve.

    Parameters
    ----------
    dict_of_pairs : dict
        Dataset in project format: {pair: {"dict_of_frames": {...}}}.
    frames : list[str] | None
        Restrict sanitization to these frame names (None = all frames).
    min_spacing_ratio : float
        Fraction of the expected bar interval that a bar must be spaced by.
    time_col : str
        Column used for bar time (default ``close_time``).

    Returns
    -------
    dict
        Same dict structure, with each sanitized frame replaced by a cleaned
        copy. Frames with an unrecognized interval are left untouched.
    """
    for pair in dict_of_pairs:
        pair_frames = dict_of_pairs[pair].get("dict_of_frames", {})
        for frame_name, frame_df in list(pair_frames.items()):
            if frames is not None and frame_name not in frames:
                continue
            if frame_df is None or frame_df.empty or time_col not in frame_df.columns:
                continue
            try:
                cleaned = _sanitize_bar_cadence(
                    frame_df=frame_df,
                    time_col=time_col,
                    execution_frame=frame_name,
                    min_spacing_ratio=min_spacing_ratio,
                )
            except ValueError:
                # Unknown frame interval -> leave the frame as-is.
                continue
            dict_of_pairs[pair]["dict_of_frames"][frame_name] = cleaned
    return dict_of_pairs


def compile_backtest_payload(
    dict_of_pairs: dict,
    execution_frame: str,
    strategy_specs: list[dict[str, str]],
    gate_col: str = "allowed_to_trade",
    time_col: str = "close_time",
    price_columns: tuple[str, str, str, str] = ("open", "high", "low", "close"),
    aux_columns: list[str] | None = None,
    context_columns: list[str] | None = None,
    exposure_multiplier_col: str | None = "btc_exposure_multiplier",
    enforce_bar_cadence: bool = True,
    min_spacing_ratio: float = 0.90,
) -> dict:
    """
    Compile a backtest-ready payload for a selected execution frame.

    Output is array-first and strategy-agnostic:
    - one global timeline
    - one pair axis
    - per-strategy entry/exit bool matrices
    - optional aux float arrays (stop_price, quote_vol_per_unit_change, rank_score, etc.)

    Parameters
    ----------
    aux_columns : list[str] | None
        Extra float columns to compile as (T, N) float64 arrays.
        Each column is filled with NaN where the pair has no data.
        Available in payload["aux"][column_name].
    context_columns : list[str] | None
        Extra per-bar context columns to compile as (T, N) object arrays.
        These are not used by the engine for decisions, but can be attached
        to positions/trades at entry time (for example: regime).
    exposure_multiplier_col : str | None
        Optional float column (per pair / per bar) compiled as an aux array
        for every strategy. The engine uses it to scale entry sizing
        (BTC-regime capital allocation). NaN values are treated as 1.0
        (neutral) by the engine. Set to None to disable.
    """
    if aux_columns is None:
        aux_columns = []
    if context_columns is None:
        context_columns = []

    # The exposure multiplier is always compiled (when enabled) so the engine
    # can scale entry sizing even if the caller's aux_columns list omits it.
    compile_aux_columns = list(aux_columns)
    if exposure_multiplier_col and exposure_multiplier_col not in compile_aux_columns:
        compile_aux_columns.append(exposure_multiplier_col)

    if execution_frame not in {"1hourly", "4hourly"}:
        raise ValueError("execution_frame must be one of: '1hourly', '4hourly'")

    if not strategy_specs:
        raise ValueError("strategy_specs must not be empty")

    if not isinstance(gate_col, str) or not gate_col:
        raise TypeError(
            f"gate_col must be a non-empty string, got {gate_col!r} "
            f"(type {type(gate_col).__name__}). A stray trailing comma in an "
            f"assignment like `gate_col=\"allowed_to_trade\",` makes it a 1-tuple, "
            f"which compiles every entry as gated OFF."
        )

    required_signal_cols = {gate_col}
    for spec in strategy_specs:
        required_signal_cols.add(spec["entry_col"])
        required_signal_cols.add(spec["exit_col"])

    timelines: list[pd.Series] = []
    eligible_pairs: list[str] = []
    cadence_dropped_rows: dict[str, int] = {}
    prepared_frames: dict[str, pd.DataFrame] = {}

    for pair, pair_data in dict_of_pairs.items():
        frames = pair_data.get("dict_of_frames", {})
        if execution_frame not in frames:
            continue
        frame_df = frames[execution_frame]
        if frame_df is None or frame_df.empty or time_col not in frame_df.columns:
            continue

        if enforce_bar_cadence:
            original_len = len(frame_df)
            frame_df = _sanitize_bar_cadence(
                frame_df=frame_df,
                time_col=time_col,
                execution_frame=execution_frame,
                min_spacing_ratio=min_spacing_ratio,
            )
            cadence_dropped_rows[pair] = max(original_len - len(frame_df), 0)

        prepared_frames[pair] = frame_df
        timelines.append(frame_df[time_col])
        eligible_pairs.append(pair)

    if not eligible_pairs:
        raise ValueError(f"No eligible pairs with non-empty '{execution_frame}' frame")

    global_timeline = (
        pd.concat(timelines, axis=0)
        .dropna()
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )

    t_size = len(global_timeline)
    n_size = len(eligible_pairs)

    open_arr = np.full((t_size, n_size), np.nan, dtype=np.float64)
    high_arr = np.full((t_size, n_size), np.nan, dtype=np.float64)
    low_arr = np.full((t_size, n_size), np.nan, dtype=np.float64)
    close_arr = np.full((t_size, n_size), np.nan, dtype=np.float64)

    gate_arr = np.zeros((t_size, n_size), dtype=np.bool_)

    aux_arrs: dict[str, dict[str, np.ndarray]] = {
        spec["name"]: {
            col: np.full((t_size, n_size), np.nan, dtype=np.float64)
            for col in compile_aux_columns
        }
        for spec in strategy_specs
    }
    context_arrs: dict[str, np.ndarray] = {
        col: np.full((t_size, n_size), None, dtype=object)
        for col in context_columns
    }

    strategy_payload: dict[str, dict[str, np.ndarray]] = {
        spec["name"]: {
            "entry": np.zeros((t_size, n_size), dtype=np.bool_),
            "exit": np.zeros((t_size, n_size), dtype=np.bool_),
        }
        for spec in strategy_specs
    }

    timeline_index = pd.Index(global_timeline)
    gate_col_found = 0

    for col_idx, pair in enumerate(eligible_pairs):
        frame_df = prepared_frames[pair].copy()
        frame_df = frame_df.sort_values(time_col).drop_duplicates(subset=[time_col], keep="last")
        if gate_col in frame_df.columns:
            gate_col_found += 1

        keep_cols = [time_col, *price_columns, *required_signal_cols, *compile_aux_columns, *context_columns]
        frame_df = _ensure_columns(frame_df, keep_cols)
        pair_aligned = frame_df.set_index(time_col).reindex(timeline_index)

        open_arr[:, col_idx] = pd.to_numeric(pair_aligned[price_columns[0]], errors="coerce").to_numpy(dtype=np.float64)
        high_arr[:, col_idx] = pd.to_numeric(pair_aligned[price_columns[1]], errors="coerce").to_numpy(dtype=np.float64)
        low_arr[:, col_idx] = pd.to_numeric(pair_aligned[price_columns[2]], errors="coerce").to_numpy(dtype=np.float64)
        close_arr[:, col_idx] = pd.to_numeric(pair_aligned[price_columns[3]], errors="coerce").to_numpy(dtype=np.float64)

        gate_series = _to_bool_series(pair_aligned, gate_col, default=False)
        gate_arr[:, col_idx] = gate_series.to_numpy(dtype=np.bool_)

        for col in context_columns:
            context_arrs[col][:, col_idx] = pair_aligned[col].to_numpy(dtype=object, copy=True)

        for spec in strategy_specs:
            strategy_name = spec["name"]
            entry_col = spec["entry_col"]
            exit_col = spec["exit_col"]
            aux_prefix = entry_col

            def _as_series(column_name: str) -> pd.Series:
                selected = pair_aligned.loc[:, column_name]
                if isinstance(selected, pd.DataFrame):
                    selected = selected.iloc[:, 0]
                return selected

            entry_series = _to_bool_series(pair_aligned, entry_col, default=False)
            exit_series = _to_bool_series(pair_aligned, exit_col, default=False)

            strategy_payload[strategy_name]["entry"][:, col_idx] = (
                gate_series.to_numpy(dtype=np.bool_) & entry_series.to_numpy(dtype=np.bool_)
            )
            strategy_payload[strategy_name]["exit"][:, col_idx] = exit_series.to_numpy(dtype=np.bool_)

            for col in compile_aux_columns:
                prefixed_col = f"{aux_prefix}__{col}"
                source_col = prefixed_col if prefixed_col in pair_aligned.columns else col
                if source_col in pair_aligned.columns:
                    aux_source = _as_series(source_col)
                else:
                    aux_source = pd.Series(np.nan, index=pair_aligned.index)
                aux_arrs[strategy_name][col][:, col_idx] = pd.to_numeric(
                    aux_source,
                    errors="coerce",
                ).to_numpy(dtype=np.float64)

    if gate_col_found == 0:
        raise ValueError(
            f"gate_col={gate_col!r} was not found in any eligible pair's "
            f"'{execution_frame}' frame. Compiling would silently mask every "
            f"entry (all-False gate) — did STEP 2 attach the gate column?"
        )

    return {
        "execution_frame": execution_frame,
        "timeline": global_timeline,
        "pairs": eligible_pairs,
        "prices": {
            "open": open_arr,
            "high": high_arr,
            "low": low_arr,
            "close": close_arr,
        },
        "gate": gate_arr,
        "aux": aux_arrs,
        "context": context_arrs,
        "strategies": strategy_payload,
        "shape": {
            "T": t_size,
            "N": n_size,
        },
        "diagnostics": {
            "enforce_bar_cadence": enforce_bar_cadence,
            "min_spacing_ratio": float(min_spacing_ratio),
            "cadence_dropped_rows_by_pair": cadence_dropped_rows,
            "total_cadence_dropped_rows": int(sum(cadence_dropped_rows.values())) if cadence_dropped_rows else 0,
        },
    }


def _gain_vol_source_col(entry_col: str):
    """Map a STEP 3 entry column to its gain/vol base-signal source.

    Returns (kind, level_col) where kind in ("break", "test") -- mirroring the
    entry formulas in strategies.apply_break_strategy / apply_test_strategy --
    or None when the entry column does not follow the break_/test_ naming.
    """
    for prefix, kind in (("entry_break_", "break"), ("entry_test_", "test")):
        if entry_col.startswith(prefix):
            return kind, entry_col[len(prefix):]
    return None


def _base_entry_signal(frame: pd.DataFrame, kind: str, level_col: str):
    """Recompute the STEP 3 entry signal WITHOUT the gain/vol filters.

    break: <level>_is_broken & (rsi_slope > 0)
    test : (<level>_tested > 0) & (rsi_slope > 0) & (<level>_last_broken > 5)

    Returns None when a required level column is missing on the frame.
    NaN in any component behaves like the strategy functions (NaN > 0 -> False,
    NaN in a boolean & -> False via _to_bool_series).
    """
    if "rsi_slope" not in frame.columns:
        return None
    rsi_up = pd.to_numeric(frame["rsi_slope"], errors="coerce") > 0
    if kind == "break":
        broken_col = f"{level_col}_is_broken"
        if broken_col not in frame.columns:
            return None
        return _to_bool_series(frame, broken_col, default=False) & rsi_up
    if kind == "test":
        tested_col = f"{level_col}_tested"
        last_broken_col = f"{level_col}_last_broken"
        if tested_col not in frame.columns or last_broken_col not in frame.columns:
            return None
        tested = pd.to_numeric(frame[tested_col], errors="coerce") > 0
        last_broken = pd.to_numeric(frame[last_broken_col], errors="coerce") > 5
        return tested & rsi_up & last_broken
    return None


def remask_entry_filters(
    dict_of_pairs: dict,
    execution_frame: str,
    strategy_specs: list[dict[str, str]],
    gain_ratio: float = 1.15,
    lower_volatility_threshold: float = 0.01,
    upper_volatility_threshold: float = 0.08,
    inplace: bool = False,
) -> tuple[dict, list[str]]:
    """Recompute each strategy's entry column from a new gain/vol filter.

    The STEP 3 break/test entry signal is
        base_signal & (lower < volatility < upper) & (rolling_gain > gain_ratio)
    where base_signal depends only on the level columns and rsi_slope -- NOT on
    gain_ratio or the volatility band. So a gain/vol sweep only needs to re-mask
    the entry column: every aux column (stop_price, take_profit_price,
    rank_score, qvpuc, ...) and the exit column were baked at STEP 3 and are
    left untouched. This is the cheap equivalent of re-running the strategy
    loop with new parameters, without touching live_runtime/ or the engine.

    Strategy entry columns are recognised by name:
      'entry_break_<col>' -> base = <col>_is_broken & (rsi_slope > 0)
      'entry_test_<col>'  -> base = (<col>_tested > 0) & (rsi_slope > 0)
                                    & (<col>_last_broken > 5)
    Any other strategy is skipped (reported in the second return value) and its
    entry column is left as-is.

    Returns
    -------
    (dict_of_pairs, skipped) : tuple
        The (new or same) dataset with entry columns re-masked, and the list of
        strategy names that were skipped because they are not recognised.
    """
    if not inplace:
        dict_of_pairs = copy.deepcopy(dict_of_pairs)
    skipped: list[str] = []

    for spec in strategy_specs:
        entry_col = spec["entry_col"]
        src = _gain_vol_source_col(entry_col)
        if src is None:
            skipped.append(spec["name"])
            continue
        kind, level_col = src

        for pair, pair_data in dict_of_pairs.items():
            frame = pair_data.get("dict_of_frames", {}).get(execution_frame)
            if frame is None or frame.empty:
                continue
            if "volatility" not in frame.columns or "rolling_gain" not in frame.columns:
                continue
            base = _base_entry_signal(frame, kind, level_col)
            if base is None:
                continue
            volatile = (
                (frame["volatility"] > lower_volatility_threshold)
                & (frame["volatility"] < upper_volatility_threshold)
            )
            is_gainer = frame["rolling_gain"] > gain_ratio
            frame[entry_col] = base & volatile & is_gainer

    return dict_of_pairs, skipped


def apply_btc_drift_multiplier_to_payload(
    payload: dict,
    btc_pair: str = "BTCUSDT",
    drift_bars: int = 12,
    turn_on_threshold: float = 0.01,
    turn_off_threshold: float = -0.01,
    turn_on_bars: int = 2,
    turn_off_bars: int = 2,
    floor: float = 0.4,
) -> dict:
    """
    Apply the BTC 4h drift soft exposure multiplier directly to an ALREADY
    COMPILED backtest payload (no re-run of STEP 2/3/5 required).

    The multiplier is a BTC-only signal, computed from the BTC pair's own close
    column on the payload timeline (``payload["prices"]["close"][:, btc_idx]``),
    broadcast across all pairs, and injected into every strategy's aux dict as
    ``btc_drift_exposure_multiplier``. The engine reads it per entry via
    ``EngineConfig.exposure_multiplier_col`` and scales entry sizing. It never
    blocks an entry (floor >= 0) and never touches the gate / entry / exit
    matrices — so it is a pure sizing A/B on an existing payload.

    Notes
    -----
    - Requires ``btc_pair`` in ``payload["pairs"]`` (true for the full-dataset
      STEP 5 payload; not true for subsets that exclude BTC).
    - Values are identical to the prepare-path
      (``gating.add_btc_drift_exposure_multiplier``) because the drift is causal
      and the payload timeline is the same 4h grid.
    - Mutates ``payload["aux"]`` in place and returns the same dict.

    Returns
    -------
    payload : dict
        The same payload with ``btc_drift_exposure_multiplier`` injected into
        each strategy's aux dict.
    """
    if btc_pair not in payload["pairs"]:
        raise ValueError(
            f"apply_btc_drift_multiplier_to_payload requires '{btc_pair}' in "
            f"payload['pairs'] (found {len(payload['pairs'])} pairs)"
        )

    # Local import: no circular dependency (gating does not import backtest_prep).
    import gating

    btc_idx = payload["pairs"].index(btc_pair)
    timeline = pd.to_datetime(payload["timeline"])
    btc_close = payload["prices"]["close"][:, btc_idx]

    # Compute the drift on BTC's OWN contiguous (non-NaN) bars — mirroring the
    # prepare path, which runs add_btc_drift_exposure_multiplier on the full BTC
    # 4h frame. The payload timeline is the UNION of all pairs' bars, so the BTC
    # close column can contain NaN gaps that would corrupt the rolling/hysteresis
    # if fed directly. We then as-of align (backward) the clean multiplier onto
    # the full timeline, exactly like attach_btc_drift_to_execution_frame.
    valid = np.isfinite(btc_close)
    if not valid.any():
        raise ValueError(f"'{btc_pair}' close column is entirely NaN in the payload")
    btc_clean = pd.DataFrame(
        {"close_time": timeline[valid], "close": btc_close[valid]}
    )
    drift = gating.add_btc_drift_exposure_multiplier(
        btc_clean,
        drift_bars=drift_bars,
        turn_on_threshold=turn_on_threshold,
        turn_off_threshold=turn_off_threshold,
        turn_on_bars=turn_on_bars,
        turn_off_bars=turn_off_bars,
        floor=floor,
    )
    aligned = pd.merge_asof(
        pd.DataFrame({"close_time": timeline}).sort_values("close_time"),
        drift[["close_time", "btc_drift_exposure_multiplier"]].sort_values("close_time"),
        on="close_time",
        direction="backward",
    )
    mult = aligned["btc_drift_exposure_multiplier"].to_numpy(dtype=np.float64)
    # Bars before the first BTC bar have no mapped multiplier -> neutral 1.0
    # (mirrors the fill_value used by attach_btc_drift_to_execution_frame).
    mult[pd.isna(aligned["btc_drift_exposure_multiplier"]).to_numpy()] = 1.0

    T = int(payload["shape"]["T"])
    N = int(payload["shape"]["N"])
    mult_2d = np.broadcast_to(mult[:, None], (T, N)).copy()

    for strat_name in payload["aux"]:
        payload["aux"][strat_name]["btc_drift_exposure_multiplier"] = mult_2d.copy()

    return payload
