import os
import pickle
import picklejar
import re
import time

import numpy as np
import pandas as pd
import talib
from scipy.signal import find_peaks

from binance import AsyncClient

from features import (
    prepare_df,
    attach_sqn_to_dataset,
    add_hh_ll_columns,
    populate_historical_levels_fast,
    get_common_aux_columns,
    attach_trend_lines_to_dataset,
    attach_trend_lines,
)
from gating import compute_4h_structure_for_dataset
from helpers import level_test_buys_with_last_broken, get_simple_slope_2, level_finder
from data_import import importer, get_frame_import_interval

def read_pickle_object(path):
    try:
        with open(f"{path}.pickle", 'rb') as open_object:
            object =  pickle.load(open_object)
            open_object.close()
        
        return object
    except Exception as e:
        print(e)
        return False
    
def store_pickle_object(path, object_to_store):

    try:
        with open(f"{path}.pickle", "wb") as open_object:
            pickle.dump(object_to_store, open_object, protocol=pickle.HIGHEST_PROTOCOL)
            open_object.close() 
        
        return True
    except Exception as e:
        print(e)
        return False


# ---------------------------------------------------------------------------
# Dataset refresh: bring raw + prepared pickles up to the latest bar
# ---------------------------------------------------------------------------
# Mirrors tester.ipynb's prep pipeline (prepare_df -> SQN/structure -> daily
# merge -> levels/retest/break -> aux columns) so the stored 'prepared' pickle
# is byte-for-byte equivalent to what the backtest notebook would produce.

# Execution-frame prep config (STEP 1b + the prep cell in tester.ipynb).
_EXECUTION_FRAME = "4hourly"
_DAILY_FRAME = "1daily"

# Causal swing trend lines (tester.ipynb's attach_trend_lines_to_dataset call;
# the same kwargs live_runtime/live_prep.py passes to attach_trend_lines).
_TREND_FRAMES = (_EXECUTION_FRAME, _DAILY_FRAME)
_TREND_KWARGS = dict(
    distance=3,
    window_peaks=4,
    margin=1.0,
    break_margin=0.0,
    spread_col="short_spread",
)
_TREND_COLUMNS = ("upper_trend", "lower_trend")
_TREND_MERGED_COLUMNS = tuple(f"{col}_{_DAILY_FRAME}" for col in _TREND_COLUMNS)

_COLUMNS_FOR_MERGE = [
    "ma25", "ma50", "ma100", "ma200", "ema25", "ema50", "ema100", "ema200",
    "rsi", "rsi_slope", "max", "min", "boll_upper", "boll_mid", "boll_lower",
]
_COLUMNS_FOR_MERGE += list(_TREND_COLUMNS)

_COLS_FOR_RETEST = [
    "ma50", "boll_lower", "ema50", "ma200", "boll_mid_1daily", "ema100",
    "ma100", "ema200", "ema25_1daily", "prev_low", "min", "ema25",
    "yearly_open", "monthly_open", "weekly_open", "nearest_support",
]
_COLS_FOR_RETEST += ["lower_trend", f"lower_trend_{_DAILY_FRAME}"]

_COLS_FOR_BREAK = [
    "max", "boll_upper", "boll_upper_1daily", "ema100", "ema200", "ma100",
    "ma200", "ema100_1daily", "ema200_1daily", "ma100_1daily", "ma200_1daily",
    "prev_high", "yearly_open", "monthly_open", "weekly_open",
    "nearest_resistance",
]
_COLS_FOR_BREAK += ["upper_trend", f"upper_trend_{_DAILY_FRAME}"]

# Schema guard: the columns every prepared frame must carry. Older pickles built
# before the trend-line step lack them, and the incremental tail can only produce
# them for NEW rows (the spliced history would come back NaN), so a missing
# column family must force a full rebuild instead.
_RETEST_BREAK_PROPS = ("tested", "last_broken", "spreads_from_close", "is_broken")
# Execution frame: its own trend lines + the daily-merged copies + the retest/break
# family the strategy step consumes (lower_trend_tested, upper_trend_1daily_is_broken...).
PREPARED_REQUIRED_COLUMNS = tuple(
    _TREND_COLUMNS
    + _TREND_MERGED_COLUMNS
    + tuple(
        f"{col}_{prop}"
        for col in (_TREND_COLUMNS + _TREND_MERGED_COLUMNS)
        for prop in _RETEST_BREAK_PROPS
    )
)
# Daily frame: only what the backward merge reads off it (prepare_df outputs +
# rsi_slope + its own trend lines) — the merged/derived family lives on the
# execution frame.
_PREPARED_REQUIRED_BY_FRAME = {
    _EXECUTION_FRAME: PREPARED_REQUIRED_COLUMNS,
    _DAILY_FRAME: tuple(_COLUMNS_FOR_MERGE),
}


def missing_prepared_columns(prepared):
    """List ``(pair, frame, [missing columns])`` for under-built prepared frames.

    A non-empty result means the stored pickle predates a column family (or was
    written by an older pipeline) and must be rebuilt in full — splicing an
    incremental tail would only fill the new rows.
    """
    missing = []
    for pair, pdata in (prepared or {}).items():
        frame_dict = (pdata or {}).get("dict_of_frames") or {}
        for frame_name, required in _PREPARED_REQUIRED_BY_FRAME.items():
            frame_df = frame_dict.get(frame_name)
            if frame_df is None or not hasattr(frame_df, "columns"):
                continue
            have = set(frame_df.columns)
            gone = [col for col in required if col not in have]
            if gone:
                missing.append((pair, frame_name, gone))
    return missing

_DATASET_DIRS = {
    "halal": "data/halal",
    "complete": "data/complete",
    "validation": "data/validation",
}


def _frame_cadence(frame_name: str) -> pd.Timedelta:
    """Timedelta for a frame key ('4hourly' -> 4h, '1daily' -> 1D, ...)."""
    interval = get_frame_import_interval(frame_name)  # e.g. '4h' / '1d' / '1w'
    m = re.match(r"(\d+)([a-z]+)", interval)
    if not m:
        return pd.Timedelta(hours=1)
    n = int(m.group(1))
    unit = {"h": "hours", "d": "days", "w": "weeks", "m": "minutes"}.get(m.group(2)[0], "hours")
    return pd.Timedelta(**{unit: n})


def _rolling_marker_path(prepared_path: str) -> str:
    # same convention as read_pickle_object/store_pickle_object: path + '.pickle'
    return f"{prepared_path}.pickle.rolling_window"


def _read_rolling_marker(prepared_path: str):
    """Window length of the stored rolling prep, or None for a plain frame.

    A rolling ``prepared`` pickle must never be refreshed incrementally: the tail
    would come from the full-history pipeline and be spliced onto windowed rows.
    """
    try:
        with open(_rolling_marker_path(prepared_path), "r", encoding="utf-8") as handle:
            value = handle.read().strip()
        return int(value) if value else None
    except Exception:
        return None


def _write_rolling_marker(prepared_path: str, window_bars) -> None:
    path = _rolling_marker_path(prepared_path)
    try:
        if window_bars is None:
            if os.path.exists(path):
                os.remove(path)
            return
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(str(int(window_bars)))
    except Exception as exc:                      # pragma: no cover - best effort
        print(f"warning: could not update {path}: {exc}")


def _stale_entries(dict_of_pairs):
    """(pair, frame) entries whose last stored bar is older than one cadence."""
    now = pd.Timestamp.utcnow().tz_localize(None)
    stale = []
    for pair, pdata in dict_of_pairs.items():
        for frame_name, frame_df in (pdata.get("dict_of_frames") or {}).items():
            if frame_df is None or frame_df.empty or "close_time" not in frame_df.columns:
                continue
            last_close = pd.to_datetime(frame_df["close_time"], errors="coerce").max()
            if pd.isna(last_close):
                continue
            if last_close < now - _frame_cadence(frame_name):
                stale.append((pair, frame_name))
    return stale


async def _fetch_missing(client, dict_of_pairs, stale_entries):
    """Incrementally fetch bars newer than each stale frame's last close."""
    updated = False
    for pair, frame_name in stale_entries:
        frame_df = dict_of_pairs[pair]["dict_of_frames"][frame_name]
        last_close = pd.to_datetime(frame_df["close_time"], errors="coerce").max()
        new = await importer(
            client, pair, get_frame_import_interval(frame_name),
            start=last_close, end=None, limit=None,
        )
        if new is None or new.empty:
            continue
        combined = pd.concat([frame_df, new], ignore_index=True)
        combined = (
            combined.drop_duplicates(subset=["open_time"], keep="last")
            .sort_values("close_time")
            .reset_index(drop=True)
        )
        dict_of_pairs[pair]["dict_of_frames"][frame_name] = combined
        updated = True
        # print(f"  updated {pair} [{frame_name}]: {len(frame_df)} -> {len(combined)} bars")
    return updated


def _reprepared_dataset(dict_of_pairs):
    """Rebuild the 'prepared' dataset exactly as tester.ipynb does (cells 10-13)."""
    for pair, pdata in dict_of_pairs.items():
        for frame_name, frame_df in (pdata.get("dict_of_frames") or {}).items():
            if frame_df is None or frame_df.empty:
                continue
            frame_df = prepare_df(frame_df)
            if frame_name == _EXECUTION_FRAME and "open_time" in frame_df.columns:
                t = pd.to_datetime(frame_df["open_time"], errors="coerce")
                open_series = pd.to_numeric(frame_df["open"], errors="coerce")
                s = pd.Series(t, index=frame_df.index)
                frame_df["yearly_open"] = open_series.groupby(s.dt.to_period("Y")).transform("first")
                frame_df["monthly_open"] = open_series.groupby(s.dt.to_period("M")).transform("first")
                frame_df["weekly_open"] = open_series.groupby(s.dt.to_period("W-SUN")).transform("first")
            # rsi_slope is computed on EVERY frame here (not only the execution
            # frame) so the daily frame can contribute rsi_slope_1daily through
            # the backward merge below — same as tester.ipynb's prep cell.
            frame_df["rsi_slope"] = frame_df["rsi"].rolling(6).apply(get_simple_slope_2, raw=True)
            pdata["dict_of_frames"][frame_name] = frame_df

    # Causal swing trend lines (upper_trend / lower_trend + peak/trough counts).
    # Must run BEFORE the daily merge and the levels stage: the merge publishes
    # lower_trend_1daily / upper_trend_1daily, and the retest/break lists below
    # consume all four.
    dict_of_pairs = attach_trend_lines_to_dataset(
        dict_of_pairs, frames=list(_TREND_FRAMES), **_TREND_KWARGS
    )

    dict_of_pairs = attach_sqn_to_dataset(dict_of_pairs)
    dict_of_pairs = compute_4h_structure_for_dataset(
        dict_of_pairs,
        ema_fast_period=25,
        ema_mid_period=50,
        ema_slow_period=200,
        slope_lookback=5,
    )

    # daily -> execution-frame backward merge (columns get the _1daily suffix).
    for pair, pdata in dict_of_pairs.items():
        frames = pdata.get("dict_of_frames") or {}
        target = frames.get(_EXECUTION_FRAME)
        daily = frames.get(_DAILY_FRAME)
        if target is None or daily is None:
            continue
        source = daily[_COLUMNS_FOR_MERGE + ["close_time"]]
        frames[_EXECUTION_FRAME] = pd.merge_asof(
            target.sort_values("close_time"),
            source.sort_values("close_time"),
            left_on="close_time",
            right_on="close_time",
            direction="backward",
            suffixes=["", f"_{_DAILY_FRAME}"],
        )

    cols_to_prepare = list(set(_COLS_FOR_RETEST + _COLS_FOR_BREAK))
    for pair, pdata in dict_of_pairs.items():
        frame_df = pdata["dict_of_frames"].get(_EXECUTION_FRAME)
        if frame_df is None or frame_df.empty:
            continue
        frame_df = add_hh_ll_columns(frame_df)
        frame_df = populate_historical_levels_fast(
            ohlc_df=frame_df,
            levels_reactions_limit=12,
            lookback_bars=None,
            price_col="close",
            skip_rows=6 * 14,
        )
        for col in cols_to_prepare:
            retest_dict = level_test_buys_with_last_broken(
                frame_df, [col], confirm_breaks=False, confirm_tests=False)
            retest_frame = pd.DataFrame(retest_dict)
            retest_frame.rename(
                columns={prop: f"{col}_{prop}" for prop in retest_dict}, inplace=True)
            frame_df = pd.concat([frame_df, retest_frame], axis=1)
        frame_df["rolling_max"] = frame_df["close"].rolling(6 * 7).max()
        frame_df["rolling_min"] = frame_df["close"].rolling(6 * 7).min()
        frame_df["rolling_gain"] = frame_df["rolling_max"] / frame_df["rolling_min"]
        frame_df["obv"] = talib.OBV(frame_df["close"], frame_df["volume"])
        frame_df = get_common_aux_columns(
            frame_df, rs_boost_weight=0.5, rs_boost_regime_conditional=True)
        pdata["dict_of_frames"][_EXECUTION_FRAME] = frame_df

    return dict_of_pairs


def _reprepared_dataset_incremental(raw, prepared, warmup_bars=500):
    """Rebuild only the NEW bars of 'prepared', splicing onto the existing one.

    For each (pair, frame) whose raw has bars newer than the stored prepared
    frame, the last ``warmup_bars + n_new`` RAW bars are run through the SAME
    prep pipeline; only the newest ``n_new`` prepped rows are kept and spliced
    onto the old prepared frame (concat + drop_duplicates on close_time + sort).
    Period opens (yearly/monthly/weekly_open) are recomputed on the FULL spliced
    frame afterward (cheap) so they stay exact.

    Accuracy note: all rolling indicators (ema/ma/boll/short_spread/sqn/aux,
    trend lines) are correct because ``warmup_bars`` (default 500) covers the
    longest lookback (252-bar sqn percentile, 200-bar ema). The one residual
    difference vs a full rebuild is the LEVELS-derived columns
    (nearest_support / nearest_resistance / levels + the
    ``{col}_is_broken/_tested/_last_broken`` family): those are computed from the
    warmup window's extrema instead of all history, so the newest bars' levels
    can differ slightly. Raise ``warmup_bars`` or use ``full_rebuild=True`` for
    exact parity.

    If the stored prepared frame is missing any column family the current
    pipeline produces (see ``PREPARED_REQUIRED_COLUMNS``), the incremental path
    cannot backfill it for the OLD rows, so this falls back to a full rebuild.
    """
    if prepared is False or prepared is None:
        return _reprepared_dataset(raw)

    missing = missing_prepared_columns(prepared)
    if missing:
        print(f"prepared: schema incomplete for {len(missing)} (pair, frame) "
              f"entries (e.g. {missing[0][0]} [{missing[0][1]}]: "
              f"{', '.join(missing[0][2][:4])}) -> full rebuild")
        return _reprepared_dataset(raw)

    # Build per-pair windows of RAW bars (warmup + new) for frames that grew.
    windows = {}
    keep_new = {}
    need = False
    for pair, pdata in raw.items():
        raw_frames = pdata.get("dict_of_frames") or {}
        prep_frames = (prepared.get(pair) or {}).get("dict_of_frames") or {}
        pair_windows = {}
        pair_keep = {}
        for frame_name, raw_df in raw_frames.items():
            if raw_df is None or raw_df.empty or "close_time" not in raw_df.columns:
                continue
            old = prep_frames.get(frame_name)
            if old is None or old.empty or "close_time" not in old.columns:
                pair_windows[frame_name] = raw_df
                pair_keep[frame_name] = len(raw_df)
                need = True
                continue
            last_prep = pd.to_datetime(old["close_time"], errors="coerce").max()
            new_mask = pd.to_datetime(raw_df["close_time"], errors="coerce") > last_prep
            n_new = int(new_mask.sum())
            if n_new == 0:
                continue  # frame already current -> keep old prepared frame
            need = True
            pair_keep[frame_name] = n_new
            start = max(0, len(raw_df) - (warmup_bars + n_new))
            pair_windows[frame_name] = raw_df.iloc[start:].reset_index(drop=True)
        if pair_windows:
            # The daily -> execution backward merge (in _reprepared_dataset)
            # needs BOTH frames present so the '{daily}_{col}' columns (e.g.
            # boll_mid_1daily, boll_upper_1daily, ema25_1daily) can be produced.
            # The daily frame closes once a day while 4h closes every 4h, so a
            # typical refresh has the 4h frame grow while the daily frame is
            # already current -> without the daily frame the merge is skipped and
            # _reprepared_dataset crashes on the missing '{col}_{daily}' column.
            # Pull the daily frame's raw warmup in as a HELPER (n_new stays 0, so
            # the splicer below keeps the old prepared daily frame untouched; the
            # merge still has daily bars to asof onto the new execution bars).
            if (_EXECUTION_FRAME in pair_windows
                    and _DAILY_FRAME in raw_frames
                    and _DAILY_FRAME not in pair_windows):
                daily_df = raw_frames[_DAILY_FRAME]
                if daily_df is not None and not daily_df.empty and "close_time" in daily_df.columns:
                    d_start = max(0, len(daily_df) - warmup_bars)
                    pair_windows[_DAILY_FRAME] = daily_df.iloc[d_start:].reset_index(drop=True)
                    pair_keep.setdefault(_DAILY_FRAME, 0)
            windows[pair] = {"dict_of_frames": pair_windows}
            keep_new[pair] = pair_keep

    if not need:
        return prepared

    prepped_windows = _reprepared_dataset(windows)

    out = {}
    for pair, pdata in raw.items():
        raw_frames = pdata.get("dict_of_frames") or {}
        prep_frames = (prepared.get(pair) or {}).get("dict_of_frames") or {}
        pair_keep = keep_new.get(pair) or {}
        new_frames = {}
        for frame_name, raw_df in raw_frames.items():
            old = prep_frames.get(frame_name)
            n_new = pair_keep.get(frame_name, 0)
            pw = (prepped_windows.get(pair) or {}).get("dict_of_frames", {}).get(frame_name)
            if n_new > 0 and pw is not None and not pw.empty:
                new_rows = pw.iloc[-n_new:]
                if old is None or old.empty:
                    combined = new_rows
                else:
                    combined = (
                        pd.concat([old, new_rows], ignore_index=True)
                        .drop_duplicates(subset=["close_time"], keep="last")
                        .sort_values("close_time")
                        .reset_index(drop=True)
                    )
            else:
                combined = old if (old is not None and not old.empty) else raw_df
            # recompute period opens on the full spliced frame (exact + cheap)
            if (frame_name == _EXECUTION_FRAME and "open_time" in combined.columns
                    and "open" in combined.columns and not combined.empty):
                combined = combined.copy()
                t = pd.to_datetime(combined["open_time"], errors="coerce")
                open_series = pd.to_numeric(combined["open"], errors="coerce")
                s = pd.Series(t, index=combined.index)
                combined["yearly_open"] = open_series.groupby(s.dt.to_period("Y")).transform("first")
                combined["monthly_open"] = open_series.groupby(s.dt.to_period("M")).transform("first")
                combined["weekly_open"] = open_series.groupby(s.dt.to_period("W-SUN")).transform("first")
            new_frames[frame_name] = combined
        out[pair] = {"dict_of_frames": new_frames}
    return out


# ---------------------------------------------------------------------------
# Bounded ("rolling window") preparation — the backtest counterpart of live
# ---------------------------------------------------------------------------
# Live never prepares the full history. Its cache holds exactly
# ``Runtime._compute_frame_limits()`` bars per frame (425 at the time of writing:
# the longest strategy requirement — 200-bar indicators x ema_warmup_multiplier
# 2.0 = 400 — plus ``bootstrap_buffer_bars`` 25) and it RE-prepares that window at
# every bar close. Every history-dependent column therefore comes out of a bounded
# window, and two families react very differently:
#   * recursive indicators (ema/ma/boll/short_spread/sqn) only carry a small
#     warm-up residual;
#   * WINDOW-derived columns differ outright — ``levels`` / ``nearest_support`` /
#     ``nearest_resistance`` (built from the window's extrema) and the swing chain
#     behind ``prev_high_rsi`` / ``prev_low_rsi`` / ``HH`` / ``LL`` (``find_peaks``
#     over the window), which is exactly what momentum's ``reject_rsi_div`` reads.
# Preparing the backtest from full history instead leaves those columns different
# from live's on the same bar even with byte-identical OHLCV, which surfaces as a
# handful of entries/exits landing one bar apart.
#
# ``reprepared_dataset_rolling_window`` truncates every RAW frame to its last
# ``window_bars`` bars and THEN runs the normal pipeline, so all of those columns
# are computed from the same bounded history live uses.
#
# 425 is live's own limit, derived in ``Runtime._compute_frame_limits``:
#   base {'1daily': 100, '4hourly': 205}
#   raised to the largest strategy requirement = 200-bar indicators x
#   ``ema_warmup_multiplier`` 2.0 = 400
#   plus ``bootstrap_buffer_bars`` 25  =>  425 for BOTH frames.
# tests/test_data_management_incremental.py recomputes that number from the live
# runtime and asserts it still equals this constant, so a live config change
# (ema_warmup_multiplier, bootstrap_buffer_bars, a longer indicator) fails loudly
# here instead of silently drifting out of parity.
ROLLING_WINDOW_BARS = 425


def rolling_window_dataset(raw, window_bars: int = ROLLING_WINDOW_BARS, frames=None):
    """Copy of ``raw`` with every (pair, frame) truncated to its last ``window_bars`` bars.

    The input dict is left untouched (frames are copied). A frame shorter than the
    window is kept whole. ``frames`` optionally limits which frame names are kept.
    """
    keep = set(frames) if frames is not None else None
    out = {}
    for pair, pdata in (raw or {}).items():
        kept = {}
        for frame_name, frame_df in (pdata.get("dict_of_frames") or {}).items():
            if frame_df is None or frame_df.empty:
                continue
            if keep is not None and frame_name not in keep:
                continue
            ordered = (
                frame_df.sort_values("close_time")
                if "close_time" in frame_df.columns
                else frame_df
            )
            kept[frame_name] = ordered.tail(int(window_bars)).reset_index(drop=True)
        if kept:
            out[pair] = {"dict_of_frames": kept}
    return out


def _apply_reference_period_opens(
    dict_of_pairs,
    execution_frame: str = _EXECUTION_FRAME,
    daily_frame: str = _DAILY_FRAME,
):
    """Re-derive yearly/monthly/weekly_open from the DAILY frame's period starts.

    ``_reprepared_dataset`` derives these from the execution frame itself, which is
    fine on full history but collapses to "first open of the window" once the frame
    is bounded: a 425-bar 4h frame is ~71 days, so a bar in January would wrongly
    take the window's first open as ``yearly_open``.

    live_prep does the same thing for the same reason —
    ``add_period_open_columns(frame, reference_df=daily)`` maps each execution bar's
    period to the DAILY frame's first open of that period (its comment: "The
    execution window can be shorter than a year (e.g. 425 4h bars), which would
    otherwise make yearly_open collapse to the first open of the series"). The daily
    frame spans more than a year, so the true period start is always available.
    """
    for pair, pdata in dict_of_pairs.items():
        frames = pdata.get("dict_of_frames") or {}
        target = frames.get(execution_frame)
        daily = frames.get(daily_frame)
        if target is None or daily is None or target.empty or daily.empty:
            continue
        if "open_time" not in target.columns or "open_time" not in daily.columns:
            continue
        reference = pd.DataFrame(
            {
                "t": pd.to_datetime(daily["open_time"], errors="coerce"),
                "open": pd.to_numeric(daily["open"], errors="coerce"),
            }
        ).dropna()
        if reference.empty:
            continue
        target_time = pd.to_datetime(target["open_time"], errors="coerce")
        for column, freq in (
            ("yearly_open", "Y"),
            ("monthly_open", "M"),
            ("weekly_open", "W-SUN"),
        ):
            mapping = reference.groupby(reference["t"].dt.to_period(freq))["open"].first()
            target[column] = target_time.dt.to_period(freq).map(mapping).astype(float)
        frames[execution_frame] = target
    return dict_of_pairs


def reprepared_dataset_rolling_window(raw, window_bars: int = ROLLING_WINDOW_BARS):
    """Full rebuild in which EVERY column is computed from the last ``window_bars`` bars.

    Use this instead of ``_reprepared_dataset`` when the dataset has to line up with
    the live runtime bar-for-bar: live prepares a bounded window at every close, so a
    full-history backtest prep leaves the levels/nearest_* and swing-derived columns
    different from live's on the same bar.

    NOTE this makes the backtest *consistent with live*, not "more correct": the
    bounded window is itself an approximation of the full-history value (that is the
    price live pays for bounded memory). Align ``window_bars`` with live's
    ``Runtime._compute_frame_limits()`` output for an exact match.
    """
    trimmed = rolling_window_dataset(raw, window_bars)
    prepared = _reprepared_dataset(trimmed)
    return _apply_reference_period_opens(prepared)


# ---------------------------------------------------------------------------
# Rolling-window prepared dataset: live's per-bar window on a FULL-LENGTH frame
# ---------------------------------------------------------------------------
# Rule (user instruction 2026-09-12):
#     value_at_row_t = prep(raw[max(0, t - W + 1) .. t]).iloc[-1]
# Every row carries what the live runtime computes for that bar out of its own
# trailing W-bar window; rows younger than W use their prefix, exactly like a
# young live dataset. The frame is NEVER truncated: a 2-year dataset stays 2
# years long and every bar from row W-1 on is live-identical.
#
# Why the obvious "re-run the whole pipeline per bar" is not used: one 425-bar
# pass costs ~0.2 s (``level_test_buys_with_last_broken`` x ~35 targets is
# ~100 ms of it), i.e. ~15 h for one dataset. Only the WINDOW-SENSITIVE families
# need the per-bar pass; everything else is either row-local or a pure trailing
# rolling window - identical at row t whether it is computed on the full frame
# or on the window - so it is kept from the one vectorized pass:
#
#   per bar   : prepare_df indicators (long-memory EMAs), ema_*/structure_ok,
#               the swing chain (prev_high/prev_low/HH/LL/*_rsi, find_peaks over
#               the window), the causal trend lines, levels/nearest_* and obv
#               (cumulative inside the window)
#   vectorized: the {col}_tested/_is_broken/_spreads_from_close family plus its
#               window-aware {col}_last_broken ("was there a break in the
#               window", see _SPREAD_WARMUP_BARS)
#   kept      : ma*/boll_*/max/min/short_spread/volatility, sqn_*/regime,
#               rs_*/percentiles, rolling_*, the raw OHLCV columns and - through
#               the daily pass below - the *_1daily family
#
# Levels follow live's own routine (``level_finder`` over the bars BEFORE the
# current one, recomputed at every bar) rather than the backtest's 84-bar
# re-clustering step: on the same bars the two algorithms return different
# support/resistance, and live is the reference. ``level_engine="fast"`` keeps
# the old backtest routine for A/B work.
_ROLL_INDICATOR_COLUMNS = (
    "ema7", "ema7_high", "ema25", "ema50", "ema100", "ema200",
    "ma25", "ma50", "ma100", "ma200", "rsi", "max", "min",
    "boll_upper", "boll_mid", "boll_lower", "short_spread", "volatility",
)
_ROLL_STRUCTURE_COLUMNS = ("ema_25", "ema_50", "ema_200", "ema_200_slope", "structure_ok")
_ROLL_SWING_COLUMNS = ("prev_high", "prev_high_rsi", "HH", "higher_rsi",
                       "prev_low", "prev_low_rsi", "LL", "lower_rsi")
_ROLL_TREND_COLUMNS = ("upper_trend", "peak_count", "lower_trend", "trough_count")
_ROLL_EXEC_COLUMNS = tuple(
    _ROLL_INDICATOR_COLUMNS + ("rsi_slope",) + _ROLL_STRUCTURE_COLUMNS
    + _ROLL_SWING_COLUMNS + _ROLL_TREND_COLUMNS
    + ("levels", "nearest_support", "nearest_resistance", "obv")
)
_ROLL_DAILY_COLUMNS = tuple(_ROLL_INDICATOR_COLUMNS + ("rsi_slope",) + _ROLL_TREND_COLUMNS)
_ROLL_RECOMPUTED_COLUMNS = (
    "sqn_30", "sqn_90", "regime", "sqn_30_pct", "sqn_90_pct",
    "rs_ratio", "rs_ratio_ema50", "rs_ratio_mom60", "rs_sqn_90", "rs_ok",
    "btc_sqn_90_pct",
)
_LEVEL_REACTIONS_LIMIT = 12
_LEVEL_START_ROWS = 6 * 14            # live_levels skip_rows: first row it fills
_SPREAD_WARMUP_BARS = 6 * 14 - 1      # first window row whose MA(range, 84) exists
_PERIOD_FREQS = (("Y", "yearly_open"), ("M", "monthly_open"), ("W-SUN", "weekly_open"))
_DAILY_MERGE_SUFFIX = f"_{_DAILY_FRAME}"


def _live_nearest_levels(sorted_levels, price):
    """live_levels._nearest_levels_from_sorted: last level <= price / first above."""
    support = np.nan
    resistance = np.nan
    for level in sorted_levels:
        if level <= price:
            support = level
        elif level > price:
            resistance = level
            break
    return support, resistance


def _swing_last_row(close, rsi):
    """Exactly ``features.add_hh_ll_columns(window).iloc[-1]``, without the merges.

    The swing chain of a window only ever reads the two most recent confirmed
    swing highs / lows at or before the last bar (``shift(1)`` + asof merge +
    ffill), so the last row is computable directly from ``find_peaks``.
    """
    out = {col: np.nan for col in _ROLL_SWING_COLUMNS}
    last = len(close) - 1
    sides = (
        (close, ("prev_high", "prev_high_rsi", "HH", "higher_rsi"), False),
        (-close, ("prev_low", "prev_low_rsi", "LL", "lower_rsi"), True),
    )
    for ref, cols, is_low in sides:
        peaks = find_peaks(ref, distance=6, width=5)[0]
        if len(peaks) == 0:
            continue
        pos = int(np.searchsorted(peaks, last, side="right")) - 1
        if pos < 0:
            continue
        prev_col, prev_rsi_col, flag_col, rsi_flag_col = cols
        if pos >= 1:
            out[prev_col] = float(close[peaks[pos - 1]])
            out[prev_rsi_col] = float(rsi[peaks[pos - 1]])
            if is_low:
                out[flag_col] = bool(close[peaks[pos]] < close[peaks[pos - 1]])
                out[rsi_flag_col] = bool(rsi[peaks[pos]] < rsi[peaks[pos - 1]])
            else:
                out[flag_col] = bool(close[peaks[pos]] > close[peaks[pos - 1]])
                out[rsi_flag_col] = bool(rsi[peaks[pos]] > rsi[peaks[pos - 1]])
        else:
            # first swing of the window: no previous swing -> comparison is False
            out[flag_col] = False
            out[rsi_flag_col] = False
    return out


def _levels_row(window_frame, price, level_engine, levels_reactions_limit):
    """levels / nearest_support / nearest_resistance for the LAST row of a window.

    ``window_frame`` must be the window whose last row is being filled (0-based
    index, like live's own frames): live takes ``frame.iloc[0:idx]`` - the bars
    strictly BEFORE the current one - and compares the levels with the current
    close.
    """
    idx = len(window_frame) - 1
    if idx < _LEVEL_START_ROWS:
        return {"levels": None, "nearest_support": np.nan, "nearest_resistance": np.nan}
    if level_engine == "fast":
        filled = populate_historical_levels_fast(
            ohlc_df=window_frame, levels_reactions_limit=levels_reactions_limit,
            lookback_bars=None, price_col="close", skip_rows=_LEVEL_START_ROWS)
        return {
            "levels": filled["levels"].iloc[-1],
            "nearest_support": filled["nearest_support"].iloc[-1],
            "nearest_resistance": filled["nearest_resistance"].iloc[-1],
        }
    levels = sorted(level_finder(
        window_frame.iloc[0:idx][["close", "high", "low"]], levels_reactions_limit))
    support, resistance = _live_nearest_levels(levels, float(price))
    return {"levels": list(levels), "nearest_support": support, "nearest_resistance": resistance}


def _levels_prefix_row_values(window_raw, t, levels_reactions_limit):
    """Levels for a window younger than W (the frame's prefix) - live's rule with
    ``frame.iloc[0:idx]``, i.e. everything before the current bar."""
    if t < _LEVEL_START_ROWS:
        return {"levels": None, "nearest_support": np.nan, "nearest_resistance": np.nan}
    levels = sorted(level_finder(
        window_raw.iloc[0:t][["close", "high", "low"]], levels_reactions_limit))
    support, resistance = _live_nearest_levels(levels, float(window_raw["close"].iloc[t]))
    return {"levels": list(levels), "nearest_support": support, "nearest_resistance": resistance}


def _window_row_values(window, level_engine, levels_reactions_limit=_LEVEL_REACTIONS_LIMIT):
    """Every window-sensitive execution-frame value for the LAST row of ``window``."""
    frame = prepare_df(window)
    n = len(frame)
    close = pd.to_numeric(frame["close"], errors="coerce").to_numpy(dtype=float)
    rsi = pd.to_numeric(frame["rsi"], errors="coerce").to_numpy(dtype=float)

    out = {col: frame[col].iloc[-1] for col in _ROLL_INDICATOR_COLUMNS}
    out["rsi_slope"] = get_simple_slope_2(rsi[-6:]) if n >= 6 else np.nan

    # structure_ok (gating.add_4h_structure_ok recomputed on the window)
    ema_fast = talib.EMA(close, timeperiod=25)
    ema_mid = talib.EMA(close, timeperiod=50)
    ema_slow = talib.EMA(close, timeperiod=200)
    out["ema_25"] = ema_fast[-1]
    out["ema_50"] = ema_mid[-1]
    out["ema_200"] = ema_slow[-1]
    slope = (ema_slow[-1] - ema_slow[-6]) if n >= 6 else np.nan
    out["ema_200_slope"] = slope
    out["structure_ok"] = bool(
        (close[-1] > ema_slow[-1]) and (ema_fast[-1] > ema_mid[-1])
        and (ema_mid[-1] > ema_slow[-1]) and (slope > 0)
    )

    out.update(_swing_last_row(close, rsi))

    trends = attach_trend_lines(frame, **_TREND_KWARGS)
    for col in _ROLL_TREND_COLUMNS:
        out[col] = trends[col].iloc[-1]

    out.update(_levels_row(frame, close[-1], level_engine, levels_reactions_limit))
    out["obv"] = talib.OBV(
        close, pd.to_numeric(frame["volume"], errors="coerce").to_numpy(dtype=float))[-1]
    return out


def _daily_window_row_values(window, arrays, j):
    """Daily-frame values for the LAST row of ``window`` (indicators + trends)."""
    frame = prepare_df(window)
    rsi = pd.to_numeric(frame["rsi"], errors="coerce").to_numpy(dtype=float)
    for col in _ROLL_INDICATOR_COLUMNS:
        if col in arrays:
            arrays[col][j] = frame[col].iloc[-1]
    arrays["rsi_slope"][j] = get_simple_slope_2(rsi[-6:]) if len(frame) >= 6 else np.nan
    trends = attach_trend_lines(frame, **_TREND_KWARGS)
    for col in _ROLL_TREND_COLUMNS:
        arrays[col][j] = trends[col].iloc[-1]


def _apply_windowed_period_opens(exec_df, raw_daily, window_bars):
    """yearly/monthly/weekly_open exactly as live_prep.add_period_open_columns does.

    live builds the period map from the DAILY window (its last ``window_bars``
    daily bars) and maps each execution bar's OWN period through it. A period
    that has no daily bar inside that window yet - e.g. the first 4h bars of a
    new year, before that year's first daily bar has closed - is NaN.
    """
    exec_time = pd.to_datetime(exec_df["open_time"], errors="coerce")
    exec_close = pd.to_datetime(exec_df["close_time"], errors="coerce").to_numpy()
    daily_time = pd.to_datetime(raw_daily["open_time"], errors="coerce")
    daily_close = pd.to_datetime(raw_daily["close_time"], errors="coerce").to_numpy()
    daily_open = pd.to_numeric(raw_daily["open"], errors="coerce").to_numpy(dtype=float)
    n_daily = len(raw_daily)
    window_start = np.maximum(0, np.arange(n_daily) - window_bars + 1)
    last_daily = np.searchsorted(daily_close, exec_close, side="right") - 1

    for freq, column in _PERIOD_FREQS:
        daily_period = daily_time.dt.to_period(freq)
        codes, uniques = pd.factorize(daily_period)
        first_idx = np.full(len(uniques), -1, dtype=int)
        last_idx = np.full(len(uniques), -1, dtype=int)
        for i, code in enumerate(codes):
            if first_idx[code] < 0:
                first_idx[code] = i
            last_idx[code] = i
        bounds = {period: (int(first_idx[k]), int(last_idx[k]))
                  for k, period in enumerate(uniques)}

        values = np.full(len(exec_df), np.nan)
        exec_period = exec_time.dt.to_period(freq).to_numpy()
        for i in range(len(exec_df)):
            limits = bounds.get(exec_period[i])
            if limits is None:
                continue
            j = int(last_daily[i])
            if j < 0:
                continue
            lo = max(limits[0], int(window_start[j]))
            hi = min(limits[1], j)
            if lo <= hi:
                values[i] = daily_open[lo]
        exec_df[column] = values
    return exec_df


def _column_warmup_bars(frame_df, columns, window_bars):
    """First window row where each target becomes usable, for the ``_last_broken`` rule.

    A break at row r is only visible inside the window ending at t when the
    target was already warm at r, so a break older than
    ``window_bars - 1 - warmup`` bars has to read as NaN - exactly what live's
    own window produces.

    Two flavours of warm-up:
    * columns the pipeline computes from the frame's own leading bars (ema200,
      min, swing/trend columns, levels) start at their own first valid row, and
      that is also the window-relative warm-up;
    * ``*_1daily`` columns are merged from the separately prepared DAILY frame,
      so their leading NaN run in the frame says nothing about a window - what
      matters is how much of the trailing window they cover.
    """
    n = len(frame_df)
    last_window_start = max(n - int(window_bars), 0)
    warmup = {}
    for col in columns:
        if col not in frame_df.columns:
            warmup[col] = 0
            continue
        series = frame_df[col]
        if series.dtype == object:
            warmup[col] = 0
            continue
        first = pd.to_numeric(series, errors="coerce").first_valid_index()
        if first is None:
            warmup[col] = 0
        elif col.endswith(_DAILY_MERGE_SUFFIX):
            warmup[col] = max(0, int(first) - last_window_start)
        else:
            warmup[col] = int(first)
    return warmup


def _rebuild_retest_break_columns(frame_df, columns, window_bars, warmup_by_column):
    """Vectorized ``helpers.level_test_buys_with_last_broken`` for every target.

    ``tested`` / ``is_broken`` / ``spreads_from_close`` are row-local given the
    target column, so they are exact on the full frame. ``last_broken`` is "bars
    since the most recent break", and inside a window a break only counts when
    the target was warm at that bar - hence the per-column recency limit.

    The single-break quirk of the original is reproduced on purpose: with
    exactly ONE break inside the window ``get_ranges`` returns no ranges, so only
    the break row itself is stamped (value 0) and every other row stays NaN.
    live_helpers has the same code, so a window that saw one break behaves the
    same way on both sides.
    """
    close = pd.to_numeric(frame_df["close"], errors="coerce").to_numpy(dtype=float)
    open_ = pd.to_numeric(frame_df["open"], errors="coerce").to_numpy(dtype=float)
    low = pd.to_numeric(frame_df["low"], errors="coerce").to_numpy(dtype=float)
    spread = pd.to_numeric(frame_df["short_spread"], errors="coerce").to_numpy(dtype=float)
    pos = np.arange(len(frame_df), dtype=float)

    for col in columns:
        try:
            target = pd.to_numeric(frame_df[col], errors="coerce").to_numpy(dtype=float)
        except (TypeError, ValueError):
            continue
        if target.size != pos.size:
            continue
        tested = (close > target) & (open_ > target) & (low < target)
        tested &= target != 0.0
        broken = (close > (target + 0.15 * spread)) & (open_ < target) & (low < target)
        max_age = max(window_bars - 1 - max(int(warmup_by_column.get(col, 0)),
                                            _SPREAD_WARMUP_BARS), 0)
        last_true = np.maximum.accumulate(np.where(broken, pos, -1.0))
        age = pos - last_true
        visible = (last_true >= 0) & (age <= max_age)
        # breaks still visible inside the window ending at each row
        cumulative = np.cumsum(broken.astype(float))
        keep = max_age + 1
        if keep >= len(frame_df):
            in_window = cumulative
        else:
            in_window = cumulative - np.concatenate(
                [np.zeros(keep, dtype=float), cumulative[:-keep]])
        last_broken = np.where(
            in_window >= 2, np.where(visible, age, np.nan),
            np.where(broken & (in_window >= 1), 0.0, np.nan))
        frame_df[f"{col}_tested"] = tested.astype(int)
        frame_df[f"{col}_last_broken"] = last_broken
        frame_df[f"{col}_spreads_from_close"] = (target - close) / spread
        frame_df[f"{col}_is_broken"] = broken
    return frame_df


def _rolling_prepared_frames(pair, pair_frames, btc_frames, window_bars, level_engine):
    """Prepared (execution, daily) frames for one pair with live's window semantics."""
    dataset = {pair: {"dict_of_frames": {name: frame for name, frame in pair_frames.items()}}}
    if "BTCUSDT" not in dataset:
        dataset["BTCUSDT"] = {
            "dict_of_frames": {name: frame.copy() for name, frame in btc_frames.items()}}
    base = _reprepared_dataset(dataset)[pair]["dict_of_frames"]

    raw_exec = pair_frames[_EXECUTION_FRAME].sort_values("close_time").reset_index(drop=True)
    raw_daily = pair_frames[_DAILY_FRAME].sort_values("close_time").reset_index(drop=True)
    exec_df = base[_EXECUTION_FRAME].reset_index(drop=True).copy()
    daily_df = base[_DAILY_FRAME].reset_index(drop=True).copy()

    # ---- 1) daily frame: rows younger than a full window are already exact
    n_daily = len(raw_daily)
    if n_daily >= window_bars:
        daily_arrays = {}
        for col in _ROLL_DAILY_COLUMNS:
            if col in daily_df.columns and daily_df[col].dtype != object:
                daily_arrays[col] = daily_df[col].to_numpy(copy=True)
            else:
                daily_arrays[col] = np.full(n_daily, np.nan)
        for j in range(window_bars - 1, n_daily):
            window = raw_daily.iloc[j - window_bars + 1: j + 1].reset_index(drop=True)
            _daily_window_row_values(window, daily_arrays, j)
        for col, values in daily_arrays.items():
            daily_df[col] = values

    # ---- 2) execution frame: every row from its own trailing window
    n_exec = len(raw_exec)
    arrays = {}
    for col in _ROLL_EXEC_COLUMNS:
        if col == "levels":
            arrays[col] = (exec_df[col].to_numpy(dtype=object, copy=True)
                           if col in exec_df.columns else np.full(n_exec, None, dtype=object))
        elif col in exec_df.columns:
            arrays[col] = exec_df[col].to_numpy(copy=True)
        else:
            arrays[col] = np.full(n_exec, np.nan)

    for t in range(n_exec):
        start = t - window_bars + 1
        start = 0 if start < 0 else start
        window = raw_exec.iloc[start: t + 1].reset_index(drop=True)
        if t >= window_bars - 1:
            values = _window_row_values(window, level_engine)
        elif level_engine == "live":
            values = _levels_prefix_row_values(window, t, _LEVEL_REACTIONS_LIMIT)
        else:
            continue            # prefix rows already carry the exact (prefix) values
        for col, value in values.items():
            arrays[col][t] = value
    for col, values in arrays.items():
        exec_df[col] = values

    # ---- 3) period opens through the DAILY window (live's mapping)
    exec_df = _apply_windowed_period_opens(exec_df, raw_daily, window_bars)

    # ---- 4) sqn / regime / relative strength / percentiles on the rolled frames
    drop = [col for col in _ROLL_RECOMPUTED_COLUMNS if col in exec_df.columns]
    exec_df = exec_df.drop(columns=drop)
    drop = [col for col in _ROLL_RECOMPUTED_COLUMNS if col in daily_df.columns]
    daily_df = daily_df.drop(columns=drop)
    cross = {pair: {"dict_of_frames": {_EXECUTION_FRAME: exec_df, _DAILY_FRAME: daily_df}}}
    if pair != "BTCUSDT":
        cross["BTCUSDT"] = {
            "dict_of_frames": {name: frame.copy() for name, frame in btc_frames.items()}}
    cross = attach_sqn_to_dataset(cross, btc_pair="BTCUSDT")
    exec_df = cross[pair]["dict_of_frames"][_EXECUTION_FRAME]
    daily_df = cross[pair]["dict_of_frames"][_DAILY_FRAME]

    # ---- 5) daily -> execution merge (the *_1daily family), as in the full pass
    merge_columns = list(_COLUMNS_FOR_MERGE)
    stale = [f"{col}{_DAILY_MERGE_SUFFIX}" for col in merge_columns
             if f"{col}{_DAILY_MERGE_SUFFIX}" in exec_df.columns]
    if stale:
        exec_df = exec_df.drop(columns=stale)
    exec_df = pd.merge_asof(
        exec_df.sort_values("close_time"),
        daily_df[merge_columns + ["close_time"]].sort_values("close_time"),
        left_on="close_time",
        right_on="close_time",
        direction="backward",
        suffixes=["", _DAILY_MERGE_SUFFIX],
    ).reset_index(drop=True)

    # ---- 6) retest/break family + aux columns, vectorized on the final frame
    targets = [col for col in dict.fromkeys(list(_COLS_FOR_RETEST) + list(_COLS_FOR_BREAK))]
    warmup = _column_warmup_bars(exec_df, targets, window_bars)
    exec_df = _rebuild_retest_break_columns(exec_df, targets, window_bars, warmup)
    exec_df = get_common_aux_columns(
        exec_df, rs_boost_weight=0.5, rs_boost_regime_conditional=True)
    return exec_df, daily_df


def _rolling_pair_task(pair, pair_frames, btc_frames, window_bars, level_engine):
    """Module-level worker so the builder can run under joblib/loky."""
    exec_df, daily_df = _rolling_prepared_frames(
        pair, pair_frames, btc_frames, window_bars, level_engine)
    return {
        pair: {"dict_of_frames": {_EXECUTION_FRAME: exec_df, _DAILY_FRAME: daily_df}}
    }


def rolling_window_prepared_dataset(
    raw,
    window_bars: int = ROLLING_WINDOW_BARS,
    pairs=None,
    jobs: int = 1,
    level_engine: str = "live",
    verbose: bool = True,
):
    """Full-length prepared dataset in which every row is computed from its own
    trailing ``window_bars`` window - live's rule (see the section comment).

    Parameters
    ----------
    raw : dict
        ``dict_of_pairs`` of RAW frames (``data/<name>/raw.pickle``).
    window_bars : int
        Window length; must match live's ``Runtime._compute_frame_limits()``
        (``ROLLING_WINDOW_BARS``), otherwise the values cannot line up with the
        bot. Everything longer than the window is still kept: rows are not cut.
    pairs : iterable, optional
        Restrict to these pairs (default: every pair in ``raw``).
    jobs : int
        Parallel workers (joblib/loky). 1 = in-process loop.
    level_engine : {"live", "fast"}
        "live" reproduces ``live_runtime.live_levels`` (level_finder per bar);
        "fast" keeps ``features.populate_historical_levels_fast``.
    verbose : bool
        Print per-batch progress with an ETA.

    Returns
    -------
    dict
        Same shape as ``_reprepared_dataset`` output (``{pair: {"dict_of_frames": {...}}}``),
        full length, one row per raw bar with per-row windowed values.
    """
    names = [name for name in (list(pairs) if pairs is not None else list(raw)) if name in (raw or {})]
    if not names:
        return {}
    btc_frames = ((raw.get("BTCUSDT") or {}).get("dict_of_frames")) or {}

    workers = max(int(jobs or 1), 1)
    chunk = max(workers * 2, 1)
    prepared = {}
    started = time.time()
    for offset in range(0, len(names), chunk):
        batch = names[offset: offset + chunk]
        tasks = [
            (name, raw[name]["dict_of_frames"], btc_frames, int(window_bars), level_engine)
            for name in batch
        ]
        if workers > 1:
            try:
                from joblib import Parallel, delayed
                outputs = Parallel(n_jobs=workers, backend="loky")(
                    delayed(_rolling_pair_task)(*task) for task in tasks)
            except Exception as exc:                      # pragma: no cover - fallback
                if verbose:
                    print(f"rolling prep: parallel run failed ({exc}); falling back to one job")
                workers = 1
                outputs = [_rolling_pair_task(*task) for task in tasks]
        else:
            outputs = [_rolling_pair_task(*task) for task in tasks]
        for output in outputs:
            prepared.update(output)
        if verbose:
            done = min(offset + chunk, len(names))
            elapsed = time.time() - started
            eta = elapsed / done * (len(names) - done) if done else 0.0
            print(f"rolling prep: {done}/{len(names)} pairs  "
                  f"({elapsed / 60:.1f} min elapsed, eta {eta / 60:.1f} min)")
    return prepared


async def update_dataset_to_latest(
    dataset_name: str,
    warmup_bars: int = 500,
    full_rebuild: bool = False,
    rolling_window: int | None = None,
    jobs: int = 1,
    level_engine: str = "live",
) -> None:
    """Bring one dataset's raw + prepared pickles up to the latest bar.

    Parameters
    ----------
    dataset_name : str
        One of 'halal' | 'complete' | 'validation' (folder under ``data/``).
    warmup_bars : int, default 500
        Bars of raw history regenerated ahead of the new bars so rolling
        indicators (max lookback 252) are correct. Only used in incremental mode.
    full_rebuild : bool, default False
        True = re-run the whole prep pipeline on the full dataset (exact parity,
        slower). False = incremental tail (fast; see the levels caveat on
        ``_reprepared_dataset_incremental``).
    rolling_window : int, optional
        When set, ``prepared`` is rebuilt full-length with EVERY row computed from
        that row's own trailing window (``rolling_window_prepared_dataset``) - the
        backtest counterpart of live's bounded frame cache. Pass
        ``ROLLING_WINDOW_BARS`` (live's own limit) for bar-for-bar parity. The
        stored frame keeps the whole history; rows younger than the window use
        their prefix. This takes precedence over ``full_rebuild``/incremental.
    jobs : int, default 1
        Parallel workers for the rolling rebuild (joblib/loky). 225 pairs x ~1400
        bars is ~90 min on one core, ~15 min on 8.
    level_engine : {"live", "fast"}, default "live"
        Level routine for the rolling rebuild: "live" reproduces
        ``live_runtime.live_levels`` (level_finder per bar), "fast" keeps the
        backtest's 84-bar re-clustering.

    Loads ``data/<name>/raw`` and ``data/<name>/prepared``. If any frame's last
    stored bar is older than one cadence, downloads the missing rows from Binance
    (public klines, throwaway client) and appends them to the raw frames. Then
    refreshes ``prepared`` (full rebuild or incremental tail) and stores BOTH
    pickles back. Returns None; all data stays local to this function, so the
    frames are released from memory when it returns.

    Usage
    -----
    Notebook (kernel already has an event loop): ``await update_dataset_to_latest("halal")``
    Script: ``asyncio.run(update_dataset_to_latest("halal"))``
    """
    if dataset_name not in _DATASET_DIRS:
        raise ValueError(
            f"unknown dataset {dataset_name!r}; expected one of {list(_DATASET_DIRS)}")

    data_dir = _DATASET_DIRS[dataset_name]
    raw_path = f"{data_dir}/raw"
    prepared_path = f"{data_dir}/prepared"

    raw = read_pickle_object(raw_path)
    if raw is False or raw is None:
        raise RuntimeError(f"could not read {raw_path}.pickle")

    # ---- 1) bring raw up to the latest bar --------------------------------
    stale = _stale_entries(raw)
    if stale:
        print(f"{dataset_name}: {len(stale)} stale (pair, frame) entries; downloading...")
        client = await AsyncClient.create(requests_params={"timeout": 360})
        try:
            updated = await _fetch_missing(client, raw, stale)
        finally:
            await client.close_connection()
        if not updated:
            print(f"{dataset_name}: no new bars fetched")
    else:
        updated = False
        print(f"{dataset_name}: raw already up to date")

    if updated:
        if not store_pickle_object(raw_path, raw):
            raise RuntimeError(f"failed to store {raw_path}.pickle")
        print(f"stored updated raw -> {raw_path}.pickle")

    # ---- 2) refresh prepared (full rebuild or incremental tail) -----------
    prepared = read_pickle_object(prepared_path)
    if prepared is False or prepared is None:
        prepared = None

    stored_rolling = _read_rolling_marker(prepared_path) if prepared is not None else None
    if rolling_window is None and stored_rolling:
        print(f"prepared: the stored frame was built with the rolling-window prep "
              f"({stored_rolling} bars per row); an incremental tail would splice "
              f"full-history rows onto it, so rebuilding the rolling frame instead")
        rolling_window = stored_rolling

    if rolling_window is not None:
        # Rolling prep: full-length frame, every row from its own trailing window
        # (live recomputes exactly that window at every bar close).
        prepared = rolling_window_prepared_dataset(
            raw, int(rolling_window), jobs=jobs, level_engine=level_engine)
        print(f"prepared: rolling-window rebuild ({int(rolling_window)} bars per row, "
              f"{jobs} job(s), level_engine={level_engine!r})")
    elif full_rebuild or prepared is None:
        prepared = _reprepared_dataset(raw)
        print(f"prepared: full rebuild")
    else:
        missing = missing_prepared_columns(prepared)
        if missing:
            # The stored pickle predates a column family (e.g. the trend lines).
            # An incremental tail would only fill the NEW rows, so rebuild all of
            # it — the incremental path also reports/rebuilds this itself.
            prepared = _reprepared_dataset(raw)
            print(f"prepared: full rebuild (schema incomplete for "
                  f"{len(missing)} (pair, frame) entries; "
                  f"first gap: {missing[0][0]} [{missing[0][1]}] "
                  f"missing {', '.join(missing[0][2][:4])})")
        else:
            prepared = _reprepared_dataset_incremental(raw, prepared, warmup_bars=warmup_bars)
            print(f"prepared: incremental tail ({warmup_bars} warmup bars)")

    if not store_pickle_object(prepared_path, prepared):
        raise RuntimeError(f"failed to store {prepared_path}.pickle")
    _write_rolling_marker(prepared_path, rolling_window)
    print(f"stored prepared -> {prepared_path}.pickle")

    # nothing returned; local refs drop on exit -> memory freed