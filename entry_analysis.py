"""Entry-attribution analysis for the backtest (Phase A).

Builds a dataset of every entry that passed the current gates, labels each by
its forward path (MFE / MAE / return at several horizons), and produces
descriptive summaries to separate winners from duds *within the passing
population* (per the ChatGPT - Algotrading.pdf plan).

Backtest-only: reads a compiled backtest payload (backtest_prep.compile_backtest_payload)
and optionally the engine trade_frame. Never touches live_runtime/ or validation.py,
and never modifies trading logic -- it is a read-only analysis layer.

Dataset columns (one row per entry event):
    pair, strategy, regime, bar, time, entry_close,
    rank, stop_price, qvpuc,
    mfe_{h}, mae_{h}, ret_{h}          (percent) for h in horizons,
    bars_to_{pct}                       (bars)  for pct in pct_targets,
    max_favorable, max_adverse         (percent),
    outcome                             (immediate_winner / delayed_winner /
                                         immediate_dud / mediocre),
    was_taken, exit_close_reason, exit_pnl, exit_return_pct, exit_hold_bars,
    exit_time
                                        (only when trade_frame is provided).
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

# Date bounds are parsed by the backtest engine so the entry-analysis window is
# bit-for-bit the same window the engine traded (DD/MM/YYYY or DD/MM/YYYY HH:MM).
from backtest_engine import _build_date_mask as _engine_date_mask

DEFAULT_HORIZONS = (3, 6, 12, 24, 48)
DEFAULT_PCT_TARGETS = (1.0, 2.0, 8.0)


# ---------------------------------------------------------------------------
# Vectorized forward-path helpers (per pair column, NaN-aware)
# ---------------------------------------------------------------------------

def _forward_extreme(x: np.ndarray, h: int, mode: str) -> np.ndarray:
    """Return out[t] = max/min of x[t+1 .. t+h] (NaN-aware); NaN if no data."""
    x = np.asarray(x, dtype=np.float64)
    T = len(x)
    out = np.full(T, np.nan)
    if T <= 1:
        return out
    h = int(min(h, T - 1))
    if h <= 0:
        return out

    sentinel = -np.inf if mode == "max" else np.inf
    xf = np.where(np.isnan(x), sentinel, x).astype(np.float64)

    if mode == "max":
        rev_cum = np.maximum.accumulate(xf[::-1])[::-1]
        win_ext = sliding_window_view(xf, h).max(axis=1)
    else:
        rev_cum = np.minimum.accumulate(xf[::-1])[::-1]
        win_ext = sliding_window_view(xf, h).min(axis=1)

    # out[t] = extreme over x[t+1 .. t+h] for t in [0, T-h-1]
    out[: T - h] = win_ext[1:]
    # Tail (fewer than h bars remain): extreme over x[t+1 ..] via reversed cum
    for t in range(T - h, T - 1):
        out[t] = rev_cum[t + 1]
    # sentinel (-inf/+inf) means all-NaN window -> NaN
    out[np.isinf(out)] = np.nan
    return out


def _forward_return(close: np.ndarray, h: int) -> np.ndarray:
    """Return out[t] = close[t+h] / close[t] - 1 (fraction); NaN near the tail."""
    close = np.asarray(close, dtype=np.float64)
    T = len(close)
    out = np.full(T, np.nan)
    if T <= h:
        return out
    out[: T - h] = close[h:] / close[: T - h] - 1.0
    return out


def _bars_to_target(high: np.ndarray, close: np.ndarray, pct: float, h: int) -> np.ndarray:
    """out[t] = first k in [1..h] with high[t+k] >= close[t]*(1+pct/100); else NaN."""
    high = np.asarray(high, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    T = len(high)
    out = np.full(T, np.nan)
    h = int(min(h, T - 1))
    if h <= 0:
        return out

    xf = np.where(np.isnan(high), -np.inf, high).astype(np.float64)
    thresh = close * (1.0 + pct / 100.0)

    # Full h-bar windows for entries that still have one ahead.
    n_full = T - h
    if n_full > 0:
        windows = sliding_window_view(xf, h)          # windows[i] = high[i .. i+h-1]
        starts = np.arange(1, n_full + 1)             # entry at bar t -> window start t+1
        sub_windows = windows[starts]                 # (n_full, h)
        hit = sub_windows >= thresh[:n_full, None]
        any_hit = hit.any(axis=1)
        first = np.argmax(hit, axis=1)
        out[:n_full] = np.where(any_hit, first + 1, np.nan)

    # Tail: fewer than h bars remain; at most h iterations per column.
    for t in range(max(n_full, 0), T - 1):
        fut = xf[t + 1:]
        idx = np.flatnonzero(fut >= thresh[t])
        if idx.size:
            out[t] = idx[0] + 1
    return out


# ---------------------------------------------------------------------------
# Entry dataset construction
# ---------------------------------------------------------------------------

def _build_date_mask(timeline, start_date: Optional[str], end_date: Optional[str]) -> Optional[np.ndarray]:
    """Optional date window mask over the payload timeline (identical to the engine's).

    Accepts ``DD/MM/YYYY`` (whole day, inclusive) or ``DD/MM/YYYY HH:MM`` (edge
    pinned to that exact timestamp). The engine owns the parsing rules so the
    entry dataset lines up with the bars the backtest actually traded.
    """
    if start_date is None and end_date is None:
        return None
    return _engine_date_mask(pd.Series(timeline), start_date, end_date)


def build_entry_analysis_dataset(
    payload: dict,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    pct_targets: tuple[float, ...] = DEFAULT_PCT_TARGETS,
    target_horizon: int = 48,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    trade_frame: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """One row per gated entry (payload strategies are already gate-masked).

    Every column of the payload is a (T, N) array on a shared timeline, so the
    forward path of each entry is read straight off the high/low/close arrays.
    """
    T, N = payload["shape"]["T"], payload["shape"]["N"]
    timeline = payload["timeline"]
    pairs = payload["pairs"]
    close_arr = np.asarray(payload["prices"]["close"], dtype=np.float64)
    high_arr = np.asarray(payload["prices"]["high"], dtype=np.float64)
    low_arr = np.asarray(payload["prices"]["low"], dtype=np.float64)
    context = payload.get("context", {})
    regime_arr = context.get("regime")
    aux = payload.get("aux", {})

    date_mask = _build_date_mask(timeline, start_date, end_date)

    # Per-column caches so each forward array is computed once, not per entry.
    extreme_cache: dict[tuple[int, int, str], np.ndarray] = {}
    ret_cache: dict[tuple[int, int], np.ndarray] = {}
    target_cache: dict[tuple[int, float], np.ndarray] = {}

    def _extreme(n: int, h: int, mode: str) -> np.ndarray:
        key = (n, h, mode)
        if key not in extreme_cache:
            source = high_arr[:, n] if mode == "max" else low_arr[:, n]
            extreme_cache[key] = _forward_extreme(source, h, mode)
        return extreme_cache[key]

    def _ret(n: int, h: int) -> np.ndarray:
        key = (n, h)
        if key not in ret_cache:
            ret_cache[key] = _forward_return(close_arr[:, n], h)
        return ret_cache[key]

    def _target(n: int, pct: float) -> np.ndarray:
        key = (n, pct)
        if key not in target_cache:
            target_cache[key] = _bars_to_target(high_arr[:, n], close_arr[:, n], pct, target_horizon)
        return target_cache[key]

    rows: list[dict[str, Any]] = []

    for strat_name, signals in payload["strategies"].items():
        entry = signals.get("entry")
        if entry is None or entry.ndim != 2:
            continue
        ts, ns = np.nonzero(entry)

        strat_aux = aux.get(strat_name, {}) if isinstance(aux, dict) and any(
            isinstance(v, dict) for v in aux.values()
        ) else aux
        rank_arr = strat_aux.get("rank_score")
        stop_arr = strat_aux.get("stop_price")
        qvpuc_arr = strat_aux.get("quote_vol_per_unit_change")

        for t, n in zip(ts, ns):
            if date_mask is not None and not date_mask[t]:
                continue
            entry_close = close_arr[t, n]
            if not np.isfinite(entry_close) or entry_close <= 0:
                continue

            row: dict[str, Any] = {
                "pair": pairs[n],
                "strategy": strat_name,
                "bar": int(t),
                "time": timeline[t],
                "entry_close": float(entry_close),
            }

            if regime_arr is not None:
                row["regime"] = regime_arr[t, n]
            else:
                row["regime"] = None

            for col_name, arr in (("rank", rank_arr), ("stop_price", stop_arr), ("qvpuc", qvpuc_arr)):
                if arr is not None:
                    v = arr[t, n]
                    row[col_name] = float(v) if np.isfinite(v) else np.nan
                else:
                    row[col_name] = np.nan

            for h in horizons:
                mfe = _extreme(n, h, "max")[t]
                mae = _extreme(n, h, "min")[t]
                row[f"mfe_{h}"] = (mfe / entry_close - 1.0) * 100.0 if np.isfinite(mfe) else np.nan
                row[f"mae_{h}"] = (mae / entry_close - 1.0) * 100.0 if np.isfinite(mae) else np.nan
                row[f"ret_{h}"] = float(_ret(n, h)[t] * 100.0)

            for pct in pct_targets:
                b = _target(n, pct)[t]
                row[f"bars_to_{int(pct)}"] = float(b) if np.isfinite(b) else np.nan

            rows.append(row)

    df = pd.DataFrame(rows)

    if not df.empty:
        mfe_cols = [f"mfe_{h}" for h in horizons]
        mae_cols = [f"mae_{h}" for h in horizons]
        df["max_favorable"] = df[mfe_cols].max(axis=1)
        df["max_adverse"] = df[mae_cols].min(axis=1)
        df = classify_outcomes(df, horizons=horizons)

    if trade_frame is not None and not trade_frame.empty and not df.empty:
        df = _attach_exit_attribution(df, trade_frame)

    return df


def _attach_exit_attribution(df: pd.DataFrame, trade_frame: pd.DataFrame) -> pd.DataFrame:
    """Merge actual-trade outcome (was it taken, exit reason, pnl, hold bars).

    Matches engine trade_frame rows on (pair, strategy, entry_time). This is the
    exit-route attribution that separates "bad entry" from "exit killed it".
    """
    df = df.copy()
    tf = trade_frame.copy()
    tf["entry_time"] = pd.to_datetime(tf["entry_time"], errors="coerce")
    tf = tf.dropna(subset=["entry_time"])

    tf_key = tf.set_index(["pair", "strategy", "entry_time"])
    tf_key = tf_key[~tf_key.index.duplicated(keep="first")]

    df["_time_dt"] = pd.to_datetime(df["time"], errors="coerce")
    keys = list(zip(df["pair"], df["strategy"], df["_time_dt"]))
    joined = tf_key.reindex(keys).reset_index(drop=True)

    df["was_taken"] = joined["entry_bar"].notna().to_numpy()
    df["exit_close_reason"] = joined["close_reason"].to_numpy()
    df["exit_pnl"] = pd.to_numeric(joined["pnl"], errors="coerce").to_numpy()
    df["exit_return_pct"] = pd.to_numeric(joined["return_pct"], errors="coerce").to_numpy()
    df["exit_hold_bars"] = pd.to_numeric(joined["hold_bars"], errors="coerce").to_numpy()
    df["exit_price"] = pd.to_numeric(joined["exit_price"], errors="coerce").to_numpy()
    if "exit_time" in joined.columns:
        df["exit_time"] = pd.to_datetime(joined["exit_time"], errors="coerce").to_numpy()
    df = df.drop(columns=["_time_dt"])
    return df


# ---------------------------------------------------------------------------
# Outcome classification
# ---------------------------------------------------------------------------

def classify_outcomes(
    df: pd.DataFrame,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    immediate_winner_mfe: float = 3.5,
    delayed_winner_mfe: float = 10.0,
    immediate_dud_mae: float = -3.5,
) -> pd.DataFrame:
    """Label each entry by its forward path (not just eventual P&L).

    - immediate_dud   : MAE within 6 bars drops to <= immediate_dud_mae
    - immediate_winner: MFE within 6 bars reaches >= immediate_winner_mfe
    - delayed_winner  : 6-bar MFE small, but 24-bar MFE reaches delayed_winner_mfe
    - mediocre        : everything else
    """
    df = df.copy()
    h6 = 6 if 6 in horizons else horizons[0]
    h24 = 24 if 24 in horizons else horizons[-1]
    mfe6 = pd.to_numeric(df.get(f"mfe_{h6}"), errors="coerce")
    mae6 = pd.to_numeric(df.get(f"mae_{h6}"), errors="coerce")
    mfe24 = pd.to_numeric(df.get(f"mfe_{h24}"), errors="coerce")

    out = np.full(len(df), "mediocre", dtype=object)
    out[(mae6 <= immediate_dud_mae).to_numpy()] = "immediate_dud"
    out[(mfe6 >= immediate_winner_mfe).to_numpy()] = "immediate_winner"
    out[((mfe6 < immediate_winner_mfe) & (mfe24 >= delayed_winner_mfe)).to_numpy()] = "delayed_winner"
    df["outcome"] = out
    return df


# ---------------------------------------------------------------------------
# Descriptive summaries
# ---------------------------------------------------------------------------

def summarize_by_strategy(
    df: pd.DataFrame,
    target_pct: float = 1.0,
    within: int = 12,
) -> pd.DataFrame:
    """Per-strategy matrix: entries, hit rate, MFE6/MAE6, ret12, rank, outcome shares."""
    if df.empty:
        return pd.DataFrame()
    work = df.copy()
    work["hit"] = work[f"bars_to_{int(target_pct)}"] <= within

    g = work.groupby("strategy", dropna=False)
    out = g.agg(
        entries=("pair", "size"),
        hit_pct=("hit", "mean"),
        mean_mfe6=("mfe_6", "mean"),
        mean_mae6=("mae_6", "mean"),
        mean_ret12=("ret_12", "mean"),
        mean_rank=("rank", "mean"),
        immed_dud_share=("outcome", lambda s: (s == "immediate_dud").mean()),
        winner_share=("outcome", lambda s: s.isin(["immediate_winner", "delayed_winner"]).mean()),
    )
    out["hit_pct"] *= 100.0
    out["immed_dud_share"] *= 100.0
    out["winner_share"] *= 100.0
    return out.sort_values("entries", ascending=False)


def summarize_by_strategy_regime(
    df: pd.DataFrame,
    target_pct: float = 1.0,
    within: int = 12,
) -> pd.DataFrame:
    """Strategy x regime matrix (GPT Phase 6): a strategy may be good in one
    regime and terrible in another even though the overall numbers look fine."""
    if df.empty:
        return pd.DataFrame()
    work = df.copy()
    work["hit"] = work[f"bars_to_{int(target_pct)}"] <= within

    g = work.groupby(["strategy", "regime"], dropna=False)
    out = g.agg(
        entries=("pair", "size"),
        hit_pct=("hit", "mean"),
        mean_mfe6=("mfe_6", "mean"),
        mean_mae6=("mae_6", "mean"),
        mean_ret12=("ret_12", "mean"),
        mean_rank=("rank", "mean"),
        immed_dud_share=("outcome", lambda s: (s == "immediate_dud").mean()),
        winner_share=("outcome", lambda s: s.isin(["immediate_winner", "delayed_winner"]).mean()),
    )
    out["hit_pct"] *= 100.0
    out["immed_dud_share"] *= 100.0
    out["winner_share"] *= 100.0
    return out.sort_values(["strategy", "entries"], ascending=[True, False])


def summarize_rank_vs_expectancy(
    df: pd.DataFrame,
    q: int = 10,
    target_pct: float = 1.0,
    within: int = 12,
) -> pd.DataFrame:
    """Does the rank score actually predict forward quality? (GPT Phase 7.)

    Buckets entries into q rank quantiles; if rank 1 consistently beats rank q,
    the ranking is predictive of quality, not just useful for selection.
    """
    work = df.dropna(subset=["rank"]).copy()
    if work.empty:
        return pd.DataFrame()
    work["rank_bucket"] = pd.qcut(work["rank"], q=q, labels=False, duplicates="drop")
    work["hit"] = work[f"bars_to_{int(target_pct)}"] <= within

    g = work.groupby("rank_bucket")
    out = g.agg(
        entries=("pair", "size"),
        mean_mfe6=("mfe_6", "mean"),
        mean_mae6=("mae_6", "mean"),
        mean_ret12=("ret_12", "mean"),
        hit_pct=("hit", "mean"),
        winner_share=("outcome", lambda s: s.isin(["immediate_winner", "delayed_winner"]).mean()),
    )
    out["hit_pct"] *= 100.0
    out["winner_share"] *= 100.0
    out.index.name = "rank_quantile"
    return out


def summarize_outcome_by_exit(df: pd.DataFrame) -> pd.DataFrame:
    """For traded entries, cross-tab forward-path outcome x exit reason.

    Direct answer to "what causes the timeouts": a time_stop trade whose MFE6
    was large is a round-trip the exit machinery failed to capture (exit
    problem), while a time_stop with tiny MFE6 is a genuine dud entry.
    """
    if df.empty or "was_taken" not in df.columns:
        return pd.DataFrame()
    work = df[df["was_taken"]].copy()
    if work.empty:
        return pd.DataFrame()
    g = work.groupby(["outcome", "exit_close_reason"], dropna=False)
    out = g.agg(
        trades=("exit_pnl", "size"),
        mean_pnl=("exit_pnl", "mean"),
        mean_ret_pct=("exit_return_pct", "mean"),
        mean_hold_bars=("exit_hold_bars", "mean"),
        mean_mfe6=("mfe_6", "mean"),
        mean_mae6=("mae_6", "mean"),
        mean_ret12=("ret_12", "mean"),
    )
    return out.sort_values("trades", ascending=False)


def timeout_round_trip_analysis(df: pd.DataFrame, mfe_threshold: float = 3.0) -> Optional[dict]:
    """Partition time_stop trades into round-trips vs genuine duds.

    A timeout whose MFE6 >= mfe_threshold reached a meaningful favorable
    excursion the exit never captured (round-trip / exit problem). A timeout
    with MFE6 < mfe_threshold barely moved (genuine dud entry).
    """
    if df.empty or "was_taken" not in df.columns:
        return None
    work = df[(df["was_taken"]) & (df["exit_close_reason"] == "time_stop")].copy()
    if work.empty:
        return None
    n = len(work)
    n_round = int((work["mfe_6"] >= mfe_threshold).sum())
    return {
        "timeout_trades": n,
        "mfe_threshold": mfe_threshold,
        "round_trip_trades": n_round,
        "round_trip_pct": 100.0 * n_round / n,
        "genuine_dud_trades": int(n - n_round),
        "genuine_dud_pct": 100.0 * (n - n_round) / n,
        "mean_mfe6": float(work["mfe_6"].mean()),
        "median_mfe6": float(work["mfe_6"].median()),
        "mean_ret_pct_timeout": float(work["exit_return_pct"].mean()),
    }


def trailing_arm_sweep(
    df: pd.DataFrame,
    arm_levels: tuple[float, ...] = (4.0, 6.0, 8.0, 12.0, 15.0),
    peak_col: str = "max_favorable",
) -> pd.DataFrame:
    """How late is the current +15% trailing-arm threshold for your timeouts?

    For traded entries, the share of each exit class whose peak favorable
    excursion (peak_col, default = max over 48 bars) reached at least L%. A
    trailing stop armed at +L% can only ever capture trades that peaked at or
    above L, so this shows how much of the timeout population an earlier arm
    (e.g. 4-8%) could have captured before it rounded back down.
    """
    if df.empty or "was_taken" not in df.columns or peak_col not in df.columns:
        return pd.DataFrame()
    work = df[df["was_taken"]].copy()
    if work.empty:
        return pd.DataFrame()

    groups = {"all_traded": work, "time_stop": work[work["exit_close_reason"] == "time_stop"]}
    rows = []
    for label, sub in groups.items():
        if sub.empty:
            continue
        peak = pd.to_numeric(sub[peak_col], errors="coerce").dropna()
        row = {
            "population": label,
            "trades": len(sub),
            "median_peak_pct": float(peak.median()) if len(peak) else np.nan,
            "mean_peak_pct": float(peak.mean()) if len(peak) else np.nan,
        }
        for L in arm_levels:
            row[f"reached_{L:g}pct"] = 100.0 * (peak >= L).mean() if len(peak) else np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("population")


def mae_time_path(
    df: pd.DataFrame,
    label_col: str = "outcome",
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
) -> pd.DataFrame:
    """Adverse-excursion time path by outcome class.

    Tests the MAE-before-MFE hypothesis: if winners dip early then recover
    (deep mae_6 but mae_48 ~= mae_6, large mfe) while duds keep grinding down
    (mae keeps deepening, small mfe), then a hard stop below entry kills the
    dip-then-rise winners, while a recovery-based exit would not.
    """
    if df.empty or label_col not in df.columns:
        return pd.DataFrame()
    work = df.copy()
    agg: dict = {"trades": ("pair", "size")}
    for h in horizons:
        agg[f"mean_mae_{h}"] = (f"mae_{h}", "mean")
        agg[f"median_mae_{h}"] = (f"mae_{h}", "median")
    for h in (6, 24, 48):
        agg[f"mean_mfe_{h}"] = (f"mfe_{h}", "mean")
    if "max_adverse" in work.columns:
        agg["dip_depth_median"] = ("max_adverse", "median")

    g = work.groupby(label_col, dropna=False)
    # kwargs form required: this pandas version rejects the dict form of
    # named aggregation ({new: (col, func)}), but accepts **agg.
    out = g.agg(**agg)
    # more negative = kept grinding down after the first 6 bars
    out["addl_adverse_6_to_48_mean"] = out["mean_mae_48"] - out["mean_mae_6"]
    return out


def classify_outcomes_percentile(
    df: pd.DataFrame,
    winner_pct: float = 0.90,
    dud_pct: float = 0.10,
) -> pd.DataFrame:
    """Relative outcome labels within the passing population.

    The absolute-threshold classifier (classify_outcomes) is useless when the
    market is very volatile -- everything looks like a winner. This one labels:
      - immediate_winner : MFE6 in the top (1 - winner_pct) quantile
      - immediate_dud    : MAE6 in the bottom dud_pct quantile (most adverse)
      - mediocre         : everyone else
    Best used to build a clean winner-vs-dud contrast for discriminator
    analysis (Phase B), not as a hard trading rule.
    """
    df = df.copy()
    mfe6 = pd.to_numeric(df.get("mfe_6"), errors="coerce")
    mae6 = pd.to_numeric(df.get("mae_6"), errors="coerce")
    out = np.full(len(df), "mediocre", dtype=object)
    if mae6 is not None and mae6.notna().any():
        dud_thr = mae6.quantile(dud_pct)
        out[(mae6 <= dud_thr).to_numpy()] = "immediate_dud"
    if mfe6 is not None and mfe6.notna().any():
        win_thr = mfe6.quantile(winner_pct)
        out[(mfe6 >= win_thr).to_numpy()] = "immediate_winner"
    df["outcome"] = out
    return df


def mfe_mae_profile(
    df: pd.DataFrame,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    pct_targets: tuple[float, ...] = DEFAULT_PCT_TARGETS,
):
    """Mean/median MFE and MAE at each horizon, plus % reaching each profit target."""
    if df.empty:
        return pd.DataFrame(), {}
    rows = []
    for h in horizons:
        rows.append({
            "horizon": h,
            "mean_mfe": df[f"mfe_{h}"].mean(),
            "median_mfe": df[f"mfe_{h}"].median(),
            "mean_mae": df[f"mae_{h}"].mean(),
            "median_mae": df[f"mae_{h}"].median(),
            "mean_ret": df[f"ret_{h}"].mean(),
        })
    profile = pd.DataFrame(rows).set_index("horizon")
    targets = {
        f"pct_to_{int(p)}": df[f"bars_to_{int(p)}"].notna().mean() * 100.0
        for p in pct_targets
    }
    return profile, targets


def entry_analysis_report(
    df: pd.DataFrame,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    pct_targets: tuple[float, ...] = DEFAULT_PCT_TARGETS,
) -> dict:
    """Print the Phase A report and return the tables in a dict."""
    print(f"ENTRY ATTRIBUTION REPORT — {len(df)} gated entries")
    print("=" * 70)

    if df.empty:
        print("No entries in the dataset.")
        return {}

    outcome_dist = df["outcome"].value_counts()
    print("\nOutcome distribution (by forward path, not just P&L):")
    for label in ["immediate_winner", "delayed_winner", "mediocre", "immediate_dud"]:
        n = int(outcome_dist.get(label, 0))
        print(f"   {label:18s}: {n:6d}  ({100.0 * n / len(df):5.1f}%)")

    profile, targets = mfe_mae_profile(df, horizons=horizons, pct_targets=pct_targets)
    print("\nMFE / MAE profile (percent, over N 4h bars):")
    print(profile.round(2).to_string())
    print(f"\nShare reaching profit targets within {max(horizons)} bars:")
    for k, v in targets.items():
        print(f"   {k:10s}: {v:5.1f}%")

    print("\nBy strategy:")
    by_strategy = summarize_by_strategy(df)
    print(by_strategy.round(2).to_string() if not by_strategy.empty else "   (empty)")

    print("\nBy strategy x regime (watch for good-here/bad-there combos):")
    by_strat_regime = summarize_by_strategy_regime(df)
    print(by_strat_regime.round(2).to_string() if not by_strat_regime.empty else "   (empty)")

    print("\nRank vs forward expectancy (does rank predict quality?):")
    rank_tab = summarize_rank_vs_expectancy(df)
    print(rank_tab.round(2).to_string() if not rank_tab.empty else "   (empty)")

    if "was_taken" in df.columns:
        taken = df[df["was_taken"]]
        print(f"\nExit-route attribution ({len(taken)} entries actually traded):")
        if not taken.empty:
            exit_tab = taken.groupby("exit_close_reason").agg(
                trades=("exit_pnl", "size"),
                mean_pnl=("exit_pnl", "mean"),
                mean_ret_pct=("exit_return_pct", "mean"),
                mean_hold_bars=("exit_hold_bars", "mean"),
            )
            print(exit_tab.round(2).to_string())
        print(f"Entries that passed gates but never became trades: {int((~df['was_taken']).sum())}")

    return {
        "outcome_distribution": outcome_dist,
        "mfe_mae_profile": profile,
        "profit_targets": targets,
        "by_strategy": by_strategy,
        "by_strategy_regime": by_strat_regime,
        "rank_vs_expectancy": rank_tab,
    }


# ---------------------------------------------------------------------------
# Phase B: pre-entry feature snapshots + volume run-up analysis
# ---------------------------------------------------------------------------

def _snapshot_by_time(
    df: pd.DataFrame,
    dict_of_pairs: dict,
    execution_frame: str,
    columns: list[str],
    time_col: str,
) -> pd.DataFrame:
    """Pull frame columns at each row's ``time_col`` (matched by close_time).

    Shared engine for both the entry-time snapshot (time_col="time") and the
    TP-hit-time snapshot (time_col="tp_hit_time"). Columns not present on a
    pair's frame are left NaN; unmatched timestamps are NaN.
    """
    df = df.copy()
    if df.empty or time_col not in df.columns:
        return df
    for col in columns:
        df[col] = np.nan
    for pair, sub in df.groupby("pair"):
        pdata = dict_of_pairs.get(pair)
        if not pdata:
            continue
        frame = pdata.get("dict_of_frames", {}).get(execution_frame)
        if frame is None or frame.empty or "close_time" not in frame.columns:
            continue
        f = frame.sort_values("close_time").reset_index(drop=True)
        close_time = pd.to_datetime(f["close_time"], errors="coerce").to_numpy()
        times = pd.to_datetime(sub[time_col], errors="coerce").to_numpy()
        pos = np.searchsorted(close_time, times)
        pos = np.clip(pos, 0, len(f) - 1)
        exact = close_time[pos] == times
        for col in columns:
            if col not in f.columns:
                continue
            vals = pd.to_numeric(f[col], errors="coerce").to_numpy(dtype=float)
            df.loc[sub.index, col] = np.where(exact, vals[pos], np.nan)
    return df


def attach_snapshot_features(
    df: pd.DataFrame,
    dict_of_pairs: dict,
    execution_frame: str,
    columns: list[str],
) -> pd.DataFrame:
    """Attach pre-entry snapshot values of the given frame columns to each entry.

    For every entry (pair, time) the execution-frame row at that close_time is
    found and the requested columns are pulled (e.g. obv, volume, volatility,
    rolling_gain, rsi_slope). Columns not present on a pair's frame are left
    NaN. Phase B foundation for discriminator analysis.
    """
    return _snapshot_by_time(df, dict_of_pairs, execution_frame, columns, time_col="time")


def attach_snapshot_at_tp(
    df: pd.DataFrame,
    dict_of_pairs: dict,
    execution_frame: str,
    columns: list[str],
    offsets: tuple[int, ...] = (0, -1),
) -> pd.DataFrame:
    """Snapshot frame columns AT the TP-hit bar (and bars around it).

    Answers the dynamic-TP question -- "if these conditions are met when the TP
    is reached (or a bar earlier, since the TP fill is intrabar) the trade is
    better than it seems -- hold; otherwise take profit" -- by capturing each
    column at the TP-hit bar and the completed bar(s) before it instead of at
    entry. Requires ``tp_hit_time`` from ``attach_take_profit_features``.

    Adds ``{col}@at_tp`` (offset 0 = the TP-hit bar) and ``{col}@at_tp_prev``
    (offset -1 = the bar before it). Any other offset gets ``{col}@at_tp{+/-n}``.
    Entries that never hit the TP stay NaN.
    """
    df = df.copy()
    if df.empty or "tp_hit_time" not in df.columns:
        return df

    def _suffix(o: int) -> str:
        return {0: "at_tp", -1: "at_tp_prev"}.get(o, f"at_tp{o:+d}")

    for col in columns:
        for o in offsets:
            df[f"{col}@{_suffix(o)}"] = np.nan

    for pair, sub in df.groupby("pair"):
        pdata = dict_of_pairs.get(pair)
        if not pdata:
            continue
        frame = pdata.get("dict_of_frames", {}).get(execution_frame)
        if frame is None or frame.empty or "close_time" not in frame.columns:
            continue
        f = frame.sort_values("close_time").reset_index(drop=True)
        close_time = pd.to_datetime(f["close_time"], errors="coerce").to_numpy()
        hit_times = pd.to_datetime(sub["tp_hit_time"], errors="coerce").to_numpy()
        pos = np.searchsorted(close_time, hit_times)
        pos = np.clip(pos, 0, len(f) - 1)
        exact = close_time[pos] == hit_times
        n = len(f)
        for col in columns:
            if col not in f.columns:
                continue
            vals = pd.to_numeric(f[col], errors="coerce").to_numpy(dtype=float)
            for o in offsets:
                tgt = pos + o
                ok = exact & (tgt >= 0) & (tgt < n)
                tgt_c = np.clip(tgt, 0, n - 1)
                df.loc[sub.index, f"{col}@{_suffix(o)}"] = np.where(ok, vals[tgt_c], np.nan)
    return df


def attach_volume_ratio_at_tp(
    df: pd.DataFrame,
    dict_of_pairs: dict,
    execution_frame: str,
    baseline_window: int = 20,
) -> pd.DataFrame:
    """Volume at / just before the TP hit, normalized against its recent norm.

    Raw volume is not comparable across tokens, so normalize it:
      vol_ratio_at_tp      = volume at the TP-hit bar / mean volume over the prior
                             ``baseline_window`` bars
      vol_ratio_at_tp_prev = volume at the completed bar before the hit / same norm
    Requires ``tp_hit_time`` from ``attach_take_profit_features``.
    """
    df = df.copy()
    if df.empty or "tp_hit_time" not in df.columns:
        return df
    bw = max(int(baseline_window), 2)
    df["vol_ratio_at_tp"] = np.nan
    df["vol_ratio_at_tp_prev"] = np.nan

    for pair, sub in df.groupby("pair"):
        pdata = dict_of_pairs.get(pair)
        if not pdata:
            continue
        frame = pdata.get("dict_of_frames", {}).get(execution_frame)
        if frame is None or frame.empty or "close_time" not in frame.columns or "volume" not in frame.columns:
            continue
        f = frame.sort_values("close_time").reset_index(drop=True)
        close_time = pd.to_datetime(f["close_time"], errors="coerce").to_numpy()
        vol = pd.to_numeric(f["volume"], errors="coerce").to_numpy(dtype=float)
        n = len(f)
        base = pd.Series(vol).rolling(bw, min_periods=max(bw // 2, 2)).mean().to_numpy(dtype=float)

        hit_times = pd.to_datetime(sub["tp_hit_time"], errors="coerce").to_numpy()
        pos = np.searchsorted(close_time, hit_times)
        pos = np.clip(pos, 0, n - 1)
        exact = close_time[pos] == hit_times

        for i, j, ok in zip(sub.index, pos, exact):
            if not ok or j < 1 or not np.isfinite(base[j]) or base[j] <= 0:
                continue
            if np.isfinite(vol[j]):
                df.loc[i, "vol_ratio_at_tp"] = float(vol[j] / base[j])
            if j - 1 >= 0 and np.isfinite(vol[j - 1]):
                df.loc[i, "vol_ratio_at_tp_prev"] = float(vol[j - 1] / base[j])
    return df


def attach_tp_close_features(
    df: pd.DataFrame,
    dict_of_pairs: dict,
    execution_frame: str,
    ema_cols: tuple[str, ...] = ("ema7_high", "ema25", "ema50", "ema100", "ema200"),
) -> pd.DataFrame:
    """Capture close-derived indicators at the TP hit AND at the bar before it.

    Live-feasibility: at the intrabar TP fill the CURRENT bar is still forming,
    so its close-based values (rsi, short_spread, EMA's) are NOT yet known --
    you would have to recompute every one at tp_monitor. The PREVIOUS completed
    bar's values ARE available. So this captures both and derives normalized
    prev-bar distances a live rule can actually act on.

    Attaches for close / rsi / short_spread / volatility / each ema_col:
      {col}@at_tp      = value at the TP-hit bar (backtest hindsight only)
      {col}@at_tp_prev = value at the completed bar BEFORE the hit (live-usable)
    Plus prev-bar distance features (computed from the completed bar):
      spread_dist_prev : (close_prev - ema7_high_prev) / short_spread_prev
                         -- how far above the ema7 TP anchor the prev close sat,
                         in short_spread units (the fpt-equivalent distance the
                         move had already covered going into the TP bar)
      ema25/50/100/200_dist_prev : (close_prev - ema_prev) / close_prev
    Requires ``tp_hit_time`` from ``attach_take_profit_features``.
    """
    df = attach_snapshot_at_tp(
        df, dict_of_pairs, execution_frame,
        columns=["close", "rsi", "short_spread", "volatility", *ema_cols],
        offsets=(0, -1),
    )
    c_prev = pd.to_numeric(df.get("close@at_tp_prev"), errors="coerce") if "close@at_tp_prev" in df.columns else None
    ss_prev = pd.to_numeric(df.get("short_spread@at_tp_prev"), errors="coerce") if "short_spread@at_tp_prev" in df.columns else None
    e7_prev = pd.to_numeric(df.get("ema7_high@at_tp_prev"), errors="coerce") if "ema7_high@at_tp_prev" in df.columns else None
    if c_prev is not None and ss_prev is not None and e7_prev is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            df["spread_dist_prev"] = (c_prev - e7_prev) / ss_prev
    for ema in ema_cols:
        if ema == "ema7_high":
            continue
        e_prev = pd.to_numeric(df.get(f"{ema}@at_tp_prev"), errors="coerce") if f"{ema}@at_tp_prev" in df.columns else None
        if c_prev is not None and e_prev is not None:
            with np.errstate(divide="ignore", invalid="ignore"):
                df[f"{ema}_dist_prev"] = (c_prev - e_prev) / c_prev
    return df


# ---------------------------------------------------------------------------
# Phase B part 5: stop-event snapshot (recovery / hold-or-abandon gate)
# "at the bar BEFORE the stop fires, what separates the recoverable from the dud?"
# ---------------------------------------------------------------------------

def attach_snapshot_at_stop(
    df: pd.DataFrame,
    dict_of_pairs: dict,
    execution_frame: str,
    columns: list[str],
    offsets: tuple[int, ...] = (0, -1),
    exit_reasons: tuple[str, ...] = ("stop_loss",),
) -> pd.DataFrame:
    """Snapshot frame columns AT the stop exit bar (and the bars around it).

    Mirror of ``attach_snapshot_at_tp`` for the stop cohort -- the live-feasible
    decision point of a recovery/hold gate is the completed bar BEFORE the stop
    fires (offset -1): at that close the stop had not yet triggered, so its
    indicators are the last "healthy" read. Offset 0 is the stop bar itself
    (close already through the stop; hindsight).

    Requires ``exit_time`` + ``exit_close_reason`` (from ``build_entry_analysis_dataset``
    with ``trade_frame``). Rows whose exit reason is not in ``exit_reasons`` stay NaN.

    Adds ``{col}@at_stop`` (offset 0) and ``{col}@at_stop_prev`` (offset -1); any
    other offset gets ``{col}@at_stop{+/-n}``.
    """
    df = df.copy()
    if df.empty or "exit_time" not in df.columns or "exit_close_reason" not in df.columns:
        return df
    mask = df["exit_close_reason"].isin(exit_reasons)

    def _suffix(o: int) -> str:
        return {0: "at_stop", -1: "at_stop_prev"}.get(o, f"at_stop{o:+d}")

    for col in columns:
        for o in offsets:
            df[f"{col}@{_suffix(o)}"] = np.nan

    for pair, sub in df[mask].groupby("pair"):
        pdata = dict_of_pairs.get(pair)
        if not pdata:
            continue
        frame = pdata.get("dict_of_frames", {}).get(execution_frame)
        if frame is None or frame.empty or "close_time" not in frame.columns:
            continue
        f = frame.sort_values("close_time").reset_index(drop=True)
        close_time = pd.to_datetime(f["close_time"], errors="coerce").to_numpy()
        exit_times = pd.to_datetime(sub["exit_time"], errors="coerce").to_numpy()
        pos = np.searchsorted(close_time, exit_times)
        pos = np.clip(pos, 0, len(f) - 1)
        exact = close_time[pos] == exit_times
        n = len(f)
        for col in columns:
            if col not in f.columns:
                continue
            vals = pd.to_numeric(f[col], errors="coerce").to_numpy(dtype=float)
            for o in offsets:
                tgt = pos + o
                ok = exact & (tgt >= 0) & (tgt < n)
                tgt_c = np.clip(tgt, 0, n - 1)
                df.loc[sub.index, f"{col}@{_suffix(o)}"] = np.where(ok, vals[tgt_c], np.nan)
    return df


def attach_volume_ratio_at_stop(
    df: pd.DataFrame,
    dict_of_pairs: dict,
    execution_frame: str,
    baseline_window: int = 20,
    exit_reasons: tuple[str, ...] = ("stop_loss",),
) -> pd.DataFrame:
    """Volume at / just before the stop exit, normalized against its recent norm.

    Mirror of ``attach_volume_ratio_at_tp`` for the stop cohort:
      vol_ratio_at_stop      = volume at the stop bar / mean volume over the prior
                               ``baseline_window`` bars
      vol_ratio_at_stop_prev = volume at the completed bar before the stop / same norm
    Requires ``exit_time`` + ``exit_close_reason``.
    """
    df = df.copy()
    if df.empty or "exit_time" not in df.columns or "exit_close_reason" not in df.columns:
        return df
    mask = df["exit_close_reason"].isin(exit_reasons)
    bw = max(int(baseline_window), 2)
    df["vol_ratio_at_stop"] = np.nan
    df["vol_ratio_at_stop_prev"] = np.nan

    for pair, sub in df[mask].groupby("pair"):
        pdata = dict_of_pairs.get(pair)
        if not pdata:
            continue
        frame = pdata.get("dict_of_frames", {}).get(execution_frame)
        if frame is None or frame.empty or "close_time" not in frame.columns or "volume" not in frame.columns:
            continue
        f = frame.sort_values("close_time").reset_index(drop=True)
        close_time = pd.to_datetime(f["close_time"], errors="coerce").to_numpy()
        vol = pd.to_numeric(f["volume"], errors="coerce").to_numpy(dtype=float)
        n = len(f)
        base = pd.Series(vol).rolling(bw, min_periods=max(bw // 2, 2)).mean().to_numpy(dtype=float)

        exit_times = pd.to_datetime(sub["exit_time"], errors="coerce").to_numpy()
        pos = np.searchsorted(close_time, exit_times)
        pos = np.clip(pos, 0, n - 1)
        exact = close_time[pos] == exit_times

        for i, j, ok in zip(sub.index, pos, exact):
            if not ok or j < 1 or not np.isfinite(base[j]) or base[j] <= 0:
                continue
            if np.isfinite(vol[j]):
                df.loc[i, "vol_ratio_at_stop"] = float(vol[j] / base[j])
            if j - 1 >= 0 and np.isfinite(vol[j - 1]):
                df.loc[i, "vol_ratio_at_stop_prev"] = float(vol[j - 1] / base[j])
    return df


def attach_stop_close_features(
    df: pd.DataFrame,
    dict_of_pairs: dict,
    execution_frame: str,
    ema_cols: tuple[str, ...] = ("ema7_high", "ema25", "ema50", "ema100", "ema200"),
    exit_reasons: tuple[str, ...] = ("stop_loss",),
) -> pd.DataFrame:
    """Close-derived indicators at the stop bar AND at the bar before it.

    Mirror of ``attach_tp_close_features``: captures close / rsi / short_spread /
    volatility / each ema_col at the stop bar and the completed bar before it
    ({col}@at_stop / {col}@at_stop_prev), plus prev-bar distances with a distinct
    ``_stop_prev`` suffix so they never collide with the TP-analysis columns:
      spread_dist_stop_prev     : (close_prev - ema7_high_prev) / short_spread_prev
      ema{25,50,100,200}_dist_stop_prev : (close_prev - ema_prev) / close_prev
    """
    df = attach_snapshot_at_stop(
        df, dict_of_pairs, execution_frame,
        columns=["close", "rsi", "short_spread", "volatility", *ema_cols],
        offsets=(0, -1), exit_reasons=exit_reasons,
    )
    c_prev = pd.to_numeric(df.get("close@at_stop_prev"), errors="coerce") if "close@at_stop_prev" in df.columns else None
    ss_prev = pd.to_numeric(df.get("short_spread@at_stop_prev"), errors="coerce") if "short_spread@at_stop_prev" in df.columns else None
    e7_prev = pd.to_numeric(df.get("ema7_high@at_stop_prev"), errors="coerce") if "ema7_high@at_stop_prev" in df.columns else None
    if c_prev is not None and ss_prev is not None and e7_prev is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            df["spread_dist_stop_prev"] = (c_prev - e7_prev) / ss_prev
    for ema in ema_cols:
        if ema == "ema7_high":
            continue
        e_prev = pd.to_numeric(df.get(f"{ema}@at_stop_prev"), errors="coerce") if f"{ema}@at_stop_prev" in df.columns else None
        if c_prev is not None and e_prev is not None:
            with np.errstate(divide="ignore", invalid="ignore"):
                df[f"{ema}_dist_stop_prev"] = (c_prev - e_prev) / c_prev
    return df


def attach_stop_recovery_features(
    df: pd.DataFrame,
    dict_of_pairs: dict,
    execution_frame: str,
    exit_reasons: tuple[str, ...] = ("stop_loss",),
) -> pd.DataFrame:
    """In-hold peak / drawdown-from-peak at the bar BEFORE the stop fires.

    The headline variable: a trade that reached +7% and pulled back to +3% is
    different from one that is merely grinding down. Uses only bars up to and
    including the completed bar before the stop (causal, no lookahead). The peak
    is close-based to match the engine's own peak (the stop ratchets off closes,
    not highs -- see backtest_engine._evaluate_and_close_position). Adds:
      peak_before_stop_pct        : max close from entry through the prev bar /
                                    entry_close - 1
      drawdown_from_peak_stop_prev: close at the prev bar / peak - 1  (how much
                                    of the run the stop would have given back)
      bars_since_peak_stop        : bars from the peak bar to the prev bar
    """
    df = df.copy()
    if df.empty or "exit_time" not in df.columns or "exit_close_reason" not in df.columns:
        return df
    if "exit_hold_bars" not in df.columns or "time" not in df.columns:
        return df
    mask = df["exit_close_reason"].isin(exit_reasons)
    for col in ("peak_before_stop_pct", "drawdown_from_peak_stop_prev", "bars_since_peak_stop",
                "peak_incl_stop_pct", "drawdown_from_peak_stop"):
        df[col] = np.nan

    for pair, sub in df[mask].groupby("pair"):
        pdata = dict_of_pairs.get(pair)
        if not pdata:
            continue
        frame = pdata.get("dict_of_frames", {}).get(execution_frame)
        if frame is None or frame.empty or "close_time" not in frame.columns:
            continue
        if "close" not in frame.columns or "high" not in frame.columns:
            continue
        f = frame.sort_values("close_time").reset_index(drop=True)
        close_time = pd.to_datetime(f["close_time"], errors="coerce").to_numpy()
        close = pd.to_numeric(f["close"], errors="coerce").to_numpy(dtype=float)
        entry_times = pd.to_datetime(sub["time"], errors="coerce").to_numpy()
        entry_pos = np.searchsorted(close_time, entry_times)
        entry_pos = np.clip(entry_pos, 0, len(f) - 1)
        exact_entry = close_time[entry_pos] == entry_times
        hold = pd.to_numeric(sub["exit_hold_bars"], errors="coerce").to_numpy(dtype=float)
        n = len(f)
        for i, (ep, ok, h) in enumerate(zip(entry_pos, exact_entry, hold)):
            idx = sub.index[i]
            if not ok or not np.isfinite(h) or int(h) < 1:
                continue
            prev = int(ep) + int(h) - 1          # completed bar before the stop fires
            if prev < int(ep) or prev >= n:
                continue
            window = close[int(ep):prev + 1]
            if not np.isfinite(window).any():
                continue
            peak = float(np.nanmax(window))
            entry_close = float(close[int(ep)])
            if not np.isfinite(peak) or peak <= 0 or not np.isfinite(entry_close) or entry_close <= 0:
                continue
            df.loc[idx, "peak_before_stop_pct"] = (peak / entry_close - 1.0) * 100.0
            c_prev = float(close[prev])
            if np.isfinite(c_prev) and c_prev > 0:
                df.loc[idx, "drawdown_from_peak_stop_prev"] = (c_prev / peak - 1.0) * 100.0
            peak_idx = int(np.nanargmax(window))
            df.loc[idx, "bars_since_peak_stop"] = float(prev - (int(ep) + peak_idx))

            # stop-bar-inclusive variants: at the stop close (offset 0) the stop
            # has just fired but this bar's close IS known, so these are the
            # live-feasible reads at the moment you would postpone the stop.
            stop = int(ep) + int(h)
            if stop < int(ep) or stop >= n:
                continue
            window_stop = close[int(ep):stop + 1]
            if not np.isfinite(window_stop).any():
                continue
            peak_stop = float(np.nanmax(window_stop))
            if not np.isfinite(peak_stop) or peak_stop <= 0:
                continue
            df.loc[idx, "peak_incl_stop_pct"] = (peak_stop / entry_close - 1.0) * 100.0
            c_stop = float(close[stop])
            if np.isfinite(c_stop) and c_stop > 0:
                df.loc[idx, "drawdown_from_peak_stop"] = (c_stop / peak_stop - 1.0) * 100.0
    return df


def attach_stop_scaled_features(
    df: pd.DataFrame,
    dict_of_pairs: dict,
    execution_frame: str,
    norm_window: int = 100,
    slope_window: int = 6,
    exit_reasons: tuple[str, ...] = ("stop_loss",),
) -> pd.DataFrame:
    """Cross-token normalized + OBV-slope features at entry and at the stop bar.

    Raw ``obv`` and ``quote_vol_per_unit_change`` are absolute and NOT comparable
    across tokens (a large-cap token trivially has higher values), which is why
    the raw median gap in ``stop_win_loss_compare`` was huge but meaningless.
    This normalizes each by that token's own TRAILING mean over ``norm_window``
    bars (causal -- only prior bars, so live-usable), giving "how far above/below
    its own norm" (1.0 = at norm). OBV is a running sum, so its LEVEL is less
    informative than its SLOPE (accumulation rate) -- added with the same rolling
    slope used for ``obv_slope`` elsewhere.

    Adds, at ENTRY and at the STOP bar close + the bar before it:
      obv_norm / obv_norm@at_stop / obv_norm@at_stop_prev
      qvpuc_norm / qvpuc_norm@at_stop / qvpuc_norm@at_stop_prev
      obv_slope / obv_slope@at_stop / obv_slope@at_stop_prev   (raw polyfit slope)
      obv_slope_norm / obv_slope_norm@at_stop / _prev  (slope / obv_mean =
        fractional OBV accumulation per bar -- the CROSS-TOKEN comparable one;
        raw obv_slope scales with token size so its delta is meaningless)
    Only rows whose exit reason is in ``exit_reasons`` are filled.
    """
    df = df.copy()
    if df.empty or "exit_time" not in df.columns or "exit_close_reason" not in df.columns:
        return df
    if "time" not in df.columns:
        return df
    mask = df["exit_close_reason"].isin(exit_reasons)
    nw = max(int(norm_window), 5)
    sw = max(int(slope_window), 3)
    for col in ("obv_norm", "qvpuc_norm", "obv_slope", "obv_slope_norm"):
        for suffix in ("", "@at_stop", "@at_stop_prev"):
            df[f"{col}{suffix}"] = np.nan

    for pair, sub in df[mask].groupby("pair"):
        pdata = dict_of_pairs.get(pair)
        if not pdata:
            continue
        frame = pdata.get("dict_of_frames", {}).get(execution_frame)
        if frame is None or frame.empty or "close_time" not in frame.columns:
            continue
        f = frame.sort_values("close_time").reset_index(drop=True)
        close_time = pd.to_datetime(f["close_time"], errors="coerce").to_numpy()
        n = len(f)

        has_obv = "obv" in f.columns
        has_qv = "quote_vol_per_unit_change" in f.columns

        obv = obv_mean = obv_slope_series = None
        if has_obv:
            obv = pd.to_numeric(f["obv"], errors="coerce").ffill().to_numpy(dtype=float)
            obv_ser = pd.Series(obv)
            obv_mean = obv_ser.rolling(nw, min_periods=max(nw // 4, 3)).mean().to_numpy(dtype=float)
            obv_slope_series = obv_ser.rolling(sw, min_periods=3).apply(_rolling_slope, raw=True).to_numpy(dtype=float)

        qv = qv_mean = None
        if has_qv:
            qv = pd.to_numeric(f["quote_vol_per_unit_change"], errors="coerce").to_numpy(dtype=float)
            qv_mean = pd.Series(qv).rolling(nw, min_periods=max(nw // 4, 3)).mean().to_numpy(dtype=float)

        entry_times = pd.to_datetime(sub["time"], errors="coerce").to_numpy()
        ep = np.searchsorted(close_time, entry_times)
        ep = np.clip(ep, 0, n - 1)
        exact_entry = close_time[ep] == entry_times

        exit_times = pd.to_datetime(sub["exit_time"], errors="coerce").to_numpy()
        spos = np.searchsorted(close_time, exit_times)
        spos = np.clip(spos, 0, n - 1)
        exact_exit = close_time[spos] == exit_times

        for i, ep_i, sp_i, ok_e, ok_s in zip(sub.index, ep, spos, exact_entry, exact_exit):
            if not ok_e:
                continue
            if obv_mean is not None and np.isfinite(obv_mean[ep_i]) and obv_mean[ep_i] > 0 and np.isfinite(obv[ep_i]):
                df.loc[i, "obv_norm"] = float(obv[ep_i] / obv_mean[ep_i])
            if qv_mean is not None and np.isfinite(qv_mean[ep_i]) and qv_mean[ep_i] > 0 and np.isfinite(qv[ep_i]):
                df.loc[i, "qvpuc_norm"] = float(qv[ep_i] / qv_mean[ep_i])
            if obv_slope_series is not None and np.isfinite(obv_slope_series[ep_i]):
                df.loc[i, "obv_slope"] = float(obv_slope_series[ep_i])
                if np.isfinite(obv_mean[ep_i]) and obv_mean[ep_i] > 0:
                    df.loc[i, "obv_slope_norm"] = float(obv_slope_series[ep_i] / obv_mean[ep_i])
            if not ok_s:
                continue
            for off, suffix in ((0, "@at_stop"), (-1, "@at_stop_prev")):
                tgt = sp_i + off
                if tgt < 0 or tgt >= n:
                    continue
                if obv_mean is not None and np.isfinite(obv_mean[tgt]) and obv_mean[tgt] > 0 and np.isfinite(obv[tgt]):
                    df.loc[i, f"obv_norm{suffix}"] = float(obv[tgt] / obv_mean[tgt])
                if qv_mean is not None and np.isfinite(qv_mean[tgt]) and qv_mean[tgt] > 0 and np.isfinite(qv[tgt]):
                    df.loc[i, f"qvpuc_norm{suffix}"] = float(qv[tgt] / qv_mean[tgt])
                if obv_slope_series is not None and np.isfinite(obv_slope_series[tgt]):
                    df.loc[i, f"obv_slope{suffix}"] = float(obv_slope_series[tgt])
                    if np.isfinite(obv_mean[tgt]) and obv_mean[tgt] > 0:
                        df.loc[i, f"obv_slope_norm{suffix}"] = float(obv_slope_series[tgt] / obv_mean[tgt])
    return df


def _rolling_slope(arr: np.ndarray) -> float:
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 2:
        return np.nan
    try:
        return float(np.polyfit(np.arange(len(arr)), arr, 1)[0])
    except Exception:
        return np.nan


def add_forward_volume_features(
    df: pd.DataFrame,
    dict_of_pairs: dict,
    execution_frame: str,
    window: int = 6,
    baseline_window: int = 20,
    obv_slope_window: int = 6,
) -> pd.DataFrame:
    """Volume/OBV measures around the entry (attribution, not prediction).

    Adds, per entry:
      vol_ratio_fwd  : mean volume over the next `window` bars / mean volume
                       over the prior `baseline_window` bars (volume in the rise).
      obv_chg_fwd    : OBV change over the next `window` bars (accumulation in
                       the rise).
      obv_slope      : slope of OBV over the prior `obv_slope_window` bars at
                       entry (pre-entry accumulation proxy, usable live).

    Requires 'volume' (and optionally 'obv') on the execution frames.
    """
    df = df.copy()
    if df.empty:
        return df
    w = max(int(window), 1)
    bw = max(int(baseline_window), 2)
    df["vol_ratio_fwd"] = np.nan
    df["obv_chg_fwd"] = np.nan
    df["obv_slope"] = np.nan

    for pair, sub in df.groupby("pair"):
        pdata = dict_of_pairs.get(pair)
        if not pdata:
            continue
        frame = pdata.get("dict_of_frames", {}).get(execution_frame)
        if frame is None or frame.empty or "close_time" not in frame.columns or "volume" not in frame.columns:
            continue
        f = frame.sort_values("close_time").reset_index(drop=True)
        close_time = pd.to_datetime(f["close_time"], errors="coerce").to_numpy()
        vol = pd.to_numeric(f["volume"], errors="coerce").to_numpy(dtype=float)
        n = len(f)
        base = pd.Series(vol).rolling(bw, min_periods=max(bw // 2, 2)).mean().to_numpy(dtype=float)

        has_obv = "obv" in f.columns
        obv = None
        obv_slope_series = None
        if has_obv:
            obv = pd.to_numeric(f["obv"], errors="coerce").ffill().to_numpy(dtype=float)
            obv_slope_series = pd.Series(obv).rolling(
                obv_slope_window, min_periods=3
            ).apply(_rolling_slope, raw=True).to_numpy(dtype=float)

        times = pd.to_datetime(sub["time"], errors="coerce").to_numpy()
        pos = np.searchsorted(close_time, times)
        pos = np.clip(pos, 0, n - 1)
        exact = close_time[pos] == times

        for i, j, ok in zip(sub.index, pos, exact):
            if not ok or j < 1:
                continue
            if obv is not None and np.isfinite(obv_slope_series[j]):
                df.loc[i, "obv_slope"] = float(obv_slope_series[j])
            if (j + w) >= n:
                continue
            seg = vol[j + 1: j + w + 1]
            if np.isfinite(seg).sum() == 0 or not np.isfinite(base[j]) or base[j] <= 0:
                continue
            df.loc[i, "vol_ratio_fwd"] = float(np.nanmean(seg) / base[j])
            if obv is not None and np.isfinite(obv[j + w]) and np.isfinite(obv[j]):
                df.loc[i, "obv_chg_fwd"] = float(obv[j + w] - obv[j])
    return df


def runup_volume_continuation(
    df: pd.DataFrame,
    arm_pct: float = 4.0,
    reached_horizon: int = 12,
    q: int = 4,
    continuation_buffer: float = 2.0,
) -> pd.DataFrame:
    """Does run-up volume predict continuation past the trailing arm?

    Among entries whose favorable excursion reached +arm_pct% within
    reached_horizon bars (i.e. the trailing arm engaged), bucket by forward
    volume ratio (vol_ratio_fwd). If high-volume rises keep going (larger
    mfe_24 / mfe_48 and a higher "kept going" share) while low-volume rises
    retrace (smaller mfe, deeper max_adverse), then a wider trailing
    retracement is justified for high-volume entries.
    """
    if df.empty or "vol_ratio_fwd" not in df.columns or "mfe_12" not in df.columns:
        return pd.DataFrame()
    work = df[df["mfe_12"] >= arm_pct].dropna(subset=["vol_ratio_fwd"]).copy()
    if work.empty:
        return pd.DataFrame()
    work["vol_bucket"] = pd.qcut(work["vol_ratio_fwd"], q=q, labels=False, duplicates="drop")
    g = work.groupby("vol_bucket")
    out = g.agg(
        n=("vol_ratio_fwd", "size"),
        mean_vol_ratio=("vol_ratio_fwd", "mean"),
        mean_mfe12=("mfe_12", "mean"),
        mean_mfe24=("mfe_24", "mean"),
        mean_mfe48=("mfe_48", "mean"),
        mean_ret24=("ret_24", "mean"),
        mean_max_adverse=("max_adverse", "mean"),
        kept_going_pct=("mfe_24", lambda s: 100.0 * (s >= arm_pct + continuation_buffer).mean()),
    )
    out.index.name = "vol_quantile"
    return out


# ---------------------------------------------------------------------------
# Phase B, part 2: take-profit (TP) + price-extension analysis
# ---------------------------------------------------------------------------

def _first_dynamic_cross(
    high: np.ndarray,
    level: np.ndarray,
    h: int,
) -> np.ndarray:
    """out[t] = first k in [1..h] with high[t+k] > level[t+k]; NaN if none.

    Matches the engine's TP rule: the take-profit level is DYNAMIC (re-derived
    every bar from ema7_high + fpt*short_spread) and a bar fills when that bar's
    high exceeds that bar's level -- unlike _bars_to_target which tests a fixed
    price.
    """
    high = np.asarray(high, dtype=np.float64)
    level = np.asarray(level, dtype=np.float64)
    T = len(high)
    out = np.full(T, np.nan)
    h = int(min(h, T - 1))
    if h <= 0:
        return out
    cross = high > level
    for d in range(1, h + 1):
        c = cross[d:]                                   # index j -> time j+d
        n = T - d
        hit_now = c[:n] & np.isnan(out[:n])
        out[:n] = np.where(hit_now, d, out[:n])
    return out


def attach_take_profit_features(
    df: pd.DataFrame,
    payload: dict,
    tp_horizon: int = 24,
    extension_horizon: int = 24,
) -> pd.DataFrame:
    """Attach take-profit level / hit / post-TP extension columns per entry.

    Reads the payload's per-strategy dynamic ``take_profit_price`` aux arrays
    (the same ones the engine fills against) and answers, per entry:
      tp_distance_pct        : (tp@entry / entry_close - 1) * 100  -- how far away
                               the target is at the entry bar.
      tp_hit                 : high crossed the dynamic TP within ``tp_horizon`` bars.
      bars_to_tp             : first bar (1-based) where high > tp; NaN if never.
      tp_hit_price           : the TP level in force at the crossing bar (~the fill).
      mfe_at_tp_pct          : (tp_hit_price / entry_close - 1) * 100 -- realized
                               gain if filled at the TP.
      extension_past_tp_pct  : max high over the next ``extension_horizon`` bars
                               AFTER the hit, relative to tp_hit_price, in % --
                               the upside the engine leaves on the table by
                               filling at TP (the "TP too tight / round-trip"
                               measure). NaN when the TP is never hit.
      overshoot_entry_tp_pct : max high over [t+1, t+extension_horizon] relative
                               to the entry-bar TP, in % (always defined) -- a
                               simpler cross-check that needs no hit detection.
      tp_hit_bar / tp_hit_time : payload bar index / timeline timestamp of the
                               TP-hit bar -- lets downstream capture frame
                               features AT the hit (see attach_snapshot_at_tp).
    """
    df = df.copy()
    if df.empty:
        return df
    T, N = payload["shape"]["T"], payload["shape"]["N"]
    timeline = payload["timeline"]
    high_arr = np.asarray(payload["prices"]["high"], dtype=np.float64)
    pair_index = {p: i for i, p in enumerate(payload["pairs"])}
    aux = payload.get("aux", {})
    nested = isinstance(aux, dict) and any(isinstance(v, dict) for v in aux.values())

    for col in ("tp_distance_pct", "bars_to_tp", "tp_hit_price", "mfe_at_tp_pct",
                "extension_past_tp_pct", "overshoot_entry_tp_pct", "tp_hit_bar"):
        df[col] = np.nan
    df["tp_hit"] = False
    df["tp_hit_time"] = pd.NaT

    # Per-column caches so each dynamic-cross / forward-max array is computed once.
    cross_cache: dict[tuple[int, int], np.ndarray] = {}
    fwdmax_cache: dict[int, np.ndarray] = {}

    for idx, row in df.iterrows():
        strat = row["strategy"]
        strat_aux = aux.get(strat, {}) if nested else aux
        tp_arr = strat_aux.get("take_profit_price")
        if tp_arr is None:
            continue
        tp_arr = np.asarray(tp_arr, dtype=np.float64)
        n = pair_index.get(row["pair"])
        t = int(row["bar"])
        if n is None or not (0 <= t < T):
            continue

        key = (n, t)  # cache per column (not per t); see below
        cross_cache.setdefault(n, _first_dynamic_cross(high_arr[:, n], tp_arr[:, n], tp_horizon))
        fwdmax_cache.setdefault(n, _forward_extreme(high_arr[:, n], extension_horizon, "max"))
        fc = cross_cache[n]
        fwdmax = fwdmax_cache[n]
        tp_entry = tp_arr[t, n]

        if np.isfinite(tp_entry) and tp_entry > 0 and np.isfinite(row["entry_close"]) and row["entry_close"] > 0:
            df.at[idx, "tp_distance_pct"] = (tp_entry / row["entry_close"] - 1.0) * 100.0

        k = fc[t]
        if np.isfinite(k):
            kk = int(k)
            hit_bar = t + kk
            if hit_bar < T:
                tp_hit_price = tp_arr[hit_bar, n]
                df.at[idx, "tp_hit"] = True
                df.at[idx, "bars_to_tp"] = float(kk)
                df.at[idx, "tp_hit_bar"] = float(hit_bar)
                df.at[idx, "tp_hit_time"] = timeline[hit_bar]
                if np.isfinite(tp_hit_price) and tp_hit_price > 0:
                    df.at[idx, "tp_hit_price"] = float(tp_hit_price)
                    if np.isfinite(row["entry_close"]) and row["entry_close"] > 0:
                        df.at[idx, "mfe_at_tp_pct"] = (tp_hit_price / row["entry_close"] - 1.0) * 100.0
                    if np.isfinite(fwdmax[hit_bar]):
                        df.at[idx, "extension_past_tp_pct"] = (fwdmax[hit_bar] / tp_hit_price - 1.0) * 100.0

        if np.isfinite(tp_entry) and tp_entry > 0 and np.isfinite(fwdmax[t]):
            df.at[idx, "overshoot_entry_tp_pct"] = (fwdmax[t] / tp_entry - 1.0) * 100.0

    return df


def tp_hit_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Overall TP-hit stats, plus the same stats sliced by actual exit reason."""
    if df.empty or "tp_hit" not in df.columns:
        return pd.DataFrame()
    rows = [{
        "population": "all_entries",
        "n": len(df),
        "tp_hit_pct": 100.0 * df["tp_hit"].mean(),
        "median_bars_to_tp": df["bars_to_tp"].median(),
        "median_tp_distance_pct": df["tp_distance_pct"].median(),
        "median_extension_past_tp_pct": df.loc[df["tp_hit"], "extension_past_tp_pct"].median(),
    }]
    if "was_taken" in df.columns:
        for reason in ("vol_take_profit", "time_stop", "trailing_stop", "stop_loss", "exit_signal"):
            sub = df[(df["was_taken"]) & (df["exit_close_reason"] == reason)]
            if sub.empty:
                continue
            rows.append({
                "population": f"exit={reason}",
                "n": len(sub),
                "tp_hit_pct": 100.0 * sub["tp_hit"].mean(),
                "median_bars_to_tp": sub["bars_to_tp"].median(),
                "median_tp_distance_pct": sub["tp_distance_pct"].median(),
                "median_extension_past_tp_pct": sub.loc[sub["tp_hit"], "extension_past_tp_pct"].median(),
            })
    return pd.DataFrame(rows).set_index("population")


def tp_round_trip_analysis(
    df: pd.DataFrame,
    extension_bands: tuple[float, ...] = (1.0, 2.0, 3.0, 5.0),
) -> pd.DataFrame:
    """Of the trades that actually FILLED at the TP, how much did we leave behind?

    A ``vol_take_profit`` exit whose high ran well above the fill after the hit is
    the TP-equivalent of a round-trip: the target was reached but the move kept
    going, so a wider fpt / dynamic TP would have captured more. ``mfe_48`` is
    shown next to ``mfe_at_tp`` to quantify the gap.
    """
    if df.empty or "was_taken" not in df.columns:
        return pd.DataFrame()
    work = df[(df["was_taken"]) & (df["exit_close_reason"] == "vol_take_profit")].copy()
    if work.empty:
        return pd.DataFrame()
    ext = pd.to_numeric(work["extension_past_tp_pct"], errors="coerce")
    rows = {
        "tp_exits": len(work),
        "with_extension_data": int(ext.notna().sum()),
        "median_extension_past_tp_pct": float(ext.median()) if ext.notna().any() else np.nan,
        "mean_extension_past_tp_pct": float(ext.mean()) if ext.notna().any() else np.nan,
    }
    for b in extension_bands:
        rows[f"extended_{b:g}pct_or_more_pct"] = (100.0 * (ext >= b).mean()) if ext.notna().any() else np.nan
    rows["median_mfe_at_tp_pct"] = float(work["mfe_at_tp_pct"].median())
    rows["median_mfe_48_pct"] = float(work["mfe_48"].median())
    return pd.DataFrame([rows], index=["tp_round_trip"])


def tp_hit_x_exit(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-tab TP-hit (ever reached within the horizon) x actual exit reason.

    A trade that hit the TP but exited elsewhere (time/trailing/stop) is a sign
    the exit machinery let a reached target slip; a ``vol_take_profit`` row here
    just confirms the normal path.
    """
    if df.empty or "was_taken" not in df.columns:
        return pd.DataFrame()
    work = df[df["was_taken"]].copy()
    if work.empty:
        return pd.DataFrame()
    g = work.groupby(["tp_hit", "exit_close_reason"], dropna=False)
    out = g.agg(
        trades=("exit_pnl", "size"),
        mean_pnl=("exit_pnl", "mean"),
        mean_ret_pct=("exit_return_pct", "mean"),
        mean_mfe48=("mfe_48", "mean"),
        mean_extension=("extension_past_tp_pct", "mean"),
    )
    return out


def tp_extension_by_feature(
    df: pd.DataFrame,
    feature: str = "vol_ratio_fwd",
    q: int = 4,
    min_samples: int = 30,
) -> pd.DataFrame:
    """Bucket TP-hit entries by a feature quantile; is post-TP extension predicted?

    Discriminator test for a dynamic TP: among entries that REACHED the TP, does
    high volume / volatility / RSI predict continued extension past the target
    (=> a wider fpt when the feature is high captures more), while low ones stall
    right at the target (=> current TP is about right)? Feature columns come from
    ``attach_snapshot_features`` / ``add_forward_volume_features`` (e.g.
    vol_ratio_fwd, volatility, rsi, rsi_slope).
    """
    if df.empty or feature not in df.columns or "tp_hit" not in df.columns:
        return pd.DataFrame()
    work = df[df["tp_hit"]].dropna(subset=[feature]).copy()
    if len(work) < min_samples:
        return pd.DataFrame()
    try:
        work["feat_bucket"] = pd.qcut(work[feature], q=q, labels=False, duplicates="drop")
    except Exception:
        return pd.DataFrame()
    g = work.groupby("feat_bucket")
    out = g.agg(
        n=(feature, "size"),
        mean_feat=(feature, "mean"),
        median_tp_distance=("tp_distance_pct", "median"),
        median_bars_to_tp=("bars_to_tp", "median"),
        median_extension=("extension_past_tp_pct", "median"),
        mean_extension=("extension_past_tp_pct", "mean"),
        pct_extended_2plus=("extension_past_tp_pct", lambda s: 100.0 * (s >= 2.0).mean()),
        median_mfe48=("mfe_48", "median"),
        median_ret24=("ret_24", "median"),
    )
    out.index.name = f"{feature}_quantile"
    return out


def tp_condition_matrix(
    df: pd.DataFrame,
    features: list[str],
    q: int = 2,
    min_samples: int = 30,
) -> pd.DataFrame:
    """Stacked met-vs-not-met table for several TP-hit-time conditions.

    Directly answers the dynamic-TP question: "if these conditions are met when
    the TP is hit the trade is better than it seems -- hold; otherwise take
    profit now." For each feature (expected to be a TP-hit-time snapshot like
    ``rsi@at_tp`` / ``vol_ratio_at_tp`` / ``volatility@at_tp_prev``), TP-hit
    entries are split into high / low buckets (quantile q) and compared on how
    far price kept extending past the TP.

    A condition where the high bucket extends MORE (median_extension /
    pct_extended_2plus / median_mfe48 all higher) supports a hold-on-this-
    condition rule; one where the high bucket is flat or worse supports
    take-profit-now.
    """
    rows = []
    for feat in features:
        tab = tp_extension_by_feature(df, feature=feat, q=q, min_samples=min_samples)
        if tab.empty:
            continue
        idx_name = tab.index.name or "bucket"
        tab = tab.reset_index().rename(columns={idx_name: "bucket"})
        tab["feature"] = feat
        keep = ["feature", "bucket", "n", "median_extension",
                "pct_extended_2plus", "median_mfe48", "median_ret24"]
        rows.append(tab[[c for c in keep if c in tab.columns]])
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out = out.set_index(["feature", "bucket"]).sort_index()
    return out


def stop_recovery_by_feature(
    df: pd.DataFrame,
    feature: str,
    q: int = 2,
    recovery_mfe: float = 3.0,
    dud_mae: float = -3.5,
    min_samples: int = 30,
    exit_reasons: tuple[str, ...] = ("stop_loss",),
) -> pd.DataFrame:
    """Bucket stop exits by a feature quantile; does the feature predict recovery?

    Among the stop cohort (``exit_reasons``), split by the snapshot feature
    (expected to be an ``@at_stop_prev`` column, ``drawdown_from_peak_stop_prev``,
    ``vol_ratio_at_stop_prev``, ...) into quantile buckets and compare recovery
    share + would-have-held return:
      recoverable_pct  : mfe_6 >= recovery_mfe  (the exit leaked a real run)
      genuine_dud_pct  : mae_6 <= dud_mae and never ran (bad ENTRY)
      median_ret12     : return if held to bar 12 ("should we have held?")
      mean_exit_ret_pct: what the stop actually realized
    A feature where the high bucket has HIGHER recoverable_pct AND median_ret12
    supports a hold-when-high rule; a bucket that is flat or worse supports
    keeping the stop (the other ~54%).
    """
    if df.empty or feature not in df.columns:
        return pd.DataFrame()
    work = df[df["exit_close_reason"].isin(exit_reasons)].dropna(subset=[feature]).copy()
    if "mfe_6" not in work.columns or "mae_6" not in work.columns:
        return pd.DataFrame()
    if len(work) < min_samples:
        return pd.DataFrame()
    try:
        work["feat_bucket"] = pd.qcut(work[feature], q=q, labels=False, duplicates="drop")
    except Exception:
        return pd.DataFrame()
    had_run = pd.to_numeric(work["mfe_6"], errors="coerce") >= recovery_mfe
    deep_dip = pd.to_numeric(work["mae_6"], errors="coerce") <= dud_mae
    work["_recoverable"] = had_run
    work["_dud"] = deep_dip & ~had_run
    agg_spec = {
        "n": (feature, "size"),
        "mean_feat": (feature, "mean"),
        "recoverable_pct": ("_recoverable", lambda s: 100.0 * s.mean()),
        "genuine_dud_pct": ("_dud", lambda s: 100.0 * s.mean()),
        "median_mfe6": ("mfe_6", "median"),
        "median_mae6": ("mae_6", "median"),
        "mean_exit_ret_pct": ("exit_return_pct", lambda s: 100.0 * s.mean()),
    }
    if "ret_12" in work.columns:
        agg_spec["median_ret12"] = ("ret_12", "median")
    out = work.groupby("feat_bucket").agg(**agg_spec)
    out.index.name = f"{feature}_quantile"
    return out


def stop_condition_matrix(
    df: pd.DataFrame,
    features: list[str],
    q: int = 2,
    recovery_mfe: float = 3.0,
    dud_mae: float = -3.5,
    min_samples: int = 30,
    exit_reasons: tuple[str, ...] = ("stop_loss",),
) -> pd.DataFrame:
    """Stacked high-vs-low table: which stop-time conditions predict recovery?

    For each snapshot feature (``@at_stop_prev`` / ``drawdown_from_peak_stop_prev`` /
    etc.), split the stop cohort into high / low buckets (quantile ``q``) and
    report recoverable_pct / genuine_dud_pct / median_ret12 / mean_exit_ret_pct.
    A bucket with HIGHER recoverable_pct + median_ret12 = the condition supports a
    hold / one-bar-reprieve rule; a bucket that is flat or worse = keep the stop.
    """
    rows = []
    for feat in features:
        tab = stop_recovery_by_feature(
            df, feat, q=q, recovery_mfe=recovery_mfe, dud_mae=dud_mae,
            min_samples=min_samples, exit_reasons=exit_reasons,
        )
        if tab.empty:
            continue
        idx_name = tab.index.name or "bucket"
        tab = tab.reset_index().rename(columns={idx_name: "bucket"})
        tab["feature"] = feat
        keep = ["feature", "bucket", "n", "mean_feat", "recoverable_pct",
                "genuine_dud_pct", "median_mfe6", "median_mae6",
                "median_ret12", "mean_exit_ret_pct"]
        rows.append(tab[[c for c in keep if c in tab.columns]])
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out = out.set_index(["feature", "bucket"]).sort_index()
    return out


def stop_win_loss_compare(
    df: pd.DataFrame,
    features: list[str],
    label: str = "mfe",
    floor: float = 3.0,
    exit_reasons: tuple[str, ...] = ("stop_loss",),
    min_samples: int = 20,
) -> pd.DataFrame:
    """Simple two-group comparison: stop exits that would have made money vs not.

    This is the direct, readable table for "what separates the recoverable stops".
    For every feature (mix of ``@at_stop`` / ``@at_stop_prev`` snapshot columns,
    entry-time snapshot columns, ``vol_ratio_at_stop``, ``drawdown_from_peak_stop``,
    ``position_in_range``, ...) it returns one row:
      n_win / n_loss          : group sizes
      median_win / median_loss: group medians
      mean_win / mean_loss    : group means
      delta_median            : median_win - median_loss (the separator)
    Rows are sorted by |delta_median| descending, so the strongest separating
    features are on top. A positive delta = higher in the made-money group;
    negative = higher in the losers.

    ``label`` picks the "made money" definition:
      "mfe"   -> had a real in-hold run (mfe_6 > floor; floor defaults to 3.0,
                 the same "had a run" threshold as ``exit_cohort_autopsy``)
      "ret12" -> would have made money if held to bar 12 (ret_12 > floor;
                 pass floor=0 for "any profit if held")
    """
    if df.empty or "exit_close_reason" not in df.columns:
        return pd.DataFrame()
    work = df[df["exit_close_reason"].isin(exit_reasons)].copy()
    if label == "ret12":
        if "ret_12" not in work.columns:
            return pd.DataFrame()
        made_money = pd.to_numeric(work["ret_12"], errors="coerce") > floor
    else:
        if "mfe_6" not in work.columns:
            return pd.DataFrame()
        made_money = pd.to_numeric(work["mfe_6"], errors="coerce") > floor
    work["_made_money"] = made_money
    if int(work["_made_money"].sum()) < min_samples or int((~work["_made_money"]).sum()) < min_samples:
        return pd.DataFrame()

    rows = []
    for feat in features:
        if feat not in work.columns:
            continue
        vals = pd.to_numeric(work[feat], errors="coerce")
        mask = vals.notna()
        if int(mask.sum()) < min_samples:
            continue
        g = work.loc[mask].groupby("_made_money")[feat].agg(["median", "mean", "size"])
        if not {True, False}.issubset(g.index):
            continue
        rows.append({
            "feature": feat,
            "n_win": int(g.loc[True, "size"]),
            "n_loss": int(g.loc[False, "size"]),
            "median_win": float(g.loc[True, "median"]),
            "median_loss": float(g.loc[False, "median"]),
            "mean_win": float(g.loc[True, "mean"]),
            "mean_loss": float(g.loc[False, "mean"]),
            "delta_median": float(g.loc[True, "median"] - g.loc[False, "median"]),
        })
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out["_abs_delta"] = out["delta_median"].abs()
    out = out.sort_values("_abs_delta", ascending=False).drop(columns=["_abs_delta"])
    return out.reset_index(drop=True)


def stop_cohort_by_strategy(
    df: pd.DataFrame,
    label: str = "mfe",
    floor: float = 3.0,
    exit_reasons: tuple[str, ...] = ("stop_loss",),
    min_samples: int = 10,
) -> pd.DataFrame:
    """Per-strategy stop cohort sizes and made-money share.

    When the pooled stop cohort mixes strategies with different exit behavior,
    per-strategy signals can cancel out. This gives the population picture per
    strategy: n_stops, n_win, made_money_pct (label/floor same as
    ``stop_win_loss_compare``), mean_exit_ret_pct and median_ret12 (if present).
    """
    if df.empty or "strategy" not in df.columns or "exit_close_reason" not in df.columns:
        return pd.DataFrame()
    work = df[df["exit_close_reason"].isin(exit_reasons)].copy()
    if label == "ret12":
        if "ret_12" not in work.columns:
            return pd.DataFrame()
        made_money = pd.to_numeric(work["ret_12"], errors="coerce") > floor
    else:
        if "mfe_6" not in work.columns:
            return pd.DataFrame()
        made_money = pd.to_numeric(work["mfe_6"], errors="coerce") > floor
    work["_made_money"] = made_money
    spec = {
        "n_stops": ("pair", "size"),
        "n_win": ("_made_money", "sum"),
        "made_money_pct": ("_made_money", lambda s: 100.0 * s.mean()),
        "mean_exit_ret_pct": ("exit_return_pct", lambda s: 100.0 * s.mean()),
    }
    if "ret_12" in work.columns:
        spec["median_ret12"] = ("ret_12", "median")
    g = work.groupby("strategy").agg(**spec)
    g["made_money_pct"] = g["made_money_pct"].fillna(0.0)
    g = g[g["n_stops"] >= min_samples].sort_values("n_stops", ascending=False)
    return g.reset_index()


def stop_win_loss_by_strategy(
    df: pd.DataFrame,
    features: list[str],
    label: str = "mfe",
    floor: float = 3.0,
    exit_reasons: tuple[str, ...] = ("stop_loss",),
    min_samples: int = 10,
    top_n: int | None = None,
) -> pd.DataFrame:
    """Run the winners-vs-losers compare per strategy, stacked.

    For every strategy with enough stops in BOTH groups, runs
    ``stop_win_loss_compare`` and stacks the result with a ``strategy`` column
    (rows per strategy already sorted by |delta_median|; ``top_n`` keeps only the
    top-N separators per strategy). Use when the pooled population is clouding
    the data -- a feature that looks weak overall may separate strongly within a
    specific strategy (and vice versa).
    """
    if df.empty or "strategy" not in df.columns or "exit_close_reason" not in df.columns:
        return pd.DataFrame()
    rows: list[pd.DataFrame] = []
    for strategy, sub in df.groupby("strategy"):
        tab = stop_win_loss_compare(
            sub, features, label=label, floor=floor,
            exit_reasons=exit_reasons, min_samples=min_samples,
        )
        if tab.empty:
            continue
        tab = tab.copy()
        tab.insert(0, "strategy", strategy)
        if top_n is not None and len(tab) > top_n:
            tab = tab.head(top_n)
        rows.append(tab)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def stop_recovery_rule(
    df: pd.DataFrame,
    feature: str,
    thresholds: list[float] | None = None,
    by: str = "all",
    held_col: str = "ret_12",
    exit_reasons: tuple[str, ...] = ("stop_loss",),
    min_samples: int = 20,
) -> pd.DataFrame:
    """Evaluate a "postpone the stop" rule over a feature threshold grid.

    Analysis-only (NOT a simulation, and no engine change): for every stop exit,
    if the rule fires (``feature >= threshold``) the position is assumed held to
    ``held_col`` (default ``ret_12`` = return if held to bar 12, in percent)
    instead of being stopped; otherwise it keeps its actual stop return
    (``exit_return_pct``, a fraction -- converted to percent internally).

    Reports per (threshold, group):
      n_stops / n_fired / fired_pct : how many stops the rule touches
      saved / dragged              : of the fired, would-holding-have-been-better
                                     (ret_held > stop return) vs worse
      saved_pct / dragged_pct      : shares of the fired group
      mean_exit_ret_pct            : what the stop actually realized (all stops)
      mean_held_ret_pct            : mean return if held (fired group)
      mean_delta_pct / total_delta_pct : per-trade / summed %-points the rule
                                     adds across ALL stops (positive = worth it)

    ``by``: "all" | "family" (break vs test, from the strategy name) | "strategy".
    ``thresholds``: explicit list, or None for quantiles [0.4..0.8] of the feature
    over the whole stop cohort.
    """
    if df.empty or "exit_close_reason" not in df.columns or feature not in df.columns:
        return pd.DataFrame()
    if "exit_return_pct" not in df.columns or held_col not in df.columns:
        return pd.DataFrame()
    work = df[df["exit_close_reason"].isin(exit_reasons)].copy()
    if len(work) < min_samples:
        return pd.DataFrame()
    work["_exit_ret_pct"] = pd.to_numeric(work["exit_return_pct"], errors="coerce") * 100.0
    work["_held_pct"] = pd.to_numeric(work[held_col], errors="coerce")
    work["_feat"] = pd.to_numeric(work[feature], errors="coerce")

    if thresholds is None:
        vals = work["_feat"].dropna()
        if len(vals) < min_samples:
            return pd.DataFrame()
        thresholds = [float(vals.quantile(q)) for q in (0.4, 0.5, 0.6, 0.7, 0.8)]

    if by == "family":
        strat = work["strategy"].astype(str)
        work["_group"] = np.where(
            strat.str.startswith("entry_break_"), "break",
            np.where(strat.str.startswith("entry_test_"), "test", "other"),
        )
    elif by == "strategy":
        work["_group"] = work["strategy"].astype(str)
    else:
        work["_group"] = "all"

    rows: list[dict[str, Any]] = []
    for group, g in work.groupby("_group"):
        if len(g) < min_samples:
            continue
        exit_ret = g["_exit_ret_pct"].to_numpy(dtype=float)
        held = g["_held_pct"].to_numpy(dtype=float)
        feat = g["_feat"].to_numpy(dtype=float)
        for thr in thresholds:
            fired = np.isfinite(feat) & (feat >= thr)
            n_fired = int(fired.sum())
            if n_fired == 0:
                continue
            would = np.where(fired, held, exit_ret)
            delta = would - exit_ret
            better = fired & np.isfinite(held) & (held > exit_ret)
            worse = fired & np.isfinite(held) & (held <= exit_ret)
            saved = int(better.sum())
            dragged = int(worse.sum())
            valid = np.isfinite(delta)
            rows.append({
                "group": group,
                "threshold": thr,
                "n_stops": int(len(g)),
                "n_fired": n_fired,
                "fired_pct": 100.0 * n_fired / len(g),
                "saved": saved,
                "dragged": dragged,
                "saved_pct": 100.0 * saved / n_fired,
                "dragged_pct": 100.0 * dragged / n_fired,
                "mean_exit_ret_pct": float(np.nanmean(exit_ret)),
                "mean_held_ret_pct": float(np.nanmean(held[fired])) if n_fired else np.nan,
                "mean_delta_pct": float(np.nanmean(delta[valid])) if valid.any() else 0.0,
                "total_delta_pct": float(np.nansum(delta[valid])) if valid.any() else 0.0,
            })
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).sort_values(["group", "threshold"])
    return out.reset_index(drop=True)


def attach_price_extension_features(
    df: pd.DataFrame,
    dict_of_pairs: dict,
    execution_frame: str,
    runup_window: int = 180,
) -> pd.DataFrame:
    """Attach pre-entry "extension / position within the move" measures (Phase B).

    The plan's hypothesis: highly extended entries (already far into the move at
    trigger time) are late/failed and have a worse forward path -- "High extension
    -> bad", reject "excessively extended entries" as a dud filter. Uses the
    frame's existing 42-bar rolling_min/rolling_max plus a longer rolling base.
    Adds per entry:
      position_in_range       : (close - rolling_min)/(rolling_max - rolling_min), 0..1
      pct_above_rolling_low   : % the entry sits above the rolling low (move already run)
      pct_below_rolling_high  : % the entry sits below the rolling high
      pct_above_runup_low     : % above a longer (runup_window) rolling low
      bars_since_rolling_high : bars since close last equalled the rolling high
    """
    df = df.copy()
    if df.empty:
        return df
    cols = ("position_in_range", "pct_above_rolling_low", "pct_below_rolling_high",
            "pct_above_runup_low", "bars_since_rolling_high")
    for col in cols:
        df[col] = np.nan

    for pair, sub in df.groupby("pair"):
        pdata = dict_of_pairs.get(pair)
        if not pdata:
            continue
        frame = pdata.get("dict_of_frames", {}).get(execution_frame)
        if frame is None or frame.empty or "close_time" not in frame.columns:
            continue
        f = frame.sort_values("close_time").reset_index(drop=True)
        if "close" not in f.columns or "rolling_min" not in f.columns or "rolling_max" not in f.columns:
            continue
        close = pd.to_numeric(f["close"], errors="coerce")
        rmin = pd.to_numeric(f["rolling_min"], errors="coerce")
        rmax = pd.to_numeric(f["rolling_max"], errors="coerce")
        rng = (rmax - rmin)
        pos = np.where(rng > 0, (close - rmin) / rng, np.nan)
        pct_above_low = np.where(rmin > 0, (close / rmin - 1.0) * 100.0, np.nan)
        pct_below_high = np.where(close > 0, (rmax / close - 1.0) * 100.0, np.nan)
        runup_min = close.rolling(int(runup_window), min_periods=max(int(runup_window) // 4, 2)).min()
        pct_above_runup = np.where(runup_min > 0, (close / runup_min - 1.0) * 100.0, np.nan)

        high_idx = np.arange(len(f))
        is_high = (close.to_numpy() == rmax.to_numpy()) & rmax.notna().to_numpy()
        last_high = pd.Series(np.where(is_high, high_idx, np.nan)).ffill().to_numpy()
        bars_since = np.where(np.isfinite(last_high), high_idx - last_high, np.nan)

        close_time = pd.to_datetime(f["close_time"], errors="coerce").to_numpy()
        times = pd.to_datetime(sub["time"], errors="coerce").to_numpy()
        p = np.searchsorted(close_time, times)
        p = np.clip(p, 0, len(f) - 1)
        exact = close_time[p] == times
        arrays = {
            "position_in_range": pos,
            "pct_above_rolling_low": pct_above_low,
            "pct_below_rolling_high": pct_below_high,
            "pct_above_runup_low": pct_above_runup,
            "bars_since_rolling_high": bars_since,
        }
        for col, arr in arrays.items():
            df.loc[sub.index, col] = np.where(exact, arr[p], np.nan)
    return df


def price_extension_vs_forward(df: pd.DataFrame, q: int = 4, min_samples: Optional[int] = None) -> pd.DataFrame:
    """Bucket entries by pre-entry extension; does extension predict the path?

    Tests the plan's "High extension -> bad": if the highest extension bucket has
    the worst MFE / MAE / TP-hit / win rate, a cap on position_in_range (or
    pct_above_rolling_low) is a candidate dud filter. Requires
    ``attach_price_extension_features`` to have run.
    """
    if df.empty or "position_in_range" not in df.columns:
        return pd.DataFrame()
    work = df.dropna(subset=["position_in_range"]).copy()
    if len(work) < (min_samples if min_samples is not None else q * 10):
        return pd.DataFrame()
    try:
        work["ext_bucket"] = pd.qcut(work["position_in_range"], q=q, labels=False, duplicates="drop")
    except Exception:
        return pd.DataFrame()

    agg = {
        "n": ("position_in_range", "size"),
        "median_position": ("position_in_range", "median"),
        "median_pct_above_low": ("pct_above_rolling_low", "median"),
        "median_mfe6": ("mfe_6", "median"),
        "median_mfe24": ("mfe_24", "median"),
        "median_mae6": ("mae_6", "median"),
        "median_ret24": ("ret_24", "median"),
        "tp_hit_pct": ("tp_hit", lambda s: 100.0 * s.mean()),
        "median_extension_past_tp": ("extension_past_tp_pct", "median"),
    }
    if "was_taken" in work.columns and "exit_return_pct" in work.columns:
        work["_win"] = work["was_taken"] & (work["exit_return_pct"] > 0)
        agg["traded_win_rate_pct"] = ("_win", lambda s: 100.0 * s.mean())

    out = work.groupby("ext_bucket").agg(**agg)
    out.index.name = "extension_quantile"
    return out


# ---------------------------------------------------------------------------
# Phase B, part 3: gain-ratio & volatility bucket analysis
# ("what makes winners winners" in the gain/vol plane)
# ---------------------------------------------------------------------------

def _prep_forward_work(
    work: pd.DataFrame,
    target_pct: float = 1.0,
    within: int = 12,
) -> pd.DataFrame:
    """Add the derived columns the bucket tables need (hit, traded flags).

    ``_taken_win`` and ``_x_{reason}`` are NaN for untraded entries so their
    means (used below) are taken over the traded subset only -- otherwise the
    untraded rows (exit_close_reason == NaN) would pollute exit-route shares.
    """
    work["hit"] = work[f"bars_to_{int(target_pct)}"] <= within
    if "was_taken" in work.columns and "exit_return_pct" in work.columns:
        wt = work["was_taken"].to_numpy(dtype=bool)
        work["_taken_win"] = np.where(wt, (work["exit_return_pct"] > 0).to_numpy(), np.nan)
        for reason in ("stop_loss", "time_stop", "trailing_stop", "vol_take_profit"):
            work[f"_x_{reason}"] = np.where(
                wt, (work["exit_close_reason"] == reason).to_numpy(), np.nan
            )
    return work


def _forward_bucket_spec(work: pd.DataFrame) -> dict:
    """Named-aggregation dict for the per-bucket forward-path table.

    ``winner_share`` / ``immed_dud_share`` read the ``outcome`` column
    (immediate_winner / delayed_winner / immediate_dud / mediocre). Exit-route
    shares (stop_loss_pct, time_stop_pct, ...) are computed among TRADED rows
    only via the NaN-masked ``_x_{reason}`` columns from ``_prep_forward_work``.
    """
    spec = {
        "n": ("pair", "size"),
        "mean_mfe6": ("mfe_6", "mean"),
        "mean_mfe24": ("mfe_24", "mean"),
        "mean_mae6": ("mae_6", "mean"),
        "median_ret24": ("ret_24", "median"),
        "hit_pct": ("hit", lambda s: 100.0 * s.mean()),
        "winner_share": ("outcome", lambda s: 100.0 * s.isin(["immediate_winner", "delayed_winner"]).mean()),
        "immed_dud_share": ("outcome", lambda s: 100.0 * (s == "immediate_dud").mean()),
    }
    if "ret_48" in work.columns:
        spec["median_ret48"] = ("ret_48", "median")
    if "tp_hit" in work.columns:
        spec["tp_hit_pct"] = ("tp_hit", lambda s: 100.0 * s.mean())
    if "was_taken" in work.columns:
        spec["n_traded"] = ("was_taken", "sum")
        spec["traded_win_rate_pct"] = ("_taken_win", lambda s: 100.0 * s.mean())
        spec["mean_exit_ret_pct"] = ("exit_return_pct", "mean")
        # ``_x_{reason}`` is already the NaN-masked boolean indicator for that
        # reason (True/False among traded, NaN among untraded) -> sum / notna.
        for reason, label in (("stop_loss", "stop_loss_pct"), ("time_stop", "time_stop_pct"),
                              ("trailing_stop", "trailing_pct"), ("vol_take_profit", "vol_tp_pct")):
            spec[label] = (f"_x_{reason}",
                           lambda s: 100.0 * (s.sum() / s.notna().sum()) if s.notna().any() else np.nan)
    return spec


def feature_bucket_vs_forward(
    df: pd.DataFrame,
    feature: str,
    q: int = 4,
    min_samples: Optional[int] = None,
    target_pct: float = 1.0,
    within: int = 12,
) -> pd.DataFrame:
    """Bucket entries by ANY attached feature; does it predict the forward path?

    Generalizes the rank-vs-expectancy / price-extension ideas to any feature
    column already on the entry dataset (``rolling_gain``, ``volatility``,
    ``rsi``, ``obv_slope``, ...). Splits into ``q`` quantiles and reports per
    bucket the numbers that separate winners from duds:

      n / mean_feature,
      hit_pct               (bars_to_{target_pct} <= within),
      mean_mfe6 / mean_mfe24 / mean_mae6,
      median_ret24 / median_ret48,
      winner_share / immed_dud_share  (from the outcome column),
      tp_hit_pct             (if tp_hit present),
    and -- when the dataset has been joined to the engine trade_frame --
      n_traded / traded_win_rate_pct / mean_exit_ret_pct,
      stop_loss_pct / time_stop_pct / trailing_pct / vol_tp_pct
      (the actual exit-route mix, among traded entries in the bucket).

    Reading it: a monotone trend across buckets (high feature -> more winners,
    fewer duds, lower stop-loss share) identifies the feature as a live-usable
    entry filter to tighten; a flat table says it adds nothing.

    Requires ``feature`` attached (e.g. via ``attach_snapshot_features``) and the
    standard forward-path columns from ``build_entry_analysis_dataset``.
    """
    if df.empty or feature not in df.columns:
        return pd.DataFrame()
    work = df.dropna(subset=[feature]).copy()
    if len(work) < (min_samples if min_samples is not None else q * 10):
        return pd.DataFrame()
    try:
        work["feat_bucket"] = pd.qcut(work[feature], q=q, labels=False, duplicates="drop")
    except Exception:
        return pd.DataFrame()

    work = _prep_forward_work(work, target_pct=target_pct, within=within)
    spec = _forward_bucket_spec(work)
    out = work.groupby("feat_bucket").agg(**spec)
    out.insert(0, "mean_feature", work.groupby("feat_bucket")[feature].mean())
    out.index.name = f"{feature}_quantile"
    return out


def gain_vol_bucket_analysis(
    df: pd.DataFrame,
    gain_q: int = 4,
    vol_q: int = 4,
    min_samples: Optional[int] = None,
    target_pct: float = 1.0,
    within: int = 12,
) -> pd.DataFrame:
    """2D bucket by entry gain ratio (rolling_gain) x volatility.

    The two live-usable entry filters STEP 3 already applies. One row per
    (gain_quantile, vol_quantile), with the same forward-path columns as
    ``feature_bucket_vs_forward``. Directly answers "what makes winners winners":
    the block where winner_share is high / stop_loss_pct is low is where entries
    are good; the block where duds + stop_loss cluster is what a tighter gain/vol
    gate would filter out.

    Requires ``rolling_gain`` and ``volatility`` columns (attach via
    ``attach_snapshot_features(df, dict_of_pairs, execution_frame,
    columns=["rolling_gain", "volatility"])``).
    """
    if df.empty or "rolling_gain" not in df.columns or "volatility" not in df.columns:
        return pd.DataFrame()
    work = df.dropna(subset=["rolling_gain", "volatility"]).copy()
    if len(work) < (min_samples if min_samples is not None else gain_q * vol_q * 5):
        return pd.DataFrame()
    try:
        work["gain_bucket"] = pd.qcut(work["rolling_gain"], q=gain_q, labels=False, duplicates="drop")
        work["vol_bucket"] = pd.qcut(work["volatility"], q=vol_q, labels=False, duplicates="drop")
    except Exception:
        return pd.DataFrame()

    work = _prep_forward_work(work, target_pct=target_pct, within=within)
    spec = _forward_bucket_spec(work)
    keys = ["gain_bucket", "vol_bucket"]
    out = work.groupby(keys).agg(**spec)
    out.insert(0, "mean_gain", work.groupby(keys)["rolling_gain"].mean())
    out.insert(1, "mean_vol", work.groupby(keys)["volatility"].mean())
    out.index.names = ["gain_quantile", "vol_quantile"]
    return out


# ---------------------------------------------------------------------------
# Phase B part 4: dud vs recoverable autopsy (renovation #3)
# "how much of stop_loss / time_stop is bad ENTRY vs a fixable EXIT problem?"
# ---------------------------------------------------------------------------

_EXT_COHORT_LABELS = ("recoverable_round_trip", "recoverable_dip_then_up",
                      "genuine_dud", "grinder")


def _classify_cohort(mfe: pd.Series, mae: pd.Series,
                     recovery_mfe: float, dud_mae: float) -> np.ndarray:
    """Mutually-exclusive dud/recoverable labels from the forward path.

    - recoverable_round_trip : had a real run (mfe >= recovery_mfe) without a
                               deep dip -> the exit machinery let a winner slip.
    - recoverable_dip_then_up : dipped deep (mae <= dud_mae) AND also ran -> a
                               wider / vol-aware stop would have survived it.
    - genuine_dud             : dipped deep and NEVER ran -> bad ENTRY (filter).
    - grinder                 : neither ran nor dipped deep -> bled to the exit.
    """
    had_run = pd.to_numeric(mfe, errors="coerce") >= recovery_mfe
    deep_dip = pd.to_numeric(mae, errors="coerce") <= dud_mae
    return np.where(
        had_run,
        np.where(deep_dip, "recoverable_dip_then_up", "recoverable_round_trip"),
        np.where(deep_dip, "genuine_dud", "grinder"),
    )


def exit_cohort_autopsy(
    df: pd.DataFrame,
    exit_reason: str = "stop_loss",
    mfe_horizon: int = 6,
    mae_horizon: int = 6,
    recovery_mfe: float = 3.0,
    dud_mae: float = -3.5,
    min_samples: int = 20,
) -> Optional[dict]:
    """Split one exit cohort (stop_loss / time_stop) into genuine duds vs recoverable.

    Reads the forward path (``mfe_{mfe_horizon}`` / ``mae_{mae_horizon}``, i.e. the
    IN-HOLD window, NOT the 48-bar peak) of every traded entry that exited via
    ``exit_reason`` and classifies each into one of:
      recoverable_round_trip  : had a real run the exit didn't capture (exit leak)
      recoverable_dip_then_up : dipped deep but also ran (a wider/vol-aware stop
                                would have survived it -> exit leak)
      genuine_dud             : dipped deep and never ran (bad ENTRY -> filter)
      grinder                 : small both ways, bled to the exit (marginal)
    The headline number is ``recoverable_pct`` = share with a real run -- the
    ceiling on what exit-side fixes can recover. ``median_ret12`` per class is
    what holding to bar 12 would have returned (proxy for 'should we have held?').

    Returns None when the cohort is too small or the dataset has no trade
    attribution (``was_taken`` / ``exit_close_reason`` / forward-path columns).
    """
    if df.empty or "was_taken" not in df.columns or "exit_close_reason" not in df.columns:
        return None
    mfe_col = f"mfe_{int(mfe_horizon)}"
    mae_col = f"mae_{int(mae_horizon)}"
    if mfe_col not in df.columns or mae_col not in df.columns:
        return None
    work = df[(df["was_taken"]) & (df["exit_close_reason"] == exit_reason)].copy()
    if len(work) < min_samples:
        return None
    mfe = pd.to_numeric(work[mfe_col], errors="coerce")
    mae = pd.to_numeric(work[mae_col], errors="coerce")
    work["_class"] = _classify_cohort(mfe, mae, recovery_mfe, dud_mae)
    had_run = mfe >= recovery_mfe
    deep_dip = mae <= dud_mae

    by = work.groupby("_class").agg(
        n=("pair", "size"),
        mean_exit_ret_pct=("exit_return_pct", lambda s: 100.0 * s.mean()),
        mean_mfe=(mfe_col, "mean"),
        mean_mae=(mae_col, "mean"),
        median_ret12=("ret_12", "median"),
    ).reindex(_EXT_COHORT_LABELS).dropna(subset=["n"])

    return {
        "exit_reason": exit_reason,
        "cohort_size": int(len(work)),
        "recoverable_pct": float(100.0 * had_run.mean()),
        "genuine_dud_pct": float(100.0 * (deep_dip & ~had_run).mean()),
        "never_touched_minus1_pct": float(100.0 * (mae > -1.0).mean()),
        "mean_exit_ret_pct": float(100.0 * work["exit_return_pct"].mean()),
        "mean_mfe_pct": float(mfe.mean()),
        "mean_mae_pct": float(mae.mean()),
        "by_class": by,
    }


def exit_cohort_by_feature(
    df: pd.DataFrame,
    exit_reason: str = "stop_loss",
    feature: str = "volatility",
    q: int = 4,
    recovery_mfe: float = 3.0,
    dud_mae: float = -3.5,
    min_samples: int = 20,
) -> pd.DataFrame:
    """Cross-tab one exit cohort by a feature quantile vs recoverable/dud shares.

    Confirms WHERE the recoverable exits live -- e.g. if the high-volatility
    bucket of stop_losses is mostly recoverable, the fix is EXIT-side (vol-aware
    TP/stop), not an entry filter. ``recoverable_pct`` = share that had a real run,
    ``genuine_dud_pct`` = share that dipped deep and never ran.
    """
    if df.empty or "was_taken" not in df.columns or "exit_close_reason" not in df.columns:
        return pd.DataFrame()
    if feature not in df.columns or "mfe_6" not in df.columns or "mae_6" not in df.columns:
        return pd.DataFrame()
    work = df[(df["was_taken"]) & (df["exit_close_reason"] == exit_reason)].dropna(subset=[feature]).copy()
    if len(work) < min_samples:
        return pd.DataFrame()
    had_run = pd.to_numeric(work["mfe_6"], errors="coerce") >= recovery_mfe
    deep_dip = pd.to_numeric(work["mae_6"], errors="coerce") <= dud_mae
    work["_recoverable"] = had_run
    work["_dud"] = deep_dip & ~had_run
    try:
        work["feat_bucket"] = pd.qcut(work[feature], q=q, labels=False, duplicates="drop")
    except Exception:
        return pd.DataFrame()

    g = work.groupby("feat_bucket")
    out = g.agg(
        n=(feature, "size"),
        mean_feat=(feature, "mean"),
        recoverable_pct=("_recoverable", lambda s: 100.0 * s.mean()),
        genuine_dud_pct=("_dud", lambda s: 100.0 * s.mean()),
        mean_exit_ret_pct=("exit_return_pct", lambda s: 100.0 * s.mean()),
        median_mfe6=("mfe_6", "median"),
        median_mae6=("mae_6", "median"),
    )
    out.index.name = f"{feature}_quantile"
    return out


_COHORT_RECOVERY_GROUPS = ("exit_capturable", "stop_dependent", "genuine_dud", "grinder")


def cohort_recovery_potential(
    df: pd.DataFrame,
    exit_reason: str = "stop_loss",
    mfe_horizon: int = 12,
    capture_frac: float = 0.5,
    recovery_mfe: float = 3.0,
    dud_mae: float = -3.5,
    fee_slippage_bps: float = 20.0,
    min_samples: int = 20,
) -> Optional[pd.DataFrame]:
    """Upper-bound sizing of the P&L an exit cohort leaves on the table.

    Read-only sizing guide for the exit work (renovation #5) -- NOT a simulation.
    For every traded entry that exited via ``exit_reason``, classify by the
    in-hold forward path (mfe_6 / mae_6) and read the in-hold peak
    ``mfe_{mfe_horizon}``:

      exit_capturable : had a run (mfe_6 >= recovery_mfe) with NO deep dip
                        (mae_6 > dud_mae). A TP/trailing can reach this peak.
                        would-have ret ~= capture_frac*mfe_h - roundtrip costs.
      stop_dependent  : deep dip (mae_6 <= dud_mae) BUT also ran. The peak is
                        reachable ONLY if a wider/vol-aware stop survives the
                        dip -- mean_mae6 = the stop width that would be needed.
      genuine_dud     : deep dip, never ran -> NOT recoverable by exits (filter).
      grinder         : marginal.

    Columns: n, share_pct, actual_mean_ret_pct, mean_mfe_h, mean_mae6,
    would_have_mean_ret_pct, delta_ret_pct_per_trade, aggregate_delta_ret_pct
    (= n * per-trade delta, in %-points). The COHORT_TOTAL row weights the
    would-have by share (duds keep their actual loss) -> aggregate_delta is the
    cohort-level ceiling on per-trade %-points the exit work could add.
    """
    if df.empty or "was_taken" not in df.columns or "exit_close_reason" not in df.columns:
        return None
    if "exit_return_pct" not in df.columns:
        return None
    mfe6_c, mae6_c, mfeh_c = "mfe_6", "mae_6", f"mfe_{int(mfe_horizon)}"
    if not {mfe6_c, mae6_c, mfeh_c} <= set(df.columns):
        return None
    work = df[(df["was_taken"]) & (df["exit_close_reason"] == exit_reason)].copy()
    if len(work) < min_samples:
        return None

    work["_class"] = _classify_cohort(work[mfe6_c], work[mae6_c], recovery_mfe, dud_mae)
    work["_grp"] = np.where(
        work["_class"] == "recoverable_round_trip", "exit_capturable",
        np.where(work["_class"] == "recoverable_dip_then_up", "stop_dependent",
                 work["_class"]),
    )
    work["_actual"] = 100.0 * pd.to_numeric(work["exit_return_pct"], errors="coerce")
    mfeh = pd.to_numeric(work[mfeh_c], errors="coerce")
    mae6 = pd.to_numeric(work[mae6_c], errors="coerce")
    cost = float(fee_slippage_bps) / 100.0
    work["_would"] = np.where(
        work["_grp"] == "genuine_dud", np.nan,
        np.maximum(capture_frac * mfeh - cost, 0.0),
    )

    rows = []
    for grp in _COHORT_RECOVERY_GROUPS:
        sub = work[work["_grp"] == grp]
        if sub.empty:
            continue
        n = len(sub)
        actual = float(sub["_actual"].mean())
        w = sub["_would"].mean()
        would = float(w) if pd.notna(w) else np.nan
        rows.append({
            "group": grp,
            "n": n,
            "share_pct": 100.0 * n / len(work),
            "actual_mean_ret_pct": actual,
            "mean_mfe_h": float(mfeh.loc[sub.index].mean()),
            "mean_mae6": float(mae6.loc[sub.index].mean()),
            "would_have_mean_ret_pct": would,
            "delta_ret_pct_per_trade": float(w - actual) if pd.notna(w) else 0.0,
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return None
    out["aggregate_delta_ret_pct"] = out["n"] * out["delta_ret_pct_per_trade"]

    total_actual = float(work["_actual"].mean())
    total_would = sum(
        (r["would_have_mean_ret_pct"] if pd.notna(r["would_have_mean_ret_pct"])
         else r["actual_mean_ret_pct"]) * r["share_pct"] / 100.0
        for _, r in out.iterrows()
    )
    total = pd.DataFrame([{
        "group": "COHORT_TOTAL",
        "n": int(len(work)),
        "share_pct": 100.0,
        "actual_mean_ret_pct": total_actual,
        "mean_mfe_h": float(mfeh.mean()),
        "mean_mae6": float(mae6.mean()),
        "would_have_mean_ret_pct": total_would,
        "delta_ret_pct_per_trade": total_would - total_actual,
        "aggregate_delta_ret_pct": len(work) * (total_would - total_actual),
    }])
    return pd.concat([out, total], ignore_index=True).set_index("group")


# =============================================================================
# Phase B part 6: entry CANDLE STRUCTURE vs the reacted level (MAE/MFE study)
#
# Frame-derived, NO payload compile and NO engine run: the entry population is
# the STEP 3 strategy column (entry_break_* / entry_test_*) ANDed with each gate
# column straight off the execution frame, and the forward MAE/MFE path is
# computed from the frame's own O/H/L/C with the module's vectorized helpers.
# Deliberately NO trade attribution (was_taken / exit reason) -- forward path
# only, so the same study runs under every gate (structure_ok / allowed_to_trade).
# =============================================================================

CANDLE_ANATOMY_FEATURES = [
    "body_pct", "range_pct", "body_share", "upper_wick_pct", "lower_wick_pct",
    "close_pos_in_range", "body_spreads", "range_spreads", "n_bull_consec",
    "prev_body_pct", "gap_pct", "n_green_last_6",
]
BREAK_LEVEL_FEATURES = [
    "body_above_L_share", "open_below_L_spreads", "low_below_L_spreads",
    "close_above_L_spreads", "penetration_pct", "L_pos_in_range",
    "prev_close_rel_L_spreads",
]
TEST_LEVEL_FEATURES = [
    "wick_below_L_spreads", "close_above_L_spreads", "open_above_L_spreads",
    "bounce_pct", "test_wick_share", "last_broken_bars",
]


def _family_level_from_entry(entry_col: str):
    """'entry_break_<col>' -> ('break', <col>); 'entry_test_<col>' -> ('test', <col>)."""
    for prefix, family in (("entry_break_", "break"), ("entry_test_", "test")):
        if entry_col.startswith(prefix):
            return family, entry_col[len(prefix):]
    return None, None


def _bool_mask(series: pd.Series) -> np.ndarray:
    """Non-zero/True -> True, NaN/False -> False (mirrors backtest_prep._to_bool_series)."""
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).to_numpy(dtype=bool)
    v = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    return np.where(np.isnan(v), False, v != 0.0)


def _candle_anatomy(
    o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray, spread: np.ndarray,
) -> dict[str, np.ndarray]:
    """Body/wick anatomy of each bar (percent / share / spread units). NaN-safe."""
    n = len(c)
    rng = h - l
    body = c - o
    eps = 1e-12
    rng_s = np.where(np.abs(rng) < eps, np.nan, rng)
    o_s = np.where(np.abs(o) < eps, np.nan, o)
    l_s = np.where(np.abs(l) < eps, np.nan, l)
    spread_s = np.where(np.abs(spread) < eps, np.nan, spread)

    prev_close = np.full(n, np.nan)
    prev_close[1:] = c[:-1]
    prev_body = np.full(n, np.nan)
    prev_body[1:] = body[:-1]
    prev_open = np.full(n, np.nan)
    prev_open[1:] = o[:-1]

    green = (c > o).astype(int)
    n_bull = np.zeros(n, dtype=float)
    run = 0
    for i in range(n):
        run = run + 1 if green[i] else 0
        n_bull[i] = run

    return {
        "body_pct": body / o_s * 100.0,
        "range_pct": rng / l_s * 100.0,
        "body_share": np.abs(body) / rng_s,
        "upper_wick_pct": (h - np.maximum(o, c)) / o_s * 100.0,
        "lower_wick_pct": (np.minimum(o, c) - l) / l_s * 100.0,
        "close_pos_in_range": (c - l) / rng_s,
        "body_spreads": body / spread_s,
        "range_spreads": rng / spread_s,
        "n_bull_consec": n_bull,
        "prev_body_pct": prev_body / prev_open * 100.0,
        "gap_pct": (o - prev_close) / prev_close * 100.0,
        "n_green_last_6": pd.Series(green).rolling(6, min_periods=1).sum().to_numpy(dtype=float),
    }


def _break_level_relation(
    o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray,
    spread: np.ndarray, level: np.ndarray,
) -> dict[str, np.ndarray]:
    """Break entry: level sits INSIDE the body (open<L, low<L, close>L+0.15sp)."""
    n = len(c)
    rng = h - l
    body = c - o
    L = np.asarray(level, dtype=float)
    eps = 1e-12
    rng_s = np.where(np.abs(rng) < eps, np.nan, rng)
    body_s = np.where(np.abs(body) < eps, np.nan, body)
    spread_s = np.where(np.abs(spread) < eps, np.nan, spread)
    L_s = np.where(np.abs(L) < eps, np.nan, L)
    prev_close = np.full(n, np.nan)
    prev_close[1:] = c[:-1]

    return {
        "body_above_L_share": (c - L) / body_s,        # fraction of body above the level (candle_rel_size)
        "open_below_L_spreads": (L - o) / spread_s,    # >0: opened below the level
        "low_below_L_spreads": (L - l) / spread_s,     # >0: wick dipped below the level
        "close_above_L_spreads": (c - L) / spread_s,   # >0: close above the level
        "penetration_pct": (c - L) / L_s * 100.0,
        "L_pos_in_range": (L - l) / rng_s,             # where the level sits in the candle range
        "prev_close_rel_L_spreads": (L - prev_close) / spread_s,  # >0: prior close below the level (clean cross)
    }


def _test_level_relation(
    o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray,
    spread: np.ndarray, level: np.ndarray, last_broken: np.ndarray,
) -> dict[str, np.ndarray]:
    """Test entry: level BELOW the body, lower wick reacts to it (open>L, close>L, low<L)."""
    rng = h - l
    L = np.asarray(level, dtype=float)
    eps = 1e-12
    rng_s = np.where(np.abs(rng) < eps, np.nan, rng)
    spread_s = np.where(np.abs(spread) < eps, np.nan, spread)
    L_s = np.where(np.abs(L) < eps, np.nan, L)

    return {
        "wick_below_L_spreads": (L - l) / spread_s,    # >0: lower wick pierced below the level (the reaction)
        "close_above_L_spreads": (c - L) / spread_s,   # >0: bounce already captured at entry
        "open_above_L_spreads": (o - L) / spread_s,    # >0: opened above the level
        "bounce_pct": (c - L) / L_s * 100.0,
        "test_wick_share": (L - l) / rng_s,            # how deep the wick dipped vs the candle range
        "last_broken_bars": np.asarray(last_broken, dtype=float),
    }


def build_candle_structure_entries(
    dict_of_pairs: dict,
    execution_frame: str,
    strategy_specs: list[dict[str, str]],
    gates: tuple[str, ...] = ("structure_ok", "allowed_to_trade"),
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    pct_targets: tuple[float, ...] = DEFAULT_PCT_TARGETS,
    time_col: str = "close_time",
) -> pd.DataFrame:
    """One row per (pair, strategy, gate, bar) entry + candle structure + forward path.

    BUILT FROM THE FRAMES ONLY -- no payload compile, no engine run. The entry
    population is the STEP 3 strategy column ANDed with each gate column on the
    execution frame; the forward MAE/MFE comes from the frame's own O/H/L/C via
    the module's vectorized helpers. Deliberately NO trade attribution.

    Columns: pair, strategy, family, gate, bar, time, entry_close,
    mfe_{h}/mae_{h}/ret_{h}, bars_to_{pct}, max_favorable, max_adverse, outcome,
    + CANDLE_ANATOMY_FEATURES (shared) + the family's level-relation features
    (BREAK_LEVEL_FEATURES or TEST_LEVEL_FEATURES).
    """
    rows: list[dict[str, Any]] = []
    t_horizon = max(horizons)

    for pair, pdata in dict_of_pairs.items():
        fr = pdata.get("dict_of_frames", {}).get(execution_frame)
        if fr is None or fr.empty or time_col not in fr.columns or "close" not in fr.columns:
            continue
        fr = fr.sort_values(time_col).reset_index(drop=True)

        if "open" in fr.columns:
            o = pd.to_numeric(fr["open"], errors="coerce").to_numpy(dtype=float)
        else:
            continue
        if "high" in fr.columns:
            h = pd.to_numeric(fr["high"], errors="coerce").to_numpy(dtype=float)
        else:
            continue
        if "low" in fr.columns:
            l = pd.to_numeric(fr["low"], errors="coerce").to_numpy(dtype=float)
        else:
            continue
        c = pd.to_numeric(fr["close"], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(c).any():
            continue
        spread = (pd.to_numeric(fr["short_spread"], errors="coerce").to_numpy(dtype=float)
                  if "short_spread" in fr.columns else np.full(len(c), np.nan))
        time_a = pd.to_datetime(fr[time_col], errors="coerce").to_numpy()

        # forward path per pair (once)
        fwd: dict[str, np.ndarray] = {}
        for hh in horizons:
            fwd[f"mfe_{hh}"] = (_forward_extreme(h, hh, "max") / c - 1.0) * 100.0
            fwd[f"mae_{hh}"] = (_forward_extreme(l, hh, "min") / c - 1.0) * 100.0
            fwd[f"ret_{hh}"] = _forward_return(c, hh) * 100.0
        for pct in pct_targets:
            fwd[f"bars_to_{int(pct)}"] = _bars_to_target(h, c, pct, t_horizon)

        anat = _candle_anatomy(o, h, l, c, spread)

        gates_present = {g: _bool_mask(fr[g]) for g in gates if g in fr.columns}
        if not gates_present:
            continue

        for spec in strategy_specs:
            entry_col = spec.get("entry_col", "")
            if entry_col not in fr.columns:
                continue
            family, level_col = _family_level_from_entry(entry_col)
            if family is None or level_col is None or level_col not in fr.columns:
                continue
            entry_mask = _bool_mask(fr[entry_col])
            if not entry_mask.any():
                continue

            L = pd.to_numeric(fr[level_col], errors="coerce").to_numpy(dtype=float)
            if family == "break":
                rel = _break_level_relation(o, h, l, c, spread, L)
            else:
                lb = (pd.to_numeric(fr[f"{level_col}_last_broken"], errors="coerce").to_numpy(dtype=float)
                      if f"{level_col}_last_broken" in fr.columns else np.full(len(c), np.nan))
                rel = _test_level_relation(o, h, l, c, spread, L, lb)

            for gate, gmask in gates_present.items():
                idx = np.flatnonzero(entry_mask & gmask)
                for t in idx:
                    ec = c[t]
                    if not np.isfinite(ec) or ec <= 0:
                        continue
                    row: dict[str, Any] = {
                        "pair": pair,
                        "strategy": spec.get("name", entry_col),
                        "family": family,
                        "gate": gate,
                        "bar": int(t),
                        "time": time_a[t],
                        "entry_close": float(ec),
                    }
                    for hh in horizons:
                        row[f"mfe_{hh}"] = float(fwd[f"mfe_{hh}"][t])
                        row[f"mae_{hh}"] = float(fwd[f"mae_{hh}"][t])
                        row[f"ret_{hh}"] = float(fwd[f"ret_{hh}"][t])
                    for pct in pct_targets:
                        row[f"bars_to_{int(pct)}"] = float(fwd[f"bars_to_{int(pct)}"][t])
                    for feat, arr in anat.items():
                        row[feat] = float(arr[t])
                    for feat, arr in rel.items():
                        row[feat] = float(arr[t])
                    rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        mfe_cols = [f"mfe_{h}" for h in horizons]
        mae_cols = [f"mae_{h}" for h in horizons]
        df["max_favorable"] = df[mfe_cols].max(axis=1)
        df["max_adverse"] = df[mae_cols].min(axis=1)
        df = classify_outcomes(df, horizons=horizons)
    return df


def candle_anatomy_summary(
    df: pd.DataFrame,
    gate: Optional[str] = None,
    features: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Per-family anatomy overview: n, outcome shares, forward path, feature medians.

    Optionally filter to one gate. ``features`` default = CANDLE_ANATOMY_FEATURES;
    any listed feature absent from the rows is skipped.
    """
    work = df if gate is None else df[df["gate"] == gate]
    if work.empty or "family" not in work.columns or "outcome" not in work.columns:
        return pd.DataFrame()
    feats = [f for f in (list(features) if features else list(CANDLE_ANATOMY_FEATURES)) if f in work.columns]

    spec: dict = {
        "n": ("pair", "size"),
        "winner_share": ("outcome", lambda s: 100.0 * s.isin(["immediate_winner", "delayed_winner"]).mean()),
        "immed_dud_share": ("outcome", lambda s: 100.0 * (s == "immediate_dud").mean()),
        "mean_mfe6": ("mfe_6", "mean"),
        "mean_mae6": ("mae_6", "mean"),
        "median_ret24": ("ret_24", "median"),
    }
    for f in feats:
        spec[f"median_{f}"] = (f, "median")
    return work.groupby("family").agg(**spec)


def candle_bucket_vs_forward(
    df: pd.DataFrame,
    feature: str,
    q: int = 4,
    gate: Optional[str] = None,
    min_samples: Optional[int] = None,
    target_pct: float = 1.0,
    within: int = 12,
) -> pd.DataFrame:
    """Quantile-bucket one candle/level feature vs the forward path, BY FAMILY.

    Mirrors ``feature_bucket_vs_forward`` but splits break vs test (they have
    different level-relation semantics) and optionally filters to one gate.
    Rows = (family, feature_quantile); columns = the standard forward-path
    buckets from ``_forward_bucket_spec`` (no trade columns -- no attribution).
    """
    work = df if gate is None else df[df["gate"] == gate]
    if work.empty or feature not in work.columns or "family" not in work.columns:
        return pd.DataFrame()
    work = work.dropna(subset=[feature]).copy()
    min_s = min_samples if min_samples is not None else q * 10

    frames = []
    for fam, sub in work.groupby("family"):
        if len(sub) < min_s:
            continue
        try:
            sub["feat_bucket"] = pd.qcut(sub[feature], q=q, labels=False, duplicates="drop")
        except Exception:
            continue
        sub = _prep_forward_work(sub, target_pct=target_pct, within=within)
        out = sub.groupby("feat_bucket").agg(**_forward_bucket_spec(sub))
        out.insert(0, "mean_feature", sub.groupby("feat_bucket")[feature].mean())
        out.insert(0, "family", fam)
        frames.append(out.reset_index())
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
