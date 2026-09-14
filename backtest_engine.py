from __future__ import annotations

import warnings
from dataclasses import MISSING, dataclass, field, fields, replace
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd


# start_date / end_date bound formats accepted by ``_parse_date``: a whole-day
# ``DD/MM/YYYY`` (legacy) or ``DD/MM/YYYY HH:MM``[:SS] to pin the edge to an
# exact timestamp on the execution timeline.
_DATE_FORMATS = ("%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M", "%d/%m/%Y")
_DATE_ONLY_FORMAT = _DATE_FORMATS[-1]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(init=False)
class EngineConfig:
    """
    All engine parameters in one place.

    Attributes
    ----------
    initial_cash : float
        Starting portfolio cash.
    fee_bps : float
        Fee per side in basis points (e.g. 10 = 0.10%).
    max_net_worth_loss_ratio : float
        Max fraction of equity that can be lost at the stop.
        Used in stop-loss risk cap: cash_by_stop = equity * ratio / stop_loss_ratio.
    max_slippage_allowed : float
        Max fraction of quote_vol_per_unit_change that can be used as position size.
        cash_by_slippage = max_slippage_allowed * qvpuc
    slippage_multiplier : float
        Calibration factor applied to the qvpuc-based slippage model (default 1.0 =
        unchanged). Live orderbook-vs-qvpuc validation showed real slippage is ~2-3x
        the qvpuc model (median ratio 2.83), so set >1 to make the backtest honest
        about liquidity cost. Scales BOTH the estimated fill slippage (entry and
        exit) and, inversely, the slippage sizing cap so ``max_slippage_allowed``
        still bounds the REAL (post-multiplier) slippage:
        cash_by_slippage = (max_slippage_allowed / multiplier) * qvpuc.
    max_number_of_positions : int
        Hard cap on simultaneous open positions across all pairs.
        Excess candidates are rejected (sorted by rank_score first).
    maximum_position_of_total_net : float
        Max fraction of equity per position.
        cash_by_count = min(equity / max_number_of_positions,
                            equity * maximum_position_of_total_net)
    stop_price_col : str
        Name of the precomputed stop-price aux column.
    take_profit_price_col : str
        Name of the precomputed take-profit-price aux column.
    qvpuc_col : str
        Name of the quote_vol_per_unit_change aux column.
    rank_score_col : str
        Name of the rank_score aux column.
    trailing_stop_pct_col : str
        Name of the per-entry trailing stop percentage aux column.
    max_hold_bars_col : str
        Name of the per-entry time-stop horizon aux column.
    vol_take_profit_threshold_col : str
        Name of the per-entry profit-spike threshold aux column.
    min_signal_hold_bars_col : str
        Name of the per-entry minimum hold bars aux column.
    benchmark_pair : str | None
        Optional symbol used for benchmark-relative monthly performance.
    metrics_resample_period : str
        Resample rule for periodic return comparison metrics.
    start_date : str | None
        Optional inclusive lower bound, ``DD/MM/YYYY`` or ``DD/MM/YYYY HH:MM``.
        Date-only bounds start at 00:00 of that day.
    end_date : str | None
        Optional inclusive upper bound, ``DD/MM/YYYY`` or ``DD/MM/YYYY HH:MM``.
        A date-only bound covers the WHOLE day (legacy behaviour); adding a time
        pins the edge to that exact timestamp.
    """
    initial_cash: float = 10_000.0
    fee_bps: float = 10.0
    max_net_worth_loss_ratio: float = 0.05
    max_slippage_allowed: float = 0.005
    slippage_multiplier: float = 1.0
    max_number_of_positions: int = 5
    ceil_number_of_positions: int = 8
    maximum_position_of_total_net: float = 0.20
    # ---- Trailing stop: SHORT_SPREAD MULTIPLES (NOT % of price) ------------
    # short_spread = 84-bar MA of (high - low) on the execution frame, read at
    # the CURRENT bar. Arm  : (close - entry)/short_spread >  trailing_arm_spread_multiple
    #                         Exit : (peak - close)/short_spread >= trailing_exit_spread_multiple
    trailing_arm_spread_multiple: float = 1.0
    trailing_exit_spread_multiple: float = 0.2
    max_hold_bars_val : int = 24
    stop_price_col: str = "stop_price"
    take_profit_price_col: str = "take_profit_price"
    qvpuc_col: str = "quote_vol_per_unit_change"
    rank_score_col: str = "rank_score"
    # Per-entry HARD-STOP RATCHET as a PERCENTAGE of peak price (peak*(1-pct));
    # it raises stop_price and exits via "stop_loss". DISTINCT from the
    # spread-multiple trailing above. The compiled aux column keeps its stable
    # name (strategies.py STRATEGY_AUX_COLUMNS "trailing_stop_pct").
    trailing_stop_pct_col: str = "trailing_stop_pct"
    max_hold_bars_col: str = "max_hold_bars"
    vol_take_profit_threshold_col: str = "vol_take_profit_threshold"
    min_signal_hold_bars_col: str = "min_signal_hold_bars"
    # ---- Entry filter: TP already behind the price (2026-09-10) ------------
    # Reject an entry whose strategy TP line is AT/BELOW the signal bar's close.
    # Such a position has no upside left: the very next evaluation already sees
    # ``h > tp`` satisfied, so it is force-closed on its first eligible bar
    # (``vol_take_profit_below_entry``, always hold_bars == 1). Measured over
    # 2020-10 -> 2021-12: 10 such trades, ALL 1-bar holds, mean return -15.5 %,
    # total pnl -28,442 - the worst exit-reason cohort of the run.
    # Mirrors the live guard (entry_engine.evaluate_live_entry ->
    # EntryDecision.tp_below_entry) so live and backtest select the same trades.
    # A missing/zero TP (payload without the column) never blocks an entry.
    # False = legacy behaviour (take the entry).
    skip_entries_with_tp_below_close: bool = True
    # ---- Live parity mode (2026-09-12) ------------------------------------
    # When True, ``run_portfolio_backtest`` reproduces a real live run instead of
    # being a clean simulation:
    #   * ``parity_seed`` (built by ``live_parity.build_parity_seed``) supplies the
    #     live cash/equity at the window anchor, the positions live held at the
    #     anchor (seeded with a NEGATIVE entry_bar so hold-bar logic keeps counting),
    #     the manual Override closes with their recorded fills, the (pair, bar)
    #     entries whose live order Binance rejected, and the live trade ledger used
    #     for the reconciliation report.
    #   * An intrabar take-profit pass runs BEFORE each bar's exit evaluation, the
    #     way live's 15s TP monitor does: ``high >= take_profit`` closes the position
    #     at the TP line (or at the recorded live sell when one matches) instead of
    #     waiting for the bar close to fill at ``close``.
    #   * Rejected live entries suppress the matching backtest entry on that bar.
    #   * Positions still open at the end are appended to ``trade_frame`` with
    #     ``close_reason="still_open"`` and NaN gain (metrics stay closed-only).
    # False (default) leaves every code path bit-identical to before.
    ensure_parity: bool = False
    parity_seed: dict | None = None
    include_open_in_trade_frame: bool = True
    exposure_multiplier_col: str = "btc_exposure_multiplier"
    benchmark_pair: str | None = "BTCUSDT"
    metrics_resample_period: str = "ME"
    start_date: str | None = None
    end_date: str | None = None
    cash_infusion_amount : float = 0.0
    cash_infusion_count : int = 0
    # new candidate must exceed weakest position's rank_score by this fraction; 0 = disabled
    replacement_score_margin: float = 0.0
    min_hold_before_replacement: int = 0
    # Optional global override for the per-strategy min_signal_hold_bars aux.
    # When > 0, every position uses this value instead of the per-entry column.
    # Used to sweep the minimum-hold-before-exit dimension without recompiling.
    min_signal_hold_bars_override: int = 0
    tp_modifier: float = 1.0
    tp_modifier_caps : list[float] = field(default_factory=lambda: [0.5, 1.5])
    # Strategies that get ENTRY priority 1 (sorted ahead of the default priority 10)
    # when several strategies signal the same pair on the same bar. Membership is
    # what matters (a `strat_name in [...]` test), so REORDERING this list is a
    # no-op; the winner among equal-priority co-signals is resolved by the payload
    # strategy order (backtest) / strategy_id (runtime). Empty list = all default.
    # Default mirrors the previously-hardcoded list: boll_upper_1daily, max, boll_upper.
    priority_1_strategies: list[str] = field(default_factory=lambda: [
        "entry_break_boll_upper_1daily__exit_break_boll_upper_1daily",
        "entry_break_max__exit_break_max",
        "entry_break_boll_upper__exit_break_boll_upper",
    ])

    # Legacy trailing-stop names (percent-era) map onto the spread-multiple knobs
    # above so old call sites and in-flight sweeps keep parsing unchanged.
    _LEGACY_TRAILING_ALIASES = {
        "trailing_stop_threshold": "trailing_arm_spread_multiple",
        "trailing_stop_pct": "trailing_exit_spread_multiple",
    }

    def __init__(self, **kwargs):
        # Remap the two legacy trailing kwarg names with a deprecation warning.
        for legacy, canonical in self._LEGACY_TRAILING_ALIASES.items():
            if legacy in kwargs:
                warnings.warn(
                    f"EngineConfig.{legacy} is deprecated; use {canonical} "
                    "(units: multiples of short_spread).",
                    DeprecationWarning,
                    stacklevel=2,
                )
                kwargs[canonical] = kwargs.pop(legacy)
        for f in fields(type(self)):
            if not f.init:
                continue
            if f.name in kwargs:
                setattr(self, f.name, kwargs.pop(f.name))
            elif f.default is not MISSING:
                setattr(self, f.name, f.default)
            elif f.default_factory is not MISSING:
                setattr(self, f.name, f.default_factory())
            else:
                raise TypeError(f"EngineConfig missing required field {f.name!r}")
        if kwargs:
            raise TypeError(
                f"EngineConfig got unexpected keyword argument(s): {sorted(kwargs)}"
            )

    @property
    def trailing_stop_pct(self) -> float:
        """Deprecated read-only alias for ``trailing_exit_spread_multiple``."""
        return self.trailing_exit_spread_multiple

    @property
    def trailing_stop_threshold(self) -> float:
        """Deprecated read-only alias for ``trailing_arm_spread_multiple``."""
        return self.trailing_arm_spread_multiple


# ---------------------------------------------------------------------------
# State objects
# ---------------------------------------------------------------------------

@dataclass
class PositionState:
    pair: str
    strategy: str
    regime: Any
    entry_bar: int
    entry_time: Any
    entry_price: float
    initial_stop_price: float
    stop_price: float
    peak_price: float
    # % hard-stop ratchet (peak*(1 - trailing_ratchet_pct)) that raises stop_price
    # and exits via "stop_loss". DISTINCT from the cfg spread-multiple trailing.
    trailing_ratchet_pct: float
    max_hold_bars: int
    vol_take_profit_threshold: float
    min_signal_hold_bars: int
    amount: float       # units of the asset
    entry_value: float  # cash deployed (before fees)
    entry_fee: float    # fee paid at entry
    entry_rank_score: float = 0.0


@dataclass
class TradeRecord:
    pair: str
    strategy: str
    regime: Any
    entry_bar: int
    entry_time: Any
    entry_price: float
    quantity: float
    initial_stop_price: float
    stop_price: float
    entry_value: float
    entry_fee: float
    exit_bar: int
    exit_time: Any
    exit_price: float
    exit_value: float
    exit_fee: float
    pnl: float          # net of both fees
    return_pct: float
    hold_bars: int
    close_reason: str


# ---------------------------------------------------------------------------
# Sizing
# ---------------------------------------------------------------------------

def _effective_slippage_multiplier(cfg: EngineConfig) -> float:
    """Sanitised EngineConfig.slippage_multiplier (NaN/<=0 -> 1.0 neutral)."""
    mult = float(getattr(cfg, "slippage_multiplier", 1.0))
    if not np.isfinite(mult) or mult <= 0.0:
        return 1.0
    return mult


def compute_entry_cash(
    equity: float,
    cash: float,
    close_price: float,
    stop_price: float,
    qvpuc: float,
    open_position_count: int,
    cfg: EngineConfig,
    exposure_multiplier: float | None = 1.0,
) -> float:
    """
    Return the maximum cash to deploy for one entry given three sizing caps.

    Caps
    ----
    cash_by_stop : risk the most we are willing to lose at the stop.
    cash_by_slippage : avoid moving the market too much.
    cash_by_count : enforce position-count and concentration limits.

    exposure_multiplier : float | None
        Regime-based capital-allocation score in [0, 1] (BTC SQN tier).
        Scales the final min-of-caps result. None/NaN -> 1.0 (neutral).
    """
    if close_price <= 0 or stop_price <= 0 or stop_price >= close_price:
        return 0.0

    stop_loss_ratio = max(1.0 - stop_price / close_price, 1e-6)

    cash_by_stop = (equity * cfg.max_net_worth_loss_ratio) / stop_loss_ratio

    if (qvpuc > 0) & (cfg.max_slippage_allowed > 0):
        # Divide by the calibration multiplier so ``max_slippage_allowed`` still bounds
        # the REAL (post-multiplier) slippage: at the cap, slippage = (cap/qvpuc)*mult
        # = max_slippage_allowed. A multiplier > 1 therefore shrinks positions.
        mult = _effective_slippage_multiplier(cfg)
        cash_by_slippage = (cfg.max_slippage_allowed / mult) * qvpuc
    else:
        cash_by_slippage = cash_by_stop  # no volume info: unconstrained by slippage

    cash_by_count = min(
        equity / max(cfg.max_number_of_positions, 1),
        equity * cfg.maximum_position_of_total_net,
    )

    final_cash = float(min(cash_by_stop, cash_by_slippage, cash_by_count, cash))

    # Regime-based exposure scaling (BTC SQN -> total capital to deploy).
    multiplier = 1.0 if exposure_multiplier is None else exposure_multiplier
    if not np.isfinite(multiplier):
        multiplier = 1.0
    multiplier = float(np.clip(multiplier, 0.0, 1.0))
    final_cash = final_cash * multiplier

    return final_cash


def _parse_date(value: str | None, field_name: str) -> tuple[pd.Timestamp | None, bool]:
    """Parse a ``start_date``/``end_date`` bound into ``(timestamp, has_time)``.

    Accepted formats:
      * ``DD/MM/YYYY``           - whole-day bound (:func:`_build_date_mask`
        keeps the legacy semantics: ``end_date`` includes the entire end day)
      * ``DD/MM/YYYY HH:MM``     - bound pinned to that exact minute
      * ``DD/MM/YYYY HH:MM:SS``  - same, with seconds

    An already-parsed ``Timestamp``/``datetime`` is accepted as-is, and ``None``
    (or an empty string) means "no bound".
    """
    if value is None:
        return None, False
    if isinstance(value, str) and not value.strip():
        return None, False
    if isinstance(value, (pd.Timestamp, datetime)):
        ts = pd.Timestamp(value)
        if ts.tz is not None:
            ts = ts.tz_localize(None)
        return ts, ts != ts.normalize()

    text = str(value).strip()
    for fmt in _DATE_FORMATS:
        try:
            ts = pd.to_datetime(text, format=fmt)
            break
        except (ValueError, TypeError):
            continue
    else:
        raise ValueError(
            f"{field_name} must be a date string in DD/MM/YYYY or "
            f"DD/MM/YYYY HH:MM format"
        ) from None
    return ts, fmt != _DATE_ONLY_FORMAT


def _build_date_mask(timeline: pd.Series, start_date: str | None, end_date: str | None) -> np.ndarray:
    if timeline.empty:
        return np.zeros(0, dtype=bool)

    start_ts, start_has_time = _parse_date(start_date, "start_date")
    end_ts, end_has_time = _parse_date(end_date, "end_date")

    if start_ts is not None and end_ts is not None and start_ts > end_ts:
        raise ValueError("start_date must be earlier than or equal to end_date")

    timeline_ts = pd.to_datetime(
        pd.Series(timeline).reset_index(drop=True), errors="coerce"
    )
    if getattr(timeline_ts.dt, "tz", None) is not None:
        timeline_ts = timeline_ts.dt.tz_localize(None)
    mask = pd.Series(True, index=np.arange(len(timeline_ts)))

    if start_ts is not None:
        # Date-only start = 00:00 of that day; a timed start is exact.
        mask &= timeline_ts >= (start_ts if start_has_time else start_ts.normalize())
    if end_ts is not None:
        # Date-only end stays INCLUSIVE of the whole day (legacy behaviour); a
        # timed end cuts at that exact timestamp.
        end_edge = end_ts if end_has_time else _end_of_day(end_ts)
        mask &= timeline_ts <= end_edge

    return mask.to_numpy(dtype=bool)


def _end_of_day(ts: pd.Timestamp) -> pd.Timestamp:
    """Last representable timestamp of ``ts``'s calendar day."""
    return ts.normalize() + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)


def _is_nested_aux(aux: dict) -> bool:
    return bool(aux) and all(isinstance(v, dict) for v in aux.values())


def _get_strategy_aux_arrays(
    aux: dict,
    strategy_name: str,
    date_mask: np.ndarray,
    filtered_T: int,
    n_size: int,
    cfg: EngineConfig,
) -> dict[str, np.ndarray]:
    strategy_aux = aux.get(strategy_name, {}) if _is_nested_aux(aux) else aux

    def _get(col: str, default: float = 0.0) -> np.ndarray:
        if col in strategy_aux:
            return strategy_aux[col][date_mask, :]
        return np.full((filtered_T, n_size), default, dtype=np.float64)

    return {
        "stop_arr": _get(cfg.stop_price_col),
        "tp_arr": _get(cfg.take_profit_price_col),
        "qvpuc_arr": _get(cfg.qvpuc_col),
        "rank_arr": _get(cfg.rank_score_col),
        "max_hold_bars_arr": _get(cfg.max_hold_bars_col),
        "vol_take_profit_threshold_arr": _get(cfg.vol_take_profit_threshold_col),
        "min_signal_hold_bars_arr": _get(cfg.min_signal_hold_bars_col),
        # Neutral default (1.0) when the column is absent, so backtests that
        # predate the exposure multiplier are not flattened to zero size.
        "exposure_arr": _get(cfg.exposure_multiplier_col, default=1.0),
        # Same array under its literal column name, so engine-side code can read
        # strat_aux["btc_exposure_multiplier"] directly (mirrors payload aux).
        "btc_exposure_multiplier": _get(cfg.exposure_multiplier_col, default=1.0),
        # short_spread (common execution-frame column) exposed to exit evaluation.
        "short_spread_arr": _get("short_spread"),
        # Renamed for clarity: per-entry PERCENT hard-stop ratchet (stable column).
        "trailing_ratchet_pct_arr": _get(cfg.trailing_stop_pct_col),
    }


# ---------------------------------------------------------------------------
# Core simulation loop
# ---------------------------------------------------------------------------

def _run_one_strategy(
    strategy_name: str,
    entry_arr: np.ndarray,   # (T, N) bool
    exit_arr: np.ndarray,    # (T, N) bool
    close_arr: np.ndarray,   # (T, N) float64
    high_arr: np.ndarray,    # (T, N) float64
    timeline: pd.Series,
    pairs: list[str],
    stop_arr: np.ndarray,    # (T, N) float64
    tp_arr: np.ndarray,      # (T, N) float64
    short_spread_arr: np.ndarray,   # (T, N) float64 (exposed to exit eval like close_arr)
    qvpuc_arr: np.ndarray,   # (T, N) float64
    rank_arr: np.ndarray,    # (T, N) float64
    trailing_ratchet_pct_arr: np.ndarray,   # (T, N) float64  per-entry % stop ratchet
    max_hold_bars_arr: np.ndarray,       # (T, N) float64
    vol_take_profit_threshold_arr: np.ndarray,  # (T, N) float64
    min_signal_hold_bars_arr: np.ndarray,       # (T, N) float64
    exposure_arr: np.ndarray | None,            # (T, N) float64
    regime_arr: np.ndarray | None,
    cfg: EngineConfig,
) -> tuple[list[TradeRecord], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Walk the timeline for one strategy and return closed trades, the equity
    curve, and the per-bar state series (cash / open positions / closed
    trades) captured at the same cadence as the equity mark.

    Single-strategy runner used by ``run_backtest``. It delegates to the shared
    portfolio simulation loop (``_run_portfolio``) with the same helpers used by
    ``run_portfolio_backtest`` (``_open_position_from_candidate`` and
    ``_evaluate_and_close_position``), so the two pipelines stay in parity:

    - identical exit semantics (trailing / time / vol-take-profit / stop-loss /
      strategy-exit) with slippage on exit and take-profit-price fills,
    - identical entry sizing (stop-risk / slippage / position-count caps) plus
      regime-based exposure scaling via ``btc_exposure_multiplier``,
    - identical position-replacement and cash-infusion behavior.

    Each call simulates one strategy on its own independent capital pool.
    """
    strategy_arrays = {
        strategy_name: {"entry": entry_arr, "exit": exit_arr},
    }
    strategy_aux_arrays = {
        strategy_name: {
            "stop_arr": stop_arr,
            "tp_arr": tp_arr,
            "short_spread_arr": short_spread_arr,
            "qvpuc_arr": qvpuc_arr,
            "rank_arr": rank_arr,
            "trailing_ratchet_pct_arr": trailing_ratchet_pct_arr,
            "max_hold_bars_arr": max_hold_bars_arr,
            "vol_take_profit_threshold_arr": vol_take_profit_threshold_arr,
            "min_signal_hold_bars_arr": min_signal_hold_bars_arr,
            "exposure_arr": exposure_arr,
        }
    }

    # Per-strategy mode is not parity-aware: keep the historical 5-tuple contract.
    return _run_portfolio(
        strategy_arrays=strategy_arrays,
        strategy_aux_arrays=strategy_aux_arrays,
        close_arr=close_arr,
        high_arr=high_arr,
        timeline=timeline,
        pairs=pairs,
        regime_arr=regime_arr,
        cfg=cfg,
    )[:5]

import numpy as np
from numba import njit


@njit(cache=True)
def rolling_max_runup(close_arr: np.ndarray, window: int) -> np.ndarray:
    """
    Computes the maximum low->high gain within a rolling window.

    Parameters
    ----------
    close_arr : float64[:, :]
        Shape (T, N)

    window : int
        Rolling window length (e.g. 180 for 30 days of 4h candles)

    Returns
    -------
    runup : float64[:, :]
        Shape (T, N)

        runup[t,n] is the maximum gain obtainable by buying at the
        lowest price and selling later within the previous `window`
        candles ending at t.

        Invalid values are NaN.
    """

    T, N = close_arr.shape

    runup = np.full((T, N), np.nan)

    for pair in range(N):

        for t in range(window - 1, T):

            lowest = np.inf
            best_gain = -np.inf
            valid = False

            start = t - window + 1

            for i in range(start, t + 1):

                price = close_arr[i, pair]

                if np.isnan(price):
                    continue

                if price < lowest:
                    lowest = price

                if lowest > 0:

                    gain = price / lowest - 1.0

                    if gain > best_gain:
                        best_gain = gain
                        valid = True

            if valid:
                runup[t, pair] = best_gain

    return runup


@njit(cache=True)
def rolling_top_n(runup: np.ndarray, top_n: int) -> np.ndarray:
    """
    Returns indices of the top-N gainers.

    Output shape:
        (T, top_n)

    Invalid entries are -1.
    """

    T, N = runup.shape

    result = np.full((T, top_n), -1, dtype=np.int32)

    for t in range(T):

        top_idx = np.full(top_n, -1, dtype=np.int32)
        top_val = np.full(top_n, -np.inf)

        for pair in range(N):

            value = runup[t, pair]

            if np.isnan(value):
                continue

            if value <= top_val[-1]:
                continue

            pos = top_n - 1

            while pos > 0 and value > top_val[pos - 1]:

                top_val[pos] = top_val[pos - 1]
                top_idx[pos] = top_idx[pos - 1]

                pos -= 1

            top_val[pos] = value
            top_idx[pos] = pair

        result[t] = top_idx

    return result


def get_rolling_top_gainers(
    close_arr: np.ndarray,
    days: int = 30,
    candles_per_day: int = 6,
    top_n: int = 10,
):
    """
    Returns
    -------
    top_idx : int32[:, :]
        Shape (T, top_n)

    runup : float64[:, :]
        Shape (T, N)
    """

    window = days * candles_per_day

    runup = rolling_max_runup(close_arr, window)

    top_idx = rolling_top_n(runup, top_n)

    return top_idx, runup


# ---------------------------------------------------------------------------
# Live parity helpers (see EngineConfig.ensure_parity)
# ---------------------------------------------------------------------------

def _parity_timeline_index(timeline_full: pd.Series, value: Any) -> int | None:
    """Index of the bar whose close equals ``value`` (falls back to the last one before it)."""
    ts = pd.to_datetime(value, errors="coerce", utc=True)
    if ts is None or pd.isna(ts):
        return None
    stamps = pd.to_datetime(timeline_full, errors="coerce", utc=True)
    if stamps.isna().all():
        return None
    exact = np.flatnonzero(stamps.values == np.datetime64(ts.tz_convert("UTC").tz_localize(None)))
    if exact.size:
        return int(exact[0])
    naive = stamps.dt.tz_localize(None).values
    position = int(np.searchsorted(naive, np.datetime64(ts.tz_localize(None)), side="right")) - 1
    return position if position >= 0 else None


PARITY_PASS_LATENCY = pd.Timedelta(minutes=10)


def _parity_bar_index(timeline_full: pd.Series, value: Any, interval: str = "4h") -> int | None:
    """Index of the bar whose **processing pass** an unscheduled event belongs to.

    ``_parity_timeline_index`` maps a stamp to the bar that closed at/before it, which
    is right for a fill of a decision taken at that close (live's entry fills land ~7
    minutes after the close). A manual override is unscheduled: if it lands inside a bar
    — i.e. more than ``CLOSE_TOLERANCE`` after the previous close — it takes effect
    during the bar being processed, whose close comes NEXT. Mapping it to the bar that
    closed just before put it one bar early, where the position it closed did not exist
    yet (2026-09-12: the TFUELUSDT override at 09-11 08:12:06 was applied at bar 18,
    before the bar-18 entry, and never matched).
    """
    moment = pd.to_datetime(value, errors="coerce", utc=True)
    if moment is None or pd.isna(moment):
        return None
    close_in_progress = moment.ceil(interval) - pd.Timedelta(milliseconds=1)
    previous_close = close_in_progress - pd.Timedelta(interval)
    # The live pass for a bar runs ~7-8 min after its close (measured: 07:59:59.999 ->
    # 08:07:15), so anything arriving inside that window is handled by THAT pass; later
    # than that, the bar being processed is the next one.
    within_pass = (moment - previous_close) <= PARITY_PASS_LATENCY
    target = previous_close if within_pass else close_in_progress
    return _parity_timeline_index(timeline_full, target)


def _parity_entry_stamp(value: Any) -> str:
    """Bar-granularity identity of an entry stamp (tz-agnostic)."""
    moment = pd.to_datetime(value, errors="coerce", utc=True)
    if moment is None or pd.isna(moment):
        return "NaT"
    return str(moment.floor("4h"))


def _align_timestamp(value: Any, reference: Any) -> Any:
    """Return ``value`` as a timestamp matching ``reference``'s tz-awareness.

    Live records are tz-aware UTC (live_parity reads the bot's own stamps) while
    the payload timeline is usually tz-naive. A TradeRecord list / trade_frame
    column cannot hold both (``pd.to_datetime`` raises "Cannot mix tz-aware with
    tz-naive values"), so recorded live stamps are converted onto the engine's
    convention: naive reference -> naive UTC wall clock, aware -> aware UTC.
    """
    if value is None:
        return None
    moment = pd.to_datetime(value, errors="coerce", utc=True)
    if moment is None or pd.isna(moment):
        return value
    try:
        reference_naive = pd.Timestamp(reference).tz is None
    except (ValueError, TypeError):
        reference_naive = True
    return moment.tz_localize(None) if reference_naive else moment


def _parity_aux_scalars(
    payload: dict, strat_name: str, t_full: int, n: int, cfg: EngineConfig
) -> dict[str, float]:
    """Aux values for one (strategy, pair, bar) on the UNMASKED payload.

    Used to rebuild a seeded position's stop / TP anchors / ratchet / hold caps from
    the same arrays a normal entry would read, so the seeded position behaves exactly
    like one the engine opened itself.
    """
    aux = payload.get("aux", {}) or {}
    strategy_aux = aux.get(strat_name, {}) if _is_nested_aux(aux) else aux

    def _get(col: str, default: float = 0.0) -> float:
        arr = strategy_aux.get(col)
        if arr is None:
            return float(default)
        value = arr[t_full, n]
        return float(value) if value is not None and not np.isnan(value) else float(default)

    return {
        "stop": _get(cfg.stop_price_col),
        "tp": _get(cfg.take_profit_price_col),
        "qvpuc": _get(cfg.qvpuc_col),
        "rank": _get(cfg.rank_score_col),
        "trailing_ratchet_pct": _get(cfg.trailing_stop_pct_col),
        "max_hold_bars": _get(cfg.max_hold_bars_col),
        "vol_take_profit_threshold": _get(cfg.vol_take_profit_threshold_col),
        "min_signal_hold_bars": _get(cfg.min_signal_hold_bars_col),
    }


def _resolve_parity_strategy(strategy_id: Any, strategy_names: list[str]) -> str | None:
    """Map a live strategy id onto a payload strategy key.

    Live stores catalog ids (``momentum_hh_hl``, ``break_max``, ``test_prev_low``, ...)
    while the payload keys are ``f"{entry_col}__{exit_col}"`` (``entry_momentum_hh_hl__
    exit_momentum_hh_hl``), so try the exact name, the notebook's naming convention,
    then a unique suffix match.
    """
    if strategy_id is None:
        return None
    sid = str(strategy_id)
    if sid in strategy_names:
        return sid
    candidate = f"entry_{sid}__exit_{sid}"
    if candidate in strategy_names:
        return candidate
    matches = [
        name
        for name in strategy_names
        if name.endswith(f"__{sid}") or name.startswith(f"{sid}__") or name.split("__")[0] == f"entry_{sid}"
    ]
    if len(matches) == 1:
        return matches[0]
    return None


def _build_parity_context(
    payload: dict,
    cfg: EngineConfig,
    *,
    date_mask: np.ndarray,
    timeline_full: pd.Series,
    pairs: list[str],
    pair_index: dict[str, int],
    strategy_names: list[str] | None = None,
    interval: str = "4h",
) -> dict[str, Any]:
    """Normalise ``cfg.parity_seed`` into masked-timeline indices the loop can use."""
    context: dict[str, Any] = {
        "active": bool(cfg.ensure_parity),
        "cash": None,
        "positions": [],
        "overrides": [],
        "suppressed": {},
        "live_trades": [],
        "stats": {"seeded_positions": 0, "seeded_skipped": 0, "overrides": 0, "suppressed_entries": 0},
        "warnings": [],
    }
    if not cfg.ensure_parity:
        return context

    seed = cfg.parity_seed or {}
    if not seed:
        raise ValueError("ensure_parity=True requires cfg.parity_seed (see live_parity.build_parity_seed)")

    first_masked = int(np.flatnonzero(date_mask)[0]) if date_mask.size else 0
    # Convert every recorded live stamp (tz-aware) onto the timeline's convention
    # so seeded rows mix cleanly with engine-generated ones.
    parity_ref_ts = timeline_full.iloc[first_masked] if timeline_full.size > first_masked else None
    context["cash"] = seed.get("cash")
    context["live_trades"] = list(seed.get("live_trades") or [])

    for entry in seed.get("positions") or []:
        pair = str(entry.get("pair") or "").upper()
        strategy = _resolve_parity_strategy(entry.get("strategy"), strategy_names or [])
        if pair not in pair_index or not strategy:
            context["stats"]["seeded_skipped"] += 1
            context["warnings"].append(
                f"seed position skipped (pair/strategy not in payload): {pair} {entry.get('strategy')}"
            )
            continue
        quantity = pd.to_numeric(entry.get("quantity"), errors="coerce")
        entry_price = pd.to_numeric(entry.get("entry_price"), errors="coerce")
        if quantity is None or entry_price is None or pd.isna(quantity) or pd.isna(entry_price):
            context["stats"]["seeded_skipped"] += 1
            context["warnings"].append(f"seed position skipped (missing qty/price): {pair}")
            continue
        t_full = _parity_timeline_index(timeline_full, entry.get("entry_time"))
        aux_t = t_full
        entry_bar: int | None = int(t_full - first_masked) if t_full is not None else None
        if entry_bar is None:
            # The seeded position was opened BEFORE the payload's timeline starts.
            # Express it as a negative offset from the first in-window bar so
            # hold_bars keeps counting, and read its aux from the earliest bar we
            # have (flagged: the true entry bar's aux is not in the payload).
            moment = pd.to_datetime(entry.get("entry_time"), errors="coerce", utc=True)
            if moment is None or pd.isna(moment):
                context["stats"]["seeded_skipped"] += 1
                context["warnings"].append(f"seed position skipped (unparseable entry_time): {pair}")
                continue
            first_ts = pd.to_datetime(timeline_full.iloc[first_masked], errors="coerce", utc=True)
            bars_before = float((first_ts - moment) / pd.Timedelta(interval))
            entry_bar = -int(max(0, np.ceil(bars_before)))
            aux_t = int(first_masked)
            context["stats"]["seeded_aux_approximated"] = (
                int(context["stats"].get("seeded_aux_approximated", 0)) + 1
            )
            context["warnings"].append(
                f"seed aux approximated for {pair}: entry bar precedes the window (entry_bar={entry_bar})"
            )
        aux = _parity_aux_scalars(payload, strategy, aux_t, pair_index[pair], cfg)
        context["positions"].append(
            {
                "pair": pair,
                "strategy": strategy,
                "entry_bar": int(entry_bar),  # negative = opened before the window
                "entry_time": _align_timestamp(entry.get("entry_time"), parity_ref_ts),
                "entry_price": float(entry_price),
                "quantity": float(quantity),
                "aux": aux,
            }
        )
        context["stats"]["seeded_positions"] += 1

    for override in seed.get("overrides") or []:
        pair = str(override.get("pair") or "").upper()
        exit_time = override.get("exit_time")
        # The bar whose pass the override landed in (NOT the bar that closed before it):
        # a manual /sell takes effect as soon as it arrives.
        bar = _parity_bar_index(timeline_full, exit_time, interval) if exit_time else None
        if bar is None and exit_time:
            # The override fires INSIDE the bar whose close follows it, so fall back
            # to the bar the timestamp belongs to.
            moment = pd.to_datetime(exit_time, errors="coerce", utc=True)
            if moment is not None and not pd.isna(moment):
                bar_close = (moment.ceil(interval) - pd.Timedelta(milliseconds=1)).tz_localize(None)
                bar = int(
                    max(
                        np.searchsorted(
                            pd.to_datetime(timeline_full).dt.tz_localize(None).values,
                            np.datetime64(bar_close),
                            side="right",
                        )
                        - 1,
                        0,
                    )
                )
        if bar is None:
            context["warnings"].append(f"override skipped (no bar for {exit_time}): {pair}")
            continue
        # An override can only close a position that exists: never in the bar it was
        # opened on (the engine's override pass runs before that bar's entries).
        entry_bar_full = _parity_timeline_index(timeline_full, override.get("entry_time"))
        if entry_bar_full is not None:
            bar = max(bar, entry_bar_full + 1)
        masked_bar = int(bar - first_masked)
        if masked_bar < 0:
            context["warnings"].append(f"override before the window skipped: {pair} {exit_time}")
            continue
        context["overrides"].append(
            {
                "pair": pair,
                "strategy": override.get("strategy"),
                "bar": masked_bar,
                "exit_time": _align_timestamp(exit_time, parity_ref_ts),
                "price": pd.to_numeric(override.get("exit_price"), errors="coerce"),
                "notional": pd.to_numeric(override.get("exit_notional"), errors="coerce"),
                "quantity": pd.to_numeric(override.get("exit_quantity"), errors="coerce"),
            }
        )
        context["stats"]["overrides"] += 1

    for rejected in seed.get("rejected_entries") or []:
        pair = str(rejected.get("pair") or "").upper()
        bar = _parity_timeline_index(timeline_full, rejected.get("bar_close"))
        if bar is None:
            continue
        masked_bar = int(bar - first_masked)
        if masked_bar < 0:
            continue
        context["suppressed"][(pair, masked_bar)] = str(rejected.get("status") or "rejected")
        context["stats"]["suppressed_entries"] += 1

    return context


def _parity_position_state(entry: dict[str, Any], cfg: EngineConfig, fee_rate: float) -> PositionState:
    """Build the seeded PositionState (live prices/quantity, engine-side aux)."""
    aux = entry["aux"]
    entry_price = float(entry["entry_price"])
    quantity = float(entry["quantity"])
    entry_value = entry_price * quantity
    stop = aux["stop"] if aux["stop"] > 0 else entry_price
    min_signal_hold_bars = aux["min_signal_hold_bars"]
    if cfg.min_signal_hold_bars_override > 0:
        min_signal_hold_bars = float(cfg.min_signal_hold_bars_override)
    return PositionState(
        pair=entry["pair"],
        strategy=entry["strategy"],
        regime=None,
        entry_bar=int(entry["entry_bar"]),
        entry_time=entry["entry_time"],
        entry_price=entry_price,
        initial_stop_price=stop,
        stop_price=stop,
        peak_price=entry_price,
        trailing_ratchet_pct=float(max(aux["trailing_ratchet_pct"], 0.0)),
        max_hold_bars=int(aux["max_hold_bars"]) if aux["max_hold_bars"] > 0 else 0,
        vol_take_profit_threshold=float(max(aux["vol_take_profit_threshold"], 0.0)),
        min_signal_hold_bars=int(max(min_signal_hold_bars, 0)),
        amount=quantity,
        entry_value=entry_value,
        entry_fee=entry_value * fee_rate,
        entry_rank_score=float(aux["rank"]),
    )


def _close_position_for_parity(
    *,
    t: int,
    n: int,
    ts: Any,
    pair: str,
    pos: PositionState,
    cash: float,
    exit_price: float,
    exit_time: Any,
    close_reason: str,
    fee_rate: float,
    quantity: float | None = None,
    exit_notional: float | None = None,
) -> tuple[TradeRecord, float]:
    """Record a parity-driven close (Override / emulated TP) and credit the cash."""
    # Override / matched-live-exit stamps come from the live log (tz-aware) but the
    # rest of the TradeRecord uses the engine timeline stamp -> align it.
    exit_time = _align_timestamp(exit_time, ts)
    sold = float(quantity) if quantity is not None and quantity > 0 else float(pos.amount)
    price = float(exit_price)
    exit_value = float(exit_notional) if exit_notional is not None and exit_notional > 0 else sold * price
    exit_fee = exit_value * fee_rate
    pnl = (exit_value - exit_fee) - (pos.entry_value + pos.entry_fee)
    record = TradeRecord(
        pair=pair,
        strategy=pos.strategy,
        regime=pos.regime,
        entry_bar=pos.entry_bar,
        entry_time=pos.entry_time,
        entry_price=pos.entry_price,
        quantity=sold,
        initial_stop_price=pos.initial_stop_price,
        stop_price=pos.stop_price,
        entry_value=pos.entry_value,
        entry_fee=pos.entry_fee,
        exit_bar=t,
        exit_time=exit_time if exit_time is not None else ts,
        exit_price=price,
        exit_value=exit_value,
        exit_fee=exit_fee,
        pnl=pnl,
        return_pct=(pnl / pos.entry_value) if pos.entry_value > 0 else 0.0,
        hold_bars=int(t - pos.entry_bar),
        close_reason=close_reason,
    )
    return record, cash + exit_value - exit_fee


def _parity_live_exit(parity: dict, pair: str, pos: PositionState, bar_ts: Any) -> dict[str, Any] | None:
    """Match the live sell that closed THIS position (identity, not just pair+reason).

    Matching only on pair + reason handed the recorded sell of an EARLIER position to a
    later one entered on the same pair: the emulation then closed the new position at a
    timestamp from BEFORE its entry (2026-09-12 trade frame: NEWTUSDT and TFUELUSDT rows
    with a negative ``holding_period``, exits on 09-10 03:13 / 09-11 04:07 for positions
    opened hours later). The match therefore also requires the same entry bar, each
    recorded sell is consumed at most once, and a sell that happened after the current
    bar's close is not used here.
    """
    target = _parity_entry_stamp(pos.entry_time)
    used = parity.setdefault("used_live_exits", set())
    bar_limit = pd.to_datetime(bar_ts, errors="coerce", utc=True)
    for index, trade in enumerate(parity.get("live_trades") or []):
        if index in used:
            continue
        if str(trade.get("pair") or "").upper() != pair:
            continue
        if str(trade.get("reason")) != "take_profit":
            continue
        if _parity_entry_stamp(trade.get("entry_time")) != target:
            continue
        exit_ts = pd.to_datetime(trade.get("exit_time"), errors="coerce", utc=True)
        if not pd.isna(exit_ts) and not pd.isna(bar_limit) and exit_ts > bar_limit:
            continue
        used.add(index)
        return trade
    return None


def _evaluate_and_close_position(
    *,
    t: int,
    n: int,
    ts: Any,
    pair: str,
    pos: PositionState,
    strategy_arrays: dict[str, dict[str, np.ndarray]],
    strategy_aux_arrays: dict[str, dict[str, np.ndarray]],
    close_arr: np.ndarray,
    high_arr: np.ndarray,
    zero_arr: np.ndarray,
    fee_rate: float,
    cfg: EngineConfig,
) -> TradeRecord | None:
    c = close_arr[t, n]
    h = high_arr[t, n]

    if np.isnan(c):
        return None

    strategy_entries = [strategy_arrays[strat]["entry"][t, n] for strat in strategy_arrays]
    # previous_bar_entries = [strategy_arrays[strat]["entry"][t-1, n] for strat in strategy_arrays if "test" in strat]
    strat_aux = strategy_aux_arrays.get(pos.strategy, {})
    tp_arr = strat_aux.get("tp_arr", zero_arr)
    stop_arr = strat_aux.get("stop_arr", zero_arr)
    short_spread_arr = strat_aux.get("short_spread_arr", zero_arr)
    tp = tp_arr[t, n] 
    short_spread = short_spread_arr[t,n]

    hold_bars = t - pos.entry_bar
    trailing_exit = False
    stop_loss_exit = False
    time_exit = False
    vol_take_profit_exit = False
    close_reason = "exit_signal"

    strategy_exit = strategy_arrays[pos.strategy]["exit"][t, n]# if not np.any(strategy_entries) else False

    if cfg.max_hold_bars_val > 0:
        if not np.any(strategy_entries):
            # Grinding near entry -> exit at the global max_hold.
            time_exit_dud = hold_bars >= cfg.max_hold_bars_val and ((c - pos.entry_price) / pos.entry_price < 0.01)

            # Has gained (allowed above max_hold) but has stagnated for the last 3
            # bars INCLUDING the current bar (close_arr[t-2:t+1]) -> lock in profit.
            stagnated = np.abs((c - np.mean(close_arr[max(0, t - 2):t + 1, n])) / c) < 0.03
            time_exit_gain = hold_bars >= cfg.max_hold_bars_val and ((c - pos.entry_price) / pos.entry_price >= 0.1) and stagnated
            #
            time_exit = time_exit_dud or time_exit_gain
        # if np.any(strategy_entries):
        #     return None

    # ---- Trailing stop: SHORT_SPREAD MULTIPLES (current-bar spread) ----
    # Arm  : (close - entry)/short_spread >  cfg.trailing_arm_spread_multiple
    # Exit : (peak - close)/short_spread   >= cfg.trailing_exit_spread_multiple
    # short_spread is the 84-bar MA of (high-low) on the execution frame, read at
    # the CURRENT bar. A missing/zero/non-finite spread disables spread-trailing
    # for that bar (no divide-by-zero, no spurious exits on spread-less payloads).
    if np.isfinite(short_spread) and short_spread > 0:
        gain_in_spreads = (c - pos.entry_price) / short_spread
        if gain_in_spreads > cfg.trailing_arm_spread_multiple and not np.any(strategy_entries):
            retracement_in_spreads = (pos.peak_price - c) / short_spread
            trailing_exit = retracement_in_spreads >= cfg.trailing_exit_spread_multiple


        # if np.any(strategy_entries):
        #     return None
        
    if (h > tp):
        # if not np.any(previous_bar_entries):
            vol_take_profit_exit = True

    previous_close = close_arr[t - 1, n] if t > 0 else np.nan
    if (c < pos.stop_price) & (previous_close < pos.stop_price): 

        if ("test" in pos.strategy):
            if (hold_bars > 12):
                stop_loss_exit = True
        else:
            stop_loss_exit = True

        # stop_loss_exit = True


    strategy_exit_allowed = hold_bars >= pos.min_signal_hold_bars
    should_exit = (strategy_exit and strategy_exit_allowed) or trailing_exit or time_exit or vol_take_profit_exit or stop_loss_exit
    if not should_exit:
        if not np.isnan(c):
            pos.peak_price = max(pos.peak_price, c)
        # % HARD-STOP RATCHET (separate from the spread-multiple trailing exit):
        # ratchets the stop up to peak*(1 - trailing_ratchet_pct); a later close
        # below it exits via "stop_loss".
        if pos.trailing_ratchet_pct > 0 and pos.peak_price > 0:
            trailing_stop_price = pos.peak_price * (1.0 - pos.trailing_ratchet_pct)
            pos.stop_price = max(pos.stop_price, trailing_stop_price)
        return None

    if vol_take_profit_exit:
        if tp > pos.entry_price:
            close_reason = "vol_take_profit"

        else:
            
            close_reason = "vol_take_profit_below_entry"
            if hold_bars == 1:
                close_reason
            
    elif trailing_exit:
        close_reason = "trailing_stop"
        
    elif time_exit:
        if time_exit_dud:
            close_reason = "time_stop"
        else:
            close_reason = "profit_lock_in"

        
    elif stop_loss_exit:
        close_reason = "stop_loss"
        
    initial_exit_value = pos.amount * c
    exit_qvpuc_source = strat_aux.get("qvpuc_arr", zero_arr)
    exit_qvpuc = exit_qvpuc_source[t, n]
    _slip_mult = _effective_slippage_multiplier(cfg)
    estimated_slippage = (initial_exit_value / exit_qvpuc if exit_qvpuc > 0 else cfg.max_slippage_allowed) * _slip_mult

    exit_price = c * (1 - estimated_slippage)
    if close_reason == "vol_take_profit":
        initial_exit_value = pos.amount * tp
        estimated_slippage = (initial_exit_value / exit_qvpuc if exit_qvpuc > 0 else cfg.max_slippage_allowed) * _slip_mult
        exit_price = tp * (1 - estimated_slippage)

    exit_value = pos.amount * exit_price
    exit_fee = exit_value * fee_rate
    pnl = (exit_value - exit_fee) - (pos.entry_value + pos.entry_fee)
    ret = pnl / pos.entry_value if pos.entry_value > 0 else 0.0

    return TradeRecord(
        pair=pair,
        strategy=pos.strategy,
        regime=pos.regime,
        entry_bar=pos.entry_bar,
        entry_time=pos.entry_time,
        entry_price=pos.entry_price,
        quantity=pos.amount,
        initial_stop_price=pos.initial_stop_price,
        stop_price=pos.stop_price,
        entry_value=pos.entry_value,
        entry_fee=pos.entry_fee,
        exit_bar=t,
        exit_time=ts,
        exit_price=exit_price,
        exit_value=exit_value,
        exit_fee=exit_fee,
        pnl=pnl,
        return_pct=ret,
        hold_bars=hold_bars,
        close_reason=close_reason,
    )


def _open_position_from_candidate(
    *,
    t: int,
    n: int,
    ts: Any,
    strat_name: str,
    cash: float,
    open_positions: dict[str, PositionState],
    strategy_aux_arrays: dict[str, dict[str, np.ndarray]],
    close_arr: np.ndarray,
    high_arr: np.ndarray,
    pairs: list[str],
    pair_index: dict[str, int],
    regime_arr: np.ndarray | None,
    zero_arr: np.ndarray,
    fee_rate: float,
    cfg: EngineConfig,
) -> tuple[PositionState | None, float]:
    pair = pairs[n]
    c = close_arr[t, n]
    strat_aux = strategy_aux_arrays.get(strat_name, {})
    stop_source = strat_aux.get("stop_arr", zero_arr)
    qvpuc_source = strat_aux.get("qvpuc_arr", zero_arr)
    trailing_source = strat_aux.get("trailing_ratchet_pct_arr", zero_arr)
    max_hold_source = strat_aux.get("max_hold_bars_arr", zero_arr)
    vol_take_profit_source = strat_aux.get("vol_take_profit_threshold_arr", zero_arr)
    min_hold_source = strat_aux.get("min_signal_hold_bars_arr", zero_arr)
    rank_source = strat_aux.get("rank_arr", zero_arr)
    exposure_source = strat_aux.get("exposure_arr", None)

    
    tp_arr = strat_aux.get("tp_arr", zero_arr)
    tp = tp_arr[t, n]

    # if c > tp:
    #     return None, cash

    # --- INITIAL STOP (fixed at entry) ---
    # The initial stop is the STRATEGY stop frozen at the ENTRY bar. It does NOT
    # move with the per-bar stop_price_series (min(open,close)-1.25*short_spread),
    # which previously recomputed each bar and tracked the EMA, so stops rarely
    # got hit and positions dragged. Here the stop is locked once at entry and
    # only ever moves UP via the trailing ratchet (peak*(1-trailing_pct)).
    # This "fixed initial stop + unconditional trigger" produced the 90x result.
    # Do NOT switch to a tight flat % stop: stop = c*(1-0.02) gave only 74x
    # (too many recoverable entries were stopped out).
    stop = stop_source[t, n]
    # stop = c * (1 - cfg.max_net_worth_loss_ratio)   # 74x variant — too tight; keep OFF
    qvpuc = qvpuc_source[t, n]
    trailing_ratchet_pct = trailing_source[t, n]
    max_hold_bars = max_hold_source[t, n]
    vol_take_profit_threshold = vol_take_profit_source[t, n]
    min_signal_hold_bars = min_hold_source[t, n]
    if cfg.min_signal_hold_bars_override > 0:
        min_signal_hold_bars = float(cfg.min_signal_hold_bars_override)

    if np.isnan(stop) or stop <= 0 or np.isnan(qvpuc):
        return None, cash

    equity = cash + sum(
        open_positions[p].amount * close_arr[t, pair_index[p]]
        for p in open_positions
        if not np.isnan(close_arr[t, pair_index[p]])
    )

    exposure = 1.0
    if exposure_source is not None:
        exposure = float(exposure_source[t, n])
        if not np.isfinite(exposure):
            exposure = 1.0

    entry_cash = compute_entry_cash(
        equity=equity,
        cash=cash,
        close_price=c,
        stop_price=stop,
        qvpuc=qvpuc,
        open_position_count=len(open_positions),
        cfg=cfg,
        exposure_multiplier=exposure,
    )

    if entry_cash < 1.0:
        return None, cash

    entry_fee = entry_cash * fee_rate
    _slip_mult = _effective_slippage_multiplier(cfg)
    estimated_slippage = (entry_cash / qvpuc if qvpuc > 0 else cfg.max_slippage_allowed) * _slip_mult
    amount = entry_cash / (c * (1 + estimated_slippage))
    peak_price = c

    if not np.isnan(high_arr[t, n]):
        peak_price = max(peak_price, high_arr[t, n])

    regime = None
    if regime_arr is not None:
        regime = regime_arr[t, n]
        if pd.isna(regime):
            regime = None

    position = PositionState(
        pair=pair,
        strategy=strat_name,
        regime=regime,
        entry_bar=t,
        entry_time=ts,
        entry_price=c,
        initial_stop_price=stop,
        stop_price=stop,
        peak_price=peak_price,
        trailing_ratchet_pct=float(max(trailing_ratchet_pct, 0.0)) if not np.isnan(trailing_ratchet_pct) else 0.0,
        max_hold_bars=int(max_hold_bars) if not np.isnan(max_hold_bars) and max_hold_bars > 0 else 0,
        vol_take_profit_threshold=(
            float(vol_take_profit_threshold)
            if not np.isnan(vol_take_profit_threshold) and vol_take_profit_threshold > 0
            else 0.0
        ),
        min_signal_hold_bars=int(max(min_signal_hold_bars, 0)) if not np.isnan(min_signal_hold_bars) else 0,
        amount=amount,
        entry_value=entry_cash,
        entry_fee=entry_fee,
        entry_rank_score=float(rank_source[t, n]) if not np.isnan(rank_source[t, n]) else 0.0,
    )
    return position, cash - entry_cash


def _collect_ranked_entry_candidates(
    *,
    t: int,
    strategy_arrays: dict[str, dict[str, np.ndarray]],
    strategy_aux_arrays: dict[str, dict[str, np.ndarray]],
    close_arr: np.ndarray,
    pairs: list[str],
    open_positions: dict[str, PositionState],
    zero_arr: np.ndarray,
    cfg: EngineConfig,
) -> list[tuple[float, int, str, int]]:
    raw_candidates: list[tuple[float, int, str, int]] = []

    priority_1 = set(getattr(cfg, "priority_1_strategies", []) or [])
    skip_tp_below_close = bool(getattr(cfg, "skip_entries_with_tp_below_close", False))

    for strat_name, arrs in strategy_arrays.items():

        
        strat_aux = strategy_aux_arrays.get(strat_name, {})
        rank_source = strat_aux.get("rank_arr", zero_arr)
        tp_source = strat_aux.get("tp_arr", zero_arr)

        # if strat_name == "entry_momentum_hh_hl__exit_momentum_hh_hl":
        #     continue

        for n in range(close_arr.shape[1]):
            c = close_arr[t, n]
            if np.isnan(c):
                continue
            if not arrs["entry"][t, n]:
                continue

            # TP already at/below the signal close -> no upside left (the next
            # evaluation sees `h > tp` and force-closes on the following bar).
            # Same rule as live's EntryDecision.tp_below_entry; a missing (NaN/0)
            # TP never blocks, so TP-less payloads are unaffected.
            if skip_tp_below_close:
                tp = tp_source[t, n]
                if np.isfinite(tp) and tp > 0 and tp <= c:
                    continue

            pair = pairs[n]
            if pair in open_positions:
                continue

            strat_rank = rank_source[t, n]
            strat_priority = 1 if strat_name in priority_1 else 10

            raw_candidates.append((strat_rank, n, strat_name, strat_priority))

    raw_candidates.sort(key=lambda x: [x[3], -x[0]], reverse=False)

    seen_pairs: set[str] = set()
    candidates: list[tuple[float, int, str, int]] = []
    for rank_score, n, strat_name, strat_priority in raw_candidates:
        pair = pairs[n]
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        candidates.append((rank_score, n, strat_name, strat_priority))

    return candidates
# ---------------------------------------------------------------------------
# Shared-capital portfolio simulation loop
# ---------------------------------------------------------------------------

def _run_portfolio(
    strategy_arrays: dict[str, dict[str, np.ndarray]],  # {name: {"entry": arr, "exit": arr}}
    strategy_aux_arrays: dict[str, dict[str, np.ndarray]],
    close_arr: np.ndarray,      # (T, N) float64
    high_arr: np.ndarray,       # (T, N) float64
    timeline: pd.Series,
    pairs: list[str],
    regime_arr: np.ndarray | None,
    cfg: EngineConfig,
    slippage_arr: np.ndarray | None = None,
    parity: dict[str, Any] | None = None,
) -> tuple[list[TradeRecord], np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """
    Walk the timeline across all strategies on a single shared capital pool.

    Rules
    -----
    - One position per pair at a time, regardless of which strategy opened it.
    - A position is only exited by the strategy that opened it (engine-side exits
      trailing_stop / time_stop / vol_take_profit are always active).
    - Entry candidates from all strategies are merged and rank-sorted globally.
      When two strategies signal the same pair on the same bar, the one with the
      higher rank_score wins; the other is silently dropped.
    - PositionState.strategy records the owning strategy for exit routing.
    """
    T, N = close_arr.shape
    fee_rate = cfg.fee_bps / 10_000.0

    parity = parity or {}
    parity_active = bool(parity.get("active"))
    parity_stats: dict[str, Any] = dict(parity.get("stats") or {})
    parity_stats.setdefault("overrides_applied", 0)
    parity_stats.setdefault("overrides_unmatched", 0)
    parity_stats.setdefault("emulated_tp", 0)
    parity_stats.setdefault("emulated_tp_matched", 0)
    parity_stats.setdefault("suppressed_entries_used", 0)
    parity_warnings: list[str] = list(parity.get("warnings") or [])

    cash = cfg.initial_cash
    if parity_active and parity.get("cash") is not None:
        # Live free cash at the anchor: it already accounts for the seeded positions,
        # so nothing is deducted for them.
        cash = float(parity["cash"])
    open_positions: dict[str, PositionState] = {}
    if parity_active:
        for entry in parity.get("positions") or []:
            position = _parity_position_state(entry, cfg, fee_rate)
            open_positions[position.pair] = position
    trades: list[TradeRecord] = []
    equity_curve = np.full(T, np.nan, dtype=np.float64)
    # Per-bar state captured at the SAME point as the equity mark (the
    # carried-forward state at the start of each bar's evaluation): cash on
    # hand, number of currently open positions, and the cumulative number of
    # closed trades. All three are index-aligned with timeline / equity_curve.
    cash_curve = np.full(T, np.nan, dtype=np.float64)
    open_positions_curve = np.zeros(T, dtype=np.int64)
    closed_trades_curve = np.zeros(T, dtype=np.int64)
    zero_arr = np.zeros((T, N), dtype=np.float64)

    pair_index = {pair: idx for idx, pair in enumerate(pairs)}

    # gainers, max_gain = get_rolling_top_gainers(close_arr, days=30, candles_per_day=6, top_n=20)
    month_starts = timeline.groupby(timeline.dt.to_period('M')).head(1).index.values

    for t in range(T):
        ts = timeline.iloc[t]

        # ---- mark equity ----
        pos_value = sum(
            open_positions[p].amount * close_arr[t, pair_index[p]]
            for p in open_positions
            if not np.isnan(close_arr[t, pair_index[p]])
        )
        equity = cash + pos_value
        equity_curve[t] = equity
        # State capture at the same cadence as total equity (every 4h bar).
        cash_curve[t] = cash
        open_positions_curve[t] = len(open_positions)
        closed_trades_curve[t] = len(trades)

        exited_this_bar: set[int] = set()

        if parity_active:
            # ---- Override closes (manual /sell) for this bar, BEFORE the engine's
            # own exit evaluation: the recorded live fill is ground truth, and the
            # position is gone by the time the bar's exits/entries are decided.
            for override in [o for o in (parity.get("overrides") or []) if o.get("bar") == t]:
                pair = override["pair"]
                pos = open_positions.get(pair)
                override_price = pd.to_numeric(override.get("price"), errors="coerce")
                if pos is None or override_price is None or pd.isna(override_price) or float(override_price) <= 0:
                    parity_stats["overrides_unmatched"] += 1
                    parity_warnings.append(f"override at bar {t} unmatched for {pair}")
                    continue
                quantity = pd.to_numeric(override.get("quantity"), errors="coerce")
                notional = pd.to_numeric(override.get("notional"), errors="coerce")
                record, cash = _close_position_for_parity(
                    t=t,
                    n=pair_index[pair],
                    ts=ts,
                    pair=pair,
                    pos=pos,
                    cash=cash,
                    exit_price=float(override_price),
                    exit_time=override.get("exit_time"),
                    close_reason="Override",
                    fee_rate=fee_rate,
                    quantity=None if quantity is None or pd.isna(quantity) else float(quantity),
                    exit_notional=None if notional is None or pd.isna(notional) else float(notional),
                )
                trades.append(record)
                del open_positions[pair]
                parity_stats["overrides_applied"] += 1

            # ---- Intrabar take-profit emulation (live's 15s TP monitor): if the
            # just-closed bar's HIGH crossed the TP line, live sold inside that bar,
            # so close it here at the TP line (or at the recorded live sell) instead
            # of letting the close-based rule fill at the bar close.
            for pair in list(open_positions.keys()):
                pos = open_positions[pair]
                n = pair_index[pair]
                if (t - pos.entry_bar) < 1:
                    continue
                tp_arr_pair = strategy_aux_arrays.get(pos.strategy, {}).get("tp_arr", zero_arr)
                tp = tp_arr_pair[t, n]
                bar_high = high_arr[t, n]
                if not np.isfinite(tp) or tp <= 0 or not np.isfinite(bar_high) or bar_high < tp:
                    continue

                live_exit = _parity_live_exit(parity, pair, pos, ts)
                exit_price: Any = None
                exit_notional: Any = None
                exit_quantity: Any = None
                if live_exit is not None:
                    exit_price = pd.to_numeric(live_exit.get("exit_price"), errors="coerce")
                    exit_notional = pd.to_numeric(live_exit.get("exit_notional"), errors="coerce")
                    exit_quantity = pd.to_numeric(live_exit.get("exit_quantity"), errors="coerce")
                if exit_price is None or pd.isna(exit_price) or float(exit_price) <= 0:
                    exit_price = float(tp)
                else:
                    parity_stats["emulated_tp_matched"] += 1

                record, cash = _close_position_for_parity(
                    t=t,
                    n=n,
                    ts=ts,
                    pair=pair,
                    pos=pos,
                    cash=cash,
                    exit_price=float(exit_price),
                    exit_time=(live_exit or {}).get("exit_time"),
                    close_reason="vol_take_profit",
                    fee_rate=fee_rate,
                    quantity=None if exit_quantity is None or pd.isna(exit_quantity) else float(exit_quantity),
                    exit_notional=None if exit_notional is None or pd.isna(exit_notional) else float(exit_notional),
                )
                trades.append(record)
                del open_positions[pair]
                parity_stats["emulated_tp"] += 1

        # ---- exits: each position uses its own strategy's exit array ----
        for pair in list(open_positions.keys()):
            n = pair_index[pair]
            pos = open_positions[pair]
            closed_trade = _evaluate_and_close_position(
                t=t,
                n=n,
                ts=ts,
                pair=pair,
                pos=pos,
                strategy_arrays=strategy_arrays,
                strategy_aux_arrays=strategy_aux_arrays,
                close_arr=close_arr,
                high_arr=high_arr,
                zero_arr=zero_arr,
                fee_rate=fee_rate,
                cfg=cfg,
            )
            if closed_trade is None:
                continue

            trades.append(closed_trade)
            cash += closed_trade.exit_value - closed_trade.exit_fee
            del open_positions[pair]

            if closed_trade.close_reason == "vol_take_profit":
                
                exited_this_bar.add(n)

        # ---- entries: collect from all strategies, rank globally ----
        if len(open_positions) >= cfg.max_number_of_positions:
            if cfg.replacement_score_margin <= 0:
                continue
            # replacement pass: try to displace the weakest open position
            # candidates_for_replace = _collect_ranked_entry_candidates(
            #     t=t,
            #     strategy_arrays=strategy_arrays,
            #     strategy_aux_arrays=strategy_aux_arrays,
            #     close_arr=close_arr,
            #     pairs=pairs,
            #     open_positions=open_positions,
            #     zero_arr=zero_arr,
            #     cfg=cfg,
            # )
            # if candidates_for_replace:
            #     best_rank, best_n, best_strat, _ = candidates_for_replace[0]
            #     weakest_pair = min(
            #         open_positions,
            #         key=lambda p: open_positions[p].entry_rank_score,
            #     )
            #     weakest = open_positions[weakest_pair]
            #     hold_bars_weakest = t - weakest.entry_bar
            #     threshold = weakest.entry_rank_score * (1.0 + cfg.replacement_score_margin)
            #     if (
            #         best_rank > threshold
            #         and hold_bars_weakest >= cfg.min_hold_before_replacement
            #         and weakest_pair not in exited_this_bar
            #     ):
            #         weakest_n = pair_index[weakest_pair]
            #         c_w = close_arr[t, weakest_n]
            #         if not np.isnan(c_w):
            #             exit_value_w = weakest.amount * c_w
            #             exit_fee_w = exit_value_w * fee_rate
            #             pnl_w = (exit_value_w - exit_fee_w) - (weakest.entry_value + weakest.entry_fee)
            #             ret_w = pnl_w / weakest.entry_value if weakest.entry_value > 0 else 0.0
            #             trades.append(TradeRecord(
            #                 pair=weakest_pair,
            #                 strategy=weakest.strategy,
            #                 regime=weakest.regime,
            #                 entry_bar=weakest.entry_bar,
            #                 entry_time=weakest.entry_time,
            #                 entry_price=weakest.entry_price,
            #                 quantity=weakest.amount,
            #                 initial_stop_price=weakest.initial_stop_price,
            #                 stop_price=weakest.stop_price,
            #                 entry_value=weakest.entry_value,
            #                 entry_fee=weakest.entry_fee,
            #                 exit_bar=t,
            #                 exit_time=ts,
            #                 exit_price=c_w,
            #                 exit_value=exit_value_w,
            #                 exit_fee=exit_fee_w,
            #                 pnl=pnl_w,
            #                 return_pct=ret_w,
            #                 hold_bars=hold_bars_weakest,
            #                 close_reason="replaced",
            #             ))
            #             cash += exit_value_w - exit_fee_w
            #             del open_positions[weakest_pair]
            #             exited_this_bar.add(weakest_pair)
            #             position, cash = _open_position_from_candidate(
            #                 t=t, n=best_n, ts=ts, strat_name=best_strat,
            #                 cash=cash, open_positions=open_positions,
            #                 strategy_aux_arrays=strategy_aux_arrays,
            #                 close_arr=close_arr, high_arr=high_arr,
            #                 pairs=pairs, pair_index=pair_index,
            #                 regime_arr=regime_arr, zero_arr=zero_arr,
            #                 fee_rate=fee_rate, cfg=cfg,
            #             )
            #             if position is not None:
            #                 open_positions[position.pair] = position
            # continue

        candidates = _collect_ranked_entry_candidates(
            t=t,
            strategy_arrays=strategy_arrays,
            strategy_aux_arrays=strategy_aux_arrays,
            close_arr=close_arr,
            pairs=pairs,
            open_positions=open_positions,
            zero_arr=zero_arr,
            cfg=cfg,
        )

        for rank_score, n, strat_name, strat_priority in candidates:
            if len(open_positions) >= cfg.max_number_of_positions:
                break

            # if n in exited_this_bar:
                
            #     continue

            if parity_active and (pairs[n], t) in (parity.get("suppressed") or {}):
                # Live's order for this pair on this bar was rejected by Binance and
                # never filled, so the backtest must not take it either.
                parity_stats["suppressed_entries_used"] += 1
                continue

            position, cash = _open_position_from_candidate(
                t=t,
                n=n,
                ts=ts,
                strat_name=strat_name,
                cash=cash,
                open_positions=open_positions,
                strategy_aux_arrays=strategy_aux_arrays,
                close_arr=close_arr,
                high_arr=high_arr,
                pairs=pairs,
                pair_index=pair_index,
                regime_arr=regime_arr,
                zero_arr=zero_arr,
                fee_rate=fee_rate,
                cfg=cfg,
            )
            if position is None:
                continue

            open_positions[position.pair] = position

            if (cfg.cash_infusion_count > 0) and (t in month_starts):
                infusion_amount = cfg.cash_infusion_amount
                cash += infusion_amount
                cfg.cash_infusion_count -= 1

    parity_stats["warnings"] = parity_warnings
    parity_stats["open_at_end"] = [
        {
            "pair": pos.pair,
            "strategy": pos.strategy,
            "entry_time": str(pos.entry_time),
            "entry_price": pos.entry_price,
            "quantity": pos.amount,
            "hold_bars": int((T - 1) - pos.entry_bar),
        }
        for pos in open_positions.values()
    ]
    return (
        trades,
        equity_curve,
        cash_curve,
        open_positions_curve,
        closed_trades_curve,
        {"parity_stats": parity_stats, "open_positions": open_positions},
    )


# ---------------------------------------------------------------------------
# Public runner
# ---------------------------------------------------------------------------

def run_backtest(
    payload: dict,
    cfg: EngineConfig | None = None,
) -> dict:
    """
    Run the backtest engine over all strategy specs in the payload.

    Parameters
    ----------
    payload : dict
        Output of compile_backtest_payload.
        Must include aux arrays: stop_price, quote_vol_per_unit_change, rank_score.
    cfg : EngineConfig | None
        Engine configuration. Defaults to EngineConfig() if None.

    Returns
    -------
    dict
        {
          strategy_name: {
            "trades": list[TradeRecord],
            "equity_curve": np.ndarray (T,),
            "metrics": dict,
          },
          ...
        }
    """
    if cfg is None:
        cfg = EngineConfig()

    T, N = payload["shape"]["T"], payload["shape"]["N"]
    timeline = payload["timeline"]
    pairs = payload["pairs"]
    close_arr = payload["prices"]["close"]
    high_arr = payload["prices"]["high"]
    aux = payload.get("aux", {})
    context = payload.get("context", {})

    date_mask = _build_date_mask(timeline, cfg.start_date, cfg.end_date)
    if date_mask.size and not date_mask.any():
        raise ValueError("No bars fall inside the requested date window")

    timeline = timeline.iloc[date_mask].reset_index(drop=True)
    close_arr = close_arr[date_mask, :]
    high_arr = high_arr[date_mask, :]
    filtered_T = len(timeline)

    regime_arr = context.get("regime")
    if regime_arr is not None:
        regime_arr = regime_arr[date_mask, :]
    benchmark_close_series = _extract_benchmark_close_series(
        pairs=pairs,
        close_arr=close_arr,
        timeline=timeline,
        benchmark_pair=cfg.benchmark_pair,
    )

    results = {}
    for strat_name, signals in payload["strategies"].items():
        entry_arr = signals["entry"][date_mask, :]
        exit_arr = signals["exit"][date_mask, :]
        strategy_aux = _get_strategy_aux_arrays(aux, strat_name, date_mask, filtered_T, N, cfg)
        trades, equity_curve, cash_curve, open_positions_curve, closed_trades_curve = _run_one_strategy(
            strategy_name=strat_name,
            entry_arr=entry_arr,
            exit_arr=exit_arr,
            close_arr=close_arr,
            high_arr=high_arr,
            timeline=timeline,
            pairs=pairs,
            stop_arr=strategy_aux["stop_arr"],
            tp_arr=strategy_aux["tp_arr"],
            short_spread_arr=strategy_aux["short_spread_arr"],
            qvpuc_arr=strategy_aux["qvpuc_arr"],
            rank_arr=strategy_aux["rank_arr"],
            trailing_ratchet_pct_arr=strategy_aux["trailing_ratchet_pct_arr"],
            max_hold_bars_arr=strategy_aux["max_hold_bars_arr"],
            vol_take_profit_threshold_arr=strategy_aux["vol_take_profit_threshold_arr"],
            min_signal_hold_bars_arr=strategy_aux["min_signal_hold_bars_arr"],
            exposure_arr=strategy_aux["exposure_arr"],
            regime_arr=regime_arr,
            cfg=cfg,
        )
        results[strat_name] = {
            "trades": trades,
            "equity_curve": equity_curve,
            "cash_curve": cash_curve,
            "open_positions_curve": open_positions_curve,
            "closed_trades_curve": closed_trades_curve,
            "trade_frame": trades_to_frame(trades),
            "metrics": _compute_metrics(
                trades=trades,
                equity_curve=equity_curve,
                timeline=timeline,
                initial_cash=cfg.initial_cash,
                benchmark_close_series=benchmark_close_series,
                resample_period=cfg.metrics_resample_period,
            ),
        }

    return results


def _max_drawdown_unclipped(equity_curve: np.ndarray) -> float:
    """True max drawdown (no -50% clip).

    The engine's ``max_drawdown_pct`` metric excludes drawdowns at or below
    -50% (``drawdowns[drawdowns > -0.5].min()``), so it cannot distinguish
    configs whose real drawdowns are worse than -50%. This helper reports the
    actual minimum over the full equity path (NaN if no valid equity points).
    """
    eq = np.asarray(equity_curve, dtype=np.float64)
    eq = eq[np.isfinite(eq) & (eq > 0)]
    if eq.size == 0:
        return np.nan
    running_max = np.maximum.accumulate(eq)
    dd = (eq - running_max) / running_max
    return float(dd.min() * 100.0)


def _max_drawdown_sustained_pct(equity_curve: np.ndarray, min_episode_bars: int = 6) -> float:
    """Deepest drawdown inside an episode that lasts >= min_episode_bars bars.

    A single anomalous candle (bad close that immediately reverts) creates a
    one-bar drawdown spike that is excluded here, while a real multi-day crash
    stays below the running max for many bars and is kept. This is the metric
    to trust when the dataset may contain isolated bad candles that the
    cadence sanitizer cannot remove. Default 6 bars = 1 day on 4h bars.
    """
    eq = np.asarray(equity_curve, dtype=np.float64)
    eq = eq[np.isfinite(eq) & (eq > 0)]
    if eq.size == 0:
        return np.nan
    running_max = np.maximum.accumulate(eq)
    dd = (eq - running_max) / running_max
    below = dd < 0.0
    worst = np.nan
    i, n = 0, len(dd)
    while i < n:
        if below[i]:
            j = i
            while j < n and below[j]:
                j += 1
            if (j - i) >= min_episode_bars:
                worst = float(min(worst, dd[i:j].min())) if np.isfinite(worst) else float(dd[i:j].min())
            i = j
        else:
            i += 1
    return float(worst * 100.0) if np.isfinite(worst) else np.nan


def compare_engine_configs(
    payload: dict,
    base_cfg: EngineConfig,
    variants: list[dict],
    labels: list[str] | None = None,
    current_slippage_array: np.ndarray | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run the portfolio backtest over several EngineConfig variants and return
    a side-by-side comparison (one row per variant).

    Each entry in ``variants`` is a dict of EngineConfig field overrides applied
    on top of ``base_cfg`` via dataclasses.replace. Only EngineConfig-level
    knobs can vary here (trailing stop, max hold, exposure multiplier, ...);
    entry-side changes such as the gate column or gain_ratio need a payload
    recompile (STEP 3 / STEP 5).

    Returned columns:
        total_multiple, final_equity, max_drawdown_pct, trades, win_rate,
        expectancy_rate_pct, timeout_pct, avg_win_pct, avg_loss_pct
    """
    if labels is None:
        labels = [f"variant_{i}" for i in range(len(variants))]
    if len(labels) != len(variants):
        raise ValueError("labels and variants must have the same length")

    rows = []
    for label, overrides in zip(labels, variants):
        overrides = overrides or {}
        cfg = replace(base_cfg, **overrides)
        res = run_portfolio_backtest(
            payload, cfg=cfg, current_slippage_array=current_slippage_array
        )
        metrics = res["portfolio"]["metrics"]
        trade_frame = res["portfolio"]["trade_frame"]
        timeout_pct = 0.0
        if len(trade_frame):
            timeout_pct = 100.0 * (trade_frame["close_reason"] == "time_stop").mean()
        rows.append({
            "config": label,
            "total_multiple": metrics.get("total_profit_multiple"),
            "final_equity": metrics.get("final_equity"),
            "max_drawdown_pct": metrics.get("max_drawdown_pct"),
            "max_drawdown_true_pct": _max_drawdown_unclipped(res["portfolio"]["equity_curve"]),
            "max_drawdown_sustained_pct": _max_drawdown_sustained_pct(res["portfolio"]["equity_curve"]),
            "trades": metrics.get("trade_count"),
            "win_rate": metrics.get("win_rate"),
            "expectancy_rate_pct": metrics.get("expectancy_rate_pct"),
            "timeout_pct": timeout_pct,
            "avg_win_pct": metrics.get("avg_win_pct"),
            "avg_loss_pct": metrics.get("avg_loss_pct"),
        })
    df = pd.DataFrame(rows).set_index("config")
    if verbose:
        print(df.round(2).to_string())
    return df


def run_portfolio_backtest(
    payload: dict,
    cfg: EngineConfig | None = None,
    current_slippage_array: np.ndarray | None = None,
) -> dict:
    """
    Run all strategies in the payload on a single shared capital pool.

    Unlike run_backtest (which gives each strategy its own independent capital),
    this function merges all strategy signals into one simulation:
    - Capital is shared across all strategies.
    - Only one position per pair at a time; if two strategies signal the same pair
      on the same bar the one with the higher rank_score wins.
    - A position is only exited by the strategy that opened it (engine-side exits
      trailing_stop / time_stop / vol_take_profit remain active for all positions).

    Returns
    -------
    dict  with one key ``"portfolio"`` containing:
      - ``trades``       : list[TradeRecord]
      - ``equity_curve`` : np.ndarray (T,)          total equity per 4h bar
      - ``cash_curve``   : np.ndarray (T,)          cash on hand per 4h bar
      - ``open_positions_curve`` : np.ndarray (T,)  # of open positions per bar
      - ``closed_trades_curve``  : np.ndarray (T,)  cumulative closed trades/bar
      - ``trade_frame``  : pd.DataFrame  (strategy column shows which strategy traded)
      - ``metrics``      : dict  (combined portfolio metrics)
      - ``by_strategy``  : dict  {strategy_name: {"trade_frame": ..., "metrics": ...}}
    """
    if cfg is None:
        cfg = EngineConfig()

    T, N = payload["shape"]["T"], payload["shape"]["N"]
    timeline = payload["timeline"]
    pairs = payload["pairs"]
    close_arr = payload["prices"]["close"]
    high_arr = payload["prices"]["high"]
    aux = payload.get("aux", {})
    context = payload.get("context", {})

    date_mask = _build_date_mask(timeline, cfg.start_date, cfg.end_date)
    if date_mask.size and not date_mask.any():
        raise ValueError("No bars fall inside the requested date window")

    timeline = timeline.iloc[date_mask].reset_index(drop=True)
    close_arr = close_arr[date_mask, :]
    high_arr = high_arr[date_mask, :]
    filtered_T = len(timeline)
    

    regime_arr = context.get("regime")
    if regime_arr is not None:
        regime_arr = regime_arr[date_mask, :]
    benchmark_close_series = _extract_benchmark_close_series(
        pairs=pairs,
        close_arr=close_arr,
        timeline=timeline,
        benchmark_pair=cfg.benchmark_pair,
    )

    strategy_arrays = {
        strat_name: {
            "entry": signals["entry"][date_mask, :],
            "exit": signals["exit"][date_mask, :],
        }
        for strat_name, signals in payload["strategies"].items()
    }

    strategy_aux_arrays = {
        strat_name: _get_strategy_aux_arrays(aux, strat_name, date_mask, filtered_T, N, cfg)
        for strat_name in strategy_arrays
    }

    parity_context = _build_parity_context(
        payload,
        cfg,
        date_mask=date_mask,
        timeline_full=payload["timeline"],
        pairs=pairs,
        pair_index={pair: idx for idx, pair in enumerate(pairs)},
        strategy_names=list(strategy_arrays.keys()),
    )

    (
        trades,
        equity_curve,
        cash_curve,
        open_positions_curve,
        closed_trades_curve,
        parity_result,
    ) = _run_portfolio(
        strategy_arrays=strategy_arrays,
        strategy_aux_arrays=strategy_aux_arrays,
        close_arr=close_arr,
        high_arr=high_arr,
        timeline=timeline,
        pairs=pairs,
        regime_arr=regime_arr,
        cfg=cfg,
        slippage_arr=current_slippage_array,
        parity=parity_context,
    )

    portfolio_metrics = _compute_metrics(
        trades=trades,
        equity_curve=equity_curve,
        timeline=timeline,
        initial_cash=cfg.initial_cash,
        benchmark_close_series=benchmark_close_series,
        resample_period=cfg.metrics_resample_period,
    )

    # by_strategy: dict[str, dict] = {}
    # for strat_name in strategy_arrays:
    #     strat_trades = [tr for tr in trades if tr.strategy == strat_name]
    #     by_strategy[strat_name] = {
    #         "trade_frame": trades_to_frame(strat_trades),
    #         "metrics": _compute_metrics(
    #             trades=strat_trades,
    #             equity_curve=equity_curve,
    #             timeline=timeline,
    #             initial_cash=cfg.initial_cash,
    #             benchmark_close_series=benchmark_close_series,
    #             resample_period=cfg.metrics_resample_period,
    #         ),
    #     }

    closed_frame = trades_to_frame(trades)
    open_frame = pd.DataFrame()
    parity_report: dict[str, Any] = {"ensure_parity": bool(cfg.ensure_parity)}
    if cfg.ensure_parity:
        # Positions still open at the end: rows with NaN gain, excluded from metrics.
        open_records: list[TradeRecord] = []
        last_t = len(timeline) - 1
        for pair in pairs:
            pos = (parity_result.get("open_positions") or {}).get(pair)
            if pos is None:
                continue
            mark = close_arr[last_t, pairs.index(pair)] if last_t >= 0 else np.nan
            open_records.append(
                TradeRecord(
                    pair=pair,
                    strategy=pos.strategy,
                    regime=pos.regime,
                    entry_bar=pos.entry_bar,
                    entry_time=pos.entry_time,
                    entry_price=pos.entry_price,
                    quantity=pos.amount,
                    initial_stop_price=pos.initial_stop_price,
                    stop_price=pos.stop_price,
                    entry_value=pos.entry_value,
                    entry_fee=pos.entry_fee,
                    exit_bar=last_t,
                    exit_time=None,
                    exit_price=float(mark) if np.isfinite(mark) else np.nan,
                    exit_value=float(mark) * pos.amount if np.isfinite(mark) else np.nan,
                    exit_fee=0.0,
                    pnl=np.nan,
                    return_pct=np.nan,
                    hold_bars=int(last_t - pos.entry_bar),
                    close_reason="still_open",
                )
            )
        open_frame = trades_to_frame(open_records) if open_records else pd.DataFrame()
        closed_frame["is_open"] = False
        if not open_frame.empty:
            open_frame["is_open"] = True
        parity_report = {
            "ensure_parity": True,
            "seed_anchor": (cfg.parity_seed or {}).get("anchor_ts"),
            "seed_cash": (cfg.parity_seed or {}).get("cash"),
            "seed_equity": (cfg.parity_seed or {}).get("equity"),
            "stats": parity_result.get("parity_stats", {}),
            "reconcile_hint": "live_parity.reconcile_parity(cfg.parity_seed, result['portfolio']['trade_frame'])",
        }

    trade_frame = closed_frame
    if cfg.ensure_parity and cfg.include_open_in_trade_frame and not open_frame.empty:
        trade_frame = pd.concat([closed_frame, open_frame], ignore_index=True)

    return {
        "portfolio": {
            "trades": trades,
            "equity_curve": equity_curve,
            "cash_curve": cash_curve,
            "open_positions_curve": open_positions_curve,
            "closed_trades_curve": closed_trades_curve,
            "trade_frame": trade_frame,
            "open_positions_frame": open_frame,
            "parity_report": parity_report,
            "metrics": portfolio_metrics,
            # "by_strategy": by_strategy,
        }
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _to_datetime_tolerant(values: pd.Series) -> pd.Series:
    """``pd.to_datetime`` that survives a mix of tz-aware and tz-naive stamps.

    Parity mode can put a recorded live stamp (tz-aware UTC) in the same column as
    engine timeline stamps (usually tz-naive), which pandas refuses to parse
    ("Cannot mix tz-aware with tz-naive values") — and, worse, ``errors="coerce"``
    silently turns the offending value into NaT. So the mix is detected up front and
    only then parsed element by element; the offset is dropped, which keeps the same
    UTC wall clock and is a no-op for values that were already naive. Non-mixed
    columns take the plain path unchanged.
    """
    def _awareness(value: Any) -> bool | None:
        if value is None:
            return None
        try:
            stamp = pd.Timestamp(value)
        except (ValueError, TypeError):
            return None
        return None if pd.isna(stamp) else (stamp.tz is not None)

    known = {state for state in (_awareness(value) for value in values) if state is not None}
    if len(known) > 1:
        def _naive_utc(value: Any) -> Any:
            try:
                moment = pd.to_datetime(value, errors="coerce", utc=True)
            except (ValueError, TypeError):
                return value
            if moment is None or pd.isna(moment):
                return value
            return moment.tz_localize(None)

        return pd.Series(
            [_naive_utc(value) for value in values], index=values.index, dtype="datetime64[ns]"
        )

    try:
        return pd.to_datetime(values, errors="coerce")
    except (ValueError, TypeError):
        parsed = pd.to_datetime(values, errors="coerce", utc=True)
        return parsed.dt.tz_localize(None) if parsed.dt.tz is not None else parsed


def trades_to_frame(trades: list[TradeRecord]) -> pd.DataFrame:
    columns = [
        "pair",
        "strategy",
        "regime",
        "entry_bar",
        "entry_time",
        "entry_price",
        "initial_stop_price",
        "stop_price",
        "entry_value",
        "entry_fee",
        "exit_bar",
        "exit_time",
        "exit_price",
        "exit_value",
        "exit_fee",
        "pnl",
        "return_pct",
        "hold_bars",
        "close_reason",
    ]

    if not trades:
        return pd.DataFrame(columns=columns + [
            "open_time",
            "close_time",
            "profit",
            "gross_profit",
            "gain",
            "holding_period",
        ])

    trade_frame = pd.DataFrame([t.__dict__ for t in trades]).copy()
    trade_frame["open_time"] = _to_datetime_tolerant(trade_frame["entry_time"])
    trade_frame["close_time"] = _to_datetime_tolerant(trade_frame["exit_time"])
    trade_frame["profit"] = pd.to_numeric(trade_frame["pnl"], errors="coerce")
    trade_frame["gross_profit"] = (
        pd.to_numeric(trade_frame["exit_value"], errors="coerce")
        - pd.to_numeric(trade_frame["entry_value"], errors="coerce")
    )
    trade_frame["gain"] = 1.0 + pd.to_numeric(trade_frame["return_pct"], errors="coerce")
    try:
        trade_frame["holding_period"] = trade_frame["close_time"] - trade_frame["open_time"]
    except TypeError:
        # Still-open rows carry no exit timestamp (all-NaT close_time), which pandas
        # cannot subtract from a tz-aware open_time. Only produced by parity mode.
        trade_frame["holding_period"] = pd.NaT
    trade_frame = trade_frame.sort_values(["open_time", "close_time", "pair"]).reset_index(drop=True)
    return trade_frame


def _extract_benchmark_close_series(
    pairs: list[str],
    close_arr: np.ndarray,
    timeline: pd.Series,
    benchmark_pair: str | None,
) -> pd.Series | None:
    if benchmark_pair is None or benchmark_pair not in pairs:
        return None

    pair_idx = pairs.index(benchmark_pair)
    benchmark_series = pd.Series(
        close_arr[:, pair_idx],
        index=pd.to_datetime(timeline, errors="coerce"),
        dtype=np.float64,
    ).dropna()

    if benchmark_series.empty:
        return None

    return benchmark_series[~benchmark_series.index.duplicated(keep="last")].sort_index()


def _safe_compounded_multiple(total_multiple: float, periods: float) -> float | None:
    if periods is None or periods <= 0 or total_multiple <= 0:
        print(f"Invalid input for compounded multiple: total_multiple={total_multiple}, periods={periods}")
        return None
    return float(total_multiple ** (1.0 / periods))


def _safe_months_to_target(monthly_multiple: float | None, target_multiple: float) -> float | None:
    if monthly_multiple is None or monthly_multiple <= 1.0:
        return None
    return float(np.log(target_multiple) / np.log(monthly_multiple))


def _round_or_none(value: Any, digits: int = 2) -> Any:
    if value is None:
        return None
    if isinstance(value, pd.Timedelta):
        return value
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    return round(float(value), digits)


def _timedelta_floor_seconds(value: pd.Timedelta | None) -> pd.Timedelta | None:
    if value is None or pd.isna(value):
        return None
    return pd.to_timedelta(np.floor(value.total_seconds()), unit="s")


def format_metrics_report(metrics: dict) -> str:
    return "\n".join([
        f"total profit multiple: {metrics.get('total_profit_multiple')}",
        f"final equity: {metrics.get('final_equity'):,.0f}",
        f"max drawdown: {metrics.get('max_drawdown_pct')}%",
        f"number of trades: {metrics.get('trade_count')}",
        f"expectancy_flat: {metrics.get('expectancy_flat')}",
        f"expectancy_rate: {metrics.get('expectancy_rate_pct')}%",
        f"average_performance_vs_btc: {metrics.get('average_excess_vs_btc_pct')}%",
        f"max_win_p: {metrics.get('max_win_pct')}%",
        f"average_win: {metrics.get('avg_win_pct')}%",
        f"max_loss_p: {metrics.get('max_loss_pct')}%",
        f"average_loss: {metrics.get('avg_loss_pct')}%",
        f"win rate: {metrics.get('win_rate')}%",
        f"start date and time: {metrics.get('start_time')}",
        f"end date and time: {metrics.get('end_time')}",
        f"duration: {metrics.get('duration')}",
        f"average_holding_period: {metrics.get('average_holding_period')}",
        f"estimated daily profit multiple: {metrics.get('profit_per_day_multiple')}",
        f"estimated profit per trade multiple: {metrics.get('profit_per_trade_multiple')}",
        f"estimated monthly profit multiple: {metrics.get('profit_per_month_multiple')}",
        f"months until 100x: {metrics.get('months_until_100x')}",
        f"months until 1000x: {metrics.get('months_until_1000x')}",
        f"months until 5000x: {metrics.get('months_until_5000x')}",
    ])


def _compute_metrics(
    trades: list[TradeRecord],
    equity_curve: np.ndarray,
    timeline: pd.Series,
    initial_cash: float,
    benchmark_close_series: pd.Series | None = None,
    resample_period: str = "ME",
) -> dict:
    trade_frame = trades_to_frame(trades)
    equity_index = pd.to_datetime(timeline, errors="coerce")
    equity_series = pd.Series(equity_curve, index=equity_index, dtype=np.float64).dropna()
    equity_series = equity_series[~equity_series.index.duplicated(keep="last")].sort_index()

    trade_count = len(trade_frame)
    wins = trade_frame[trade_frame["gain"] >= 1.0] if trade_count else trade_frame
    losses = trade_frame[trade_frame["gain"] < 1.0] if trade_count else trade_frame
    win_rate_fraction = (len(wins) / trade_count) if trade_count else None

    avg_win_flat = wins["profit"].mean() if len(wins) > 0 else None
    avg_loss_flat = losses["profit"].mean() if len(losses) > 0 else None
    avg_win_pct = (wins["gain"] - 1.0).mean() * 100 if len(wins) > 0 else None
    avg_loss_pct = (losses["gain"] - 1.0).mean() * 100 if len(losses) > 0 else None

    expectancy_flat = None
    expectancy_rate_pct = None
    if win_rate_fraction is not None:
        expectancy_flat = (
            win_rate_fraction * (avg_win_flat if avg_win_flat is not None else 0.0)
            + (1.0 - win_rate_fraction) * (avg_loss_flat if avg_loss_flat is not None else 0.0)
        )
        expectancy_rate_pct = (
            win_rate_fraction * (avg_win_pct if avg_win_pct is not None else 0.0)
            + (1.0 - win_rate_fraction) * (avg_loss_pct if avg_loss_pct is not None else 0.0)
        )

    if not equity_series.empty:
        total_profit_multiple = float(equity_series.iloc[-1] / equity_series.iloc[0])
        net_return_pct = (total_profit_multiple - 1.0) * 100.0
        running_max = equity_series.cummax()
        drawdowns = (equity_series - running_max) / running_max
        max_drawdown_pct = drawdowns[drawdowns > -0.5].min() * 100 if len(drawdowns) > 0 else 0.0
    else:#
        total_profit_multiple = 1.0
        net_return_pct = 0.0
        max_drawdown_pct = 0.0

    start_time = None
    end_time = None
    duration = None
    total_days = None
    average_holding_period = None
    avg_hold_bars = None

    if trade_count:
        start_time = trade_frame.iloc[0]["open_time"]
        end_time = trade_frame.iloc[-1]["close_time"]
        duration = end_time - start_time if pd.notna(start_time) and pd.notna(end_time) else None
        average_holding_period = _timedelta_floor_seconds(trade_frame["holding_period"].mean())
        avg_hold_bars = trade_frame["hold_bars"].mean()
    elif not equity_series.empty:
        start_time = equity_series.index[0]
        end_time = equity_series.index[-1]
        duration = end_time - start_time
        final_equity = equity_series.iloc[-1]

    if duration is not None and pd.notna(duration):
        total_days = max(1, duration.total_seconds() / 86400.0)

    profit_per_day_multiple = _safe_compounded_multiple(total_profit_multiple, total_days)
    profit_per_trade_multiple = _safe_compounded_multiple(total_profit_multiple, float(trade_count)) if trade_count else None
    profit_per_month_multiple = (
        float(profit_per_day_multiple ** 30.0)
        if profit_per_day_multiple is not None
        else None
    )

    strategy_period_returns = pd.Series(dtype=np.float64)
    average_excess_vs_btc_pct = None
    downside_deviation_pct = None
    if not equity_series.empty:
        strategy_period_values = equity_series.resample(resample_period).last().dropna()
        strategy_period_returns = strategy_period_values.pct_change().dropna()
        downside = strategy_period_returns[strategy_period_returns < 0]
        if not downside.empty:
            downside_deviation_pct = downside.std() * 100.0

        if benchmark_close_series is not None and not benchmark_close_series.empty:
            benchmark_period_values = benchmark_close_series.resample(resample_period).last().dropna()
            benchmark_period_returns = benchmark_period_values.pct_change().dropna()
            aligned = pd.concat(
                [strategy_period_returns, benchmark_period_returns],
                axis=1,
                join="inner",
            ).dropna()
            if not aligned.empty:
                aligned.columns = ["strategy", "benchmark"]
                average_excess_vs_btc_pct = (aligned["strategy"] - aligned["benchmark"]).mean() * 100.0

    max_win_pct = (trade_frame["gain"] - 1.0).max() * 100 if trade_count else None
    max_loss_pct = (trade_frame["gain"] - 1.0).min() * 100 if trade_count else None

    return {
        "trade_count": trade_count,
        "final_equity": _round_or_none(equity_series.iloc[-1] if not equity_series.empty else None, 2),
        "win_rate": _round_or_none(win_rate_fraction * 100.0 if win_rate_fraction is not None else None, 2),
        "avg_win_pct": _round_or_none(avg_win_pct, 2),
        "avg_loss_pct": _round_or_none(avg_loss_pct, 2),
        "expectancy_pct": _round_or_none(expectancy_rate_pct, 2),
        "expectancy_flat": _round_or_none(expectancy_flat, 2),
        "expectancy_rate_pct": _round_or_none(expectancy_rate_pct, 2),
        "net_return_pct": _round_or_none(net_return_pct, 2),
        "total_profit_multiple": _round_or_none(total_profit_multiple, 4),
        "max_drawdown_pct": _round_or_none(max_drawdown_pct, 2),
        "avg_hold_bars": _round_or_none(avg_hold_bars, 1),
        "average_win": _round_or_none(avg_win_flat, 2),
        "average_loss": _round_or_none(avg_loss_flat, 2),
        "max_win_pct": _round_or_none(max_win_pct, 2),
        "max_loss_pct": _round_or_none(max_loss_pct, 2),
        "start_time": start_time,
        "end_time": end_time,
        "duration": duration,
        "average_holding_period": average_holding_period,
        "profit_per_day_multiple": _round_or_none(profit_per_day_multiple, 6),
        "profit_per_trade_multiple": _round_or_none(profit_per_trade_multiple, 6),
        "profit_per_month_multiple": _round_or_none(profit_per_month_multiple, 4),
        "months_until_100x": _round_or_none(_safe_months_to_target(profit_per_month_multiple, 100.0), 2),
        "months_until_1000x": _round_or_none(_safe_months_to_target(profit_per_month_multiple, 1000.0), 2),
        "months_until_5000x": _round_or_none(_safe_months_to_target(profit_per_month_multiple, 5000.0), 2),
        "average_excess_vs_btc_pct": _round_or_none(average_excess_vs_btc_pct, 4),
        "downside_deviation_pct": _round_or_none(downside_deviation_pct, 4),
        "period_return_count": int(len(strategy_period_returns)),
    }


# ---------------------------------------------------------------------------
# Summary helper
# ---------------------------------------------------------------------------

def results_summary(results: dict) -> pd.DataFrame:
    """
    Return a compact comparison DataFrame, one row per strategy.
    """
    rows = []
    for name, res in results.items():
        metrics = res["metrics"]
        row = {
            "strategy": name,
            "final_equity": metrics.get("final_equity"),
            "total_profit_multiple": metrics.get("total_profit_multiple"),
            "net_return_pct": metrics.get("net_return_pct"),
            "trade_count": metrics.get("trade_count"),
            "win_rate": metrics.get("win_rate"),
            "expectancy_flat": metrics.get("expectancy_flat"),
            "expectancy_rate_pct": metrics.get("expectancy_rate_pct"),
            "avg_win_pct": metrics.get("avg_win_pct"),
            "avg_loss_pct": metrics.get("avg_loss_pct"),
            "max_win_pct": metrics.get("max_win_pct"),
            "max_loss_pct": metrics.get("max_loss_pct"),
            "max_drawdown_pct": metrics.get("max_drawdown_pct"),
            "average_excess_vs_btc_pct": metrics.get("average_excess_vs_btc_pct"),
            "profit_per_month_multiple": metrics.get("profit_per_month_multiple"),
            "months_until_100x": metrics.get("months_until_100x"),
            "average_holding_period": metrics.get("average_holding_period"),
        }
        rows.append(row)
    return pd.DataFrame(rows).set_index("strategy")
