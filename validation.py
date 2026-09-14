import numpy as np
import pandas as pd
import json
import pickle
import sqlite3
from datetime import timezone
from pathlib import Path
from dataclasses import replace
from backtest_engine import *
# Private helpers are not re-exported by the star import above: date bounds are
# parsed by the engine's own parser so every window rebuild matches the engine.
from backtest_engine import _build_date_mask, _end_of_day, _parse_date
from backtest_prep import build_strategy_specs, compile_backtest_payload
from binance.client import Client
from features import (
    add_hh_ll_columns,
    attach_sqn_to_dataset,
    get_common_aux_columns,
    prepare_df,
    populate_historical_levels_fast,
)
from gating import compute_4h_structure_for_dataset, prepare_execution_frame_dataset
from helpers import get_simple_slope_2, level_test_buys_with_last_broken
from strategies import STRATEGY_AUX_COLUMNS, apply_break_strategy, apply_test_strategy


FRAME_INTERVALS = {
    "1daily": "1d",
    "4hourly": "4h",
}

FRAME_SECONDS = {
    "1daily": 24 * 3600,
    "4hourly": 4 * 3600,
}

VALIDATION_HISTORY_BUFFER_DAYS = 500


def _to_utc_timestamp(value) -> pd.Timestamp | None:
    ts = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(ts):
        return None
    return ts


def _to_ms(ts: pd.Timestamp) -> int:
    return int(ts.tz_convert("UTC").timestamp() * 1000)


def _build_frame_from_raw_klines(raw: list[list]) -> pd.DataFrame:
    if not raw:
        return pd.DataFrame()

    frame = pd.DataFrame(
        raw,
        columns=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_asset_volume",
            "number_of_trades",
            "market_buy",
            "taker_buy_quote_asset_volume",
            "ignore",
        ],
    )
    frame["open_time"] = pd.to_datetime(frame["open_time"], unit="ms", utc=True, errors="coerce")
    frame["close_time"] = pd.to_datetime(frame["close_time"], unit="ms", utc=True, errors="coerce")
    for column in frame.columns:
        if column not in {"open_time", "close_time"}:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.dropna(subset=["open_time", "close_time"]).reset_index(drop=True)


def _fetch_klines_sync(client: Client, pair: str, interval: str, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DataFrame:
    raw = client.get_historical_klines(
        pair,
        interval,
        start_str=_to_ms(start_ts),
        end_str=_to_ms(end_ts),
    )
    return _build_frame_from_raw_klines(raw)


def _merge_frames(existing: pd.DataFrame | None, incoming: pd.DataFrame | None) -> pd.DataFrame:
    if existing is None or existing.empty:
        return incoming.copy() if incoming is not None else pd.DataFrame()
    if incoming is None or incoming.empty:
        return existing.copy()

    merged = pd.concat([existing, incoming], axis=0, ignore_index=True)
    time_key = "open_time" if "open_time" in merged.columns else "close_time"
    merged = merged.sort_values(time_key).drop_duplicates(subset=[time_key], keep="last").reset_index(drop=True)
    return merged


def _missing_ranges_for_frame(
    frame: pd.DataFrame | None,
    required_start: pd.Timestamp,
    required_end: pd.Timestamp,
    frame_name: str,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    interval_seconds = FRAME_SECONDS[frame_name]
    interval_delta = pd.Timedelta(seconds=interval_seconds)
    tolerance = pd.Timedelta(seconds=int(interval_seconds * 1.5))
    ranges: list[tuple[pd.Timestamp, pd.Timestamp]] = []

    if frame is None or frame.empty:
        return [(required_start, required_end)]

    if "open_time" not in frame.columns:
        return [(required_start, required_end)]

    times = pd.to_datetime(frame["open_time"], errors="coerce", utc=True).dropna().sort_values().drop_duplicates().reset_index(drop=True)
    if times.empty:
        return [(required_start, required_end)]

    first = times.iloc[0]
    last = times.iloc[-1]

    if first > required_start + interval_delta:
        ranges.append((required_start, first))

    deltas = times.diff()
    for idx in range(1, len(times)):
        gap = deltas.iloc[idx]
        if pd.isna(gap) or gap <= tolerance:
            continue
        gap_start = times.iloc[idx - 1] + interval_delta
        gap_end = times.iloc[idx] - interval_delta
        if gap_end > gap_start:
            ranges.append((gap_start, gap_end))

    if last < required_end - interval_delta:
        ranges.append((last, required_end))

    # Remove tiny/inverted ranges.
    clean_ranges = []
    for start_ts, end_ts in ranges:
        if end_ts > start_ts + pd.Timedelta(minutes=1):
            clean_ranges.append((start_ts, end_ts))
    return clean_ranges


def _ensure_dataset_pair_frames(dict_of_pairs: dict, pair: str) -> None:
    if pair not in dict_of_pairs:
        dict_of_pairs[pair] = {"dict_of_frames": {frame: pd.DataFrame() for frame in FRAME_INTERVALS}}
        return
    if "dict_of_frames" not in dict_of_pairs[pair] or dict_of_pairs[pair]["dict_of_frames"] is None:
        dict_of_pairs[pair]["dict_of_frames"] = {}
    for frame in FRAME_INTERVALS:
        dict_of_pairs[pair]["dict_of_frames"].setdefault(frame, pd.DataFrame())


def _load_pickle(path: str | Path):
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def _save_pickle(path: str | Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def _ensure_within_validation(path: str | Path) -> Path:
    """Protect the curated dataset pickles from being overwritten.

    Safety guard: refuse to write into the ``data/`` folder unless the target is
    under ``data/validation`` (e.g. blocks ``data/complete/raw`` or
    ``data/halal/raw``). Arbitrary paths outside ``data/`` (tmp dirs, tests,
    logs) are allowed.
    """
    p = Path(path).expanduser().resolve()
    data_root = Path("data").expanduser().resolve()
    validation_root = Path("data/validation").expanduser().resolve()
    try:
        p.relative_to(data_root)  # inside data/ ?
    except ValueError:
        return p  # not a curated dataset path -> allowed
    try:
        p.relative_to(validation_root)
    except ValueError:
        raise ValueError(
            f"Refusing to write into the data folder outside the validation folder "
            f"(data/validation): {p}"
        )
    return p


def ensure_validation_dataset_coverage(
    *,
    dataset: dict,
    required_pairs: list[str],
    required_start: pd.Timestamp,
    required_end: pd.Timestamp,
    client: Client,
) -> tuple[dict, dict[str, int]]:
    updated = dataset.copy()
    fetched_ranges = 0
    fetched_rows = 0

    for pair in required_pairs:
        _ensure_dataset_pair_frames(updated, pair)
        for frame_name, interval in FRAME_INTERVALS.items():
            existing = updated[pair]["dict_of_frames"].get(frame_name)
            ranges = _missing_ranges_for_frame(existing, required_start, required_end, frame_name)
            if not ranges:
                continue

            for start_ts, end_ts in ranges:
                incoming = _fetch_klines_sync(client, pair, interval, start_ts, end_ts)
                existing = _merge_frames(existing, incoming)
                fetched_ranges += 1
                fetched_rows += 0 if incoming is None else int(len(incoming))

            updated[pair]["dict_of_frames"][frame_name] = existing

    return updated, {"fetched_ranges": fetched_ranges, "fetched_rows": fetched_rows}


def load_runtime_tables(db_path: str | Path) -> dict[str, pd.DataFrame]:
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"Runtime state DB not found: {db_path}")

    with sqlite3.connect(db_path) as conn:
        runtime_state = pd.read_sql_query("SELECT key, value FROM runtime_state", conn)
        events = pd.read_sql_query("SELECT id, created_at, event_type, payload FROM events ORDER BY id", conn)
        trades = pd.read_sql_query("SELECT id, created_at, pair, strategy_id, mode, side, payload FROM trades ORDER BY id", conn)
        positions = pd.read_sql_query("SELECT pair, payload, updated_at FROM positions ORDER BY pair", conn)
        cursors = pd.read_sql_query("SELECT pair, frame, cursor_value, updated_at FROM cursors", conn)
        try:
            equity_snapshots = pd.read_sql_query("SELECT id, created_at, ts, equity, cash, open_positions, mode FROM equity_snapshots ORDER BY ts", conn)
        except Exception:
            equity_snapshots = pd.DataFrame()

    for frame in (runtime_state, events, trades, positions, cursors, equity_snapshots):
        if "created_at" in frame.columns:
            frame["created_at"] = pd.to_datetime(frame["created_at"], errors="coerce", utc=True)
        if "updated_at" in frame.columns:
            frame["updated_at"] = pd.to_datetime(frame["updated_at"], errors="coerce", utc=True)
        if "ts" in frame.columns:
            frame["ts"] = pd.to_datetime(frame["ts"], errors="coerce", utc=True)

    return {
        "runtime_state": runtime_state,
        "events": events,
        "trades": trades,
        "positions": positions,
        "cursors": cursors,
        "equity_snapshots": equity_snapshots,
    }


def _safe_json_load(value):
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return {}
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return {}
    return {}


def normalize_runtime_trades(trades_df: pd.DataFrame) -> pd.DataFrame:
    if trades_df is None or trades_df.empty:
        return pd.DataFrame(
            columns=[
                "id", "ts_event", "pair", "strategy_id", "mode", "side",
                "decision_time", "reason", "reference_price", "execution_price",
                "submitted_quantity", "executed_quantity", "expected_slippage",
                "realized_slippage", "filled_quote_notional", "order_id",
            ]
        )

    rows: list[dict] = []
    for _, row in trades_df.iterrows():
        payload = _safe_json_load(row.get("payload"))
        intent = payload.get("intent") or {}
        result = payload.get("result") or {}
        metadata = intent.get("metadata") or {}
        decision = metadata.get("decision") or {}

        # Prefer the ACTUAL close reason (only set on sells) over the entry
        # decision's reason. The on_bar_close sell path stores BOTH:
        #   metadata["close_reason"] = trailing_stop / time_stop / exit_signal /
        #                             stop_loss / take_profit   (the REAL exit)
        #   metadata["decision"]     = the OWNER entry decision for that bar,
        #                             whose reason is "entry_triggered" /
        #                             "no_entry" / "gate_blocked"
        # Preferring the decision reason mislabels real closes as no_entry /
        # gate_blocked in the trade frame. Buys carry no close_reason, so they
        # correctly fall back to the entry decision reason.
        reason = metadata.get("close_reason")
        if reason is None:
            reason = decision.get("reason")

        decision_time = decision.get("latest_time")
        if decision_time is None:
            position_payload = metadata.get("position") or {}
            decision_time = position_payload.get("entry_time")

        execution_price = result.get("avg_fill_price")
        if execution_price in (None, ""):
            execution_price = result.get("submitted_limit_price")
        if execution_price in (None, ""):
            execution_price = result.get("reference_price")

        sizing = metadata.get("sizing") or {}
        sell_slippage_meta = metadata.get("slippage") or {}

        # Sells carry the pre-trade slippage estimate in metadata["slippage"].
        if isinstance(sell_slippage_meta, dict) and sell_slippage_meta.get("expected_slippage_qvpuc") is not None:
            expected_slippage = sell_slippage_meta.get("expected_slippage")
            expected_slippage_qvpuc = sell_slippage_meta.get("expected_slippage_qvpuc")
            expected_slippage_orderbook = sell_slippage_meta.get("expected_slippage_orderbook")
            slippage_source = sell_slippage_meta.get("slippage_source")
            cap_by_slippage_orderbook = sell_slippage_meta.get("cap_by_slippage_orderbook")
        else:
            expected_slippage = sizing.get("expected_slippage") if isinstance(sizing, dict) else None
            expected_slippage_qvpuc = sizing.get("expected_slippage_qvpuc") if isinstance(sizing, dict) else None
            expected_slippage_orderbook = sizing.get("expected_slippage_orderbook") if isinstance(sizing, dict) else None
            slippage_source = sizing.get("slippage_source") if isinstance(sizing, dict) else None
            cap_by_slippage_orderbook = sizing.get("cap_by_slippage_orderbook") if isinstance(sizing, dict) else None

        rows.append(
            {
                "id": row.get("id"),
                "ts_event": pd.to_datetime(row.get("created_at"), errors="coerce", utc=True),
                "pair": row.get("pair"),
                "strategy_id": row.get("strategy_id"),
                "mode": row.get("mode"),
                "side": row.get("side"),
                "decision_time": pd.to_datetime(decision_time, errors="coerce", utc=True),
                "reason": reason,
                "reference_price": pd.to_numeric(result.get("reference_price"), errors="coerce"),
                "execution_price": pd.to_numeric(execution_price, errors="coerce"),
                "submitted_quantity": pd.to_numeric(result.get("submitted_quantity"), errors="coerce"),
                "executed_quantity": pd.to_numeric(result.get("executed_qty"), errors="coerce"),
                "expected_slippage": pd.to_numeric(expected_slippage, errors="coerce"),
                "expected_slippage_qvpuc": pd.to_numeric(expected_slippage_qvpuc, errors="coerce"),
                "expected_slippage_orderbook": pd.to_numeric(expected_slippage_orderbook, errors="coerce"),
                "slippage_source": slippage_source,
                "cap_by_slippage_orderbook": pd.to_numeric(cap_by_slippage_orderbook, errors="coerce"),
                "realized_slippage": pd.to_numeric(result.get("realized_slippage"), errors="coerce"),
                "filled_quote_notional": pd.to_numeric(result.get("filled_quote_notional"), errors="coerce"),
                "order_id": result.get("order_id"),
            }
        )

    normalized = pd.DataFrame(rows)
    normalized["action_time"] = normalized["decision_time"].where(normalized["decision_time"].notna(), normalized["ts_event"])
    return normalized.sort_values(["action_time", "id"], na_position="last").reset_index(drop=True)


_BACKTEST_TRADE_FRAME_COLUMNS = [
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

_BACKTEST_TRADE_FRAME_DERIVED_COLUMNS = [
    "open_time",
    "close_time",
    "profit",
    "gross_profit",
    "gain",
    "gain_so_far",
    "holding_period",
]


def _empty_runtime_trade_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=_BACKTEST_TRADE_FRAME_COLUMNS + _BACKTEST_TRADE_FRAME_DERIVED_COLUMNS)


def _extract_position_meta(payload: dict) -> dict:
    # The runtime persists trade payloads as {"intent": {..., "metadata": ...},
    # "result": ...}, so the position lives under intent.metadata.position.
    # Fall back to a legacy top-level "metadata" key for older payloads.
    intent = payload.get("intent") or {}
    metadata = intent.get("metadata") or payload.get("metadata") or {}
    position = metadata.get("position") or {}
    return {
        "stop_price": position.get("stop_price"),
        "hold_bars": position.get("hold_bars"),
        "entry_time": position.get("entry_time"),
    }


def _maybe_float_value(value):
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_runtime_trade_frame(
    runtime_tables: dict,
    *,
    include_open: bool = True,
    bar_freq: str = "4h",
    mark_prices: dict | pd.Series | None = None,
) -> pd.DataFrame:
    """
    Build a backtest-style trade frame from the runtime `trades` table.

    Each completed buy->sell round trip on the same (pair, strategy_id) becomes one row,
    using the same column set as run_portfolio_backtest(...)["portfolio"]["trade_frame"].
    Open (still-held) positions can be included with NaN exit fields when include_open=True.

    ``mark_prices`` optionally maps pair -> current price (dict or Series). When provided,
    still-open rows get ``gain_so_far = mark_price / entry_price`` (unrealized gain multiple);
    closed rows always get NaN for ``gain_so_far`` (the mirror of ``gain``, which is NaN for
    open rows and populated for closed rows).
    """
    trades = runtime_tables.get("trades")
    if trades is None or trades.empty:
        return _empty_runtime_trade_frame()

    payload_by_id: dict = {}
    for _, row in trades.iterrows():
        payload_by_id[row.get("id")] = _safe_json_load(row.get("payload"))

    normalized = normalize_runtime_trades(trades)
    if normalized.empty:
        return _empty_runtime_trade_frame()

    def _row_value(row, key):
        value = row.get(key)
        try:
            if value is None or pd.isna(value):
                return None
        except (TypeError, ValueError):
            return None
        return value

    def _build_round_trip(entry_row, exit_row) -> dict:
        entry_id = _row_value(entry_row, "id")
        exit_id = _row_value(exit_row, "id")
        entry_payload = payload_by_id.get(entry_id) or {}
        exit_payload = payload_by_id.get(exit_id) or {}
        exit_meta = _extract_position_meta(exit_payload)

        pair = entry_row.get("pair")
        strategy = entry_row.get("strategy_id")
        entry_time = pd.to_datetime(_row_value(entry_row, "action_time"), errors="coerce", utc=True)
        # Prefer the persisted trade timestamp for the exit: normalized action_time
        # can be backfilled to the position entry_time for sells, which would collapse
        # the holding period to zero.
        exit_time = pd.to_datetime(_row_value(exit_row, "ts_event"), errors="coerce", utc=True)
        if exit_time is None or pd.isna(exit_time):
            exit_time = pd.to_datetime(_row_value(exit_row, "action_time"), errors="coerce", utc=True)

        entry_price = _maybe_float_value(_row_value(entry_row, "execution_price"))
        if entry_price is None:
            entry_price = _maybe_float_value(_row_value(entry_row, "reference_price"))
        exit_price = _maybe_float_value(_row_value(exit_row, "execution_price"))
        if exit_price is None:
            exit_price = _maybe_float_value(_row_value(exit_row, "reference_price"))

        quantity = _maybe_float_value(_row_value(entry_row, "executed_quantity"))
        if quantity is None:
            quantity = _maybe_float_value(_row_value(exit_row, "executed_quantity"))

        entry_value = _maybe_float_value(_row_value(entry_row, "filled_quote_notional"))
        if entry_value is None and entry_price is not None and quantity is not None:
            entry_value = entry_price * quantity
        exit_value = _maybe_float_value(_row_value(exit_row, "filled_quote_notional"))
        if exit_value is None and exit_price is not None and quantity is not None:
            exit_value = exit_price * quantity

        entry_value = entry_value if entry_value is not None else 0.0
        exit_value = exit_value if exit_value is not None else 0.0
        pnl = exit_value - entry_value
        return_pct = pnl / entry_value if entry_value > 0 else 0.0

        hold_bars = _maybe_float_value(exit_meta.get("hold_bars"))
        if hold_bars is None and entry_time is not None and exit_time is not None and not pd.isna(entry_time) and not pd.isna(exit_time):
            try:
                seconds_per_bar = {"4h": 4 * 3600, "1h": 3600, "1d": 24 * 3600}.get(bar_freq, 4 * 3600)
                hold_bars = max(int((exit_time - entry_time).total_seconds() // seconds_per_bar), 0)
            except (TypeError, ValueError, OverflowError):
                hold_bars = None

        stop_price = _maybe_float_value(exit_meta.get("stop_price"))
        entry_time_from_pos = pd.to_datetime(exit_meta.get("entry_time"), errors="coerce", utc=True)
        if pd.isna(entry_time_from_pos) and entry_time is not None and not pd.isna(entry_time):
            entry_time_from_pos = entry_time

        return {
            "pair": pair,
            "strategy": strategy,
            "regime": None,
            "entry_bar": None,
            "entry_time": entry_time_from_pos if not pd.isna(entry_time_from_pos) else entry_time,
            "entry_price": entry_price,
            "initial_stop_price": stop_price,
            "stop_price": stop_price,
            "entry_value": entry_value,
            "entry_fee": 0.0,
            "exit_bar": None,
            "exit_time": exit_time,
            "exit_price": exit_price,
            "exit_value": exit_value,
            "exit_fee": 0.0,
            "pnl": pnl,
            "return_pct": return_pct,
            "hold_bars": hold_bars,
            "close_reason": exit_row.get("reason"),
        }

    def _build_open_row(entry_row) -> dict:
        entry_id = _row_value(entry_row, "id")
        entry_payload = payload_by_id.get(entry_id) or {}
        entry_meta = _extract_position_meta(entry_payload)

        entry_time = pd.to_datetime(_row_value(entry_row, "action_time"), errors="coerce", utc=True)
        entry_time_from_pos = pd.to_datetime(entry_meta.get("entry_time"), errors="coerce", utc=True)
        if pd.isna(entry_time_from_pos) and entry_time is not None and not pd.isna(entry_time):
            entry_time_from_pos = entry_time

        entry_price = _maybe_float_value(_row_value(entry_row, "execution_price"))
        if entry_price is None:
            entry_price = _maybe_float_value(_row_value(entry_row, "reference_price"))
        quantity = _maybe_float_value(_row_value(entry_row, "executed_quantity"))
        entry_value = _maybe_float_value(_row_value(entry_row, "filled_quote_notional"))
        if entry_value is None and entry_price is not None and quantity is not None:
            entry_value = entry_price * quantity
        entry_value = entry_value if entry_value is not None else 0.0
        stop_price = _maybe_float_value(entry_meta.get("stop_price"))
        # Still-open position: the persisted hold_bars can be stale/missing, so
        # derive it from (now - entry_time) in bars. Falls back to the persisted
        # value only when the entry time is unavailable.
        hold_bars = _maybe_float_value(entry_meta.get("hold_bars"))
        if entry_time_from_pos is not None and not pd.isna(entry_time_from_pos):
            try:
                seconds_per_bar = {"4h": 4 * 3600, "1h": 3600, "1d": 24 * 3600}.get(bar_freq, 4 * 3600)
                elapsed_seconds = (pd.Timestamp.now(tz="UTC") - entry_time_from_pos).total_seconds()
                hold_bars = max(int(elapsed_seconds // seconds_per_bar), 0)
            except (TypeError, ValueError, OverflowError):
                pass

        return {
            "pair": entry_row.get("pair"),
            "strategy": entry_row.get("strategy_id"),
            "regime": None,
            "entry_bar": None,
            "entry_time": entry_time_from_pos if not pd.isna(entry_time_from_pos) else entry_time,
            "entry_price": entry_price,
            "initial_stop_price": stop_price,
            "stop_price": stop_price,
            "entry_value": entry_value,
            "entry_fee": 0.0,
            "exit_bar": None,
            "exit_time": None,
            "exit_price": None,
            "exit_value": None,
            "exit_fee": 0.0,
            "pnl": None,
            "return_pct": None,
            "hold_bars": hold_bars,
            "close_reason": None,
        }

    rows: list[dict] = []
    for (pair, strategy_id), group in normalized.groupby(["pair", "strategy_id"], sort=False):
        group = group.sort_values("action_time", na_position="last")
        entry = None
        for _, row in group.iterrows():
            side = str(row.get("side", "")).lower()
            if side == "buy":
                entry = row
            elif side == "sell" and entry is not None:
                rows.append(_build_round_trip(entry, row))
                entry = None
        if include_open and entry is not None:
            rows.append(_build_open_row(entry))

    frame = pd.DataFrame(rows)
    if frame.empty:
        return _empty_runtime_trade_frame()

    frame["open_time"] = pd.to_datetime(frame["entry_time"], errors="coerce", utc=True)
    frame["close_time"] = pd.to_datetime(frame["exit_time"], errors="coerce", utc=True)
    frame["profit"] = pd.to_numeric(frame["pnl"], errors="coerce")
    frame["gross_profit"] = pd.to_numeric(frame["exit_value"], errors="coerce") - pd.to_numeric(frame["entry_value"], errors="coerce")
    frame["gain"] = 1.0 + pd.to_numeric(frame["return_pct"], errors="coerce")
    # gain_so_far: unrealized gain multiple for still-open rows (mark_price / entry_price),
    # NaN for closed rows — the mirror of `gain` (NaN for open rows).
    frame["gain_so_far"] = np.nan
    if mark_prices is not None and not frame.empty:
        price_map = mark_prices.to_dict() if isinstance(mark_prices, pd.Series) else dict(mark_prices)
        open_mask = frame["close_time"].isna()
        mark_price = frame["pair"].map(price_map)
        entry_price = pd.to_numeric(frame["entry_price"], errors="coerce")
        valid = open_mask & mark_price.notna() & entry_price.notna() & entry_price.ne(0)
        frame.loc[valid, "gain_so_far"] = (mark_price[valid] / entry_price[valid]).astype(float)
    frame["holding_period"] = frame["close_time"] - frame["open_time"]

    return frame.sort_values(["open_time", "close_time", "pair"]).reset_index(drop=True)


def _runtime_round_or_none(value, digits: int = 2):
    """Local copy of backtest_engine._round_or_none (not star-exported)."""
    if value is None:
        return None
    if isinstance(value, pd.Timedelta):
        return value
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    return round(float(value), digits)


def _runtime_safe_compounded_multiple(total_multiple: float, periods: float) -> float | None:
    """Local copy of backtest_engine._safe_compounded_multiple."""
    if periods is None or periods <= 0 or total_multiple <= 0:
        return None
    return float(total_multiple ** (1.0 / periods))


def _runtime_safe_months_to_target(monthly_multiple: float | None, target_multiple: float) -> float | None:
    """Local copy of backtest_engine._safe_months_to_target."""
    if monthly_multiple is None or monthly_multiple <= 1.0:
        return None
    return float(np.log(target_multiple) / np.log(monthly_multiple))


def _runtime_timedelta_floor_seconds(value: pd.Timedelta | None) -> pd.Timedelta | None:
    """Local copy of backtest_engine._timedelta_floor_seconds."""
    if value is None or pd.isna(value):
        return None
    return pd.to_timedelta(np.floor(value.total_seconds()), unit="s")


# aliases so the metric functions read like the backtest internals
_round_or_none = _runtime_round_or_none
_safe_compounded_multiple = _runtime_safe_compounded_multiple
_safe_months_to_target = _runtime_safe_months_to_target
_timedelta_floor_seconds = _runtime_timedelta_floor_seconds


def _runtime_equity_series(
    runtime_tables: dict,
    trade_frame: pd.DataFrame,
    initial_cash: float,
) -> pd.Series:
    """Best-effort runtime equity curve.

    Prefers the persisted ``equity_snapshots`` (equity marked at each 4h bar
    close); falls back to deriving an equity curve from the closed-trade PnL
    (cumulative, starting at ``initial_cash``). A synthetic first point at
    ``initial_cash`` keeps ``total_profit_multiple = final / initial``.
    """
    snaps = runtime_tables.get("equity_snapshots")
    if snaps is not None and not snaps.empty and "equity" in snaps.columns:
        equity = pd.to_numeric(snaps["equity"], errors="coerce")
        valid = equity.dropna()
        if not valid.empty:
            index = pd.to_datetime(
                snaps.loc[valid.index, "ts"], errors="coerce", utc=True
            )
            series = pd.Series(valid.values, index=index, dtype=np.float64).sort_index()
            series = series[~series.index.duplicated(keep="last")]
            if not series.empty:
                first_ts = series.index[0]
                # Place the synthetic initial point just before the first mark so
                # downstream dedupe (keep="last") cannot drop it — this keeps
                # total_profit_multiple = final_equity / initial_cash.
                return pd.concat(
                    [pd.Series([float(initial_cash)], index=[first_ts - pd.Timedelta(1, "ns")]), series]
                )

    closed = trade_frame[trade_frame["close_time"].notna()].sort_values("close_time")
    if closed.empty:
        return pd.Series([float(initial_cash)], index=[pd.Timestamp.now(tz="UTC")])
    index = pd.to_datetime(closed["close_time"], errors="coerce", utc=True)
    cumulative = pd.to_numeric(closed["pnl"], errors="coerce").fillna(0.0).cumsum()
    series = pd.Series(float(initial_cash) + cumulative.values, index=index, dtype=np.float64)
    series = series[~series.index.duplicated(keep="last")].sort_index()
    first_ts = series.index[0]
    return pd.concat([pd.Series([float(initial_cash)], index=[first_ts]), series])


def _compute_runtime_metrics(
    trade_frame: pd.DataFrame,
    equity_series: pd.Series,
    initial_cash: float,
    benchmark_close_series: pd.Series | None = None,
    resample_period: str = "ME",
) -> dict:
    """Mirror of ``backtest_engine._compute_metrics`` for a runtime trade frame.

    Closed buy->sell round trips drive the trade statistics; open positions are
    excluded from win/loss math and reported separately as ``open_positions``.
    """
    closed = trade_frame[trade_frame["close_time"].notna()].copy() if not trade_frame.empty else trade_frame.copy()
    trade_count = len(closed)
    open_positions = int((trade_frame["close_time"].isna()).sum()) if not trade_frame.empty else 0

    wins = closed[closed["gain"] >= 1.0] if trade_count else closed
    losses = closed[closed["gain"] < 1.0] if trade_count else closed
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

    equity_series = equity_series.dropna()
    equity_series = equity_series[~equity_series.index.duplicated(keep="last")].sort_index()
    # Normalize to tz-naive so resampled period labels match a tz-naive benchmark.
    if getattr(equity_series.index, "tz", None) is not None:
        equity_series = equity_series.set_axis(equity_series.index.tz_localize(None))
    if benchmark_close_series is not None and getattr(benchmark_close_series.index, "tz", None) is not None:
        benchmark_close_series = benchmark_close_series.set_axis(benchmark_close_series.index.tz_localize(None))
    if not equity_series.empty:
        total_profit_multiple = float(equity_series.iloc[-1] / equity_series.iloc[0])
        net_return_pct = (total_profit_multiple - 1.0) * 100.0
        running_max = equity_series.cummax()
        drawdowns = (equity_series - running_max) / running_max
        max_drawdown_pct = drawdowns[drawdowns > -0.5].min() * 100 if len(drawdowns) > 0 else 0.0
    else:
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
        start_time = closed.iloc[0]["open_time"]
        end_time = closed.iloc[-1]["close_time"]
        duration = end_time - start_time if pd.notna(start_time) and pd.notna(end_time) else None
        average_holding_period = _timedelta_floor_seconds(closed["holding_period"].mean())
        avg_hold_bars = closed["hold_bars"].mean()
    elif not equity_series.empty:
        start_time = equity_series.index[0]
        end_time = equity_series.index[-1]
        duration = end_time - start_time

    if duration is not None and pd.notna(duration):
        total_days = max(1, duration.total_seconds() / 86400.0)

    profit_per_day_multiple = _safe_compounded_multiple(total_profit_multiple, total_days) if total_days else None
    profit_per_trade_multiple = (
        _safe_compounded_multiple(total_profit_multiple, float(trade_count)) if trade_count else None
    )
    profit_per_month_multiple = (
        float(profit_per_day_multiple ** 30.0) if profit_per_day_multiple is not None else None
    )

    strategy_period_returns = pd.Series(dtype=np.float64)
    average_excess_vs_btc_pct = None
    downside_deviation_pct = None
    if not equity_series.empty:
        # Short sessions (< ~2 months) do not form enough monthly periods for
        # pct_change to be meaningful, so compare on daily returns instead.
        span_days = (equity_series.index[-1] - equity_series.index[0]).total_seconds() / 86400.0
        effective_period = "D" if span_days < 60 else resample_period

        strategy_period_values = equity_series.resample(effective_period).last().dropna()
        strategy_period_returns = strategy_period_values.pct_change().dropna()
        downside = strategy_period_returns[strategy_period_returns < 0]
        if not downside.empty:
            downside_deviation_pct = downside.std() * 100.0

        if benchmark_close_series is not None and not benchmark_close_series.empty:
            benchmark_period_values = benchmark_close_series.resample(effective_period).last().dropna()
            benchmark_period_returns = benchmark_period_values.pct_change().dropna()
            aligned = pd.concat(
                [strategy_period_returns, benchmark_period_returns], axis=1, join="inner"
            ).dropna()
            if not aligned.empty:
                aligned.columns = ["strategy", "benchmark"]
                average_excess_vs_btc_pct = (aligned["strategy"] - aligned["benchmark"]).mean() * 100.0

    max_win_pct = (closed["gain"] - 1.0).max() * 100 if trade_count else None
    max_loss_pct = (closed["gain"] - 1.0).min() * 100 if trade_count else None

    return {
        "trade_count": trade_count,
        "open_positions": open_positions,
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


def runtime_metrics_report(
    runtime_tables: dict,
    *,
    initial_cash: float | None = None,
    include_open: bool = True,
    benchmark_close_series: pd.Series | None = None,
    resample_period: str = "ME",
) -> dict:
    """Build a backtest-style metrics report from the runtime DB tables.

    Parameters
    ----------
    runtime_tables:
        Output of ``load_runtime_tables`` (trades / positions / equity_snapshots / ...).
    initial_cash:
        Starting cash. Defaults to the RuntimeConfig default (10_000) — pass the
        value the runtime actually booted with for exact multiples.
    include_open:
        Include still-open positions as rows in the returned trade frame
        (they are excluded from win/loss statistics and counted separately).

    Returns
    -------
    dict with ``metrics`` (same keys as ``backtest_engine._compute_metrics``),
    ``trade_frame`` (backtest-style, see ``build_runtime_trade_frame``),
    ``equity_curve`` and ``open_positions``.
    """
    if initial_cash is None:
        initial_cash = 10_000.0

    trade_frame = build_runtime_trade_frame(runtime_tables, include_open=include_open, bar_freq="4h")
    equity_curve = _runtime_equity_series(runtime_tables, trade_frame, float(initial_cash))
    metrics = _compute_runtime_metrics(
        trade_frame,
        equity_curve,
        float(initial_cash),
        benchmark_close_series=benchmark_close_series,
        resample_period=resample_period,
    )
    return {
        "metrics": metrics,
        "trade_frame": trade_frame,
        "equity_curve": equity_curve,
        "open_positions": metrics.get("open_positions", 0),
    }


def format_runtime_metrics_report(metrics: dict) -> str:
    """Human-readable report matching ``backtest_engine.format_metrics_report``."""
    return "\n".join(
        [
            f"total profit multiple: {metrics.get('total_profit_multiple')}",
            f"final equity: {metrics.get('final_equity'):,.0f}",
            f"max drawdown: {metrics.get('max_drawdown_pct')}%",
            f"number of trades: {metrics.get('trade_count')}",
            f"open positions: {metrics.get('open_positions')}",
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
        ]
    )


def runtime_metrics_by_strategy(
    runtime_tables: dict,
    *,
    include_open: bool = True,
) -> pd.DataFrame:
    """Per-strategy trade-stat summary (mirrors ``results_summary``).

    The runtime shares one capital pool across strategies, so per-strategy
    equity-based fields (final equity / multiple / drawdown) are not well
    defined; this reports trade statistics only, indexed by strategy.
    """
    trade_frame = build_runtime_trade_frame(runtime_tables, include_open=include_open, bar_freq="4h")
    if trade_frame.empty:
        return pd.DataFrame()

    closed = trade_frame[trade_frame["close_time"].notna()].copy()
    rows = []
    for strategy, group in closed.groupby("strategy", sort=False):
        wins = group[group["gain"] >= 1.0]
        losses = group[group["gain"] < 1.0]
        win_rate = (len(wins) / len(group)) * 100.0 if len(group) else None
        avg_win = (wins["gain"] - 1.0).mean() * 100 if len(wins) else None
        avg_loss = (losses["gain"] - 1.0).mean() * 100 if len(losses) else None
        expectancy = (
            (len(wins) / len(group)) * (avg_win or 0.0)
            + (len(losses) / len(group)) * (avg_loss or 0.0)
        ) if len(group) else None
        holding = _timedelta_floor_seconds(group["holding_period"].mean())
        rows.append(
            {
                "strategy": strategy,
                "trade_count": len(group),
                "win_rate": _round_or_none(win_rate, 2),
                "avg_win_pct": _round_or_none(avg_win, 2),
                "avg_loss_pct": _round_or_none(avg_loss, 2),
                "expectancy_rate_pct": _round_or_none(expectancy, 2),
                "total_pnl": _round_or_none(group["pnl"].sum(), 2),
                "average_holding_period": holding,
            }
        )
    return pd.DataFrame(rows).set_index("strategy")


def resolve_runtime_validation_window(normalized_trades: pd.DataFrame, events_df: pd.DataFrame | None = None) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    candidates: list[pd.Timestamp] = []

    if normalized_trades is not None and not normalized_trades.empty:
        for col in ("action_time", "ts_event"):
            if col in normalized_trades.columns:
                values = pd.to_datetime(normalized_trades[col], errors="coerce", utc=True).dropna()
                if not values.empty:
                    candidates.extend(values.tolist())

    if events_df is not None and not events_df.empty and "created_at" in events_df.columns:
        values = pd.to_datetime(events_df["created_at"], errors="coerce", utc=True).dropna()
        if not values.empty:
            candidates.extend(values.tolist())

    if not candidates:
        return None, None

    start = min(candidates)
    end = max(candidates)
    return start, end


def store_validation_artifact(obj, output_path: str | Path) -> Path:
    output_path = _ensure_within_validation(Path(output_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        pickle.dump(obj, handle)
    return output_path


def run_tester_cells_11_20_equivalent(
    dict_of_pairs_ori: dict,
    engine_cfg,
    *,
    execution_frame: str = "4hourly",
    gate_col: str = "structure_ok",
    data_set_name: str = "validation",
    output_dir: str | Path = "data/validation",
    current_slippage_array=None,
) -> dict:
    """
    Reproduce tester.ipynb cells 11-20 exactly after data-fetch step.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_list = ["1daily", "4hourly"]
    dict_of_pairs_ori = dict_of_pairs_ori.copy()

    # Equivalent to tester preparation section before strategy application.
    for pair in dict_of_pairs_ori:
        for frame in dict_of_pairs_ori[pair]["dict_of_frames"]:
            dict_of_pairs_ori[pair]["dict_of_frames"][frame] = prepare_df(dict_of_pairs_ori[pair]["dict_of_frames"][frame])

            if frame == "4hourly":
                frame_df = dict_of_pairs_ori[pair]["dict_of_frames"][frame]
                time_col = "open_time" if "open_time" in frame_df.columns else None
                if time_col is not None:
                    time_index = pd.to_datetime(frame_df[time_col], errors="coerce")
                    time_series = pd.Series(time_index, index=frame_df.index)
                    open_series = pd.to_numeric(frame_df["open"], errors="coerce")

                    year_key = time_series.dt.to_period("Y")
                    month_key = time_series.dt.to_period("M")
                    week_key = time_series.dt.to_period("W-SUN")

                    frame_df["yearly_open"] = open_series.groupby(year_key).transform("first")
                    frame_df["monthly_open"] = open_series.groupby(month_key).transform("first")
                    frame_df["weekly_open"] = open_series.groupby(week_key).transform("first")

                dict_of_pairs_ori[pair]["dict_of_frames"][frame] = frame_df

    dict_of_pairs_ori = attach_sqn_to_dataset(dict_of_pairs_ori)
    dict_of_pairs_ori = compute_4h_structure_for_dataset(
        dict_of_pairs_ori,
        ema_fast_period=25,
        ema_mid_period=50,
        ema_slow_period=200,
        slope_lookback=5,
    )

    columns_for_merge = ["ma25", "ma50", "ma100", "ma200", "ema25", "ema50", "ema100", "ema200", "rsi", "max", "min", "boll_upper", "boll_mid", "boll_lower"]
    for pair in dict_of_pairs_ori:
        for i in range(0, frame_list.index(execution_frame)):
            frame = frame_list[i]

            target = dict_of_pairs_ori[pair]["dict_of_frames"][execution_frame]
            source = dict_of_pairs_ori[pair]["dict_of_frames"][frame][columns_for_merge + ["close_time"]]

            dict_of_pairs_ori[pair]["dict_of_frames"][execution_frame] = pd.merge_asof(
                target.sort_values("close_time"),
                source.sort_values("close_time"),
                left_on="close_time",
                right_on="close_time",
                direction="backward",
                suffixes=["", f"_{frame}"],
            )

    cols_for_retest = [
        "ma50", "boll_lower", "ema50", "ma200", "boll_mid_1daily", "ema100", "ma100", "ema200", "ema25_1daily",
        "prev_low", "min", "ema25", "yearly_open", "monthly_open", "weekly_open", "nearest_support",
    ]
    cols_for_break = [
        "max", "boll_upper", "boll_upper_1daily", "ema100", "ema200", "ma100", "ma200",
        "ema100_1daily", "ema200_1daily", "ma100_1daily", "ma200_1daily", "prev_high",
        "yearly_open", "monthly_open", "weekly_open", "nearest_resistance",
    ]
    cols_to_prepare = cols_for_retest + cols_for_break

    for pair in dict_of_pairs_ori:
        frame_df = dict_of_pairs_ori[pair]["dict_of_frames"][execution_frame]
        frame_df = add_hh_ll_columns(frame_df)
        frame_df = populate_historical_levels_fast(
            ohlc_df=frame_df,
            levels_reactions_limit=12,
            lookback_bars=None,
            price_col="close",
            skip_rows=6 * 14,
        )

        for col in list(set(cols_to_prepare)):
            retest_dict = level_test_buys_with_last_broken(frame_df, [col], confirm_breaks=False, confirm_tests=False)
            retest_frame = pd.DataFrame(retest_dict)
            rename_dict = {prop: f"{col}_{prop}" for prop in retest_dict}
            retest_frame.rename(columns=rename_dict, inplace=True)
            frame_df = pd.concat([frame_df, retest_frame], axis=1)

        frame_df["rolling_max"] = frame_df["close"].rolling(6 * 7).max()
        frame_df["rolling_min"] = frame_df["close"].rolling(6 * 7).min()
        frame_df["rolling_gain"] = frame_df["rolling_max"] / frame_df["rolling_min"]
        frame_df["rsi_slope"] = frame_df["rsi"].rolling(6).apply(get_simple_slope_2, raw=True)
        frame_df = get_common_aux_columns(frame_df)
        dict_of_pairs_ori[pair]["dict_of_frames"][execution_frame] = frame_df

    dict_of_pairs_ori = prepare_execution_frame_dataset(
        dict_of_pairs=dict_of_pairs_ori,
        execution_frame=execution_frame,
        btc_pair="BTCUSDT",
        structure_col="structure_ok",
    )

    dict_of_pairs = dict_of_pairs_ori.copy()
    for pair in dict_of_pairs:
        df = dict_of_pairs[pair]["dict_of_frames"][execution_frame]
        if df is not None and not df.empty:
            df["all_true_gate"] = True
            dict_of_pairs[pair]["dict_of_frames"][execution_frame] = df

    list_of_strategies = []

    cols_for_break = [
        "boll_upper_1daily",
        "ema100", "ema200", "ma100", "ma200",
        "ema100_1daily", "ema200_1daily", "ma100_1daily", "ma200_1daily",
        "yearly_open", "monthly_open", "weekly_open",
    ]
    for col in cols_for_break:
        strategy_dict = {
            "name": f"break_{col}",
            "entry_col": f"entry_break_{col}",
            "exit_col": f"exit_break_{col}",
        }
        list_of_strategies.append(strategy_dict)
        dict_of_pairs = apply_break_strategy(
            dict_of_pairs=dict_of_pairs,
            execution_frame=execution_frame,
            entry_col=strategy_dict["entry_col"],
            exit_col=strategy_dict["exit_col"],
            break_column=col,
            aux_prefix=strategy_dict["entry_col"],
            fast_profit_threshold=1.5,
            gain_ratio=1.15,
            lower_volatility_threshold=0.01,
            upper_volatility_threshold=0.08,
            use_rsi_slope_exit=False,
        )

    cols_for_retest = [
        "ma50", "boll_lower", "ema50", "ma200", "boll_mid_1daily", "ema100", "ma100", "ema200", "ema25_1daily",
        "yearly_open", "monthly_open",
    ]
    for col in cols_for_retest:
        strategy_dict = {
            "name": f"test_{col}",
            "entry_col": f"entry_test_{col}",
            "exit_col": f"exit_test_{col}",
        }
        list_of_strategies.append(strategy_dict)
        dict_of_pairs = apply_test_strategy(
            dict_of_pairs=dict_of_pairs,
            execution_frame=execution_frame,
            entry_col=strategy_dict["entry_col"],
            exit_col=strategy_dict["exit_col"],
            retest_column=col,
            aux_prefix=strategy_dict["entry_col"],
            fast_profit_threshold=1.5,
            lower_volatility_threshold=0.02,
            upper_volatility_threshold=0.08,
            gain_ratio=1.15,
            use_rsi_slope_exit=False,
        )

    extra_break_cols = ["max", "boll_upper", "nearest_resistance"]
    for col in extra_break_cols:
        strategy_dict = {
            "name": f"break_{col}",
            "entry_col": f"entry_break_{col}",
            "exit_col": f"exit_break_{col}",
        }
        list_of_strategies.append(strategy_dict)
        dict_of_pairs = apply_break_strategy(
            dict_of_pairs=dict_of_pairs,
            execution_frame=execution_frame,
            entry_col=strategy_dict["entry_col"],
            exit_col=strategy_dict["exit_col"],
            break_column=col,
            aux_prefix=strategy_dict["entry_col"],
            fast_profit_threshold=2.5,
            gain_ratio=1.15,
            lower_volatility_threshold=0.01,
            upper_volatility_threshold=0.08,
        )

    entry_columns = [strat["entry_col"] for strat in list_of_strategies]
    exit_columns = [strat["exit_col"] for strat in list_of_strategies]
    strategy_specs = build_strategy_specs(
        entry_columns=entry_columns,
        exit_columns=exit_columns,
        mode="pairwise",
    )

    backtest_payload = compile_backtest_payload(
        dict_of_pairs=dict_of_pairs,
        execution_frame=execution_frame,
        strategy_specs=strategy_specs,
        gate_col=gate_col,
        time_col="close_time",
        price_columns=("open", "high", "low", "close"),
        aux_columns=STRATEGY_AUX_COLUMNS,
        context_columns=["regime"],
        enforce_bar_cadence=False,
        min_spacing_ratio=0.90,
    )

    backtest_results = run_portfolio_backtest(
        backtest_payload,
        cfg=engine_cfg,
        current_slippage_array=current_slippage_array,
    )

    store_validation_artifact(dict_of_pairs_ori, output_dir / f"{data_set_name}_prepared_pairs.pkl")
    store_validation_artifact(dict_of_pairs, output_dir / f"{data_set_name}_strategy_applied_pairs.pkl")
    store_validation_artifact(strategy_specs, output_dir / f"{data_set_name}_strategy_specs.pkl")
    store_validation_artifact(backtest_payload, output_dir / f"{data_set_name}_backtest_payload.pkl")
    store_validation_artifact(backtest_results, output_dir / f"{data_set_name}_backtest_results.pkl")

    return {
        "dict_of_pairs_prepared": dict_of_pairs_ori,
        "dict_of_pairs_strategy_applied": dict_of_pairs,
        "list_of_strategies": list_of_strategies,
        "strategy_specs": strategy_specs,
        "backtest_payload": backtest_payload,
        "backtest_results": backtest_results,
    }


def _normalize_strategy_key(value: str | None) -> str:
    text = "" if value is None else str(value)
    if text.startswith("entry_") and "__exit_" in text:
        return text.split("__", 1)[0].replace("entry_", "", 1)
    return text


def extract_expected_actions_from_backtest(backtest_results: dict, *, bar_freq: str = "4h") -> pd.DataFrame:
    portfolio = backtest_results.get("portfolio", {})
    trade_frame = portfolio.get("trade_frame")
    if trade_frame is None:
        return pd.DataFrame(columns=["pair", "strategy_key", "side", "action_time", "bar_time", "reason"])

    frame = pd.DataFrame(trade_frame).copy()
    if frame.empty:
        return pd.DataFrame(columns=["pair", "strategy_key", "side", "action_time", "bar_time", "reason"])

    rows: list[dict] = []
    for _, row in frame.iterrows():
        pair = row.get("pair")
        strategy_key = _normalize_strategy_key(row.get("strategy"))

        entry_time = pd.to_datetime(row.get("entry_time"), errors="coerce", utc=True)
        exit_time = pd.to_datetime(row.get("exit_time"), errors="coerce", utc=True)

        if pd.notna(entry_time):
            rows.append(
                {
                    "pair": pair,
                    "strategy_key": strategy_key,
                    "side": "buy",
                    "action_time": entry_time,
                    "bar_time": entry_time.floor(bar_freq),
                    "reason": "entry_triggered",
                }
            )

        if pd.notna(exit_time):
            rows.append(
                {
                    "pair": pair,
                    "strategy_key": strategy_key,
                    "side": "sell",
                    "action_time": exit_time,
                    "bar_time": exit_time.floor(bar_freq),
                    "reason": row.get("close_reason"),
                }
            )

    return pd.DataFrame(rows).sort_values(["bar_time", "pair", "strategy_key", "side"]).reset_index(drop=True)


def extract_runtime_actions_for_parity(normalized_runtime_trades: pd.DataFrame, *, bar_freq: str = "4h") -> pd.DataFrame:
    if normalized_runtime_trades is None or normalized_runtime_trades.empty:
        return pd.DataFrame(columns=["pair", "strategy_key", "side", "action_time", "bar_time", "reason"])

    out = normalized_runtime_trades.copy()
    out["strategy_key"] = out["strategy_id"].map(lambda x: _normalize_strategy_key(x))
    out["action_time"] = pd.to_datetime(out["action_time"], errors="coerce", utc=True)
    out["bar_time"] = out["action_time"].dt.floor(bar_freq)
    out["reason"] = out["reason"].fillna("")
    return out[["pair", "strategy_key", "side", "action_time", "bar_time", "reason"]].sort_values(
        ["bar_time", "pair", "strategy_key", "side"]
    ).reset_index(drop=True)


def compare_trigger_parity(expected_actions: pd.DataFrame, runtime_actions: pd.DataFrame) -> dict[str, pd.DataFrame]:
    group_cols = ["pair", "strategy_key", "side", "bar_time"]

    expected_counts = (
        expected_actions.groupby(group_cols, dropna=False).size().reset_index(name="expected_count")
        if expected_actions is not None and not expected_actions.empty
        else pd.DataFrame(columns=group_cols + ["expected_count"])
    )
    runtime_counts = (
        runtime_actions.groupby(group_cols, dropna=False).size().reset_index(name="runtime_count")
        if runtime_actions is not None and not runtime_actions.empty
        else pd.DataFrame(columns=group_cols + ["runtime_count"])
    )

    merged = expected_counts.merge(runtime_counts, on=group_cols, how="outer")
    merged["expected_count"] = merged["expected_count"].fillna(0).astype(int)
    merged["runtime_count"] = merged["runtime_count"].fillna(0).astype(int)
    merged["delta"] = merged["runtime_count"] - merged["expected_count"]

    mismatches = merged.loc[merged["delta"] != 0].copy()
    if mismatches.empty:
        mismatch_summary = pd.DataFrame(columns=["mismatch_type", "count"])
        return {
            "summary": mismatch_summary,
            "mismatches": mismatches,
            "merged_counts": merged,
        }

    def classify(row):
        side = str(row["side"]).lower()
        if row["expected_count"] > row["runtime_count"]:
            return "expected_buy_runtime_no_buy" if side == "buy" else "expected_sell_runtime_held"
        return "runtime_buy_expected_no_buy" if side == "buy" else "runtime_sell_expected_hold"

    mismatches["mismatch_type"] = mismatches.apply(classify, axis=1)
    mismatches["missing_count"] = (mismatches["expected_count"] - mismatches["runtime_count"]).abs()
    mismatch_summary = mismatches.groupby("mismatch_type").size().reset_index(name="count").sort_values("count", ascending=False)

    return {
        "summary": mismatch_summary,
        "mismatches": mismatches.sort_values(["bar_time", "pair", "strategy_key", "side"]),
        "merged_counts": merged.sort_values(["bar_time", "pair", "strategy_key", "side"]),
    }


def build_slippage_calibration_report(normalized_runtime_trades: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if normalized_runtime_trades is None or normalized_runtime_trades.empty:
        empty = pd.DataFrame()
        return {
            "rows": empty,
            "overall": empty,
            "by_side": empty,
            "by_pair": empty,
            "by_strategy": empty,
        }

    rows = normalized_runtime_trades.copy()
    rows["expected_slippage"] = pd.to_numeric(rows["expected_slippage"], errors="coerce")
    rows["realized_slippage"] = pd.to_numeric(rows["realized_slippage"], errors="coerce")
    rows = rows.dropna(subset=["expected_slippage", "realized_slippage"]).copy()

    if rows.empty:
        empty = pd.DataFrame()
        return {
            "rows": rows,
            "overall": empty,
            "by_side": empty,
            "by_pair": empty,
            "by_strategy": empty,
        }

    rows["slippage_error"] = rows["realized_slippage"] - rows["expected_slippage"]
    rows["abs_slippage_error"] = rows["slippage_error"].abs()

    overall = pd.DataFrame(
        {
            "n": [int(len(rows))],
            "expected_mean": [float(rows["expected_slippage"].mean())],
            "realized_mean": [float(rows["realized_slippage"].mean())],
            "mae": [float(rows["abs_slippage_error"].mean())],
            "rmse": [float(np.sqrt(np.mean(np.square(rows["slippage_error"]))))],
            "bias": [float(rows["slippage_error"].mean())],
            "corr": [float(rows[["expected_slippage", "realized_slippage"]].corr().iloc[0, 1]) if len(rows) > 1 else np.nan],
        }
    )

    by_side = (
        rows.groupby("side", dropna=False)
        .agg(
            n=("side", "size"),
            expected_mean=("expected_slippage", "mean"),
            realized_mean=("realized_slippage", "mean"),
            mae=("abs_slippage_error", "mean"),
            bias=("slippage_error", "mean"),
        )
        .reset_index()
        .sort_values("n", ascending=False)
    )

    by_pair = (
        rows.groupby("pair", dropna=False)
        .agg(
            n=("pair", "size"),
            expected_mean=("expected_slippage", "mean"),
            realized_mean=("realized_slippage", "mean"),
            mae=("abs_slippage_error", "mean"),
            bias=("slippage_error", "mean"),
        )
        .reset_index()
        .sort_values(["n", "mae"], ascending=[False, False])
    )

    by_strategy = (
        rows.groupby("strategy_id", dropna=False)
        .agg(
            n=("strategy_id", "size"),
            expected_mean=("expected_slippage", "mean"),
            realized_mean=("realized_slippage", "mean"),
            mae=("abs_slippage_error", "mean"),
            bias=("slippage_error", "mean"),
        )
        .reset_index()
        .sort_values(["n", "mae"], ascending=[False, False])
    )

    return {
        "rows": rows.sort_values(["action_time", "id"], na_position="last").reset_index(drop=True),
        "overall": overall,
        "by_side": by_side,
        "by_pair": by_pair,
        "by_strategy": by_strategy,
    }


def slippage_disconnect_summary(
    runtime_tables: dict,
    *,
    max_slippage_allowed: float = 0.004,
) -> dict:
    """Quantify the orderbook-vs-qvpuc slippage disconnect in live trade payloads.

    ``expected_slippage_qvpuc`` and ``expected_slippage_orderbook`` are both slippage
    FRACTIONS (multiply by 1e4 for bps), so their ratio is apples-to-apples:
      - qvpuc     : 7-bar average traded-volume liquidity model -- this is EXACTLY the
                    backtest's slippage baseline (entry_cash / qvpuc, capped at
                    max_slippage_allowed on the entry side).
      - orderbook : instantaneous resting order-book depth at execution time.
    Resting depth is a fraction of traded volume, so the ratio is systematically > 1
    and the backtest's exit-slippage baseline structurally UNDERSTATES exit cost.

    ``order_exceeds_book`` flags trades whose notional exceeds the book's max
    absorbable size within ``max_slippage_allowed`` (cap_by_slippage_orderbook). For
    those, the orderbook estimate was INF and got recorded as 0.0 -- so a 0.0
    orderbook estimate is AMBIGUOUS (real zero-slip fill vs blows-through-the-book).

    Returns {"rows": per-trade detail, "summary": aggregate stats}. Unlike
    build_slippage_calibration_report this needs no realized fills, so it works in
    TEST_LIVE / paper mode.
    """
    trades = runtime_tables.get("trades")
    if trades is None or trades.empty:
        return {"rows": pd.DataFrame(), "summary": pd.DataFrame()}
    norm = normalize_runtime_trades(trades)
    if norm.empty:
        return {"rows": pd.DataFrame(), "summary": pd.DataFrame()}
    need = ["expected_slippage_qvpuc", "expected_slippage_orderbook",
            "cap_by_slippage_orderbook", "filled_quote_notional", "slippage_source"]
    if not set(need) <= set(norm.columns):
        return {"rows": pd.DataFrame(), "summary": pd.DataFrame()}

    rows = norm[["pair", "strategy_id", "side", "reason"] + need].copy()
    q = pd.to_numeric(rows["expected_slippage_qvpuc"], errors="coerce")
    ob = pd.to_numeric(rows["expected_slippage_orderbook"], errors="coerce")
    cap = pd.to_numeric(rows["cap_by_slippage_orderbook"], errors="coerce")
    notional = pd.to_numeric(rows["filled_quote_notional"], errors="coerce")

    rows["qvpuc_bps"] = 1e4 * q
    rows["orderbook_bps"] = 1e4 * ob
    rows["ratio"] = ob / q.replace(0, np.nan)
    rows["order_exceeds_book"] = (cap > 0) & (notional > cap)

    valid = rows[rows["orderbook_bps"].notna()].copy()
    summary = pd.DataFrame([{
        "n": int(len(valid)),
        "median_ratio": float(valid["ratio"].median()),
        "p90_ratio": float(valid["ratio"].quantile(0.90)),
        "max_ratio": float(valid["ratio"].max()),
        "median_orderbook_bps": float(valid["orderbook_bps"].median()),
        "p90_orderbook_bps": float(valid["orderbook_bps"].quantile(0.90)),
        "max_orderbook_bps": float(valid["orderbook_bps"].max()),
        "orderbook_zero_pct": float(100.0 * (valid["orderbook_bps"] == 0).mean()),
        "order_exceeds_book_pct": float(100.0 * valid["order_exceeds_book"].mean()),
        "over_cap_pct": float(100.0 * (valid["orderbook_bps"] > 1e4 * max_slippage_allowed).mean()),
    }]) if not valid.empty else pd.DataFrame()

    return {
        "rows": rows.sort_values("ratio", ascending=False, na_position="last").reset_index(drop=True),
        "summary": summary,
    }


def _apply_time_window(actions: pd.DataFrame, start_ts, end_ts, *, time_col: str) -> pd.DataFrame:
    if actions is None or actions.empty:
        return actions

    out = actions.copy()
    out[time_col] = pd.to_datetime(out[time_col], errors="coerce", utc=True)
    mask = pd.Series(True, index=out.index)
    if start_ts is not None:
        mask &= out[time_col] >= start_ts
    if end_ts is not None:
        mask &= out[time_col] <= end_ts
    return out.loc[mask].reset_index(drop=True)


def run_runtime_backtest_validation(
    dict_of_pairs_ori: dict | None = None,
    engine_cfg=None,
    *,
    runtime_db_path: str | Path = "live_runtime/live_state.sqlite",
    output_dir: str | Path = "data/validation",
    raw_data_path: str | Path | None = None,
    data_set_name: str = "validation",
    execution_frame: str = "4hourly",
    gate_col: str = "structure_ok",
    current_slippage_array=None,
    bar_freq: str = "4h",
    start_ts=None,
    end_ts=None,
) -> dict:
    """
    Notebook-ready orchestration for post-step-6 validation:
    - run tester-equivalent baseline
    - extract expected/runtime actions
    - compute trigger parity mismatches
    - compute slippage calibration diagnostics

    Parameters
    ----------
    dict_of_pairs_ori:
        Fetched raw dataset object. If None, the function loads from `raw_data_path` when present,
        otherwise it bootstraps an empty cumulative dataset and fetches required ranges.
    raw_data_path:
        Pickle path to fetched raw dict_of_pairs artifact.
    """

    if engine_cfg is None:
        engine_cfg = EngineConfig(
            initial_cash=10_000.0,
            fee_bps=10.0,
            max_net_worth_loss_ratio=0.05,
            max_slippage_allowed=0.004,
            max_number_of_positions=5,
            maximum_position_of_total_net=0.5,
            max_hold_bars_val=12,
            trailing_stop_pct=0.03,
            trailing_stop_threshold=0.15,
            cash_infusion_amount=1000.0,
            cash_infusion_count=0,
            replacement_score_margin=0.0,
            min_hold_before_replacement=0,
        )

    output_dir = _ensure_within_validation(Path(output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    if raw_data_path is None:
        raw_data_path = _ensure_within_validation(output_dir / f"{data_set_name}_raw_pairs.pickle")
    else:
        raw_data_path = _ensure_within_validation(Path(raw_data_path))

    if dict_of_pairs_ori is None:
        if raw_data_path.exists():
            dict_of_pairs_ori = _load_pickle(raw_data_path)
        else:
            dict_of_pairs_ori = {}

    runtime_tables = load_runtime_tables(runtime_db_path)
    normalized_runtime_trades = normalize_runtime_trades(runtime_tables["trades"])

    inferred_start, inferred_end = resolve_runtime_validation_window(
        normalized_trades=normalized_runtime_trades,
        events_df=runtime_tables.get("events"),
    )

    if start_ts is None:
        start_ts = inferred_start
    else:
        start_ts = pd.to_datetime(start_ts, errors="coerce", utc=True)

    if end_ts is None:
        end_ts = inferred_end
    else:
        end_ts = pd.to_datetime(end_ts, errors="coerce", utc=True)

    required_start = start_ts if start_ts is not None else inferred_start
    if required_start is None:
        required_start = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=30)

    # Backfill enough history for EMA/MA warmups used by tester cells 11-20.
    coverage_start = required_start - pd.Timedelta(days=VALIDATION_HISTORY_BUFFER_DAYS)

    required_end = pd.Timestamp.now(tz="UTC")

    runtime_pairs = []
    if normalized_runtime_trades is not None and not normalized_runtime_trades.empty:
        runtime_pairs = sorted({str(pair) for pair in normalized_runtime_trades["pair"].dropna().tolist()})

    dataset_pairs = sorted(list(dict_of_pairs_ori.keys())) if isinstance(dict_of_pairs_ori, dict) else []
    required_pairs = sorted(set(dataset_pairs).union(runtime_pairs).union({"BTCUSDT"}))

    binance_client = Client()
    try:
        dict_of_pairs_ori, fetch_stats = ensure_validation_dataset_coverage(
            dataset=dict_of_pairs_ori,
            required_pairs=required_pairs,
            required_start=coverage_start,
            required_end=required_end,
            client=binance_client,
        )
    finally:
        try:
            binance_client.close_connection()
        except Exception:
            pass

    _save_pickle(raw_data_path, dict_of_pairs_ori)

    tester_out = run_tester_cells_11_20_equivalent(
        dict_of_pairs_ori=dict_of_pairs_ori,
        engine_cfg=engine_cfg,
        execution_frame=execution_frame,
        gate_col=gate_col,
        data_set_name=data_set_name,
        output_dir=output_dir,
        current_slippage_array=current_slippage_array,
    )

    expected_actions = extract_expected_actions_from_backtest(tester_out["backtest_results"], bar_freq=bar_freq)
    runtime_actions = extract_runtime_actions_for_parity(normalized_runtime_trades, bar_freq=bar_freq)

    expected_actions = _apply_time_window(expected_actions, start_ts, end_ts, time_col="action_time")
    runtime_actions = _apply_time_window(runtime_actions, start_ts, end_ts, time_col="action_time")
    normalized_runtime_trades = _apply_time_window(normalized_runtime_trades, start_ts, end_ts, time_col="action_time")

    parity = compare_trigger_parity(expected_actions, runtime_actions)
    slippage = build_slippage_calibration_report(normalized_runtime_trades)

    tp_runtime_rows = runtime_actions.loc[
        runtime_actions["reason"].astype(str).str.contains("take_profit", case=False, na=False)
    ].copy()
    tp_expected_rows = expected_actions.loc[
        expected_actions["reason"].astype(str).str.contains("take_profit|vol_take_profit", case=False, na=False)
    ].copy()
    tp_summary = pd.DataFrame(
        {
            "runtime_tp_actions": [int(len(tp_runtime_rows))],
            "expected_tp_actions": [int(len(tp_expected_rows))],
            "tp_action_delta": [int(len(tp_runtime_rows) - len(tp_expected_rows))],
        }
    )

    return {
        "runtime_tables": runtime_tables,
        "runtime_trades_normalized": normalized_runtime_trades,
        "dataset": {
            "raw_data_path": str(raw_data_path),
            "required_pairs": required_pairs,
            "fetch_stats": fetch_stats,
            "coverage_start": coverage_start,
            "required_start": required_start,
            "required_end": required_end,
        },
        "window": {
            "start_ts": start_ts,
            "end_ts": end_ts,
            "inferred_start_ts": inferred_start,
            "inferred_end_ts": inferred_end,
        },
        "tester_pipeline": tester_out,
        "expected_actions": expected_actions,
        "runtime_actions": runtime_actions,
        "parity": parity,
        "slippage": slippage,
        "tp_approx_summary": tp_summary,
    }

def clone_payload(payload: dict) -> dict:
    out = {
        "execution_frame": payload["execution_frame"],
        "timeline": payload["timeline"].copy(),
        "pairs": list(payload["pairs"]),
        "shape": dict(payload["shape"]),
        "prices": {k: v.copy() for k, v in payload["prices"].items()},
        "gate": payload["gate"].copy(),
        "aux": {k: v.copy() for k, v in payload.get("aux", {}).items()},
        "strategies": {
            name: {"entry": arrs["entry"].copy(), "exit": arrs["exit"].copy()}
            for name, arrs in payload["strategies"].items()
        },
    }
    if "context" in payload:
        out["context"] = {k: v.copy() for k, v in payload["context"].items()}
    return out


def shift_entries_one_bar(payload: dict) -> dict:
    p = clone_payload(payload)
    for strat_name, arrs in p["strategies"].items():
        e = arrs["entry"]
        shifted = np.zeros_like(e, dtype=np.bool_)
        shifted[1:, :] = e[:-1, :]
        arrs["entry"] = shifted
    return p


def run_passes(payload: dict, engine_cfg):
    reports = {}

    # 1) Baseline
    res_base = run_portfolio_backtest(payload, cfg=engine_cfg)
    reports["baseline"] = res_base["portfolio"]["metrics"]

    # 2) Timing stress: execute 1 bar later
    payload_t1 = shift_entries_one_bar(payload)
    res_t1 = run_portfolio_backtest(payload_t1, cfg=engine_cfg)
    reports["entry_shift_plus_1_bar"] = res_t1["portfolio"]["metrics"]

    # 3) Slippage stress (path-aware proxy via higher fees)
    # If you want +10 bps extra ROUND-TRIP friction, add +5 bps per side.
    cfg_slip = replace(engine_cfg, fee_bps=engine_cfg.fee_bps + 5.0)
    res_slip = run_portfolio_backtest(payload, cfg=cfg_slip)
    reports["extra_roundtrip_10bps"] = res_slip["portfolio"]["metrics"]

    # 4) Harsher slippage stress (+20 bps round-trip)
    cfg_slip2 = replace(engine_cfg, fee_bps=engine_cfg.fee_bps + 10.0)
    res_slip2 = run_portfolio_backtest(payload, cfg=cfg_slip2)
    reports["extra_roundtrip_20bps"] = res_slip2["portfolio"]["metrics"]

    summary = pd.DataFrame(reports).T[
        [
            "trade_count",
            "win_rate",
            "expectancy_rate_pct",
            "total_profit_multiple",
            "final_equity",
            "max_drawdown_pct",
        ]
    ]
    return summary, {
        "baseline": res_base,
        "t1": res_t1,
        "slip10": res_slip,
        "slip20": res_slip2,
    }



import mplfinance as mpf
import matplotlib.pyplot as plt


def plot_equity_curve(backtest_results=None, backtest_payload=None, start_date=None, end_date=None, plot_btc=False, dict_of_pairs=None, from_runtime=False, equity_curve=None):
    """Plot the portfolio equity curve.

    Two input modes:
      * backtest mode (default): ``backtest_results`` + ``backtest_payload``
        (behavior unchanged).
      * runtime mode (``from_runtime=True``): ``equity_curve`` is a pd.Series of
        equity indexed by timestamp — e.g. the ``equity_curve`` returned by
        ``runtime_metrics_report(...)``. A bare array/list is accepted too
        (timestamps become 0..n-1).

    ``start_date``/``end_date`` accept ``DD/MM/YYYY`` (whole day, inclusive — the
    same bounds the engine takes) or ``DD/MM/YYYY HH:MM`` to cut at an exact
    timestamp.
    """

    # Shared with the engine so the rebuilt window matches its bar mask exactly.
    s, s_has_time = _parse_date(start_date, "start_date")
    e, e_has_time = _parse_date(end_date, "end_date")

    if not from_runtime:
        eq = np.asarray(backtest_results["portfolio"]["equity_curve"])[:]
        timeline = pd.to_datetime(backtest_payload["timeline"], errors="coerce")

        # Rebuild the engine's date filter only for timeline alignment.
        mask = _build_date_mask(pd.Series(timeline), start_date, end_date)

        timeline_f = timeline[mask].reset_index(drop=True)
        if len(timeline_f) != len(eq):
            # print(f"Warning: Timeline length ({len(timeline_f)}) does not match equity length ({len(eq)}). Truncating equity to match timeline.")
            eq = eq[:-(len(eq) - len(timeline_f))]

        # eq is already filtered by the engine, so do not mask eq again
        if len(timeline_f) != len(eq):
            raise ValueError(f"Timeline/equity length mismatch: {len(timeline_f)} vs {len(eq)}")

    else:
        # Runtime mode: accept a pd.Series (index = timestamps) or a plain
        # array/list. The date window is applied on the series directly so
        # equity and its timestamps stay aligned.
        series = equity_curve
        if not isinstance(series, pd.Series):
            series = pd.Series(np.asarray(series, dtype=np.float64))
        series = pd.to_numeric(series, errors="coerce").dropna()
        series = series[~series.index.duplicated(keep="last")].sort_index()

        index = pd.to_datetime(series.index, errors="coerce")
        if getattr(index, "tz", None) is not None:
            index = index.tz_localize(None)
        if s is not None:
            series = series[index.to_numpy() >= (s if s_has_time else s.normalize())]
        if e is not None:
            index = pd.to_datetime(series.index, errors="coerce")
            if getattr(index, "tz", None) is not None:
                index = index.tz_localize(None)
            edge = e if e_has_time else _end_of_day(e)
            series = series[index.to_numpy() <= edge]

        final_index = pd.to_datetime(series.index, errors="coerce")
        if getattr(final_index, "tz", None) is not None:
            final_index = final_index.tz_localize(None)
        eq = series.to_numpy(dtype=np.float64)
        timeline_f = pd.Series(final_index.to_numpy())
        if len(timeline_f) != len(eq):
            raise ValueError(f"Timeline/equity length mismatch: {len(timeline_f)} vs {len(eq)}")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(timeline_f, eq, label="Portfolio Equity", linewidth=1.8)
    ax.axhline(eq[0], linestyle="--", alpha=0.6, label="Initial Equity")
    ax.set_title("Equity Curve")
    ax.set_xlabel("Time")
    ax.set_ylabel("Equity")
    ax.grid(True, alpha=0.25)
    ax.legend()



    if plot_btc:

        if dict_of_pairs is None:
            raise ValueError(
                "dict_of_pairs must be provided when plot_btc=True"
            )

        btc_df = (
            dict_of_pairs["BTCUSDT"]
            ["dict_of_frames"]
            ["4hourly"]
            [["open_time", "close"]]
            .copy()
        )

        btc_df["open_time"] = pd.to_datetime(
            btc_df["open_time"],
            errors="coerce"
        )

        btc_df = btc_df.dropna(
            subset=["open_time", "close"]
        ).sort_values("open_time")

        # Create dataframe from equity timeline
        timeline_df = pd.DataFrame({
            "timeline": timeline_f
        }).sort_values("timeline")

        # Match each equity timestamp to the most recent BTC candle
        aligned_btc = pd.merge_asof(
            timeline_df,
            btc_df,
            left_on="timeline",
            right_on="open_time",
            direction="backward"
        )

        ax_btc = ax.twinx()

        ax_btc.plot(
            aligned_btc["timeline"],
            aligned_btc["close"],
            label="BTC",
            linewidth=1.2,
            alpha=0.8,
            color="green"
        )

        ax_btc.set_ylabel("BTCUSDT Price")

        # Combine legends from both axes
        lines_1, labels_1 = ax.get_legend_handles_labels()
        lines_2, labels_2 = ax_btc.get_legend_handles_labels()

        ax.legend(
            lines_1 + lines_2,
            labels_1 + labels_2,
            loc="best"
        )

    else:
        ax.legend(loc="best")

    plt.tight_layout()
    plt.show()


def plot_trade_mpf(trade, dict_of_pairs):
    execution_frame = "4hourly"
    bars_margin = 20
    rsi_period = 14

    pair = trade["pair"]
    strategy = str(trade.get("strategy", ""))
    entry_time = pd.to_datetime(trade["entry_time"].tz_localize(None), errors="coerce")
    exit_time = pd.to_datetime(trade["exit_time"].tz_localize(None).ceil("4h"), errors="coerce")
    exit_reason = trade.get("close_reason", "unknown")

    if pair not in dict_of_pairs:
        raise KeyError(f"Pair not found in dict_of_pairs: {pair}")

    frame = dict_of_pairs[pair]["dict_of_frames"].get(execution_frame)
    if frame is None or frame.empty:
        raise ValueError(f"No data for {pair} on frame '{execution_frame}'")

    df = frame.copy()
    time_col = "close_time" if "close_time" in df.columns else "open_time"
    if time_col not in df.columns:
        raise ValueError("Frame must contain close_time or open_time")

    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col]).sort_values(time_col).drop_duplicates(subset=[time_col], keep="last")
    df = df.set_index(time_col)

    for c in ["open", "high", "low", "close"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")
        df[c] = pd.to_numeric(df[c], errors="coerce")

    vol_candidates = ["quote_asset_volume"]
    vol_col = next((c for c in vol_candidates if c in df.columns), None)

    plot_df = pd.DataFrame(index=df.index)
    plot_df["Open"] = df["open"]
    plot_df["High"] = df["high"]
    plot_df["Low"] = df["low"]
    plot_df["Close"] = df["close"]
    plot_df["Volume"] = pd.to_numeric(df[vol_col], errors="coerce").fillna(0.0) if vol_col else 0.0
    plot_df["spread"] = df["short_spread"]

    entry_idx = plot_df.index.get_indexer([entry_time], method="nearest")[0]
    exit_idx = plot_df.index.get_indexer([exit_time], method="nearest")[0]

    left = max(0, min(entry_idx, exit_idx) - bars_margin)
    right = min(len(plot_df) - 1, max(entry_idx, exit_idx) + bars_margin)
    w = plot_df.iloc[left:right + 1].copy()

    buy_marker = np.full(len(w), np.nan)
    sell_marker = np.full(len(w), np.nan)
    rel_entry = entry_idx - left
    rel_exit = exit_idx - left

    if 0 <= rel_entry < len(w):
        buy_marker[rel_entry] = w["Low"].iloc[rel_entry]  - 2*w["spread"].iloc[rel_entry]
    if 0 <= rel_exit < len(w):
        sell_marker[rel_exit] = w["High"].iloc[rel_exit] + 2*w["spread"].iloc[rel_entry]

    if "rsi" in df.columns:
        rsi_series = pd.to_numeric(df["rsi"], errors="coerce").loc[w.index]
    else:
        delta = w["Close"].diff()
        gain = delta.clip(lower=0).rolling(rsi_period).mean()
        loss = (-delta.clip(upper=0)).rolling(rsi_period).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi_series = 100 - (100 / (1 + rs))

    ma_cols = []
    if "mr_v1" in strategy:
        for c in ["mr_ema_fast_20", "mr_ema_slow_50", "ema20", "ema50"]:
            if c in df.columns:
                ma_cols.append(c)
    elif "test_" in strategy:
        entry_part = strategy.split("__")[0]
        retest_col = entry_part.replace("entry_test_", "").replace("test_", "")
        for c in [retest_col, "ema7_high", "ema25"]:
            if c in df.columns and c not in ma_cols:
                ma_cols.append(c)
    elif "break_" in strategy:
        entry_part = strategy.split("__")[0]
        break_col = entry_part.replace("entry_break_", "").replace("break_", "")
        for c in [break_col, "ema7_high", "ema25"]:
            if c in df.columns and c not in ma_cols:
                ma_cols.append(c)
    else:
        for c in ["ema25", "ema50", "ema100", "ema200", "ma25", "ma50", "ma100", "ma200"]:
            if c in df.columns:
                ma_cols.append(c)

    addplots = []
    ma_colors = ["dodgerblue", "orange", "violet", "gold", "deepskyblue", "tomato", "limegreen", "magenta"]

    for i, c in enumerate(ma_cols[:8]):
        s = pd.to_numeric(df[c], errors="coerce").loc[w.index]
        addplots.append(mpf.make_addplot(s, panel=0, color=ma_colors[i], width=1.1))

    profit_multiple = 1.5
    if (("max" in strategy) or ("boll_upper" in strategy) or ("nearest_resistance" in strategy) or ("nearest_support" in strategy)) and ("1daily" not in strategy):
        profit_multiple = 2.5
        
    take_profit_line = df["ema7_high"] + profit_multiple * df.shift(1)["short_spread"]
    take_profit_line_num = pd.to_numeric(take_profit_line, errors="coerce").loc[w.index]
    addplots.append(mpf.make_addplot(take_profit_line_num, panel=0, color="black", width=1.1))

    # stop_price_series = (df[["open", "close"]].min(axis=1) - 1.25*df["short_spread"])#.rolling(window=10).max()
    # stop_price_series_num = pd.to_numeric(stop_price_series, errors="coerce").loc[w.index]
    # addplots.append(mpf.make_addplot(stop_price_series_num, panel=0, color="black", width=1.1))

    stop_price = None
    entry_row = w.iloc[rel_entry]
    if "test" in strategy:
        try:
            stop_price = (entry_row[retest_col] - 1.25 * entry_row["spread"])
        except:
            pass
    elif "break" in strategy:
        # stop_price = (entry_row[["Open", "Close"]].min(axis=1) - 1.25*entry_row["spread"])
        try:
            stop_price = (np.min([entry_row["Open"], entry_row["Close"]]) - 1.25*entry_row["spread"])
        except:
            pass

    # if stop_price is not None:
    #     # plt.axhline(y=stop_price, color='r', linestyle='--')
    #     addplots.append(mpf.make_addplot([stop_price], type="line", marker="--", markersize=120, color="red", panel=0))

    addplots.append(mpf.make_addplot(buy_marker, type="scatter", marker="^", markersize=120, color="green", panel=0))
    addplots.append(mpf.make_addplot(sell_marker, type="scatter", marker="v", markersize=120, color="red", panel=0))

    addplots.append(mpf.make_addplot(rsi_series, panel=2, color="cyan", width=1.0, ylabel="RSI"))
    addplots.append(mpf.make_addplot(pd.Series(70, index=w.index), panel=2, color="gray", linestyle="--", width=0.8))
    addplots.append(mpf.make_addplot(pd.Series(30, index=w.index), panel=2, color="gray", linestyle="--", width=0.8))

    def fmt_hour_ceil(ts):
        if pd.isna(ts):
            return "N/A"
        return pd.Timestamp(ts).ceil("4h").strftime("%d/%m/%Y %H")

    entry_str = fmt_hour_ceil(entry_time)
    exit_str = fmt_hour_ceil(exit_time)
    pnl = trade.get("pnl", np.nan)
    gain = ((trade.get("exit_price", np.nan) / trade.get("entry_price", np.nan)) - 1.0) * 100.0
    hold_bars = trade.get("hold_bars", np.nan)
    regime = trade.get("regime", None)

    title = (
        f"{pair} | {strategy}\n"
        f"Entry: {entry_str}  Exit: {exit_str}  \n"
        f"Exit Reason: {exit_reason}  "
        # f"PnL: {pnl:.2f}  Hold bars: {hold_bars}  Regime: {regime}"
        f"gain: {gain:.2f}%  Hold bars: {hold_bars}  Regime: {regime}"
        if pd.notna(pnl) else
        f"{pair} | {strategy}\n"
        f"Entry: {entry_str}  Exit: {exit_str}  Exit Reason: {exit_reason}"
    )

    fig, axes = mpf.plot(
        w,
        type="candle",
        style="yahoo",
        addplot=addplots,
        volume=True,
        panel_ratios=(6, 2, 2),
        figsize=(15, 9),
        title=title,
        ylabel="Price",
        ylabel_lower="Volume",
        returnfig=True,
            hlines=dict(
        hlines=[stop_price],
        colors=['red'],
        linestyle='--'
    ) if stop_price is not None else {},
        # colors=["red"]
    )

    main_ax = axes[0]

    # Exit reason annotation near sell marker
    if 0 <= rel_exit < len(w) and not np.isnan(sell_marker[rel_exit]):
        main_ax.annotate(
            f"Exit: {exit_reason}",
            xy=(w.index[rel_exit], sell_marker[rel_exit]),
            xytext=(10, 15),
            textcoords="offset points",
            color="red",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="red", alpha=0.8),
            arrowprops=dict(arrowstyle="->", color="red", lw=0.8),
        )

    legend_lines = []
    legend_labels = []

    for i, c in enumerate(ma_cols[:8]):
        legend_lines.append(plt.Line2D([0], [0], color=ma_colors[i], lw=1.5))
        legend_labels.append(c)

    legend_lines.append(plt.Line2D([0], [0], marker="^", color="green", linestyle="None", markersize=10))
    legend_labels.append("Buy")
    legend_lines.append(plt.Line2D([0], [0], marker="v", color="red", linestyle="None", markersize=10))
    legend_labels.append("Sell")

    if legend_lines:
        main_ax.legend(legend_lines, legend_labels, loc="upper left")

    # Intentionally no plt.tight_layout() to avoid mplfinance axes warning
    plt.show()




def detect_equity_shock_recovery_events(
    backtest_results,
    backtest_payload,
    drop_threshold=-0.60,
    recovery_target=0.98,
    max_recovery_bars=200,
    top_k_trades=5,
    top_k_bar_moves=5,
    min_bars_between_events=5,
    start_date=None,
    end_date=None,
    verbose=True,
    ):
    """
    Detect large equity drawdowns that recover quickly, then explain likely drivers.

    drop_threshold: drawdown level (negative), e.g. -0.60 means >=60% drawdown
    recovery_target: fraction of prior peak considered 'recovered', e.g. 0.98
    max_recovery_bars: only keep episodes that recover within this many bars
    """
    eq = np.asarray(backtest_results["portfolio"]["equity_curve"], dtype=np.float64)

    # Prefer timeline from results if present; otherwise rebuild from payload timeline + date window.
    if "timeline" in backtest_results["portfolio"]:
        timeline = pd.to_datetime(backtest_results["portfolio"]["timeline"], errors="coerce")
    else:
        timeline = pd.to_datetime(backtest_payload["timeline"], errors="coerce")

        if len(timeline) != len(eq):
            # Use explicit args if passed; else fallback to notebook globals.
            s_raw = start_date if start_date is not None else globals().get("start_date", None)
            e_raw = end_date if end_date is not None else globals().get("end_date", None)

            mask = _build_date_mask(timeline, s_raw, e_raw)
            timeline = timeline[mask].reset_index(drop=True)

    trades = backtest_results["portfolio"]["trade_frame"].copy()

    if len(eq) == 0:
        raise ValueError("Equity curve is empty")

    if len(timeline) != len(eq):
        raise ValueError(
            f"Timeline/equity mismatch: {len(timeline)} vs {len(eq)}. "
            "Pass start_date/end_date to this function in DD/MM/YYYY or "
            "DD/MM/YYYY HH:MM format."
        )

    if np.isnan(eq).any():
        raise ValueError("Equity curve contains NaN values")

    if not trades.empty:
        trades["entry_time"] = pd.to_datetime(trades["entry_time"], errors="coerce")
        trades["exit_time"] = pd.to_datetime(trades["exit_time"], errors="coerce")

    eq_series = pd.Series(eq, index=timeline, name="equity")
    peak_equity = eq_series.cummax()
    drawdown = (eq_series / peak_equity) - 1.0
    bar_ret = eq_series.pct_change().fillna(0.0)

    events = []
    i = 1
    T = len(eq_series)

    while i < T:
        if drawdown.iloc[i] <= drop_threshold and drawdown.iloc[i - 1] > drop_threshold:
            peak_idx = int(np.argmax(eq[: i + 1]))
            peak_val = float(eq[peak_idx])

            j = i
            trough_idx = i
            trough_val = float(eq[i])
            recover_idx = None

            while j < T:
                if eq[j] < trough_val:
                    trough_val = float(eq[j])
                    trough_idx = j

                if eq[j] >= recovery_target * peak_val:
                    recover_idx = j
                    break
                j += 1

            if recover_idx is not None and (recover_idx - trough_idx) <= max_recovery_bars:
                start_t = timeline.iloc[peak_idx]
                trough_t = timeline.iloc[trough_idx]
                end_t = timeline.iloc[recover_idx]

                drop_pct = ((trough_val / peak_val) - 1.0) * 100.0
                recov_pct_from_trough = np.nan if trough_val <= 0 else ((eq[recover_idx] / trough_val) - 1.0) * 100.0

                span_bar_ret = bar_ret.iloc[peak_idx : recover_idx + 1]
                worst_bars = span_bar_ret.nsmallest(top_k_bar_moves)
                best_bars = span_bar_ret.nlargest(top_k_bar_moves)

                drop_trades = pd.DataFrame()
                recovery_trades = pd.DataFrame()
                if not trades.empty and "exit_time" in trades.columns:
                    drop_mask = (trades["exit_time"] >= start_t) & (trades["exit_time"] <= trough_t)
                    rec_mask = (trades["exit_time"] > trough_t) & (trades["exit_time"] <= end_t)

                    drop_trades = trades.loc[drop_mask].sort_values("pnl", ascending=True).head(top_k_trades)
                    recovery_trades = trades.loc[rec_mask].sort_values("pnl", ascending=False).head(top_k_trades)

                # Time-step sanity checks for likely data quality problems
                event_times = timeline.iloc[peak_idx : recover_idx + 1]
                dt_seconds = event_times.diff().dt.total_seconds().dropna()
                has_sub_hour_bars = bool((dt_seconds < 3600).any()) if not dt_seconds.empty else False
                has_zero_equity = trough_val <= 0

                events.append({
                    "peak_idx": peak_idx,
                    "trough_idx": trough_idx,
                    "recover_idx": recover_idx,
                    "peak_time": start_t,
                    "trough_time": trough_t,
                    "recovery_time": end_t,
                    "bars_peak_to_trough": trough_idx - peak_idx,
                    "bars_trough_to_recovery": recover_idx - trough_idx,
                    "peak_equity": peak_val,
                    "trough_equity": trough_val,
                    "recovery_equity": float(eq[recover_idx]),
                    "drop_pct": drop_pct,
                    "recovery_pct_from_trough": recov_pct_from_trough,
                    "worst_bars": worst_bars,
                    "best_bars": best_bars,
                    "drop_trades": drop_trades,
                    "recovery_trades": recovery_trades,
                    "has_sub_hour_bars": has_sub_hour_bars,
                    "has_zero_equity": has_zero_equity,
                })

                i = recover_idx + min_bars_between_events
                continue
        i += 1

    summary_rows = []
    for k, e in enumerate(events, 1):
        summary_rows.append({
            "event_id": k,
            "peak_time": e["peak_time"],
            "trough_time": e["trough_time"],
            "recovery_time": e["recovery_time"],
            "drop_pct": e["drop_pct"],
            "bars_peak_to_trough": e["bars_peak_to_trough"],
            "bars_trough_to_recovery": e["bars_trough_to_recovery"],
            "peak_equity": e["peak_equity"],
            "trough_equity": e["trough_equity"],
            "recovery_equity": e["recovery_equity"],
            "has_sub_hour_bars": e["has_sub_hour_bars"],
            "has_zero_equity": e["has_zero_equity"],
        })

    summary = pd.DataFrame(summary_rows)

    def _print_df(df: pd.DataFrame) -> None:
        if df is None or df.empty:
            print("(empty)")
            return
        print(df.to_string())

    if verbose:
        if summary.empty:
            print("No fast shock-recovery events found for current thresholds.")
        else:
            print("Detected fast shock-recovery events:")
            _print_df(summary)

            for idx, e in enumerate(events, 1):
                print("\n" + "=" * 90)
                print(f"Event #{idx}: {e['peak_time']} -> {e['trough_time']} -> {e['recovery_time']}")
                print(f"Drop: {e['drop_pct']:.2f}% | Bars to trough: {e['bars_peak_to_trough']} | Bars to recover: {e['bars_trough_to_recovery']}")
                if e["has_zero_equity"] or e["has_sub_hour_bars"]:
                    print("Data-quality warning:")
                    if e["has_zero_equity"]:
                        print("  - Trough equity reached 0 (can create infinite rebound percentages).")
                    if e["has_sub_hour_bars"]:
                        print("  - Sub-hour bar spacing detected inside event (unexpected for 4H execution).")

                print("\nWorst bar returns inside event:")
                _print_df((100.0 * e["worst_bars"]).rename("bar_return_pct").to_frame())

                print("Top rebound bar returns inside event:")
                _print_df((100.0 * e["best_bars"]).rename("bar_return_pct").to_frame())

                if not e["drop_trades"].empty:
                    cols = [c for c in ["pair", "strategy", "entry_time", "exit_time", "pnl", "return_pct", "close_reason"] if c in e["drop_trades"].columns]
                    print("Worst closed trades during drop phase:")
                    _print_df(e["drop_trades"][cols])
                else:
                    print("No closed trades in drop phase.")

                if not e["recovery_trades"].empty:
                    cols = [c for c in ["pair", "strategy", "entry_time", "exit_time", "pnl", "return_pct", "close_reason"] if c in e["recovery_trades"].columns]
                    print("Best closed trades during recovery phase:")
                    _print_df(e["recovery_trades"][cols])
                else:
                    print("No closed trades in recovery phase.")

    return summary, events


def get_viable_entries_summary(backtest_payload, start_date=None, end_date=None):
    """
    Compute a summary of viable entry signals for each strategy within a specified date range.

    Parameters:
    - backtest_payload: dict, the backtest payload containing strategies and their signals.
    - start_date: str or None, the start date in "DD/MM/YYYY" or "DD/MM/YYYY HH:MM" format.
      If None, no lower bound is applied.
    - end_date: str or None, the end date in "DD/MM/YYYY" or "DD/MM/YYYY HH:MM" format.
      If None, no upper bound is applied.

    Returns:
    - viability_summary: pd.DataFrame, a summary of viable entries for each strategy.
    """
    # Engine-shared window (DD/MM/YYYY = whole day, DD/MM/YYYY HH:MM = exact edge).
    timeline = pd.to_datetime(backtest_payload["timeline"], errors="coerce")
    mask = pd.Series(_build_date_mask(timeline, start_date, end_date))

    # print("Bars in window:", int(mask.sum()))

    # 1) Raw entry signals per strategy (after payload build)
    rows = []
    for strat, sig in backtest_payload["strategies"].items():
        entry = sig["entry"][mask.to_numpy(), :]   # (T_window, N)
        rows.append({
            "strategy": strat,
            "entry_signals": int(entry.sum()),
            "bars_with_any_entry": int(entry.any(axis=1).sum()),
        })

    entry_summary = pd.DataFrame(rows).sort_values("entry_signals", ascending=False)
    # display(entry_summary)

    # 2) Optional: entry viability checks used by engine filters
    # Aux is now strategy-specific, so inspect the first strategy by default.
    first_strategy = next(iter(backtest_payload["strategies"].keys()))
    strategy_aux = backtest_payload["aux"][first_strategy]
    close_arr = backtest_payload["prices"]["close"][mask.to_numpy(), :]
    stop_arr = strategy_aux["stop_price"][mask.to_numpy(), :]
    qvpuc_arr = strategy_aux["quote_vol_per_unit_change"][mask.to_numpy(), :]

    basic_viable = (
        ~np.isnan(close_arr) &
        ~np.isnan(stop_arr) &
        ~np.isnan(qvpuc_arr) &
        (stop_arr > 0) &
        (stop_arr < close_arr)
    )

    rows2 = []
    for strat, sig in backtest_payload["strategies"].items():
        entry = sig["entry"][mask.to_numpy(), :]
        viable_entries = entry & basic_viable
        rows2.append({
            "strategy": strat,
            "entry_signals": int(entry.sum()),
            "viable_entries_basic": int(viable_entries.sum()),
        })

    viability_summary = pd.DataFrame(rows2).sort_values("viable_entries_basic", ascending=False)

    return viability_summary


def diagnose_frame_integrity(
    dict_of_pairs,
    execution_frame,
    expected_bar_hours=4,
    jump_threshold=0.50,
    zscore_threshold=12.0,
    top_k=20,
    verbose=True
    ):
    """
    Scan per-pair execution-frame data for anomalies that can create implausible equity shocks.

    Checks:
    - non-monotonic or duplicate close_time
    - sub-expected bar spacing (e.g., <4h for 4H frame)
    - non-positive OHLC values
    - extreme close-to-close jumps
    - close-price outliers via robust z-score on log returns
    """
    summary_rows = []
    jump_rows = []

    def _print_df(df: pd.DataFrame) -> None:
        if df is None or df.empty:
            print("(empty)")
            return
        
        if verbose:
            print(df.to_string())

    expected_seconds = int(expected_bar_hours * 3600)

    for pair, pdata in dict_of_pairs.items():
        frame = pdata.get("dict_of_frames", {}).get(execution_frame)
        if frame is None or frame.empty:
            continue

        df = frame.copy()
        if "close_time" not in df.columns:
            continue

        df["close_time"] = pd.to_datetime(df["close_time"], errors="coerce")
        df = df.dropna(subset=["close_time"]).sort_values("close_time").reset_index(drop=True)

        # Ensure numeric OHLC
        for c in ["open", "high", "low", "close"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            else:
                df[c] = np.nan

        dt = df["close_time"].diff().dt.total_seconds()
        dt_valid = dt.dropna()

        dup_count = int(df["close_time"].duplicated(keep=False).sum())
        non_monotonic_count = int((dt_valid <= 0).sum())
        sub_expected_count = int((dt_valid < expected_seconds).sum())
        huge_gap_count = int((dt_valid > (expected_seconds * 3)).sum())

        non_positive_count = int(((df[["open", "high", "low", "close"]] <= 0).any(axis=1)).sum())

        c2c = df["close"].pct_change()
        abs_jump = c2c.abs()
        jump_mask = abs_jump > jump_threshold
        jump_count = int(jump_mask.sum())

        # Robust z-score on log returns
        lr = np.log(df["close"]).diff().replace([np.inf, -np.inf], np.nan)
        med = np.nanmedian(lr)
        mad = np.nanmedian(np.abs(lr - med))
        if mad > 0 and np.isfinite(mad):
            robust_z = 0.6745 * (lr - med) / mad
            outlier_mask = robust_z.abs() > zscore_threshold
            outlier_count = int(outlier_mask.sum())
        else:
            robust_z = pd.Series(np.nan, index=df.index)
            outlier_mask = pd.Series(False, index=df.index)
            outlier_count = 0

        suspicious = any([
            dup_count > 0,
            non_monotonic_count > 0,
            sub_expected_count > 0,
            non_positive_count > 0,
            jump_count > 0,
            outlier_count > 0,
        ])

        summary_rows.append({
            "pair": pair,
            "rows": len(df),
            "dup_timestamps": dup_count,
            "non_monotonic_dt": non_monotonic_count,
            "sub_expected_dt": sub_expected_count,
            "huge_gaps": huge_gap_count,
            "non_positive_ohlc": non_positive_count,
            "abs_jump_gt_threshold": jump_count,
            "robust_z_outliers": outlier_count,
            "suspicious": suspicious,
        })

        if jump_count > 0 or outlier_count > 0 or non_positive_count > 0:
            flagged = df.loc[jump_mask | outlier_mask | ((df[["open", "high", "low", "close"]] <= 0).any(axis=1))].copy()
            flagged["pair"] = pair
            flagged["abs_close_to_close_return"] = abs_jump.loc[flagged.index]
            flagged["close_to_close_return"] = c2c.loc[flagged.index]
            flagged["robust_z_logret"] = robust_z.loc[flagged.index]
            jump_rows.append(flagged[[
                "pair", "close_time", "open", "high", "low", "close",
                "close_to_close_return", "abs_close_to_close_return", "robust_z_logret"
            ]])

    summary_df = pd.DataFrame(summary_rows)
    if summary_df.empty:
        print("No frame data found for diagnostics.")
        return summary_df, pd.DataFrame()

    summary_df = summary_df.sort_values(
        ["suspicious", "sub_expected_dt", "abs_jump_gt_threshold", "robust_z_outliers", "dup_timestamps"],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)

    flagged_rows_df = pd.concat(jump_rows, ignore_index=True) if jump_rows else pd.DataFrame()
    if not flagged_rows_df.empty:
        flagged_rows_df = flagged_rows_df.sort_values(
            ["abs_close_to_close_return", "robust_z_logret"],
            ascending=[False, False],
        ).head(top_k).reset_index(drop=True)

    print("Pair-level integrity summary:")
    _print_df(summary_df.head(top_k))

    suspicious_count = int(summary_df["suspicious"].sum())
    print(f"Suspicious pairs: {suspicious_count} / {len(summary_df)}")

    if not flagged_rows_df.empty:
        print("Top suspicious rows:")
        _print_df(flagged_rows_df)
    else:
        print("No suspicious price/jump rows found with current thresholds.")

    return summary_df, flagged_rows_df

# === Trade Comparator Helpers ===

def _cliffs_delta_from_samples(a: pd.Series, b: pd.Series) -> float:
    a = pd.to_numeric(a, errors="coerce").dropna()
    b = pd.to_numeric(b, errors="coerce").dropna()
    n_a = len(a)
    n_b = len(b)
    if n_a == 0 or n_b == 0:
        return np.nan

    combined = pd.concat([a.reset_index(drop=True), b.reset_index(drop=True)], ignore_index=True)
    ranks = combined.rank(method="average")
    r_a = ranks.iloc[:n_a].sum()
    u_a = r_a - (n_a * (n_a + 1) / 2.0)
    delta = (2.0 * u_a / (n_a * n_b)) - 1.0
    return float(delta)

def build_entry_feature_table(
    trade_frame: pd.DataFrame,
    dict_of_pairs: dict,
    execution_frame: str,
    max_match_seconds: int = 60,
    base_feature_cols: list[str] | None = None,
) -> pd.DataFrame:
    if base_feature_cols is None:
        base_feature_cols = [
            "close", "open", "high", "low",
            "volatility", "adx", "sqn_30", "sqn_90",
            "rsi", "rsi_slope", "short_spread",
            "allowed_to_trade", "allowed_daily", "pair_tradable", "structure_ok",
            "regime", "max", "min",
        ]

    tf = trade_frame.copy()
    tf["entry_time"] = pd.to_datetime(tf["entry_time"], errors="coerce")
    tf["pair"] = tf["pair"].astype(str)
    tf["strategy"] = tf["strategy"].astype(str)
    tf["entry_col"] = tf["strategy"].str.split("__").str[0]
    tf["win"] = pd.to_numeric(tf["pnl"], errors="coerce") > 0

    unique_entry_cols = sorted(tf["entry_col"].dropna().unique().tolist())
    per_pair = []

    for pair, sub in tf.groupby("pair", sort=False):
        frame = dict_of_pairs.get(pair, {}).get("dict_of_frames", {}).get(execution_frame)
        if frame is None or frame.empty:
            continue

        f = frame.copy()
        if "close_time" not in f.columns:
            continue

        f["close_time"] = pd.to_datetime(f["close_time"], errors="coerce")
        f = f.dropna(subset=["close_time"]).sort_values("close_time")
        if f.empty:
            continue

        candidate_cols = ["close_time", *base_feature_cols, *unique_entry_cols]
        existing_cols = [c for c in candidate_cols if c in f.columns]
        f = f[existing_cols].drop_duplicates(subset=["close_time"], keep="last")
        f = f.rename(columns={"close_time": "frame_time"})

        left = sub.sort_values("entry_time").copy()
        merged = pd.merge_asof(
            left,
            f.sort_values("frame_time"),
            left_on="entry_time",
            right_on="frame_time",
            direction="nearest",
            tolerance=pd.Timedelta(seconds=max_match_seconds),
        )
        per_pair.append(merged)

    if not per_pair:
        return pd.DataFrame()

    out = pd.concat(per_pair, ignore_index=True)
    out["entry_time_match_diff_s"] = (out["entry_time"] - out["frame_time"]).dt.total_seconds().abs()

    out["gain_ratio_feature"] = np.where(
        (pd.to_numeric(out.get("min"), errors="coerce") > 0),
        pd.to_numeric(out.get("max"), errors="coerce") / pd.to_numeric(out.get("min"), errors="coerce"),
        np.nan,
    )

    def _entry_signal_value(row: pd.Series) -> float:
        col = row.get("entry_col")
        if isinstance(col, str) and col in row.index:
            return row[col]
        return np.nan

    out["entry_signal_value"] = out.apply(_entry_signal_value, axis=1)

    return out

def apply_default_strategies(dict_of_pairs, execution_frame):


    for pair in dict_of_pairs:
        df = dict_of_pairs[pair]["dict_of_frames"][execution_frame]
        if df is not None and not df.empty:
            df["all_true_gate"] = True
            dict_of_pairs[pair]["dict_of_frames"][execution_frame] = df

    list_of_strategies = []


    cols_for_break = [
                    #   "prev_high",
                    "boll_upper_1daily",
                    "ema100", "ema200", "ma100", "ma200",
                    "ema100_1daily", "ema200_1daily", "ma100_1daily", "ma200_1daily",
                    "yearly_open", "monthly_open","weekly_open", 
                    ]

    for col in cols_for_break:

        strategy_dict = {
            "name": f"break_{col}",
            'entry_col': f"entry_break_{col}",
            'exit_col': f"exit_break_{col}"
        }

        list_of_strategies.append(strategy_dict)

        dict_of_pairs = apply_break_strategy(
            dict_of_pairs=dict_of_pairs,
            execution_frame=execution_frame,
            entry_col=strategy_dict["entry_col"],
            exit_col=strategy_dict["exit_col"],
            break_column=col,
            aux_prefix=strategy_dict["entry_col"],
            fast_profit_threshold=1.5,
            gain_ratio = 1.15,
            lower_volatility_threshold = 0.01,
            upper_volatility_threshold = 0.08,
            use_rsi_slope_exit = False
        )

    cols_for_retest = [
        'ma50',
        'boll_lower',
        'ema50',
        'ma200',
        'boll_mid_1daily',
        'ema100',
        'ma100',
        'ema200',
        'ema25_1daily',
        "yearly_open", "monthly_open"
    ]

    for col in cols_for_retest:

        strategy_dict = {
            "name": f"test_{col}",
            'entry_col': f"entry_test_{col}",
            'exit_col': f"exit_test_{col}"
        }

        list_of_strategies.append(strategy_dict)

        dict_of_pairs = apply_test_strategy(
            dict_of_pairs=dict_of_pairs,
            execution_frame=execution_frame,
            entry_col=strategy_dict["entry_col"],
            exit_col=strategy_dict["exit_col"],
            retest_column=col,
            aux_prefix=strategy_dict["entry_col"],
            fast_profit_threshold=1.5,
            lower_volatility_threshold = 0.02,
            upper_volatility_threshold = 0.08,
            gain_ratio = 1.15,
            use_rsi_slope_exit = False
        )

    extra_break_cols = ["max", "boll_upper", "nearest_resistance"]
    for col in extra_break_cols:
        specific_column = col
        strategy_dict = {
            "name": f"break_{specific_column}",
            'entry_col': f"entry_break_{specific_column}",
            'exit_col': f"exit_break_{specific_column}"
        }

        list_of_strategies.append(strategy_dict)

        dict_of_pairs = apply_break_strategy(
        dict_of_pairs=dict_of_pairs,
        execution_frame=execution_frame,
        entry_col=strategy_dict["entry_col"],
        exit_col=strategy_dict["exit_col"],
        break_column=specific_column,
        aux_prefix=strategy_dict["entry_col"],
        fast_profit_threshold=2.5,
        gain_ratio = 1.15,
        lower_volatility_threshold = 0.01,
        upper_volatility_threshold = 0.08,
        )

    return dict_of_pairs, list_of_strategies

def prepare_dataset_for_validation(dict_of_pairs_ori):

    for pair in dict_of_pairs_ori:
        for frame in dict_of_pairs_ori[pair]['dict_of_frames']:
            dict_of_pairs_ori[pair]['dict_of_frames'][frame] = prepare_df(dict_of_pairs_ori[pair]['dict_of_frames'][frame])

            if frame == "4hourly":
                frame_df = dict_of_pairs_ori[pair]['dict_of_frames'][frame]
                time_col = "open_time" if "open_time" in frame_df.columns else None
                if time_col is not None:
                    time_index = pd.to_datetime(frame_df[time_col], errors="coerce")
                    time_series = pd.Series(time_index, index=frame_df.index)
                    open_series = pd.to_numeric(frame_df["open"], errors="coerce")

                    year_key = time_series.dt.to_period("Y")
                    month_key = time_series.dt.to_period("M")
                    week_key = time_series.dt.to_period("W-SUN")  # Weeks start on Monday.

                    frame_df["yearly_open"] = open_series.groupby(year_key).transform("first")
                    frame_df["monthly_open"] = open_series.groupby(month_key).transform("first")
                    frame_df["weekly_open"] = open_series.groupby(week_key).transform("first")

                dict_of_pairs_ori[pair]['dict_of_frames'][frame] = frame_df

    dict_of_pairs_ori = attach_sqn_to_dataset(dict_of_pairs_ori)

    dict_of_pairs_ori = compute_4h_structure_for_dataset(
        dict_of_pairs_ori,
        ema_fast_period=25,
        ema_mid_period=50,
        ema_slow_period=200,
        slope_lookback=5
    )

    frame_list = ["1daily", "4hourly"]

    execution_frame = "4hourly"

    columns_for_merge = ["ma25", "ma50", "ma100", "ma200", "ema25", "ema50", "ema100", "ema200", "rsi", 'max',
        'min', 'boll_upper', 'boll_mid', 'boll_lower']


    for pair in dict_of_pairs_ori:

        for i in range(0,frame_list.index(execution_frame)):
            
            frame = frame_list[i]

            target = dict_of_pairs_ori[pair]['dict_of_frames'][execution_frame]
            source = dict_of_pairs_ori[pair]['dict_of_frames'][frame][columns_for_merge + ["close_time"]]

            dict_of_pairs_ori[pair]['dict_of_frames'][execution_frame] = pd.merge_asof(
                target.sort_values("close_time"),
                source.sort_values("close_time"),
                left_on="close_time",
                right_on="close_time",
                direction="backward",
                suffixes=[f"", f"_{frame}"]
            )
        

    cols_for_retest = ['ma50',
    'boll_lower',
    'ema50',
    'ma200',
    'boll_mid_1daily',
    'ema100',
    'ma100',
    'ema200',
    'ema25_1daily', 'prev_low', "min", "ema25",
    "yearly_open", "monthly_open", "weekly_open", "nearest_support"]

    cols_for_break = [
                    "max","boll_upper", "boll_upper_1daily",
                    "ema100", "ema200", "ma100", "ma200",
                    "ema100_1daily", "ema200_1daily", "ma100_1daily", "ma200_1daily", "prev_high",
                    "yearly_open", "monthly_open", "weekly_open", "nearest_resistance"
                    ]

    cols_to_prepare = cols_for_retest + cols_for_break

    for pair in dict_of_pairs_ori:
        frame_df = dict_of_pairs_ori[pair]['dict_of_frames'][execution_frame]
        frame_df = add_hh_ll_columns(frame_df)

        frame_df = populate_historical_levels_fast(
            ohlc_df=frame_df,
            levels_reactions_limit=12,
            lookback_bars=None,   # or e.g. 6*90 for last 90 days on 4h bars
            price_col="close",
            skip_rows = 6*14
        )

        for col in list(set(cols_to_prepare)):

            retest_dict = level_test_buys_with_last_broken(frame_df, [col], confirm_breaks=False, confirm_tests=False)
            retest_frame = pd.DataFrame(retest_dict)

            rename_dict = {prop: f"{col}_{prop}" for prop in retest_dict}
            retest_frame.rename(columns=rename_dict, inplace=True)

            frame_df = pd.concat([frame_df, retest_frame], axis=1)

        frame_df["rolling_max"] = frame_df["close"].rolling(6*7).max()
        frame_df["rolling_min"] = frame_df["close"].rolling(6*7).min()
        frame_df["rolling_gain"] = frame_df["rolling_max"] / frame_df["rolling_min"]
        frame_df["rsi_slope"] = frame_df["rsi"].rolling(6).apply(get_simple_slope_2, raw=True)



        frame_df = get_common_aux_columns(frame_df)
        
        dict_of_pairs_ori[pair]['dict_of_frames'][execution_frame] = frame_df

    dict_of_pairs_ori = prepare_execution_frame_dataset(
        dict_of_pairs=dict_of_pairs_ori,
        execution_frame=execution_frame,
        btc_pair="BTCUSDT",
        structure_col="structure_ok",
    )

    return dict_of_pairs_ori, execution_frame


def build_specs_and_compile(dict_of_pairs, execution_frame, list_of_strategies, gate_col):

    entry_columns = [strat['entry_col'] for strat in list_of_strategies]
    exit_columns = [strat['exit_col'] for strat in list_of_strategies]

    strategy_specs = build_strategy_specs(
        entry_columns=entry_columns,
        exit_columns=exit_columns,
        mode="pairwise"
    )

    backtest_payload = compile_backtest_payload(
        dict_of_pairs=dict_of_pairs,
        execution_frame=execution_frame,
        strategy_specs=strategy_specs,
        gate_col=gate_col,
        time_col="close_time",
        price_columns=("open", "high", "low", "close"),
        aux_columns=STRATEGY_AUX_COLUMNS,
        context_columns=["regime"],
        enforce_bar_cadence=False,
        min_spacing_ratio= 0.90
    )

    return backtest_payload


def get_strategy_performance_summary(backtest_results, top_k=10):
    strategy_regime_stats = (
        backtest_results["portfolio"]["trade_frame"].assign(win=backtest_results["portfolio"]["trade_frame"]["pnl"] > 0)
        .groupby(["strategy"])
        .agg(
            trades=("win", "size"),
            wins=("win", "sum"),
            win_rate_pct=("win", "mean"),
            avg_return_pct=("return_pct", "mean"),
            total_pnl=("pnl", "sum"),
        )
    )

    strategy_regime_stats["win_rate_pct"] *= 100

    return strategy_regime_stats.sort_values(["total_pnl"], ascending=[False]).head(top_k)