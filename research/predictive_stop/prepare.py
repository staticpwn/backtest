"""Reconstruct the multi-strategy notebook book without optional live imports."""
import ast
import json
import os
import sys
import time
import warnings
from pathlib import Path
from importlib import reload

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import talib
from threadpoolctl import threadpool_limits
import features
import gating
import helpers
import strategies
from backtest_prep import compile_backtest_payload, build_strategy_specs


def main():
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    universe = pd.read_csv(ROOT / "binance_usdt_tokens.csv").option_symbol.tolist()
    data = {}
    coverage = []
    timestamp_changes = []
    for symbol in universe:
        source = HERE / "cache" / (symbol + ".pkl")
        item = pd.read_pickle(source)
        # Historical exchange maintenance closed some klines early, with a
        # distinct millisecond timestamp per symbol. Their open times remain
        # on-grid. Evaluate the known OHLCV at the nominal scheduled close,
        # preventing hundreds of phantom cross-pair timeline rows. No prices
        # are changed, interpolated, or filled from later candles.
        for name, cadence in [("4hourly", pd.Timedelta(hours=4)), ("1daily", pd.Timedelta(days=1))]:
            raw_frame = item["dict_of_frames"][name]
            if len(raw_frame):
                nominal = raw_frame.open_time + cadence - pd.Timedelta(milliseconds=1)
                assert (raw_frame.close_time <= nominal).all(), "Cannot move a later observation earlier"
                assert (raw_frame.open_time.dt.minute == 0).all() and (raw_frame.open_time.dt.second == 0).all()
                changed = raw_frame.close_time != nominal
                if changed.any():
                    timestamp_changes.append(dict(pair=symbol,frame=name,changed_bars=int(changed.sum())))
                raw_frame["close_time"] = nominal
        frame = item["dict_of_frames"]["4hourly"]
        daily = item["dict_of_frames"]["1daily"]
        coverage.append(dict(pair=symbol, bars_4h=len(frame), bars_1d=len(daily),
                             first_bar=str(frame.close_time.min()), last_bar=str(frame.close_time.max())))
        if len(frame) >= 201 and len(daily) >= 2:
            data[symbol] = item
    pd.DataFrame(coverage).to_csv(HERE / "data_coverage.csv", index=False)
    pd.DataFrame(timestamp_changes).to_csv(HERE / "timestamp_alignment.csv", index=False)
    print("Pairs with usable pre-2022 history", len(data), flush=True)
    ns = dict(np=np, pd=pd, talib=talib, reload=reload)
    for module in [features, gating, helpers]:
        ns.update({k: v for k, v in vars(module).items() if not k.startswith("__")})
    # Execute the repository's exact preparation function and its constants;
    # do not import missing telegram/live-only utility modules.
    tree = ast.parse((ROOT / "data_management.py").read_text())
    allowed = {"_EXECUTION_FRAME", "_DAILY_FRAME", "_TREND_FRAMES", "_TREND_KWARGS",
               "_TREND_COLUMNS", "_TREND_MERGED_COLUMNS", "_COLUMNS_FOR_MERGE",
               "_COLS_FOR_RETEST", "_COLS_FOR_BREAK"}
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in allowed for t in node.targets):
            nodes.append(node)
        elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name) and node.target.id in allowed:
            nodes.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name == "_reprepared_dataset":
            nodes.append(node)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(ROOT / "data_management.py"), "exec"), ns)
    original_levels = ns["populate_historical_levels_fast"]
    count = [0]
    def progress_levels(*args, **kwargs):
        result = original_levels(*args, **kwargs)
        count[0] += 1
        if count[0] % 5 == 0:
            print("Prepared levels", count[0], "/", len(data), flush=True)
        return result
    ns["populate_historical_levels_fast"] = progress_levels
    checkpoint = HERE / "prepared.pkl"
    if checkpoint.exists():
        data = pd.read_pickle(checkpoint)
    else:
        with threadpool_limits(limits=1):
            data = ns["_reprepared_dataset"](data)
        pd.to_pickle(data, checkpoint)
    notebook = json.loads((ROOT / "tester.ipynb").read_text())
    ns.update(dict(dict_of_pairs_ori=data, execution_frame="4hourly"))
    # The multi-strategy book that matches the notebook's saved portfolio run.
    # Cells 22/23 are independent single-strategy experiments and are excluded.
    for cell in [17, 18, 20, 21]:
        print("Notebook cell", cell, flush=True)
        exec(compile("".join(notebook["cells"][cell]["source"]), "tester.ipynb:cell"+str(cell), "exec"), ns)
    data = ns["dict_of_pairs"]
    specs = build_strategy_specs([s["entry_col"] for s in ns["list_of_strategies"]],
                                 [s["exit_col"] for s in ns["list_of_strategies"]])
    # Indicators already computed with warm-up; retain only modelling/evaluation bars.
    for item in data.values():
        f = item["dict_of_frames"]["4hourly"]
        item["dict_of_frames"]["4hourly"] = f[f.close_time >= "2019-01-01"].reset_index(drop=True)
    payload = compile_backtest_payload(data, "4hourly", specs, gate_col="structure_ok",
        time_col="close_time", price_columns=("open", "high", "low", "close"),
        aux_columns=strategies.STRATEGY_AUX_COLUMNS, context_columns=["regime"],
        exposure_multiplier_col="__none__", enforce_bar_cadence=False)
    nominal_times = pd.DatetimeIndex(payload["timeline"])
    assert np.all(nominal_times.minute == 59) and np.all(nominal_times.second == 59)
    assert np.all(nominal_times.hour % 4 == 3)
    assert np.all(np.diff(nominal_times.asi8) >= pd.Timedelta(hours=4).value)
    pd.to_pickle(payload, HERE / "payload.pkl")
    # Keep compact causal ML features. Exclude full-history swing/cluster outputs.
    rows = []
    for pair, item in data.items():
        f = item["dict_of_frames"]["4hourly"]
        c, o, h, l = (f[k].astype(float) for k in ["close", "open", "high", "low"])
        q = f.quote_asset_volume.astype(float)
        x = pd.DataFrame({"time": f.close_time, "pair": pair})
        for key in ["rsi", "rsi_slope", "volatility", "rolling_gain", "sqn_30", "sqn_90",
                    "btc_sqn_90_pct", "rs_ratio_mom60", "rs_sqn_90", "rank_score"]:
            if key in f:
                x[key] = pd.to_numeric(f[key], errors="coerce")
        for window in [1, 3, 6, 12, 42]:
            x["return_"+str(window)] = c.pct_change(window, fill_method=None)
        rng = (h-l).replace(0, np.nan)
        x["range_pct"] = (h-l)/c
        x["body_pct"] = (c-o)/o
        x["close_location"] = (c-l)/rng
        x["upper_wick"] = (h-np.maximum(c,o))/rng
        x["lower_wick"] = (np.minimum(c,o)-l)/rng
        for window in [7, 25, 50, 100, 200]:
            ema = f["ema"+str(window)]
            x["ema_distance_"+str(window)] = (c-ema)/c
            x["ema_slope_"+str(window)] = ema.pct_change(6, fill_method=None)
        for window in [7, 20, 42]:
            x["qvolume_ratio_"+str(window)] = q/q.rolling(window).mean().replace(0,np.nan)
        x["boll_width"] = (f.boll_upper-f.boll_lower)/c
        rows.append(x)
    pd.concat(rows, ignore_index=True).to_pickle(HERE / "features.pkl")
    metadata = dict(universe=len(universe), historical_pairs=len(data), strategies=list(payload["strategies"]),
        gate="structure_ok", start="2020-10-01", end="2021-12-31", training_start="2019-01-01",
        warmup_start="2017-08-17 (or pair listing)", data_last_bar=str(payload["timeline"].max()),
        source_commit=os.popen("git -C '"+str(ROOT)+"' rev-parse HEAD").read().strip())
    (HERE / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
