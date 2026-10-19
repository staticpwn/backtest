"""Build a findings report and preserve fitted models/results as a single bundle."""
import hashlib
import importlib.metadata
import json
import sys
import zipfile
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BRANCH = "research/predictive-stop-2020-2021-20261019"


def table(frame):
    lines = ["| " + " | ".join(frame.columns) + " |", "| " + " | ".join(["---"]*len(frame.columns)) + " |"]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in frame.itertuples(index=False,name=None))
    return "\n".join(lines)


def main():
    s = pd.read_csv(HERE/"summary.csv")
    reasons = pd.read_csv(HERE/"close_reasons.csv")
    meta = json.loads((HERE/"metadata.json").read_text())
    parity = json.loads((HERE/"parity.json").read_text())
    audit = json.loads((HERE/"prefix_audit.json").read_text())
    diag = json.loads((HERE/"label_diagnostics.json").read_text())
    wf = pd.read_csv(HERE/"walk_forward.csv")
    coverage = pd.read_csv(HERE/"data_coverage.csv")
    original = pd.read_csv(HERE/"original_engine_summary.csv")
    fee_comparison = original[["arm","net_multiple"]].rename(columns={"net_multiple":"Original engine multiple"}).merge(
        s[["arm","net_multiple"]].rename(columns={"net_multiple":"Fee-corrected net multiple"}),on="arm")
    for col in fee_comparison.columns[1:]:
        fee_comparison[col]=fee_comparison[col].map(lambda x:f"{x:.6f}x")
    ml = s[s.arm.str.startswith("ml_")].sort_values("net_multiple",ascending=False).iloc[0]
    baseline = s.iloc[0]
    fixed = s[s.arm.str.startswith("fixed_")].sort_values("net_multiple",ascending=False).iloc[0]
    formatted = pd.DataFrame({"Arm":s.arm,"Net multiple":s.net_multiple.map(lambda x:f"{x:.6f}x"),
        "Trades":s.trades.astype(str),"Expectancy":s.expectancy_pct.map(lambda x:f"{x:.4f}%"),
        "Delta expectancy (pp)":s.delta_expectancy_pct_pp.map(lambda x:f"{x:+.4f}"),
        "Win rate":s.win_rate_pct.map(lambda x:f"{x:.3f}%"),
        "Delta win rate (pp)":s.delta_win_rate_pct_pp.map(lambda x:f"{x:+.3f}"),
        "Max drawdown":s.max_drawdown_pct.map(lambda x:f"{x:.3f}%")})
    r = reasons.copy()
    r["Count (share)"] = r.apply(lambda row:f"{row['count']} ({row['pct']:.2f}%)",axis=1)
    pivot=r.pivot(index="reason",columns="arm",values="Count (share)").reindex(columns=s.arm).fillna("0 (0.00%)").reset_index()
    decisions=s[["arm","tight_candidate_pct","wide_candidate_pct","tight_executed_pct","wide_executed_pct"]].copy()
    for col in decisions.columns[1:]:
        decisions[col]=decisions[col].map(lambda x:f"{x:.3f}%")
    inconsistent=len(set((x["pair"],x["decision_time"]) for x in audit["differences"]))
    outcome = (f"The strongest ML arm was {ml.arm}: {ml.net_multiple:.6f}x versus "
        f"{baseline.net_multiple:.6f}x for baseline ({ml.relative_multiple_change_pct:+.3f}% relative final equity). "
        f"Its expectancy changed by {ml.delta_expectancy_pct_pp:+.4f} percentage points and win rate "
        f"by {ml.delta_win_rate_pct_pp:+.3f} percentage points. "
        f"The best fixed alternative was {fixed.arm} at {fixed.net_multiple:.6f}x.")
    packages = ["numpy","pandas","scikit-learn","numba","TA-Lib","joblib","threadpoolctl"]
    runtime=dict(python=sys.version,packages={p:importlib.metadata.version(p) for p in packages})
    (HERE/"runtime_versions.json").write_text(json.dumps(runtime,indent=2))
    text=f"""# Predictive initial stop-loss study: October 2020–December 2021

## Findings

{outcome}

These are paired experiments in the checked-in repository book, with entry-fee cash accounting corrected equally in every arm. An inherited swing-feature prefix inconsistency prevents treating them as a clean causal live-performance estimate. No evaluation-period thresholds or model parameters were selected from these results.

## Portfolio results

{table(formatted)}

Net multiple = the fee-corrected engine's last marked portfolio equity divided by initial cash of 5,020 USDT. Both entry/exit fees and liquidity-based slippage are included. Positions are not forcibly liquidated at the segment boundary. Expectancy is the arithmetic mean net percentage return per closed trade; win rate uses net PnL > 0. Changes are percentage points. A higher win rate does not by itself imply a better portfolio result.

## Entry-fee accounting correction

The checked-in _open_position_from_candidate computes and records an entry fee but returns cash minus entry_value without debiting the fee. Exit cash debits only the exit fee. Thus trade PnL includes both fees while the original cash/equity multiple does not. The primary table above uses a research-only wrapper to debit opening fees and constrain entry spending to available cash including its fee. Gate, priority, stop/TP trigger, ratchet, signal, and time rules remain unchanged. Expected cash debits and nonnegative cash are checked on every fee-corrected entry. Labels already include both fees, so models do not need a different target for this correction.

{table(fee_comparison)}

The original-engine figures are reference reproductions, not fully fee-net multiples. The source engine file itself is unchanged; the correction is isolated in the research harness. fee_accounting_audit.json records the verification totals.

## Close-reason distribution

{table(pivot)}

The newer gain guard and accelerated stop-loss mechanic are absent from the checked-in engine. They were not recreated from memory or substituted with a different rule. The historical trade_frame_extra.csv contains an older export with accelerated_stop_loss exits and cannot serve as the baseline for this code version.

## Stop choices

{table(decisions)}

Shares labelled executed are over closed trades; open positions at the boundary are excluded. Stop factors multiply the original strategy's entry stop distance: 0.5 tightens it, 1.0 preserves it, 1.5 widens it. Alternative prices are floored at 0.001 times entry close to remain positive. No other stop or exit parameter changes.

## Model and labelling

- Two HistGradientBoostingRegressor models predict R(tight)-R(current) and R(wide)-R(current). Squared loss, 150 iterations, learning rate 0.05, 15 leaves, minimum leaf size 100, L2=10, seed 42, no early stopping or parameter search.
- Monthly walk-forward fits use at most 24 preceding months, with available entry history starting 1 January 2019. First evaluation month uses less than 24 months because the modelling history starts in 2019.
- Every training entry must have exited under all three stop policies strictly before its prediction month. No fixed three-bar purge is used in place of actual label maturity.
- {len(wf)} monthly fits per model; training samples range from {wf.training_entries.min():,} to {wf.training_entries.max():,} per month. Fitted models are saved separately by month/policy.
- Alternative stop selection requires predicted incremental return above 0.25 percentage points, fixed before observing evaluation results. The combined arm chooses the larger positive predicted improvement; otherwise it retains the current stop.
- Entry features are backward-looking candle anatomy, returns, EMA distances/slopes, volatility, participation, relative strength, stop/TP distances, strategy ratchet setting, and strategy one-hot indicators. Full-history swing/cluster outputs are not added as new ML predictors, but the inherited book can still affect entry selection and strategy stop inputs.
- Labels simulate all eligible strategy candidates, including candidates the baseline book cannot take because slots are occupied. The older closed-trade CSV alone is insufficient for this task.
- Label returns use a common 1,000 USDT nominal entry, capped by the existing 0.4% liquidity limit. This isolates stop-policy trade returns from unknown future book equity. Full book runs use actual equity, cash, stop-risk sizing, slot limits, and resulting slippage.
- A Numba kernel generates labels. {parity['checked_paths']} sampled policy paths matched the unmodified repository exit evaluator in exit bar, reason, and net return (absolute tolerance 1e-12).
- The hash check in book_integrity.json confirms that candidate gates/signals, TP arrays, ranks, and all non-stop auxiliary arrays were unchanged across book arms.

## Preserved book rules

- Source commit: {meta['source_commit']}.
- Multi-strategy notebook cells 17, 18, 20, 21 are used (30 strategies). Cells 22/23 are independent single-strategy experiments and are excluded.
- Compile gate is structure_ok, matching notebook cell 25. Daily gating fields are prepared, but allowed_to_trade is not substituted for the notebook's selected gate. Exposure scaling is disabled as in its engine configuration.
- Five positions maximum; one position per pair; 20% equity position cap; existing priority/rank ordering and 5% stop-risk cap remain active. The cash cap additionally reserves the actual opening fee in the fee-corrected replay.
- Fee: 10 bps per side. Liquidity/slippage cap: 0.004; slippage multiplier: 1.
- Initial stop is frozen at entry. Existing percentage ratchet remains active and can override a widened initial stop.
- Stop-loss requires current and previous closes below the stop; test strategies additionally require hold > 12 bars. Stop execution remains at close with the repository's slippage model.
- Dynamic per-bar TP levels and high > TP detection/fill rules remain unchanged. Spread trailing, signal exits, minimum signal hold of 8 bars, and conditional global time rules at 12 bars remain unchanged.
- Existing same-bar re-entry and exit priority rules are inherited without alteration. This is not the separate winner-rotation proposal.

## Data and endpoint

- Universe: {meta['universe']} symbols from binance_usdt_tokens.csv; {int((coverage.bars_4h>0).sum())} have historical data before 2022; {meta['historical_pairs']} have enough data for book indicators. Later listings contribute no synthetic history.
- Official public source: https://data-api.binance.vision/api/v3/klines . The main api.binance.com endpoint returned HTTP 451 from this environment.
- Only 4h and 1d candles were downloaded, from Binance inception/listing through 31 December 2021. Earlier candles are required for training, indicator warm-up, and cumulative support/resistance levels. No post-2021 candles were fetched.
- Historical maintenance candles sometimes have symbol-specific early close timestamps. Before indicators, close_time is aligned to open_time + interval - 1 millisecond; OHLCV is preserved. This delays early records to their nominal scheduled close and prevents phantom cross-pair evaluations/missing asset valuations. timestamp_alignment.csv records the changed-row counts. An initial unaligned trial was discarded and every label/model/book arm was rebuilt on the aligned timeline.
- Evaluation: 1 October 2020 through 31 December 2021 inclusive.
- Verified last bar close: {meta['data_last_bar']} UTC (1 January 2022 03:59:59.999 in Dubai).

## Incremental stop outcome opportunity

On completed evaluation candidates, the tight policy changed {diag['tight']['changed_outcome_pct']:.3f}% of outcomes; the wide policy changed {diag['wide']['changed_outcome_pct']:.3f}%. Mean incremental returns were {diag['tight']['mean_delta_pp']:+.4f} and {diag['wide']['mean_delta_pp']:+.4f} percentage points respectively. The existing ratchet and other exits constrain how much changing only the initial stop can achieve. These diagnostics use later observed outcomes for analysis only; they are not model inputs or selection thresholds.

## Material limits

1. The existing add_hh_ll_columns pipeline uses full-history scipy peak detection. In a prefix audit of BTC, ETH, and SOL, {len(audit['differences'])} feature values differed at {inconsistent}/{audit['checks']} sampled decision timestamps even after the five-bar momentum confirmation lag. Both baseline and stop arms inherit this issue. The audit does not measure the exact PnL impact or certify other feature families.
2. This single, strongly trending historical segment does not establish behaviour in 2022–2024 or 2026; those segments were intentionally not run.
3. The fixed current universe is survivor-selected, as requested, rather than a point-in-time listing universe.
4. The model optimizes conditional trade-return deltas at a common label notional, not portfolio-wide opportunity cost directly. Actual slot/sizing/compounding effects are measured only by the full book runs.
5. Research code parses as Python 3.10-compatible. Execution used Python 3.12, with exact package versions in runtime_versions.json. Included joblib models are tied to that runtime; retrain in Python 3.10 with requirements-py310.txt rather than assuming cross-version pickle compatibility.

## Reproduction and files

Research source: https://github.com/staticpwn/backtest/tree/{BRANCH}/research/predictive_stop

Run python research/predictive_stop/run_all.py from the checked-out repository. The cache included in this bundle can be extracted into research/predictive_stop/cache to avoid downloading again. The bundle contains fitted models, raw candle caches, all arm trade ledgers and equity curves, decisions, full metrics, walk-forward maturity records, data coverage, integrity/parity checks, and the prefix audit. It excludes regenerable large intermediate payload and feature/candidate pickles. ML predictions are batch-computed; the repository book with the explicit cash-fee correction consumes stop arrays. Source scripts use vectorized pandas/NumPy features and Numba labels.
"""
    (HERE/"REPORT.md").write_text(text)
    out=ROOT.parent/"deliverables"
    out.mkdir(exist_ok=True)
    dest=out/"predictive_stop_2020_2021_results.zip"
    selected=[p for p in HERE.iterdir() if (p.suffix in [".csv",".json"] or p.name=="REPORT.md")
              and p.name not in {"book_notebook_cells.json","artifact_manifest.json"}]
    selected += list((HERE/"models").glob("*.joblib"))
    selected += list((HERE/"cache").glob("*.pkl"))
    manifest=[]
    for p in selected:
        manifest.append(dict(path=str(p.relative_to(HERE)),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size))
    (HERE/"artifact_manifest.json").write_text(json.dumps(manifest,indent=2))
    selected.append(HERE/"artifact_manifest.json")
    with zipfile.ZipFile(dest,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in selected:
            z.write(p,"predictive_stop/"+str(p.relative_to(HERE)))
    with zipfile.ZipFile(dest) as z:
        assert z.testzip() is None
    print("REPORT",HERE/"REPORT.md")
    print("BUNDLE",dest,dest.stat().st_size)
    print(outcome)


if __name__=="__main__":
    main()
