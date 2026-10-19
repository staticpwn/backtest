# Predictive initial stop-loss research

Compare the existing 30-strategy notebook book against fixed 0.5x/1.5x initial
stop distances and monthly walk-forward ML selection. The repository's portfolio
engine is called directly; its gating, priority, sizing, TP, stop ratchet, and
other exit rules are preserved. This is an isolated research harness, not a live
configuration change.

The original engine records entry fees in trade PnL but fails to deduct them
from portfolio cash. Both the original engine reference and a fee-corrected
comparison are produced. The fee-corrected run debits each opening fee to cash
and caps the cash spend so the fee cannot create borrowing. No entry/exit
signals or TP/stop trigger rules change. Label returns already include both fees.

## Run

From the repository root, with Python 3.10-compatible dependencies:

```bash
python -m pip install -r research/predictive_stop/requirements-py310.txt
python research/predictive_stop/run_all.py
python research/predictive_stop/audit.py
python research/predictive_stop/report.py
```

The supplied fitted model binaries were trained with the versions recorded in
`runtime_versions.json` in the results bundle. Retrain rather than relying on
cross-version joblib compatibility. Python 3.10 syntax was checked; the study
itself ran on Python 3.12.

Only official Binance 4h/1d data through 31 December 2021 is fetched. Earlier
history supports training, warm-up, and cumulative price levels. The public
`data-api.binance.vision` endpoint requires no API key. Existing per-symbol raw
caches are reused.

Maintenance-era early candle close timestamps are aligned to their scheduled
close (`open_time + interval - 1 ms`) before preparing indicators. OHLCV is
unchanged. This prevents a cross-symbol union of millisecond-different early
closes from creating phantom portfolio evaluations and missing valuations.

## Model

Two gradient-boosted regressors estimate incremental net trade returns of
tightening/widening relative to the original stop. Predictions are monthly,
using at most 24 trailing months. Every training label must have resolved under
all three policies before the prediction month. The decision margin of 0.25
percentage points is fixed before evaluation. Model parameters are fixed.

The Numba label kernel is checked against the original exit evaluator on 360
sampled policy paths. The complete book is then rerun for each arm so changed
slots, capital, entries, liquidity costs, and compounding are included.

## Scope and limits

- Uses notebook cells 17, 18, 20, 21; ignores the independent single-strategy
  experiments in cells 22/23. Compile gate is `structure_ok`, as in cell 25.
- Uses the checked-in engine configuration with five slots and no exposure
  multiplier. The newer gain guard and entry-ML gate are not in this snapshot.
- `trade_frame_extra.csv` is an older closed-trade export, not sufficient for
  counterfactual stop labels or a fresh shared-capital replay.
- The existing swing features have a documented prefix-consistency failure,
  including after the momentum strategy's five-bar lag. This is inherited by
  both baseline and alternatives; the experiment does not certify causal live
  performance or repair the existing feature pipeline.
- The current user-supplied universe has survivor-selection bias. Symbols
  listed later are not given synthetic historical prices.
- Other exits, especially the percentage ratchet, may erase the effect of
  changing only the initial stop. They are deliberately included in the labels.

Large raw caches, intermediate pickles, fitted models, and trade/curve outputs
are excluded from Git. The separate results bundle contains the report,
fitted models, downloaded candle caches, output CSVs, and verification records.
