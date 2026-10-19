"""Read-only prefix audit of the repository's existing swing feature pipeline."""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
import features


def main():
    warnings.simplefilter("ignore")
    rows = []
    timestamps = 0
    for symbol in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]:
        raw = pd.read_pickle(HERE / "cache" / (symbol + ".pkl"))["dict_of_frames"]["4hourly"]
        raw["close_time"] = raw.open_time + pd.Timedelta(hours=4) - pd.Timedelta(milliseconds=1)
        f = features.prepare_df(raw)
        full = features.add_hh_ll_columns(f.copy())
        for end in np.linspace(max(250, len(f)//3), len(f)-1, 25).astype(int):
            timestamps += 1
            prefix = features.add_hh_ll_columns(f.iloc[:end+1].copy())
            for col in ["HH", "LL", "prev_low", "prev_high_rsi", "prev_low_rsi", "higher_rsi", "lower_rsi"]:
                a, b = full.iloc[end-5][col], prefix.iloc[end-5][col]
                if not ((pd.isna(a) and pd.isna(b)) or a == b):
                    rows.append(dict(pair=symbol, decision_time=str(f.iloc[end].close_time),
                                     column=col, full=str(a), prefix=str(b)))
    result = dict(checks=timestamps, confirmation_lag_bars=5, differences=rows)
    (HERE / "prefix_audit.json").write_text(json.dumps(result, indent=2))
    print("Prefix differences", len(rows))


if __name__ == "__main__":
    main()
