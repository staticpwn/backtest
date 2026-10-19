"""Fetch only 1d/4h spot candles through 2021 from Binance public data API."""
import concurrent.futures
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "cache"
OUT.mkdir(exist_ok=True)
START = int(pd.Timestamp("2017-08-17", tz="UTC").timestamp() * 1000)
END = int(pd.Timestamp("2022-01-01", tz="UTC").timestamp() * 1000) - 1
COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_asset_volume", "number_of_trades", "market_buy",
        "Taker_buy_quote_asset_volume", "ignore"]


def fetch(symbol):
    dest = OUT / (symbol + ".pkl")
    if dest.exists():
        return symbol, "cached"
    frames = {}
    for interval, name in [("4h", "4hourly"), ("1d", "1daily")]:
        rows = []
        start = START
        while start <= END:
            query = urllib.parse.urlencode(dict(symbol=symbol, interval=interval,
                startTime=start, endTime=END, limit=1000))
            for attempt in range(5):
                try:
                    with urllib.request.urlopen("https://data-api.binance.vision/api/v3/klines?" + query,
                                                timeout=40) as response:
                        batch = json.load(response)
                    break
                except urllib.error.HTTPError as exc:
                    if exc.code == 400:
                        detail = exc.read().decode()
                        if "Invalid symbol" in detail:
                            batch = []
                            break
                    if attempt == 4:
                        raise
                    time.sleep(2 ** attempt)
                except Exception:
                    if attempt == 4:
                        raise
                    time.sleep(2 ** attempt)
            if not batch:
                break
            rows.extend(batch)
            start = int(batch[-1][0]) + 1
            if len(batch) < 1000:
                break
        frame = pd.DataFrame(rows, columns=COLS)
        for col in COLS:
            if col in ("open_time", "close_time"):
                frame[col] = pd.to_datetime(frame[col], unit="ms")
            else:
                frame[col] = pd.to_numeric(frame[col])
        frames[name] = frame
    pd.to_pickle({"dict_of_frames": frames}, dest)
    return symbol, {k: len(v) for k, v in frames.items()}


if __name__ == "__main__":
    universe = pd.read_csv(ROOT / "binance_usdt_tokens.csv").option_symbol.tolist()
    failures = []
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        jobs = {pool.submit(fetch, s): s for s in universe}
        for i, future in enumerate(concurrent.futures.as_completed(jobs), 1):
            try:
                print(i, len(universe), future.result(), round(time.time()-t0, 1), flush=True)
            except Exception as exc:
                failures.append({"symbol": jobs[future], "error": repr(exc)})
                print("FAILED", failures[-1], flush=True)
    (OUT / "download_failures.json").write_text(json.dumps(failures, indent=2))
    if failures:
        raise SystemExit("Download incomplete")
