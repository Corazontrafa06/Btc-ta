"""Ambil histori BTCUSDT dari Binance (public) untuk backtest.py. Hasil: backtest_data/*.parquet|csv"""
import sys, time, json, urllib.request, urllib.parse
import pandas as pd

BASE = "https://data-api.binance.vision/api/v3/klines"
FAPI = "https://fapi.binance.com/fapi/v1/fundingRate"
COLS = ["open_time","open","high","low","close","volume","close_time","quote_volume","trades","taker_buy_base","taker_buy_quote","ignore"]

def get(url, params):
    q = url + "?" + urllib.parse.urlencode(params)
    for i in range(5):
        try:
            with urllib.request.urlopen(urllib.request.Request(q, headers={"User-Agent": "btc-ta-backtest"}), timeout=20) as r:
                return json.loads(r.read())
        except Exception as e:
            print("retry", e, file=sys.stderr); time.sleep(2 + i)
    raise RuntimeError(q)

def klines(interval, start):
    start_ms = int(pd.Timestamp(start, tz="UTC").value // 1_000_000)
    rows = []
    while True:
        batch = get(BASE, {"symbol": "BTCUSDT", "interval": interval, "startTime": start_ms, "limit": 1000})
        if not batch: break
        rows += batch
        start_ms = batch[-1][0] + 1
        if len(batch) < 1000: break
    df = pd.DataFrame(rows, columns=COLS)
    for c in ("open","high","low","close","volume","taker_buy_base"):
        df[c] = pd.to_numeric(df[c])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    return df[["open_time","open","high","low","close","volume","taker_buy_base"]].drop_duplicates("open_time").reset_index(drop=True)

def funding(start):
    start_ms = int(pd.Timestamp(start, tz="UTC").value // 1_000_000)
    rows = []
    while True:
        batch = get(FAPI, {"symbol": "BTCUSDT", "startTime": start_ms, "limit": 1000})
        if not batch: break
        rows += batch
        start_ms = batch[-1]["fundingTime"] + 1
        if len(batch) < 1000: break
    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
    df["rate"] = pd.to_numeric(df["fundingRate"])
    return df[["time","rate"]]

if __name__ == "__main__":
    plan = {"15m": "2025-06-01", "1h": "2025-04-01", "4h": "2024-01-01", "1d": "2022-06-01", "1w": "2018-01-01"}
    for iv, st in plan.items():
        d = klines(iv, st); d.to_pickle(f"backtest_data/{iv}.pkl"); print(iv, len(d), d.open_time.iloc[0], d.open_time.iloc[-1])
    f = funding("2025-04-01"); f.to_pickle("backtest_data/funding.pkl"); print("funding", len(f))
