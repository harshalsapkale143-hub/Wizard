"""Build daily NSE/NIFTY-50 OHLCV data without broker credentials.

Uses Yahoo Finance's public market-data endpoint through yfinance. This is a
free data source for research/backtesting; it is not an official NSE feed.
The workflow uses a retryable batch download and writes the exact CSV schema
required by daily_sepa_backtest.py.
"""
from __future__ import annotations
from pathlib import Path
import time
import pandas as pd
import yfinance as yf

START = "2024-01-01"
END = "2026-04-01"
OUT = Path("data")


def symbols():
    return [x.strip().upper() for x in Path("config/nifty50_symbols.csv").read_text().splitlines()[1:] if x.strip()]


def download(ticker: str) -> pd.DataFrame:
    for attempt in range(5):
        try:
            d = yf.download(ticker, start=START, end=END, interval="1d", auto_adjust=False, progress=False, threads=False)
            if d is not None and not d.empty:
                if isinstance(d.columns, pd.MultiIndex):
                    d.columns = d.columns.get_level_values(0)
                d = d.rename(columns={"Date":"timestamp"}).reset_index()
                if "timestamp" not in d.columns:
                    d = d.rename(columns={d.columns[0]:"timestamp"})
                d["timestamp"] = pd.to_datetime(d["timestamp"], errors="coerce").dt.tz_localize(None)
                d = d.rename(columns={"Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"})
                cols=["timestamp","open","high","low","close","volume"]
                if set(cols).issubset(d.columns):
                    return d[cols].dropna(subset=["timestamp","open","high","low","close"]).sort_values("timestamp")
        except Exception as e:
            print(f"{ticker}: attempt {attempt+1}/5 failed: {e}")
        time.sleep(3 * (attempt + 1))
    return pd.DataFrame()


def main():
    OUT.mkdir(exist_ok=True)
    syms = symbols()
    # NIFTY index is the market-regime series used by the backtest.
    jobs = [("^NSEI", "NIFTY50")] + [(s + ".NS", s) for s in syms]
    good = 0
    for ticker, name in jobs:
        print(f"Downloading {ticker}")
        d = download(ticker)
        if d.empty:
            print(f"NO DATA: {ticker}")
            continue
        d.to_csv(OUT / f"{name}.csv", index=False)
        print(f"saved {len(d):,} rows -> data/{name}.csv")
        good += 1
        time.sleep(0.5)
    if good < 2 or not (OUT / "NIFTY50.csv").exists():
        raise RuntimeError("Insufficient market data downloaded. No backtest should run.")
    print(f"Completed {good}/{len(jobs)} downloads")


if __name__ == "__main__":
    main()
