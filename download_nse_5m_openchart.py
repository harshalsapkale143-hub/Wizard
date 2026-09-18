from datetime import datetime
from pathlib import Path
import pandas as pd
from openchart import NSEData

START = datetime(2026, 7, 10)
END = datetime(2026, 9, 8)
CHUNK_DAYS = 7
OUT = Path("data/intraday5m")
OUT.mkdir(exist_ok=True)

symbols = [x.strip() for x in Path("symbols.txt").read_text().splitlines() if x.strip() and not x.startswith("#")]
nse = NSEData()

for symbol in symbols:
    clean = symbol.replace("-EQ", "").replace("/", "_").upper()
    target = OUT / f"{clean}.csv"
    if target.exists() and target.stat().st_size > 0:
        try:
            cached = pd.read_csv(target, usecols=["timestamp"])
            if len(cached) > 1000:
                print("  CACHE HIT", len(cached), "bars")
                continue
        except Exception:
            pass
    print("Downloading", symbol)
    try:
        df = pd.concat([x.reset_index() for x in [nse.historical(symbol, "EQ", s, min(s + pd.Timedelta(days=CHUNK_DAYS), END), "5m") for s in pd.date_range(START, END, freq=f"{CHUNK_DAYS}D")] if x is not None and len(x)], ignore_index=True)
        if df is None or len(df) == 0:
            print("  NO DATA")
            continue
        rename = {}
        for c in df.columns:
            lc = str(c).lower()
            if lc in {"timestamp", "datetime", "date"}: rename[c] = "timestamp"
            elif lc in {"open", "high", "low", "close", "volume"}: rename[c] = lc
        df = df.rename(columns=rename)
        required = {"timestamp", "open", "high", "low", "close", "volume"}
        if not required.issubset(df.columns):
            print("  BAD FORMAT", df.columns.tolist())
            continue
        df = df[list(required)]
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
        df.to_csv(OUT / f"{clean}.csv", index=False)
        print("  saved", len(df), "bars")
    except Exception as e:
        print("  ERROR", e)
