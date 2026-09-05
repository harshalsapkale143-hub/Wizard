from datetime import datetime
from pathlib import Path
import pandas as pd
from openchart import NSEData

START = datetime(2024, 1, 1)
END = datetime(2026, 3, 31)
OUT = Path("data")
OUT.mkdir(exist_ok=True)

symbols = [x.strip() for x in Path("symbols.txt").read_text().splitlines() if x.strip() and not x.startswith("#")]
nse = NSEData()

for symbol in symbols:
    clean = symbol.replace("-EQ", "").replace("/", "_").upper()
    print("Downloading", symbol)
    try:
        df = nse.historical(symbol, "EQ", START, END, "5m")
        if df is None or len(df) == 0:
            print("  NO DATA")
            continue
        df = df.reset_index()
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
