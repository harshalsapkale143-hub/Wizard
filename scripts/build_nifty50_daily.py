"""Download official NSE daily CM UDiFF bhavcopy files and build NIFTY50.csv.

This script intentionally uses NSE's public archives and does not require a
broker/API key. It downloads only the NIFTY 50 constituent files listed in
config/nifty50_symbols.csv plus the NIFTY index series. Run in GitHub Actions.

NSE archive layouts can change. The script tries the current UDiFF zip URL
pattern first and fails clearly if NSE changes the archive naming/layout.
"""
from __future__ import annotations
import io, zipfile
from pathlib import Path
from datetime import date, timedelta
import requests
import pandas as pd

START = date(2024, 1, 1)
END = date(2026, 3, 31)
OUT = Path("data")
RAW = Path(".cache/nse")
BASE = "https://nsearchives.nseindia.com/content/cm"
HEADERS = {"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/131 Safari/537.36","Accept":"*/*","Referer":"https://www.nseindia.com/"}


def daterange(a,b):
    d=a
    while d<=b:
        yield d
        d += timedelta(days=1)


def get(url):
    r=requests.get(url,headers=HEADERS,timeout=60)
    if r.status_code != 200 or len(r.content)<100:
        return None
    return r.content


def find_csv(blob):
    try:
        z=zipfile.ZipFile(io.BytesIO(blob))
        names=[n for n in z.namelist() if n.lower().endswith('.csv')]
        if not names: return None
        return z.read(names[0])
    except zipfile.BadZipFile:
        return None


def download_day(d):
    # Current CM UDiFF common bhavcopy naming convention.
    candidates=[
        f"BhavCopy_NSE_CM_0_0_0_{d.strftime('%Y%m%d')}_F_0000.csv.zip",
        f"BhavCopy_NSE_CM_0_0_0_{d.strftime('%Y%m%d')}_F_0000.csv",
    ]
    for name in candidates:
        blob=get(f"{BASE}/{name}")
        if blob:
            raw=find_csv(blob)
            if raw is not None:
                return pd.read_csv(io.BytesIO(raw))
    return None


def main():
    OUT.mkdir(exist_ok=True); RAW.mkdir(parents=True,exist_ok=True)
    symbols=pd.read_csv("config/nifty50_symbols.csv")["symbol"].str.upper().tolist()
    all_rows=[]
    for d in daterange(START,END):
        if d.weekday()>=5: continue
        cache=RAW/f"{d:%Y%m%d}.parquet"
        try:
            df=pd.read_parquet(cache)
        except Exception:
            df=download_day(d)
            if df is None:
                continue
            try: df.to_parquet(cache,index=False)
            except Exception: pass
        # UDiFF columns are typically TckrSymb, TradDt, OpnPric, HghPric,
        # LwPric, ClsPric, TtlTradgVol. Normalize by aliases.
        aliases={
            "symbol":["TckrSymb","SYMBOL","Symbol"],
            "date":["TradDt","TIMESTAMP","Date","DATE"],
            "open":["OpnPric","OPEN","Open"],
            "high":["HghPric","HIGH","High"],
            "low":["LwPric","LOW","Low"],
            "close":["ClsPric","CLOSE","Close"],
            "volume":["TtlTradgVol","TOTTRDQTY","Volume","volume"],
        }
        rename={}
        for target,names in aliases.items():
            for n in names:
                if n in df.columns: rename[n]=target; break
        x=df.rename(columns=rename)
        needed={"symbol","date","open","high","low","close","volume"}
        if not needed.issubset(x.columns): continue
        x["symbol"]=x.symbol.astype(str).str.upper()
        x=x[x.symbol.isin(symbols)]
        x["timestamp"]=pd.to_datetime(x.date,errors="coerce")
        for c in ["open","high","low","close","volume"]: x[c]=pd.to_numeric(x[c],errors="coerce")
        x=x.dropna(subset=["timestamp","open","high","low","close"])
        all_rows.append(x[["timestamp","symbol","open","high","low","close","volume"]])
    if not all_rows: raise RuntimeError("No NSE bhavcopy data downloaded; archive layout may have changed.")
    stocks=pd.concat(all_rows,ignore_index=True).drop_duplicates(["timestamp","symbol"])
    for sym,g in stocks.groupby("symbol"):
        g=g.drop(columns="symbol").sort_values("timestamp")
        g.to_csv(OUT/f"{sym}.csv",index=False)
    print(f"Wrote {stocks.symbol.nunique()} stocks and {len(stocks):,} rows")

if __name__=="__main__": main()
