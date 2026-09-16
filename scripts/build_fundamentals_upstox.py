from __future__ import annotations

import os, time
from pathlib import Path
import requests
import pandas as pd

API = "https://api.upstox.com/v2/fundamentals/{isin}/income-statement"
ROOT = Path("metadata")
OUT = Path("fundamentals_upstox")
MAX_SYMBOLS = int(os.environ.get("FUNDAMENTALS_MAX_SYMBOLS", "25"))

def main():
    token = os.environ.get("UPSTOX_ACCESS_TOKEN")
    if not token:
        raise SystemExit("UPSTOX_ACCESS_TOKEN is not configured")
    meta = pd.read_csv(ROOT / "universe_symbols.csv")
    isin_col = next((c for c in meta.columns if c.lower()=="isin"), None)
    sym_col = next(c for c in meta.columns if c.lower() in {"symbol","ticker","symbols"})
    if not isin_col:
        raise SystemExit("metadata/universe_symbols.csv requires an ISIN column for Upstox")
    OUT.mkdir(parents=True, exist_ok=True)
    headers={"Accept":"application/json","Authorization":f"Bearer {token}"}
    done=0
    for _, row in meta.dropna(subset=[isin_col]).head(MAX_SYMBOLS).iterrows():
        symbol=str(row[sym_col]).replace(".NS","").strip()
        isin=str(row[isin_col]).strip()
        for typ in ("consolidated","standalone"):
            try:
                r=requests.get(API.format(isin=isin),params={"type":typ,"time_period":"quarterly","fs":"true"},headers=headers,timeout=20)
                if r.status_code != 200:
                    continue
                data=r.json().get("data",{})
                cats={x.get("category"):x.get("history",[]) for x in data.get("income_statement",[])}
                eps=[]
                for item in data.get("full_statement",[]):
                    if str(item.get("particular","")).lower() in {"eps - basic","eps - diluted"}:
                        eps=item.get("history",[])
                        if eps: break
                rows=[]
                periods=set()
                for key, arr in cats.items():
                    for x in arr:
                        periods.add(x.get("period"))
                for period in sorted(periods):
                    rec={"symbol":symbol,"isin":isin,"statement_type":typ,"period":period}
                    for key,arr in cats.items():
                        rec[key]=next((x.get("value") for x in arr if x.get("period")==period),None)
                    rec["eps"]=next((x.get("value") for x in eps if x.get("period")==period),None)
                    rows.append(rec)
                if rows:
                    pd.DataFrame(rows).to_csv(OUT/f"{symbol}_{typ}.csv",index=False)
                    done+=1
                    break
            except Exception as e:
                print(symbol, typ, type(e).__name__, e)
        time.sleep(0.1)
    print(f"Upstox fundamentals files: {done}")

if __name__=="__main__":
    main()
