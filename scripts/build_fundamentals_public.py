from __future__ import annotations
import os, time, requests, pandas as pd
from pathlib import Path

# Public-provider proof of concept. Uses BharatStock only when API key is supplied;
# otherwise uses its public endpoint pattern and records provider errors.
BASE="https://bharatstockapi.com/v1/stocks/{symbol}"
OUT=Path("fundamentals_public")
MAX_SYMBOLS=int(os.environ.get("FUNDAMENTALS_MAX_SYMBOLS","10"))

def main():
    meta=pd.read_csv("metadata/universe_symbols.csv")
    scol=next(c for c in meta.columns if c.lower() in {"symbol","ticker","symbols"})
    key=os.environ.get("BHARATSTOCK_API_KEY","").strip()
    headers={"Accept":"application/json"}
    if key: headers["X-API-Key"]=key
    OUT.mkdir(parents=True,exist_ok=True)
    ok=0
    for _,row in meta.head(MAX_SYMBOLS).iterrows():
        sym=str(row[scol]).replace(".NS","").strip()
        try:
            r=requests.get(BASE.format(symbol=sym),headers=headers,timeout=15)
            if r.ok:
                data=r.json()
                (OUT/f"{sym}.json").write_text(pd.io.json.dumps(data) if hasattr(pd.io.json,"dumps") else __import__("json").dumps(data))
                ok+=1
            else:
                print(f"{sym}: provider HTTP {r.status_code}")
        except Exception as e:
            print(f"{sym}: {type(e).__name__}: {e}")
        time.sleep(.2)
    print(f"Public-provider files: {ok}/{min(len(meta),MAX_SYMBOLS)}")
    if ok==0:
        raise SystemExit("Public provider returned no usable data; configure BHARATSTOCK_API_KEY or choose another permitted provider.")

if __name__=="__main__": main()
