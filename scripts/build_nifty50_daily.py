"""Build daily OHLCV research data without broker credentials.

The metadata-first workflow supplies metadata/universe_symbols.csv containing
the union of historical top-500 NSE symbols.  The script downloads that
research universe; it falls back to the current NIFTY-50 list only when the
metadata universe is unavailable.
"""
from __future__ import annotations
from pathlib import Path
import time
import pandas as pd
import yfinance as yf

START = "2018-01-01"
END = "2026-09-07"
OUT = Path("data")

def symbols():
    p=Path('metadata/universe_symbols.csv')
    if p.exists():
        d=pd.read_csv(p)
        if 'symbol' not in d.columns: raise ValueError('metadata/universe_symbols.csv must contain symbol')
        out=d.symbol.astype(str).str.strip().str.upper().tolist()
        if len(out)<100: raise RuntimeError(f'Historical universe unexpectedly small: {len(out)} symbols')
        return sorted(set(out))
    return [x.strip().upper() for x in Path("config/nifty50_symbols.csv").read_text().splitlines()[1:] if x.strip()]

def download(ticker: str) -> pd.DataFrame:
    for attempt in range(4):
        try:
            d=yf.download(ticker,start=START,end=END,interval='1d',auto_adjust=False,progress=False,threads=False,timeout=30)
            if d is not None and not d.empty:
                if isinstance(d.columns,pd.MultiIndex): d.columns=d.columns.get_level_values(0)
                d=d.rename(columns={'Date':'timestamp'}).reset_index()
                if 'timestamp' not in d.columns: d=d.rename(columns={d.columns[0]:'timestamp'})
                d['timestamp']=pd.to_datetime(d['timestamp'],errors='coerce').dt.tz_localize(None)
                d=d.rename(columns={'Open':'open','High':'high','Low':'low','Close':'close','Volume':'volume'})
                cols=['timestamp','open','high','low','close','volume']
                if set(cols).issubset(d.columns): return d[cols].dropna(subset=['timestamp','open','high','low','close']).sort_values('timestamp')
        except Exception as e: print(f'{ticker}: attempt {attempt+1}/4 failed: {e}')
        time.sleep(min(20,2*(attempt+1)))
    return pd.DataFrame()

def main():
    OUT.mkdir(exist_ok=True)
    syms=symbols()
    jobs=[('^NSEI','NIFTY50')]+[(s+'.NS',s) for s in syms]
    good=0
    for i,(ticker,name) in enumerate(jobs,1):
        print(f'[{i}/{len(jobs)}] Downloading {ticker}')
        d=download(ticker)
        if d.empty: print(f'NO DATA: {ticker}'); continue
        d.to_csv(OUT/f'{name}.csv',index=False); good+=1
        print(f'saved {len(d):,} rows -> data/{name}.csv')
        time.sleep(0.2)
    if good<11 or not (OUT/'NIFTY50.csv').exists(): raise RuntimeError(f'Insufficient market data downloaded: {good} datasets')
    print(f'Completed {good}/{len(jobs)} downloads from historical research universe')
if __name__=='__main__': main()
