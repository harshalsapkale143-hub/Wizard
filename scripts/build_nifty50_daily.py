"""Build daily OHLCV research data without broker credentials.

The metadata-first workflow supplies metadata/universe_symbols.csv containing
the union of historical top-500 NSE symbols. The script downloads that
research universe; it falls back to the current NIFTY-200 list only when the
metadata universe is unavailable.

Cached files are reused. A symbol is downloaded again only when its stored
history is stale relative to the *research end date*; this is important for
GitHub Actions, where the data/ directory is restored from the Actions cache
before this script runs.
"""
from __future__ import annotations
from pathlib import Path
import time
import pandas as pd
import yfinance as yf

START = "2018-01-01"
END = "2026-09-08"
OUT = Path("data")
STALE_AFTER_DAYS = 3
REFRESH_OVERLAP_DAYS = 5


def symbols():
    p=Path('metadata/universe_symbols.csv')
    if p.exists():
        d=pd.read_csv(p)
        if 'symbol' not in d.columns: raise ValueError('metadata/universe_symbols.csv must contain symbol')
        out=d.symbol.astype(str).str.strip().str.upper().tolist()
        if len(out)>=100:
            return sorted(set(out))
    # Production fallback: current NIFTY 200 constituents from NSE Indices.
    # The public index page exposes a stable CSV download even when the
    # browser-oriented NSE JSON endpoint is unavailable in CI.
    import requests
    url='https://www.niftyindices.com/IndexConstituent/ind_nifty200list.csv'
    try:
        r=requests.get(url,headers={'User-Agent':'Mozilla/5.0','Accept':'text/csv,*/*'},timeout=30)
        r.raise_for_status()
        from io import StringIO
        d=pd.read_csv(StringIO(r.content.decode('utf-8-sig')))
        col=next((c for c in d.columns if str(c).strip().lower() in {'symbol','ticker'}),None)
        if col is None:
            raise ValueError(f'NIFTY 200 CSV has no symbol column: {list(d.columns)}')
        out=sorted({str(x).strip().upper() for x in d[col].dropna() if str(x).strip()})
        if len(out)>=100:
            print(f'Using current NIFTY 200 universe from NSE Indices: {len(out)} symbols')
            Path('metadata').mkdir(exist_ok=True)
            pd.DataFrame({'symbol':out}).to_csv('metadata/universe_symbols.csv',index=False)
            return out
        raise ValueError(f'NIFTY 200 CSV returned only {len(out)} symbols')
    except Exception as e:
        print(f'NIFTY 200 CSV fallback failed: {e}')
    return [x.strip().upper() for x in Path("config/nifty50_symbols.csv").read_text().splitlines()[1:] if x.strip()]


def download(ticker: str, start: str = START) -> pd.DataFrame:
    for attempt in range(4):
        try:
            d=yf.download(ticker,start=start,end=END,interval='1d',auto_adjust=False,progress=False,threads=False,timeout=30)
            if d is not None and not d.empty:
                if isinstance(d.columns,pd.MultiIndex): d.columns=d.columns.get_level_values(0)
                d=d.rename(columns={'Date':'timestamp'}).reset_index()
                if 'timestamp' not in d.columns: d=d.rename(columns={d.columns[0]:'timestamp'})
                d['timestamp']=pd.to_datetime(d['timestamp'],errors='coerce').dt.tz_localize(None)
                d=d.rename(columns={'Open':'open','High':'high','Low':'low','Close':'close','Volume':'volume'})
                cols=['timestamp','open','high','low','close','volume']
                if set(cols).issubset(d.columns):
                    return d[cols].dropna(subset=['timestamp','open','high','low','close']).sort_values('timestamp')
        except Exception as e: print(f'{ticker}: attempt {attempt+1}/4 failed: {e}')
        time.sleep(min(20,2*(attempt+1)))
    return pd.DataFrame()


def refresh(ticker: str, name: str) -> tuple[bool, bool]:
    """Return (success, network_refresh_performed)."""
    path=OUT/f'{name}.csv'
    if path.exists():
        try:
            old=pd.read_csv(path)
            old['timestamp']=pd.to_datetime(old['timestamp'],errors='coerce').dt.tz_localize(None)
            old=old.dropna(subset=['timestamp']).sort_values('timestamp')
            research_end=pd.Timestamp(END).normalize()
            if len(old)>=250:
                age_days=(research_end-old['timestamp'].max().normalize()).days
                if age_days <= STALE_AFTER_DAYS:
                    return True, False
                start=(old['timestamp'].max()-pd.Timedelta(days=REFRESH_OVERLAP_DAYS)).strftime('%Y-%m-%d')
                fresh=download(ticker,start)
                if not fresh.empty:
                    merged=pd.concat([old[~old.timestamp.isin(fresh.timestamp)],fresh],ignore_index=True)
                    merged=merged.drop_duplicates('timestamp',keep='last').sort_values('timestamp')
                    merged.to_csv(path,index=False)
                    return True, True
                return True, True
        except Exception as e:
            print(f'{name}: cached file unusable, rebuilding: {e}')
    d=download(ticker)
    if d.empty: return False, True
    d.to_csv(path,index=False)
    return True, True


def main():
    OUT.mkdir(exist_ok=True)
    syms=symbols()
    jobs=[('^NSEI','NIFTY50')]+[(s+'.NS',s) for s in syms]
    good=0
    refreshed=0
    cached=0
    for i,(ticker,name) in enumerate(jobs,1):
        print(f'[{i}/{len(jobs)}] Checking {ticker}')
        ok, did_refresh=refresh(ticker,name)
        if ok:
            rows=len(pd.read_csv(OUT/f'{name}.csv'))
            good+=1
            if did_refresh: refreshed+=1
            else: cached+=1
            state='refreshed' if did_refresh else 'cache-hit'
            print(f'{state}: {rows:,} rows -> data/{name}.csv')
        else:
            print(f'NO DATA: {ticker}')
        time.sleep(0.05 if not did_refresh else 0.2)
    if good<11 or not (OUT/'NIFTY50.csv').exists(): raise RuntimeError(f'Insufficient market data downloaded: {good} datasets')
    print(f'Completed {good}/{len(jobs)} datasets: {cached} cache hits, {refreshed} network refreshes')


if __name__=='__main__': main()
