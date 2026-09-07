"""Build daily OHLCV research data without broker credentials.

The metadata-first workflow supplies metadata/universe_symbols.csv containing
the union of historical top-500 NSE symbols. The script downloads that
research universe; it falls back to the current NIFTY-50 list only when the
metadata universe is unavailable.

Cached files are reused. A symbol is downloaded again only when its stored
history is stale; this is important for GitHub Actions, where the data/
directory is restored from the Actions cache before this script runs.
"""
from __future__ import annotations
from pathlib import Path
import time
import pandas as pd
import yfinance as yf

START = "2018-01-01"
END = "2026-09-08"
OUT = Path("data")
# A daily NSE series is considered current if its last observation is within
# this many calendar days of today. This naturally covers weekends/holidays
# without forcing a network request for every cached symbol.
STALE_AFTER_DAYS = 3
REFRESH_OVERLAP_DAYS = 5


def symbols():
    p=Path('metadata/universe_symbols.csv')
    if p.exists():
        d=pd.read_csv(p)
        if 'symbol' not in d.columns: raise ValueError('metadata/universe_symbols.csv must contain symbol')
        out=d.symbol.astype(str).str.strip().str.upper().tolist()
        if len(out)<100: raise RuntimeError(f'Historical universe unexpectedly small: {len(out)} symbols')
        return sorted(set(out))
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
            today=pd.Timestamp.now().normalize()
            if len(old)>=250:
                age_days=(today-old['timestamp'].max().normalize()).days
                if age_days <= STALE_AFTER_DAYS:
                    # Cached history is current enough. Do not hit Yahoo at all.
                    return True, False
                # Stale cache: request only a short overlap and merge it into
                # the existing history, rather than redownloading from 2018.
                start=(old['timestamp'].max()-pd.Timedelta(days=REFRESH_OVERLAP_DAYS)).strftime('%Y-%m-%d')
                fresh=download(ticker,start)
                if not fresh.empty:
                    merged=pd.concat([old[~old.timestamp.isin(fresh.timestamp)],fresh],ignore_index=True)
                    merged=merged.drop_duplicates('timestamp',keep='last').sort_values('timestamp')
                    merged.to_csv(path,index=False)
                    return True, True
                return True, True
            # Old/incomplete cache: rebuild the full history.
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
