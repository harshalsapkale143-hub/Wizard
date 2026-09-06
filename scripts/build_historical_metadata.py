from __future__ import annotations

"""Build point-in-time NSE listing and market-cap metadata.

The builder is deliberately conservative: it uses only archived NSE snapshots
and refuses to use current market cap as a historical substitute.  Market-cap
files can be supplied via METADATA_SOURCE_DIR when downloaded from NSE's
historical all-company market-cap archive.  Listing dates can be supplied via
LISTING_SOURCE_FILE from an NSE security-master export.

Expected inputs:
  listing source: symbol,listing_date
  market-cap source(s): symbol,date,market_cap_cr

Outputs:
  metadata/listing_dates.csv
  metadata/historical_market_caps.csv
  metadata/metadata_quality.csv
  metadata/source_manifest.json
"""
from pathlib import Path
import json, os, re
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'metadata'; OUT.mkdir(exist_ok=True)
SRC=Path(os.getenv('METADATA_SOURCE_DIR', ROOT/'metadata_sources'))
LISTING=Path(os.getenv('LISTING_SOURCE_FILE', SRC/'listing_dates.csv'))

SYM_RE=re.compile(r'[^A-Z0-9&-]')

def norm_symbol(x):
    x=str(x).strip().upper()
    return SYM_RE.sub('', x)

def read_table(p):
    for enc in ('utf-8-sig','utf-8','latin1'):
        try: return pd.read_csv(p, encoding=enc)
        except Exception: pass
    raise ValueError(f'Unable to read {p}')

def find_col(df,names):
    cols={str(c).strip().lower().replace(' ','_'):c for c in df.columns}
    for n in names:
        if n in cols: return cols[n]
    return None

def build_listing():
    if not LISTING.exists(): raise FileNotFoundError(f'Missing {LISTING}; provide an NSE security-master listing-date export.')
    d=read_table(LISTING); sc=find_col(d,['symbol','ticker','security_symbol']); dc=find_col(d,['listing_date','listingdate','date_of_listing'])
    if not sc or not dc: raise ValueError('Listing source needs symbol and listing_date columns')
    out=pd.DataFrame({'symbol':d[sc].map(norm_symbol),'listing_date':pd.to_datetime(d[dc],errors='coerce')}).dropna()
    out=out[out.symbol!=''].drop_duplicates('symbol')
    if out.listing_date.dt.year.lt(1900).any(): raise ValueError('Invalid listing dates detected')
    out.to_csv(OUT/'listing_dates.csv',index=False); return out

def build_caps():
    frames=[]
    for p in sorted(SRC.glob('*.csv')):
        if p.resolve()==LISTING.resolve(): continue
        try: d=read_table(p)
        except Exception: continue
        sc=find_col(d,['symbol','ticker','security_symbol']); dc=find_col(d,['date','as_of_date','market_cap_date']); mc=find_col(d,['market_cap_cr','market_cap','market_capitalization','market_capitalisation'])
        if not (sc and dc and mc): continue
        x=pd.DataFrame({'symbol':d[sc].map(norm_symbol),'date':pd.to_datetime(d[dc],errors='coerce'),'market_cap_cr':pd.to_numeric(d[mc],errors='coerce')})
        x=x.dropna(); x=x[(x.symbol!='')&(x.market_cap_cr>0)]; frames.append(x)
    if not frames: raise FileNotFoundError(f'No historical market-cap CSVs found in {SRC}')
    out=pd.concat(frames,ignore_index=True).drop_duplicates(['symbol','date'],keep='last').sort_values(['symbol','date'])
    if out.date.max()>pd.Timestamp.today()+pd.Timedelta(days=31): raise ValueError('Market-cap data contains implausible future dates')
    out.to_csv(OUT/'historical_market_caps.csv',index=False); return out

def main():
    listing=build_listing(); caps=build_caps(); symbols=sorted(set(listing.symbol)&set(caps.symbol))
    quality=[]
    for y in range(2019,2027):
        cutoff=pd.Timestamp(f'{y}-12-31'); c=caps[caps.date<=cutoff].sort_values('date').groupby('symbol').tail(1)
        covered=len(set(c.symbol)&set(symbols)); quality.append({'asof':str(cutoff.date()),'symbols_with_listing_and_cap':covered,'cap_coverage_pct':round(100*covered/max(1,len(symbols)),2)})
    pd.DataFrame(quality).to_csv(OUT/'metadata_quality.csv',index=False)
    manifest={'source_dir':str(SRC),'listing_source':str(LISTING),'listing_rows':len(listing),'market_cap_rows':len(caps),'common_symbols':len(symbols),'market_cap_min_date':str(caps.date.min().date()),'market_cap_max_date':str(caps.date.max().date()),'policy':'No current-data substitution; point-in-time snapshots only'}
    (OUT/'source_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest,indent=2)); print(pd.DataFrame(quality).to_string(index=False))
if __name__=='__main__': main()
