from __future__ import annotations

"""Build point-in-time NSE listing and market-cap metadata.

Accepts CSV/XLS/XLSX/ZIP/GZ market-cap archives.  Dates are read from the
source table when available, otherwise from the archive filename.  No current
market-cap substitution is permitted.
"""
from pathlib import Path
import json, os, re, zipfile, gzip, tempfile
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'metadata'; OUT.mkdir(exist_ok=True)
SRC=Path(os.getenv('METADATA_SOURCE_DIR', ROOT/'metadata_sources'))
LISTING=Path(os.getenv('LISTING_SOURCE_FILE', SRC/'listing_dates.csv'))

SYM_RE=re.compile(r'[^A-Z0-9&-]')
MONTHS='January|February|March|April|May|June|July|August|September|October|November|December'

def norm_symbol(x):
    return SYM_RE.sub('', str(x).strip().upper())

def canon(x):
    return re.sub(r'[^a-z0-9]+','_',str(x).strip().lower()).strip('_')

def read_csv(p):
    for enc in ('utf-8-sig','utf-8','latin1'):
        try: return pd.read_csv(p, encoding=enc)
        except Exception: pass
    raise ValueError(f'Unable to read {p}')

def read_table(p):
    s=p.suffix.lower()
    if s in ('.xls','.xlsx'):
        return pd.read_excel(p)
    return read_csv(p)

def find_col(df, aliases):
    cols={canon(c):c for c in df.columns}
    for alias in aliases:
        a=canon(alias)
        if a in cols: return cols[a]
    for c0,c in cols.items():
        if any(a in c0 for a in [canon(x) for x in aliases]): return c
    return None

def date_from_text(text):
    t=str(text)
    m=re.search(rf'({MONTHS})\s+(\d{{1,2}}),?\s+(\d{{4}})',t,re.I)
    if m: return pd.to_datetime(m.group(0),errors='coerce')
    m=re.search(rf'(\d{{1,2}})\s+({MONTHS})\s+(\d{{4}})',t,re.I)
    if m: return pd.to_datetime(m.group(0),errors='coerce',dayfirst=True)
    m=re.search(r'(20\d{2})[-_](\d{1,2})[-_](\d{1,2})',t)
    if m: return pd.Timestamp(int(m.group(1)),int(m.group(2)),int(m.group(3)))
    return pd.NaT

def build_listing():
    if not LISTING.exists(): raise FileNotFoundError(f'Missing {LISTING}')
    d=read_csv(LISTING); sc=find_col(d,['symbol','ticker','security_symbol']); dc=find_col(d,['listing_date','listingdate','date_of_listing'])
    if not sc or not dc: raise ValueError('Listing source needs symbol and listing_date columns')
    out=pd.DataFrame({'symbol':d[sc].map(norm_symbol),'listing_date':pd.to_datetime(d[dc],errors='coerce')}).dropna()
    out=out[out.symbol!=''].drop_duplicates('symbol')
    if out.listing_date.dt.year.lt(1900).any(): raise ValueError('Invalid listing dates detected')
    out.to_csv(OUT/'listing_dates.csv',index=False); return out

def iter_source_files():
    for p in sorted(SRC.iterdir()):
        if p.name == LISTING.name or p.name.startswith('.') or p.is_dir(): continue
        if p.suffix.lower() in {'.csv','.xls','.xlsx','.zip','.gz'}:
            yield p

def tables_from_archive(p):
    suffix=p.suffix.lower()
    if suffix in {'.csv','.xls','.xlsx'}:
        yield p, read_table(p)
        return
    with tempfile.TemporaryDirectory() as td:
        td=Path(td)
        if suffix=='.gz':
            q=td/(p.stem or 'source.csv'); q.write_bytes(gzip.open(p,'rb').read())
            yield q, read_table(q)
        elif suffix=='.zip':
            with zipfile.ZipFile(p) as z:
                for name in z.namelist():
                    if name.endswith('/') or Path(name).suffix.lower() not in {'.csv','.xls','.xlsx'}: continue
                    q=td/Path(name).name; q.write_bytes(z.read(name))
                    try: yield q, read_table(q)
                    except Exception: continue

def build_caps():
    frames=[]; diagnostics=[]
    for p in iter_source_files():
        try:
            for inner,d in tables_from_archive(p):
                sc=find_col(d,['symbol','ticker','security_symbol','nse_symbol']); dc=find_col(d,['date','as_of_date','market_cap_date','as_on_date','date_as_on'])
                mc=find_col(d,['market_cap_cr','market_cap','market_capitalization','market_capitalisation','market_cap_rs_cr','market_cap_rs_crore'])
                snapshot=date_from_text(p.name)
                if not (sc and mc):
                    diagnostics.append((p.name,'missing_symbol_or_market_cap',list(map(str,d.columns)))); continue
                dates=pd.to_datetime(d[dc],errors='coerce') if dc else pd.Series(snapshot,index=d.index)
                if dates.isna().all() and pd.notna(snapshot): dates=pd.Series(snapshot,index=d.index)
                x=pd.DataFrame({'symbol':d[sc].map(norm_symbol),'date':dates,'market_cap_cr':pd.to_numeric(d[mc],errors='coerce')})
                x=x.dropna(); x=x[(x.symbol!='')&(x.market_cap_cr>0)]
                if not x.empty: frames.append(x)
                diagnostics.append((p.name,f'accepted:{len(x)}',list(map(str,d.columns))))
        except Exception as e:
            diagnostics.append((p.name,f'error:{e}',[]))
    if not frames:
        raise FileNotFoundError(f'No usable historical market-cap tables found in {SRC}. Diagnostics={diagnostics}')
    out=pd.concat(frames,ignore_index=True).drop_duplicates(['symbol','date'],keep='last').sort_values(['date','market_cap_cr'],ascending=[True,False])
    if out.date.max()>pd.Timestamp.today()+pd.Timedelta(days=31): raise ValueError('Market-cap data contains implausible future dates')
    out.to_csv(OUT/'historical_market_caps.csv',index=False)
    pd.DataFrame(diagnostics,columns=['source','status','columns']).to_csv(OUT/'market_cap_parse_diagnostics.csv',index=False)
    return out

def main():
    listing=build_listing(); caps=build_caps(); symbols=sorted(set(listing.symbol)&set(caps.symbol))
    quality=[]; universe=set()
    for y in range(2019,2027):
        cutoff=pd.Timestamp(f'{y}-12-31')
        c=caps[caps.date<=cutoff].sort_values(['date','market_cap_cr']).groupby('symbol').tail(1).sort_values('market_cap_cr',ascending=False)
        covered=len(set(c.symbol)&set(symbols)); quality.append({'asof':str(cutoff.date()),'symbols_with_listing_and_cap':covered,'cap_coverage_pct':round(100*covered/max(1,len(symbols)),2)})
        universe.update(c.head(500).symbol.tolist())
    pd.DataFrame(quality).to_csv(OUT/'metadata_quality.csv',index=False)
    pd.DataFrame({'symbol':sorted(universe)}).to_csv(OUT/'universe_symbols.csv',index=False)
    # Point-in-time snapshot metadata used by the experiment.  We retain every
    # historical observation rather than collapsing it to today's value.
    caps['rank']=caps.groupby('date')['market_cap_cr'].rank(method='first',ascending=False)
    caps.to_csv(OUT/'historical_market_caps.csv',index=False)
    manifest={'source_dir':str(SRC),'listing_source':str(LISTING),'listing_rows':len(listing),'market_cap_rows':len(caps),'common_symbols':len(symbols),'universe_symbols':len(universe),'market_cap_min_date':str(caps.date.min().date()),'market_cap_max_date':str(caps.date.max().date()),'size_policy':'NSE snapshot top-500 universe; small/mid research bucket is ranks 101-500; point-in-time snapshot at/before test date','policy':'No current-data substitution; point-in-time snapshots only'}
    (OUT/'source_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest,indent=2)); print(pd.DataFrame(quality).to_string(index=False))
if __name__=='__main__': main()
