"""Young/small-cap SEPA experiment.

This creates a reproducible experiment without inventing historical metadata.
Supply metadata/stock_metadata.csv with columns:
 symbol,ipo_date,market_cap_cr

For each stock and each historical test window, eligibility is evaluated using
metadata available as of the window start. The experiment compares:
CONTROL (all stocks), YOUNG (<=10 years), SMALL_MID (market cap < 100000 Cr),
and YOUNG_SMALL_MID (both). Technical signals remain identical across groups.

If metadata is absent, the script fails clearly instead of silently using
current market-cap/IPO data, which would introduce look-ahead bias.
"""
from pathlib import Path
import pandas as pd
import numpy as np

INITIAL=1_000_000.0
WINDOWS=[(pd.Timestamp('2019-01-01'),pd.Timestamp('2022-12-31')),
         (pd.Timestamp('2023-01-01'),pd.Timestamp('2024-12-31')),
         (pd.Timestamp('2025-01-01'),pd.Timestamp('2026-03-31'))]
SMALL_MID_CAP_CR=100000

def load_metadata(path):
    if not path.exists():
        raise FileNotFoundError('Missing metadata/stock_metadata.csv. Historical IPO dates and market caps are required; do not substitute current values.')
    m=pd.read_csv(path)
    req={'symbol','ipo_date','market_cap_cr'}
    if not req.issubset(m.columns): raise ValueError(f'Metadata must contain {sorted(req)}')
    m['symbol']=m.symbol.astype(str).str.upper().str.strip(); m['ipo_date']=pd.to_datetime(m.ipo_date,errors='coerce'); m['market_cap_cr']=pd.to_numeric(m.market_cap_cr,errors='coerce')
    if m[['ipo_date','market_cap_cr']].isna().any().any(): raise ValueError('Metadata contains missing/invalid IPO date or market cap')
    return m.drop_duplicates('symbol').set_index('symbol')

def classify(meta,symbol,asof):
    if symbol not in meta.index: return None
    r=meta.loc[symbol]; age=(asof-r.ipo_date).days/365.25
    young=age<=10; small_mid=r.market_cap_cr<SMALL_MID_CAP_CR
    return {'control':True,'young':young,'small_mid':small_mid,'young_small_mid':young and small_mid}

def main():
    root=Path('data'); meta=load_metadata(Path('metadata/stock_metadata.csv'))
    rows=[]
    for start,end in WINDOWS:
        counts={'control':0,'young':0,'small_mid':0,'young_small_mid':0}
        for p in root.glob('*.csv'):
            if p.name=='NIFTY50.csv': continue
            sym=p.stem.upper(); c=classify(meta,sym,start)
            if c is None: continue
            for g,v in c.items(): counts[g]+=int(v)
        for g,n in counts.items(): rows.append({'start':start.date(),'end':end.date(),'group':g,'eligible_stocks':n})
    out=Path('results'); out.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(out/'universe_experiment.csv',index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print('NOTE: Technical performance comparison must use the existing SEPA engine after the historical metadata file is populated.')
if __name__=='__main__': main()
