from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

FY_WINDOWS=[('FY2021-22','2021-04-01','2022-03-31'),('FY2022-23','2022-04-01','2023-03-31'),('FY2023-24','2023-04-01','2024-03-31'),('FY2024-25','2024-04-01','2025-03-31'),('FY2025-26','2025-04-01','2026-03-31')]

def num(s):
    return pd.to_numeric(s,errors='coerce')

def snapshot(d, asof):
    d=d[d.filing_date<=asof].sort_values('filing_date').copy()
    if d.empty: return None
    # Prefer consolidated records when the source exposes that field.
    if 'consolidated' in d.columns:
        x=d[d.consolidated.astype(str).str.contains('consolidated',case=False,na=False)]
        if len(x): d=x
    d=d.drop_duplicates('period_end',keep='last').sort_values('period_end')
    if len(d)<5: return None
    q=d.iloc[-1]
    prev=d[d.period_end<=q.period_end-pd.DateOffset(years=1)]
    if prev.empty: return None
    p=prev.iloc[(prev.period_end-q.period_end).abs().argmin()]
    eps_yoy=(q.eps/p.eps-1) if pd.notna(q.eps) and pd.notna(p.eps) and p.eps!=0 else np.nan
    rev_yoy=(q.revenue/p.revenue-1) if pd.notna(q.revenue) and pd.notna(p.revenue) and p.revenue!=0 else np.nan
    pat_yoy=(q.pat/p.pat-1) if pd.notna(q.pat) and pd.notna(p.pat) and p.pat!=0 else np.nan
    return {'filing_date':q.filing_date,'period_end':q.period_end,'eps_yoy':eps_yoy,'revenue_yoy':rev_yoy,'pat_yoy':pat_yoy}

def main():
    root=Path('fundamentals'); out=Path('results'); out.mkdir(exist_ok=True)
    rows=[]; quality=[]
    for p in sorted(root.glob('*.csv')):
        try:
            d=pd.read_csv(p,parse_dates=['period_end','filing_date'])
        except Exception: continue
        if d.empty: continue
        for c in ['revenue','pat','eps']:
            d[c]=num(d.get(c,np.nan))
        quality.append({'symbol':p.stem,'rows':len(d),'first_filing':d.filing_date.min(),'last_filing':d.filing_date.max(),'eps_coverage':d.eps.notna().mean(),'revenue_coverage':d.revenue.notna().mean(),'pat_coverage':d.pat.notna().mean()})
        for fy,a,b in FY_WINDOWS:
            s=snapshot(d,pd.Timestamp(b))
            if s: rows.append({'symbol':p.stem,'fy':fy,**s})
    q=pd.DataFrame(quality); q.to_csv(out/'fundamental_data_quality.csv',index=False)
    s=pd.DataFrame(rows)
    if s.empty: raise SystemExit('No usable point-in-time fundamental snapshots found')
    s['eps_20']=s.eps_yoy>=.20; s['revenue_10']=s.revenue_yoy>=.10; s['pat_10']=s.pat_yoy>=.10
    s['eps_40']=s.eps_yoy>=.40; s['fundamental_core']=s.eps_20&s.revenue_10; s['fundamental_strict']=s.eps_40&s.revenue_10&s.pat_10
    s.to_csv(out/'fundamental_snapshots_by_fy.csv',index=False)
    summary=s.groupby('fy').agg(symbols=('symbol','nunique'),core_pass=('fundamental_core','sum'),strict_pass=('fundamental_strict','sum'),median_eps_yoy=('eps_yoy','median'),median_revenue_yoy=('revenue_yoy','median'),median_pat_yoy=('pat_yoy','median')).reset_index()
    summary.to_csv(out/'fundamental_sepa_summary.csv',index=False)
    print('\nPOINT-IN-TIME FUNDAMENTAL COVERAGE')
    print(summary.to_string(index=False))
    print(f'\nSymbols with filings: {len(q)}; FY snapshots: {len(s)}')

if __name__=='__main__': main()
