"""Point-in-time young/small-mid SEPA experiment.

The experiment uses the same daily SEPA engine and three historical windows.
Cohort membership is frozen at each window start using the latest market-cap
snapshot at or before that date, preventing future reclassification leakage.
Young means <=10 years from listing.  Small/mid means ranks 101-500 in the
available NSE full-market-cap snapshot; ranks 101-250 correspond to mid-cap and
251-500 to small-cap under the SEBI classification.  The 251-500 cutoff is a
research-universe boundary, not a claim that rank 501+ is absent from NSE.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import daily_sepa_backtest as engine

WINDOWS=[
 (pd.Timestamp('2019-01-01'),pd.Timestamp('2022-12-31')),
 (pd.Timestamp('2023-01-01'),pd.Timestamp('2024-12-31')),
 (pd.Timestamp('2025-01-01'),pd.Timestamp('2026-03-31')),
]
GROUPS=['control','young','small_mid','young_small_mid']
INITIAL=1_000_000.0

def load_metadata():
    listing=pd.read_csv('metadata/listing_dates.csv')
    caps=pd.read_csv('metadata/historical_market_caps.csv')
    listing['symbol']=listing.symbol.astype(str).str.upper().str.strip()
    listing['listing_date']=pd.to_datetime(listing.listing_date,errors='coerce')
    caps['symbol']=caps.symbol.astype(str).str.upper().str.strip()
    caps['date']=pd.to_datetime(caps.date,errors='coerce')
    caps['market_cap_cr']=pd.to_numeric(caps.market_cap_cr,errors='coerce')
    caps['rank']=pd.to_numeric(caps.get('rank'),errors='coerce')
    return listing.dropna(), caps.dropna(subset=['date','market_cap_cr','symbol'])

def membership(listing,caps,asof):
    # Latest full snapshot at/before window start; if multiple rows share the
    # snapshot date, rank is taken from that date directly.
    latest=caps[caps.date<=asof].copy()
    if latest.empty: return {}
    snap=latest.date.max(); c=latest[latest.date==snap].copy()
    if c['rank'].isna().all(): c['rank']=c.market_cap_cr.rank(method='first',ascending=False)
    c=c.sort_values('rank').drop_duplicates('symbol')
    l=listing.set_index('symbol')['listing_date']
    out={}
    for _,r in c.iterrows():
        sym=r.symbol
        if sym not in l.index: continue
        age=(asof-l.loc[sym]).days/365.25
        young=age>=0 and age<=10
        small_mid=101<=float(r['rank'])<=500
        out[sym]={'control':True,'young':young,'small_mid':small_mid,'young_small_mid':young and small_mid}
    return out

def metrics(trades,final,start,end):
    pnl=trades.pnl.astype(float) if not trades.empty else pd.Series(dtype=float)
    wins=pnl[pnl>0]; losses=pnl[pnl<0]
    years=max((end-start).days/365.25,1/365.25)
    ret=(final/INITIAL-1)*100
    cagr=((final/INITIAL)**(1/years)-1)*100 if final>0 else -100
    # Realized-equity drawdown is deliberately labelled as such; the engine's
    # daily open-position equity is not exposed by the legacy backtest API.
    eq=pd.Series(INITIAL,index=range(len(pnl)+1),dtype=float)
    if len(pnl): eq.iloc[1:]=INITIAL+pnl.cumsum().values
    dd=(eq/eq.cummax()-1)*100
    pf=(wins.sum()/abs(losses.sum())) if len(losses) else np.inf
    return {'start':str(start.date()),'end':str(end.date()),'return_pct':ret,'cagr_pct':cagr,'realized_max_drawdown_pct':float(dd.min()),'trades':len(pnl),'win_rate_pct':(len(wins)/len(pnl)*100 if len(pnl) else 0),'profit_factor':pf,'final_capital':final}

def main():
    listing,caps=load_metadata(); data=Path('data'); files=[p for p in data.glob('*.csv') if p.name!='NIFTY50.csv']
    rows=[]; trade_rows=[]
    for start,end in WINDOWS:
        members=membership(listing,caps,start)
        for group in GROUPS:
            if group=='control': eligible=[p for p in files if p.stem.upper() in members]
            else: eligible=[p for p in files if p.stem.upper() in members and members[p.stem.upper()][group]]
            engine.START=start; engine.END=end
            trades,final=engine.backtest(eligible,data/'NIFTY50.csv')
            m=metrics(trades,final,start,end); m.update({'group':group,'eligible_stocks':len(eligible),'snapshot_date':str(max(caps[caps.date<=start].date).date()) if not caps[caps.date<=start].empty else ''})
            rows.append(m)
            if not trades.empty:
                t=trades.copy(); t['group']=group; t['window_start']=start.date(); trade_rows.append(t)
            print(m)
    out=Path('results'); out.mkdir(exist_ok=True)
    summary=pd.DataFrame(rows)
    summary.to_csv(out/'young_small_cap_results.csv',index=False)
    if trade_rows: pd.concat(trade_rows,ignore_index=True).to_csv(out/'young_small_cap_trades.csv',index=False)
    # Benchmark is NIFTY buy-and-hold over the same windows.
    md=pd.read_csv(data/'NIFTY50.csv'); md.timestamp=pd.to_datetime(md.timestamp); md=md.set_index('timestamp')
    bench=[]
    for start,end in WINDOWS:
        x=md.close.loc[(md.index>=start)&(md.index<=end)]
        if len(x)>=2: bench.append({'start':str(start.date()),'end':str(end.date()),'benchmark_return_pct':(x.iloc[-1]/x.iloc[0]-1)*100})
    pd.DataFrame(bench).to_csv(out/'young_small_cap_benchmark.csv',index=False)
    print('\nRESULTS\n',summary.to_string(index=False))
    print('\nBENCHMARK\n',pd.DataFrame(bench).to_string(index=False))
    if summary.empty or (summary.trades<1).any(): raise RuntimeError('At least one cohort/window produced zero trades; inspect universe/metadata before accepting experiment.')
if __name__=='__main__': main()
