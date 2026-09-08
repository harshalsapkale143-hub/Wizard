from __future__ import annotations
from pathlib import Path
import itertools
import numpy as np
import pandas as pd
import robustness_backtest as rb

INITIAL=1_000_000.0
CONFIGS=list(itertools.product([0.80,0.85,0.90],[0.55,0.65,0.75],[1.25,1.50,1.75]))
FY_WINDOWS=[
 ('FY2021-22','2021-04-01','2022-03-31'),('FY2022-23','2022-04-01','2023-03-31'),
 ('FY2023-24','2023-04-01','2024-03-31'),('FY2024-25','2024-04-01','2025-03-31'),
 ('FY2025-26','2025-04-01','2026-03-31')]

def regime(market,start,end):
    c=market.loc[start:end,'close']
    if len(c)<2:return 'unknown'
    ret=(c.iloc[-1]/c.iloc[0]-1)*100
    ma200=market.loc[:end,'close'].rolling(200).mean().iloc[-1]
    dd=(c/c.cummax()-1).min()*100
    if ret>=15:return 'bull'
    if ret<=-10 or dd<=-15:return 'bear'
    return 'flat/mixed'

def main():
    root=Path('data'); mp=root/'NIFTY50.csv'; market=rb.load(mp)
    files=[p for p in root.glob('*.csv') if p.name!='NIFTY50.csv']
    rb.get_cache(files,mp); rb.build_rs_rank(files,market)
    rows=[]; grid=[]; trades=[]
    for label,a,b in FY_WINDOWS:
        for near,tighten,vm in CONFIGS:
            r,tr,_=rb.run(files,market,pd.Timestamp(a),pd.Timestamp(b),near,tighten,vm)
            nifty=market.loc[a:b,'close']; bench=(nifty.iloc[-1]/nifty.iloc[0]-1)*100 if len(nifty)>1 else 0
            r.update({'fy':label,'regime':regime(market,a,b),'nifty_return_pct':bench,'excess_pct':r['return_pct']-bench,
                      'near_high':near,'tighten':tighten,'vol_mult':vm})
            grid.append(r)
            if len(tr):
                tr=tr.copy(); tr['fy']=label; tr['near_high']=near; tr['tighten']=tighten; tr['vol_mult']=vm; trades.append(tr)
    g=pd.DataFrame(grid)
    # Fixed baseline, reported separately for every FY. No combined-period performance is used.
    base=g[(g.near_high==.85)&(g.tighten==.65)&(g.vol_mult==1.5)].copy()
    base=base.sort_values('fy')
    # For each FY, rank configurations independently, then report whether the same config is stable across regimes.
    winners=g.sort_values(['fy','return_pct'],ascending=[True,False]).groupby('fy',as_index=False).head(5)
    regime_summary=g.groupby('regime').agg(fy_count=('fy','nunique'),config_return_median=('return_pct','median'),config_pf_median=('profit_factor','median'),config_sharpe_median=('sharpe','median'),config_dd_median=('max_drawdown_pct','median')).reset_index()
    out=Path('results'); out.mkdir(exist_ok=True)
    base.to_csv(out/'yearly_baseline_summary.csv',index=False)
    g.to_csv(out/'yearly_regime_robustness_grid.csv',index=False)
    winners.to_csv(out/'yearly_top5_configs.csv',index=False)
    regime_summary.to_csv(out/'regime_summary.csv',index=False)
    if trades: pd.concat(trades,ignore_index=True).to_csv(out/'yearly_trades.csv',index=False)
    print('\nYEARLY BASELINE\n',base[['fy','regime','return_pct','cagr_pct','max_drawdown_pct','trades','win_rate_pct','profit_factor','sharpe','nifty_return_pct','excess_pct']].to_string(index=False))
    print('\nTOP 5 CONFIGS PER FY\n',winners[['fy','regime','near_high','tighten','vol_mult','return_pct','max_drawdown_pct','profit_factor','sharpe']].to_string(index=False))
    print('\nREGIME SUMMARY\n',regime_summary.to_string(index=False))
if __name__=='__main__':main()
