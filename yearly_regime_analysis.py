from __future__ import annotations
from pathlib import Path
import itertools
import pandas as pd
import robustness_backtest as rb

CONFIGS=list(itertools.product([0.80,0.85,0.90],[0.55,0.65,0.75],[1.25,1.50,1.75]))
FY_WINDOWS=[
 ('FY2021-22','2021-04-01','2022-03-31'),('FY2022-23','2022-04-01','2023-03-31'),
 ('FY2023-24','2023-04-01','2024-03-31'),('FY2024-25','2024-04-01','2025-03-31'),
 ('FY2025-26','2025-04-01','2026-03-31')]


def regime(market,start,end):
    """Descriptive market regime using return, drawdown, and 200-DMA state.

    This is classification only; it is never used to select a strategy/configuration.
    A full-year return alone can mislabel a year that had a major intra-year selloff,
    so the 200-DMA state is included explicitly.
    """
    c=market.loc[start:end,'close'].dropna()
    if len(c)<2:
        return {'regime':'unknown','return_pct':0.0,'max_dd_pct':0.0,
                'end_vs_200dma':'unknown','dma200_slope_pct':0.0,'reason':'insufficient data'}
    ret=(c.iloc[-1]/c.iloc[0]-1)*100
    dd=(c/c.cummax()-1).min()*100
    full=market.loc[:end,'close'].dropna()
    dma=full.rolling(200).mean()
    end_dma=dma.iloc[-1] if len(dma) else float('nan')
    prior_dma=dma.iloc[-21] if len(dma)>=21 else float('nan')
    slope=((end_dma/prior_dma)-1)*100 if pd.notna(end_dma) and pd.notna(prior_dma) and prior_dma else 0.0
    above=pd.notna(end_dma) and c.iloc[-1] > end_dma

    if above and ret >= 10 and slope > 0:
        label='bull'; reason='positive annual return, above rising 200-DMA'
    elif (not above) and (ret <= -5 or dd <= -15):
        label='bear'; reason='below 200-DMA with weak annual return or deep drawdown'
    else:
        label='sideways/mixed'; reason='annual return and trend state are not consistently bullish or bearish'
    return {'regime':label,'return_pct':ret,'max_dd_pct':dd,
            'end_vs_200dma':'above' if above else 'below',
            'dma200_slope_pct':slope,'reason':reason}


def main():
    root=Path('data'); mp=root/'NIFTY50.csv'; market=rb.load(mp)
    files=[p for p in root.glob('*.csv') if p.name!='NIFTY50.csv']
    rb.get_cache(files,mp); rb.build_rs_rank(files,market)
    grid=[]; trades=[]; regime_rows=[]
    for label,a,b in FY_WINDOWS:
        rg=regime(market,a,b)
        regime_rows.append({'fy':label,**rg})
        for near,tighten,vm in CONFIGS:
            r,tr,_=rb.run(files,market,pd.Timestamp(a),pd.Timestamp(b),near,tighten,vm)
            nifty=market.loc[a:b,'close'].dropna()
            bench=(nifty.iloc[-1]/nifty.iloc[0]-1)*100 if len(nifty)>1 else 0
            r.update({'fy':label,'regime':rg['regime'],'nifty_return_pct':bench,
                      'excess_pct':r['return_pct']-bench,'near_high':near,
                      'tighten':tighten,'vol_mult':vm})
            grid.append(r)
            if len(tr):
                tr=tr.copy(); tr['fy']=label; tr['near_high']=near; tr['tighten']=tighten; tr['vol_mult']=vm; trades.append(tr)
    g=pd.DataFrame(grid)
    base=g[(g.near_high==.85)&(g.tighten==.65)&(g.vol_mult==1.5)].sort_values('fy').copy()
    winners=g.sort_values(['fy','return_pct'],ascending=[True,False]).groupby('fy',as_index=False).head(5)
    regime_summary=g.groupby('regime').agg(
        fy_count=('fy','nunique'), config_return_median=('return_pct','median'),
        config_excess_median=('excess_pct','median'), config_pf_median=('profit_factor','median'),
        config_sharpe_median=('sharpe','median'), config_dd_median=('max_drawdown_pct','median')
    ).reset_index()
    out=Path('results'); out.mkdir(exist_ok=True)
    base.to_csv(out/'yearly_baseline_summary.csv',index=False)
    g.to_csv(out/'yearly_regime_robustness_grid.csv',index=False)
    winners.to_csv(out/'yearly_top5_configs.csv',index=False)
    regime_summary.to_csv(out/'regime_summary.csv',index=False)
    pd.DataFrame(regime_rows).to_csv(out/'market_regime_classification.csv',index=False)
    if trades: pd.concat(trades,ignore_index=True).to_csv(out/'yearly_trades.csv',index=False)
    print('\nMARKET REGIMES\n',pd.DataFrame(regime_rows).to_string(index=False))
    print('\nYEARLY BASELINE\n',base[['fy','regime','return_pct','max_drawdown_pct','trades','win_rate_pct','profit_factor','sharpe','nifty_return_pct','excess_pct']].to_string(index=False))
    print('\nTOP 5 CONFIGS PER FY (research only; not used for selection)\n',winners[['fy','regime','near_high','tighten','vol_mult','return_pct','max_drawdown_pct','profit_factor','sharpe']].to_string(index=False))
    print('\nREGIME SUMMARY\n',regime_summary.to_string(index=False))

if __name__=='__main__':main()
