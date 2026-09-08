from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import robustness_backtest as rb

INITIAL=1_000_000.0; COST_BPS=10.0; RISK_PCT=0.005; MAX_POSITIONS=10
CONFIGS=[(.80,.65,1.5),(.85,.65,1.5),(.90,.65,1.5),(.85,.65,1.75)]
WINDOWS=[('2019-01-01','2022-12-31'),('2023-01-01','2024-12-31'),('2025-01-01','2026-03-31'),('2021-04-01','2026-03-31')]

# Exit variants are deliberately simple and pre-specified. They alter exits only;
# entry logic, universe, sizing, costs and signal parameters remain unchanged.
VARIANTS={
 'baseline_stop_50dma': {'trigger_r':999,'trail':'none'},
 'trail20_after_1r': {'trigger_r':1.0,'trail':'20d'},
 'trail20_after_1_5r': {'trigger_r':1.5,'trail':'20d'},
 'trail20_after_2r': {'trigger_r':2.0,'trail':'20d'},
 'trail10_after_1r': {'trigger_r':1.0,'trail':'10d'},
 'trail10_after_1_5r': {'trigger_r':1.5,'trail':'10d'},
 'trail10_after_2r': {'trigger_r':2.0,'trail':'10d'},
}

def run_variant(files, market, start, end, near, tighten, vm, variant):
    # Reuse the exact production signal engine; only position exit logic changes.
    rb.build_rs_rank(files, market)
    cache=rb.DATA_CACHE
    sigs=[]
    if (str(start),str(end)) not in rb.MARKET_OK_CACHE:
        md=market.loc[:end]; rb.MARKET_OK_CACHE[(str(start),str(end))]=(md.close>md.close.rolling(200).mean()).shift(1).fillna(False)
    market_ok=rb.MARKET_OK_CACHE[(str(start),str(end))]
    for sym,d in cache.items():
        for dt,e,stop in rb.signals(sym,d,market,near,tighten,vm):
            if start<=dt<=end and bool(market_ok.reindex([dt]).fillna(False).iloc[0]): sigs.append((dt,sym,e,stop))
    sigs.sort(); cash=INITIAL; pos={}; trades=[]; curve=[]
    cfg=VARIANTS[variant]
    for dt in market.index[(market.index>=start)&(market.index<=end)]:
        for sym in list(pos):
            d=cache[sym];
            if dt not in d.index: continue
            px=float(d.loc[dt,'close']); f=rb.FEATURE_CACHE[sym]; ma50=float(f['ma50'].loc[dt]); stop=pos[sym]['stop']; r=pos[sym]['risk']
            if cfg['trail']=='20d': trail=float(f['trail20'].loc[dt])
            elif cfg['trail']=='10d': trail=float(f['low10'].loc[dt])
            else: trail=-np.inf
            effective_stop=max(stop,trail) if pos[sym]['peak_r']>=cfg['trigger_r'] else stop
            reason='stop_or_50dma'
            if cfg['trail']!='none' and pos[sym]['peak_r']>=cfg['trigger_r'] and px<=effective_stop: reason=f"{cfg['trail']}_trail_after_{cfg['trigger_r']}r"
            if px<=effective_stop or (np.isfinite(ma50) and px<ma50):
                proceeds=pos[sym]['qty']*px; fee=(pos[sym]['qty']*pos[sym]['entry']+proceeds)*COST_BPS/10000; pnl=proceeds-pos[sym]['cost']-fee; cash+=proceeds-fee
                trades.append((pos[sym]['date'],dt,sym,pos[sym]['entry'],px,pos[sym]['qty'],pnl,reason)); del pos[sym]
            else:
                pos[sym]['peak_r']=max(pos[sym]['peak_r'],(px-pos[sym]['entry'])/r)
        for sdt,sym,e,stop in sigs:
            if sdt!=dt or sym in pos or len(pos)>=MAX_POSITIONS: continue
            risk=e-stop
            equity=cash+sum(x['qty']*float(cache[k].loc[dt,'close']) for k,x in pos.items() if dt in cache[k].index)
            qty=min(int(equity*RISK_PCT/risk),int(cash/(e*(1+COST_BPS/10000)))) if risk>0 else 0
            if qty<=0: continue
            fee=qty*e*COST_BPS/10000; cost=qty*e+fee
            if cost>cash: continue
            cash-=cost; pos[sym]={'date':dt,'entry':e,'stop':stop,'risk':risk,'qty':qty,'cost':cost,'peak_r':0.0}
        curve.append((dt,cash+sum(x['qty']*float(cache[k].loc[dt,'close']) for k,x in pos.items() if dt in cache[k].index)))
    for sym,p in list(pos.items()):
        d=cache[sym]; final_dt=d.loc[:end].index[-1]; px=float(d.loc[final_dt,'close']); proceeds=p['qty']*px; fee=(p['qty']*p['entry']+proceeds)*COST_BPS/10000; cash+=proceeds-fee
        trades.append((p['date'],final_dt,sym,p['entry'],px,p['qty'],proceeds-p['cost']-fee,'end_of_test'))
    tr=pd.DataFrame(trades,columns=['entry_date','exit_date','symbol','entry','exit','qty','pnl','exit_reason'])
    eq=pd.Series({d:v for d,v in curve}).sort_index(); final=cash
    dd=float((eq/eq.cummax()-1).min()) if len(eq) else 0; years=max((eq.index[-1]-eq.index[0]).days/365.25,1/365.25) if len(eq) else 1
    cagr=(final/INITIAL)**(1/years)-1; daily=eq.pct_change().dropna(); sharpe=float(np.sqrt(252)*daily.mean()/daily.std()) if daily.std()>0 else 0
    wins=tr.loc[tr.pnl>0,'pnl']; losses=tr.loc[tr.pnl<0,'pnl']; pf=float(wins.sum()/abs(losses.sum())) if len(losses) else np.inf
    return {'start':str(start.date()),'end':str(end.date()),'near_high':near,'tighten':tighten,'vol_mult':vm,'exit_variant':variant,'return_pct':(final/INITIAL-1)*100,'cagr_pct':cagr*100,'max_drawdown_pct':dd*100,'trades':len(tr),'win_rate_pct':len(wins)/len(tr)*100 if len(tr) else 0,'profit_factor':pf,'sharpe':sharpe},tr

def main():
    root=Path('data'); market=rb.load(root/'NIFTY50.csv'); rb.MARKET_CACHE=market
    files=[p for p in root.glob('*.csv') if p.name!='NIFTY50.csv']
    rb.get_cache(files,root/'NIFTY50.csv'); rb.build_rs_rank(files,market)
    rows=[]; trades=[]
    for near,tighten,vm in CONFIGS:
        for a,b in WINDOWS:
            for variant in VARIANTS:
                r,tr=run_variant(files,market,pd.Timestamp(a),pd.Timestamp(b),near,tighten,vm,variant); rows.append(r)
                tr['start']=a; tr['end']=b; tr['near_high']=near; tr['tighten']=tighten; tr['vol_mult']=vm; tr['exit_variant']=variant; trades.append(tr)
    out=Path('results'); out.mkdir(exist_ok=True); df=pd.DataFrame(rows); df.to_csv(out/'exit_management_grid.csv',index=False); pd.concat(trades,ignore_index=True).to_csv(out/'exit_management_trades.csv',index=False)
    summary=df.groupby('exit_variant').agg(return_mean=('return_pct','mean'),return_median=('return_pct','median'),cagr_mean=('cagr_pct','mean'),dd_mean=('max_drawdown_pct','mean'),pf_median=('profit_factor','median'),sharpe_mean=('sharpe','mean'),trades_mean=('trades','mean')).reset_index().sort_values(['return_mean','sharpe_mean'],ascending=False)
    summary.to_csv(out/'exit_management_summary.csv',index=False); print('EXIT MANAGEMENT SUMMARY'); print(summary.to_string(index=False))

if __name__=='__main__': main()
