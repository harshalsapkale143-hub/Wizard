from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import robustness_backtest as rb

INITIAL=1_000_000.0; COST_BPS=10.0; RISK_PCT=0.005; MAX_POSITIONS=10
CONFIGS=[(.80,.65,1.5),(.85,.65,1.5),(.90,.65,1.5),(.85,.65,1.75)]
WINDOWS=[('2019-01-01','2022-12-31'),('2023-01-01','2024-12-31'),('2025-01-01','2026-03-31'),('2021-04-01','2026-03-31')]
THRESHOLDS=[1.0,1.5,2.0,2.5,3.0]

def fee(q,p): return q*p*COST_BPS/10000

def realize(pos,q,p,dt,reason):
    frac=q/pos['qty']; alloc=pos['cost']*frac; proceeds=q*p
    xfee=(q*pos['entry']+proceeds)*COST_BPS/10000
    pnl=proceeds-alloc-xfee
    return proceeds-xfee,(pos['date'],dt,pos['symbol'],pos['entry'],p,q,pnl,reason)

def run(files,market,start,end,near,tighten,vm,threshold):
    rb.build_rs_rank(files,market); cache=rb.DATA_CACHE; key=(str(start),str(end))
    if key not in rb.MARKET_OK_CACHE:
        md=market.loc[:end]; rb.MARKET_OK_CACHE[key]=(md.close>md.close.rolling(200).mean()).shift(1).fillna(False)
    mok=rb.MARKET_OK_CACHE[key]; sigs=[]
    for sym,d in cache.items():
        for dt,e,stop in rb.signals(sym,d,market,near,tighten,vm):
            if start<=dt<=end and bool(mok.reindex([dt]).fillna(False).iloc[0]): sigs.append((dt,sym,e,stop))
    sigs.sort(); cash=INITIAL; pos={}; trades=[]; curve=[]
    for dt in market.index[(market.index>=start)&(market.index<=end)]:
        for sym in list(pos):
            p=pos[sym]; d=cache[sym]
            if dt not in d.index: continue
            row=d.loc[dt]; px=float(row.close); high=float(row.high); low=float(row.low)
            f=rb.FEATURE_CACHE[sym]; ma50=float(f['ma50'].loc[dt])
            p['peak_r']=max(p['peak_r'],(high-p['entry'])/p['risk']); reached=p['peak_r']>=threshold
            if low<=p['stop']:
                proceeds,t=realize(p,p['qty'],p['stop'],dt,'initial_stop'); cash+=proceeds; trades.append(t); del pos[sym]; continue
            # Before the threshold, the 50-DMA is deliberately ignored. Once reached,
            # the normal 50-DMA close exit becomes active.
            if reached and np.isfinite(ma50) and px<ma50:
                proceeds,t=realize(p,p['qty'],px,dt,'50dma_after_threshold'); cash+=proceeds; trades.append(t); del pos[sym]; continue
        for sdt,sym,e,stop in sigs:
            if sdt!=dt or sym in pos or len(pos)>=MAX_POSITIONS: continue
            risk=e-stop
            equity=cash+sum(x['qty']*float(cache[k].loc[dt,'close']) for k,x in pos.items() if dt in cache[k].index)
            qty=min(int(equity*RISK_PCT/risk),int(cash/(e*(1+COST_BPS/10000)))) if risk>0 else 0
            if qty<=0: continue
            cost=qty*e+fee(qty,e)
            if cost>cash: continue
            cash-=cost; pos[sym]={'symbol':sym,'date':dt,'entry':e,'stop':stop,'risk':risk,'qty':qty,'cost':cost,'peak_r':0.0}
        curve.append((dt,cash+sum(x['qty']*float(cache[k].loc[dt,'close']) for k,x in pos.items() if dt in cache[k].index)))
    for sym,p in list(pos.items()):
        d=cache[sym]; fd=d.loc[:end].index[-1]; px=float(d.loc[fd,'close']); proceeds,t=realize(p,p['qty'],px,fd,'end_of_test'); cash+=proceeds; trades.append(t)
    tr=pd.DataFrame(trades,columns=['entry_date','exit_date','symbol','entry','exit','qty','pnl','exit_reason'])
    eq=pd.Series({d:v for d,v in curve}).sort_index(); final=cash
    dd=float((eq/eq.cummax()-1).min()) if len(eq) else 0
    years=max((eq.index[-1]-eq.index[0]).days/365.25,1/365.25) if len(eq) else 1
    cagr=(final/INITIAL)**(1/years)-1; daily=eq.pct_change().dropna(); sharpe=float(np.sqrt(252)*daily.mean()/daily.std()) if daily.std()>0 else 0
    wins=tr.loc[tr.pnl>0,'pnl']; losses=tr.loc[tr.pnl<0,'pnl']; pf=float(wins.sum()/abs(losses.sum())) if len(losses) else np.inf
    return {'start':str(start.date()),'end':str(end.date()),'near_high':near,'tighten':tighten,'vol_mult':vm,'threshold_r':threshold,'return_pct':100*(final/INITIAL-1),'cagr_pct':100*cagr,'max_drawdown_pct':100*dd,'trades':len(tr),'win_rate_pct':100*len(wins)/len(tr) if len(tr) else 0,'profit_factor':pf,'sharpe':sharpe,'net_pnl':float(tr.pnl.sum()) if len(tr) else 0},tr

def main():
    root=Path('data'); market=rb.load(root/'NIFTY50.csv'); rb.MARKET_CACHE=market
    files=[p for p in root.glob('*.csv') if p.name!='NIFTY50.csv']; rb.get_cache(files,root/'NIFTY50.csv'); rb.build_rs_rank(files,market)
    rows=[]; alltr=[]
    for near,tighten,vm in CONFIGS:
        for a,b in WINDOWS:
            for th in THRESHOLDS:
                r,tr=run(files,market,pd.Timestamp(a),pd.Timestamp(b),near,tighten,vm,th); rows.append(r)
                tr['start']=a; tr['end']=b; tr['near_high']=near; tr['tighten']=tighten; tr['vol_mult']=vm; tr['threshold_r']=th; alltr.append(tr)
    out=Path('results'); out.mkdir(exist_ok=True); df=pd.DataFrame(rows); df.to_csv(out/'delayed_50dma_threshold_grid_results.csv',index=False); pd.concat(alltr,ignore_index=True).to_csv(out/'delayed_50dma_threshold_grid_trades.csv',index=False)
    summary=df.groupby('threshold_r').agg(return_mean=('return_pct','mean'),return_median=('return_pct','median'),cagr_mean=('cagr_pct','mean'),drawdown_mean=('max_drawdown_pct','mean'),pf_median=('profit_factor','median'),sharpe_mean=('sharpe','mean'),trades_mean=('trades','mean'),net_pnl_mean=('net_pnl','mean')).reset_index()
    summary.to_csv(out/'delayed_50dma_threshold_grid_summary.csv',index=False); print(summary.to_string(index=False))
    oos=df[df.end=='2026-03-31'].copy(); print('\n2025-2026:'); print(oos.groupby('threshold_r').agg(return_mean=('return_pct','mean'),pf_median=('profit_factor','median'),drawdown_mean=('max_drawdown_pct','mean'),sharpe_mean=('sharpe','mean'),trades_mean=('trades','mean')).reset_index().to_string(index=False))
if __name__=='__main__': main()
