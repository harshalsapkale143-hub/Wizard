from __future__ import annotations
from pathlib import Path
import itertools
import numpy as np
import pandas as pd

INITIAL=1_000_000.0; COST_BPS=10.0; RISK_PCT=0.005

def load(p):
    d=pd.read_csv(p); d.timestamp=pd.to_datetime(d.timestamp); d=d.sort_values('timestamp').set_index('timestamp')
    for c in ['open','high','low','close','volume']: d[c]=pd.to_numeric(d[c],errors='coerce')
    return d.dropna(subset=['open','high','low','close','volume'])

def signals(d,market,near,tighten,vol_mult):
    c,h,l,v=d.close,d.high,d.low,d.volume
    ma50=c.rolling(50).mean(); ma150=c.rolling(150).mean(); ma200=c.rolling(200).mean()
    trend=(c>ma150)&(c>ma200)&(ma150>ma200)&(c>ma50)
    high60=h.rolling(60).max(); near_high=c>=high60*near
    r10=(h.rolling(10).max()-l.rolling(10).min())/c; r30=(h.rolling(30).max()-l.rolling(30).min())/c
    setup=trend&near_high&(r10<r30*tighten)&(v<v.rolling(20).mean()*0.75)
    pivot=h.rolling(20).max().shift(1); vol20=v.rolling(20).mean().shift(1)
    rs=c/market.close.reindex(d.index).ffill(); rsma=rs.rolling(50).mean()
    out=[]
    for i in range(1,len(d)-1):
        if not setup.iloc[i-1]: continue
        if d.close.iloc[i]<=pivot.iloc[i] or d.volume.iloc[i]<vol20.iloc[i]*vol_mult or rs.iloc[i]<=rsma.iloc[i]: continue
        dt=d.index[i]; nd=d.index[i+1]; entry=float(d.open.iloc[i+1])
        atr=(d.high-d.low).rolling(14).mean().iloc[i-1]
        stop=min(float(d.low.iloc[max(0,i-10):i].min()),float(d.close.iloc[i]-1.5*atr))
        if np.isfinite(stop) and stop<entry: out.append((nd,entry,stop))
    return out

def run(files,market,start,end,near=.85,tighten=.65,vol_mult=1.5):
    md=market.loc[:end]; market_ok=(md.close>md.close.rolling(200).mean()).shift(1).fillna(False)
    cache={p.stem:load(p) for p in files}; sigs=[]
    for p,d in cache.items():
        for dt,e,stop in signals(d,market,near,tighten,vol_mult):
            if start<=dt<=end and bool(market_ok.reindex([dt]).fillna(False).iloc[0]): sigs.append((dt,p.stem,e,stop))
    sigs.sort(); cash=INITIAL; pos={}; trades=[]; curve=[]
    for dt in market.index[(market.index>=start)&(market.index<=end)]:
        for sym in list(pos):
            d=cache[sym]; px=float(d.loc[dt,'close']) if dt in d.index else np.nan
            if not np.isfinite(px): continue
            ma50=float(d.close.rolling(50).mean().loc[dt]); stop=pos[sym]['stop']
            # Protect gains: trail stop upward with the 20-day low after a 2R move.
            r=pos[sym]['risk']; peak=float(d.close.loc[:dt].max()); trail=float(d.low.loc[:dt].tail(20).min())
            effective_stop=max(stop, trail) if peak>=pos[sym]['entry']+2*r else stop
            if px<=effective_stop or (np.isfinite(ma50) and px<ma50):
                proceeds=pos[sym]['qty']*px; fee=(pos[sym]['qty']*pos[sym]['entry']+proceeds)*COST_BPS/10000
                pnl=proceeds-pos[sym]['cost']-fee; cash+=proceeds-fee
                trades.append((pos[sym]['date'],dt,sym,pos[sym]['entry'],px,pos[sym]['qty'],pnl)); del pos[sym]
        for sdt,sym,e,stop in sigs:
            if sdt!=dt or sym in pos or len(pos)>=10: continue
            risk=e-stop
            equity=cash+sum(x['qty']*float(cache[k].loc[dt,'close']) for k,x in pos.items() if dt in cache[k].index)
            qty=min(int(equity*RISK_PCT/risk),int(cash/(e*(1+COST_BPS/10000)))) if risk>0 else 0
            if qty<=0: continue
            fee=qty*e*COST_BPS/10000; cost=qty*e+fee
            if cost>cash: continue
            cash-=cost; pos[sym]={'date':dt,'entry':e,'stop':stop,'risk':risk,'qty':qty,'cost':cost}
        curve.append((dt,cash+sum(x['qty']*float(cache[k].loc[dt,'close']) for k,x in pos.items() if dt in cache[k].index)))
    for sym,pos0 in list(pos.items()):
        d=cache[sym]; px=float(d.loc[:end].close.iloc[-1]); proceeds=pos0['qty']*px; fee=(pos0['qty']*pos0['entry']+proceeds)*COST_BPS/10000; cash+=proceeds-fee
        trades.append((pos0['date'],d.loc[:end].index[-1],sym,pos0['entry'],px,pos0['qty'],proceeds-pos0['cost']-fee))
    tr=pd.DataFrame(trades,columns=['entry_date','exit_date','symbol','entry','exit','qty','pnl']); eq=pd.Series({d:v for d,v in curve}).sort_index()
    final=cash; dd=float((eq/eq.cummax()-1).min()) if len(eq) else 0; years=max((eq.index[-1]-eq.index[0]).days/365.25,1/365.25) if len(eq) else 1; cagr=(final/INITIAL)**(1/years)-1
    daily=eq.pct_change().dropna(); sharpe=float(np.sqrt(252)*daily.mean()/daily.std()) if daily.std()>0 else 0
    wins=tr.loc[tr.pnl>0,'pnl']; losses=tr.loc[tr.pnl<0,'pnl']; pf=float(wins.sum()/abs(losses.sum())) if len(losses) else np.inf
    return {'start':str(start.date()),'end':str(end.date()),'near_high':near,'tighten':tighten,'vol_mult':vol_mult,'final_capital':final,'return_pct':(final/INITIAL-1)*100,'cagr_pct':cagr*100,'max_drawdown_pct':dd*100,'trades':len(tr),'win_rate_pct':len(wins)/len(tr)*100 if len(tr) else 0,'profit_factor':pf,'sharpe':sharpe},tr,eq

def main():
    root=Path('data'); market=load(root/'NIFTY50.csv'); files=[p for p in root.glob('*.csv') if p.name!='NIFTY50.csv']; rows=[]
    configs=list(itertools.product([.80,.85,.90],[.55,.65,.75],[1.25,1.50,1.75]))
    windows=[(pd.Timestamp('2019-01-01'),pd.Timestamp('2022-12-31')),(pd.Timestamp('2023-01-01'),pd.Timestamp('2024-12-31')),(pd.Timestamp('2025-01-01'),pd.Timestamp('2026-03-31'))]
    for near,tighten,vm in configs:
        for start,end in windows: rows.append(run(files,market,start,end,near,tighten,vm)[0])
    out=Path('results'); out.mkdir(exist_ok=True); df=pd.DataFrame(rows); df.to_csv(out/'robustness_grid.csv',index=False)
    base,tr,eq=run(files,market,pd.Timestamp('2025-04-01'),pd.Timestamp('2026-03-31')); pd.DataFrame([base]).to_csv(out/'robustness_baseline.csv',index=False); tr.to_csv(out/'robustness_trades.csv',index=False); eq.rename('equity').to_csv(out/'equity_curve.csv')
    b=market.loc['2025-04-01':'2026-03-31','close']; bench=(b.iloc[-1]/b.iloc[0]-1)*100 if len(b)>1 else 0
    pd.DataFrame([{'strategy_return_pct':base['return_pct'],'nifty_buy_hold_return_pct':bench,'strategy_minus_benchmark_pct':base['return_pct']-bench}]).to_csv(out/'benchmark.csv',index=False)
    print(df.groupby(['start','end']).agg(return_mean=('return_pct','mean'),return_median=('return_pct','median'),trades_mean=('trades','mean'),pf_median=('profit_factor','median')).to_string()); print('BASELINE',base,'BENCHMARK',bench)
if __name__=='__main__': main()
