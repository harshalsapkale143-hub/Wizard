from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import robustness_backtest as rb

INITIAL=1_000_000.0
COST_BPS=10.0
RISK_PCT=0.005
MAX_POSITIONS=10
CONFIGS=[(.80,.65,1.5),(.85,.65,1.5),(.90,.65,1.5),(.85,.65,1.75)]
WINDOWS=[('2019-01-01','2022-12-31'),('2023-01-01','2024-12-31'),('2025-01-01','2026-03-31'),('2021-04-01','2026-03-31')]

# Focused, pre-specified test of the proposed winner-preservation idea.
# All entries, sizing, costs and initial stops remain unchanged.
VARIANTS={
    'baseline': {'trigger_r': 999.0, 'trail': None, 'keep_50dma': True},
    'trail20_after_1r': {'trigger_r': 1.0, 'trail': 'trail20', 'keep_50dma': False},
    'trail20_after_2r': {'trigger_r': 2.0, 'trail': 'trail20', 'keep_50dma': False},
    'atr3_after_1r': {'trigger_r': 1.0, 'trail': 'atr3', 'keep_50dma': False},
    'atr3_after_2r': {'trigger_r': 2.0, 'trail': 'atr3', 'keep_50dma': False},
}

def close_trade(pos, px, dt, reason):
    proceeds=pos['qty']*px
    fee=(pos['qty']*pos['entry']+proceeds)*COST_BPS/10000
    pnl=proceeds-pos['cost']-fee
    return proceeds-fee, (pos['date'],dt,pos['symbol'],pos['entry'],px,pos['qty'],pnl,reason)

def run_variant(files, market, start, end, near, tighten, vm, name):
    rb.build_rs_rank(files, market)
    cache=rb.DATA_CACHE
    key=(str(start),str(end))
    if key not in rb.MARKET_OK_CACHE:
        md=market.loc[:end]
        rb.MARKET_OK_CACHE[key]=(md.close>md.close.rolling(200).mean()).shift(1).fillna(False)
    market_ok=rb.MARKET_OK_CACHE[key]
    sigs=[]
    for sym,d in cache.items():
        for dt,e,stop in rb.signals(sym,d,market,near,tighten,vm):
            if start<=dt<=end and bool(market_ok.reindex([dt]).fillna(False).iloc[0]):
                sigs.append((dt,sym,e,stop))
    sigs.sort(); cash=INITIAL; pos={}; trades=[]; curve=[]; cfg=VARIANTS[name]
    for dt in market.index[(market.index>=start)&(market.index<=end)]:
        for sym in list(pos):
            d=cache[sym]
            if dt not in d.index: continue
            row=d.loc[dt]; px=float(row.close); high=float(row.high); low=float(row.low)
            f=rb.FEATURE_CACHE[sym]; ma50=float(f['ma50'].loc[dt]); base_stop=pos[sym]['stop']; r=pos[sym]['risk']
            peak_r=float(pos[sym]['peak_r']); effective_stop=base_stop; reason='initial_stop'

            # Once the trade has earned the trigger R, stop using the 50DMA as an
            # immediate close signal and protect the winner with a trend/ATR trail.
            if peak_r >= cfg['trigger_r']:
                if cfg['trail']=='trail20':
                    effective_stop=max(effective_stop,float(f['trail20'].loc[dt])); reason='20d_trail_winner'
                elif cfg['trail']=='atr3':
                    effective_stop=max(effective_stop,px-3.0*float(f['atr14'].loc[dt])); reason='atr3_trail_winner'

            if low <= effective_stop:
                proceeds,trade=close_trade(pos[sym],effective_stop,dt,reason)
                cash+=proceeds; trades.append(trade); del pos[sym]; continue

            # Before the trigger, preserve the original regime exit. After the
            # trigger, deliberately ignore a routine close below the 50DMA.
            if cfg['keep_50dma'] or peak_r < cfg['trigger_r']:
                if np.isfinite(ma50) and px<ma50:
                    proceeds,trade=close_trade(pos[sym],px,dt,'50dma_close')
                    cash+=proceeds; trades.append(trade); del pos[sym]; continue

            pos[sym]['peak_r']=max(peak_r,(high-pos[sym]['entry'])/r)

        for sdt,sym,e,stop in sigs:
            if sdt!=dt or sym in pos or len(pos)>=MAX_POSITIONS: continue
            risk=e-stop
            equity=cash+sum(x['qty']*float(cache[k].loc[dt,'close']) for k,x in pos.items() if dt in cache[k].index)
            qty=min(int(equity*RISK_PCT/risk),int(cash/(e*(1+COST_BPS/10000)))) if risk>0 else 0
            if qty<=0: continue
            fee=qty*e*COST_BPS/10000; cost=qty*e+fee
            if cost>cash: continue
            cash-=cost
            pos[sym]={'symbol':sym,'date':dt,'entry':e,'stop':stop,'risk':risk,'qty':qty,'cost':cost,'peak_r':0.0}
        curve.append((dt,cash+sum(x['qty']*float(cache[k].loc[dt,'close']) for k,x in pos.items() if dt in cache[k].index)))

    for sym,p in list(pos.items()):
        d=cache[sym]; final_dt=d.loc[:end].index[-1]; px=float(d.loc[final_dt,'close'])
        proceeds,trade=close_trade(p,px,final_dt,'end_of_test'); cash+=proceeds; trades.append(trade)

    tr=pd.DataFrame(trades,columns=['entry_date','exit_date','symbol','entry','exit','qty','pnl','exit_reason'])
    eq=pd.Series({d:v for d,v in curve}).sort_index(); final=cash
    dd=float((eq/eq.cummax()-1).min()) if len(eq) else 0
    years=max((eq.index[-1]-eq.index[0]).days/365.25,1/365.25) if len(eq) else 1
    cagr=(final/INITIAL)**(1/years)-1
    daily=eq.pct_change().dropna(); sharpe=float(np.sqrt(252)*daily.mean()/daily.std()) if daily.std()>0 else 0
    wins=tr.loc[tr.pnl>0,'pnl']; losses=tr.loc[tr.pnl<0,'pnl']; pf=float(wins.sum()/abs(losses.sum())) if len(losses) else np.inf
    return {'start':str(start.date()),'end':str(end.date()),'near_high':near,'tighten':tighten,'vol_mult':vm,'exit_variant':name,'return_pct':(final/INITIAL-1)*100,'cagr_pct':cagr*100,'max_drawdown_pct':dd*100,'trades':len(tr),'win_rate_pct':len(wins)/len(tr)*100 if len(tr) else 0,'profit_factor':pf,'sharpe':sharpe,'net_pnl':float(tr.pnl.sum()) if len(tr) else 0},tr

def main():
    root=Path('data'); market=rb.load(root/'NIFTY50.csv'); rb.MARKET_CACHE=market
    files=[p for p in root.glob('*.csv') if p.name!='NIFTY50.csv']
    rb.get_cache(files,root/'NIFTY50.csv'); rb.build_rs_rank(files,market)
    rows=[]; all_trades=[]
    for near,tighten,vm in CONFIGS:
        for a,b in WINDOWS:
            for name in VARIANTS:
                r,tr=run_variant(files,market,pd.Timestamp(a),pd.Timestamp(b),near,tighten,vm,name)
                rows.append(r); tr['start']=a; tr['end']=b; tr['near_high']=near; tr['tighten']=tighten; tr['vol_mult']=vm; tr['exit_variant']=name; all_trades.append(tr)
    out=Path('results'); out.mkdir(exist_ok=True)
    df=pd.DataFrame(rows); df.to_csv(out/'winner_preservation_results.csv',index=False)
    pd.concat(all_trades,ignore_index=True).to_csv(out/'winner_preservation_trades.csv',index=False)
    summary=df.groupby('exit_variant').agg(return_mean=('return_pct','mean'),return_median=('return_pct','median'),cagr_mean=('cagr_pct','mean'),drawdown_mean=('max_drawdown_pct','mean'),pf_median=('profit_factor','median'),sharpe_mean=('sharpe','mean'),trades_mean=('trades','mean'),net_pnl_mean=('net_pnl','mean')).reset_index()
    summary.to_csv(out/'winner_preservation_summary.csv',index=False)
    print('WINNER PRESERVATION SUMMARY'); print(summary.to_string(index=False))

if __name__=='__main__': main()
