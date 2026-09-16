from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import robustness_backtest as rb

INITIAL=1_000_000.0; COST_BPS=10.0; RISK_PCT=0.005; MAX_POSITIONS=10
CONFIGS=[(.80,.65,1.5),(.85,.65,1.5),(.90,.65,1.5),(.85,.65,1.75)]
WINDOWS=[('2019-01-01','2022-12-31'),('2023-01-01','2024-12-31'),('2025-01-01','2026-03-31'),('2021-04-01','2026-03-31')]
TARGETS=[0.30,0.35,0.40,0.45,0.50]

# Pre-specified exit research. Entries, universe, sizing and costs are unchanged.
# Fixed profit targets are measured from entry price. Daily OHLC ambiguity is
# handled conservatively: if both stop and target are touched on the same day,
# the stop is assumed to occur first.
VARIANTS={
 'baseline_stop_50dma': {'mode':'baseline'},
 'breakeven_after_1r': {'mode':'breakeven','trigger_r':1.0},
 'breakeven_after_1_5r': {'mode':'breakeven','trigger_r':1.5},
 'trail20_after_1r': {'mode':'trail20','trigger_r':1.0},
 'trail20_after_2r': {'mode':'trail20','trigger_r':2.0},
 'trail10_after_1r': {'mode':'trail10','trigger_r':1.0},
 'atr3_after_1r': {'mode':'atr','trigger_r':1.0,'atr_mult':3.0},
 'atr2_after_1r': {'mode':'atr','trigger_r':1.0,'atr_mult':2.0},
 'atr3_after_2r': {'mode':'atr','trigger_r':2.0,'atr_mult':3.0},
 'trail20_plus_breakeven': {'mode':'trail20_be','trigger_r':1.0},
 'time60_if_below_05r': {'mode':'time','days':60,'min_r':0.5},
}
for target in TARGETS:
    VARIANTS[f'profit_target_{int(target*100)}pct']={'mode':'profit_target','target_pct':target}


def _close_trade(pos, px, dt, reason):
    proceeds=pos['qty']*px
    fee=(pos['qty']*pos['entry']+proceeds)*COST_BPS/10000
    pnl=proceeds-pos['cost']-fee
    return proceeds-fee, (pos['date'],dt,pos['symbol'],pos['entry'],px,pos['qty'],pnl,reason)


def run_variant(files, market, start, end, near, tighten, vm, variant, winner_symbols=None):
    rb.build_rs_rank(files, market)
    cache=rb.DATA_CACHE
    key=(str(start),str(end))
    if key not in rb.MARKET_OK_CACHE:
        md=market.loc[:end]; rb.MARKET_OK_CACHE[key]=(md.close>md.close.rolling(200).mean()).shift(1).fillna(False)
    market_ok=rb.MARKET_OK_CACHE[key]
    sigs=[]
    for sym,d in cache.items():
        if winner_symbols is not None and sym not in winner_symbols: continue
        for dt,e,stop in rb.signals(sym,d,market,near,tighten,vm):
            if start<=dt<=end and bool(market_ok.reindex([dt]).fillna(False).iloc[0]): sigs.append((dt,sym,e,stop))
    sigs.sort(); cash=INITIAL; pos={}; trades=[]; curve=[]; cfg=VARIANTS[variant]
    for dt in market.index[(market.index>=start)&(market.index<=end)]:
        for sym in list(pos):
            d=cache[sym]
            if dt not in d.index: continue
            row=d.loc[dt]; px=float(row.close); high=float(row.high); low=float(row.low)
            f=rb.FEATURE_CACHE[sym]; ma50=float(f['ma50'].loc[dt]); stop=pos[sym]['stop']; r=pos[sym]['risk']
            peak_r=float(pos[sym]['peak_r']); effective_stop=stop; reason='stop_or_50dma'
            mode=cfg['mode']

            # Stop is checked from the day's low. A same-day stop+target touch
            # is resolved as stop-first because daily bars cannot reveal order.
            if low <= effective_stop:
                proceeds,trade=_close_trade(pos[sym], effective_stop, dt, 'initial_stop')
                cash+=proceeds; trades.append(trade); del pos[sym]; continue

            if mode=='profit_target':
                target_px=pos[sym]['entry']*(1.0+cfg['target_pct'])
                if high >= target_px:
                    proceeds,trade=_close_trade(pos[sym], target_px, dt, f"profit_target_{int(cfg['target_pct']*100)}pct")
                    cash+=proceeds; trades.append(trade); del pos[sym]; continue
            else:
                if peak_r>=1.0 and mode=='breakeven': effective_stop=max(effective_stop,pos[sym]['entry']); reason='breakeven_after_1r'
                elif peak_r>=cfg.get('trigger_r',999) and mode=='trail20': effective_stop=max(effective_stop,float(f['trail20'].loc[dt])); reason='20d_trail'
                elif peak_r>=cfg.get('trigger_r',999) and mode=='trail10': effective_stop=max(effective_stop,float(f['low10'].loc[dt])); reason='10d_trail'
                elif peak_r>=cfg.get('trigger_r',999) and mode=='atr': effective_stop=max(effective_stop,px-cfg['atr_mult']*float(f['atr14'].loc[dt])); reason=f"atr{cfg['atr_mult']}_trail"
                elif peak_r>=cfg.get('trigger_r',999) and mode=='trail20_be': effective_stop=max(effective_stop,pos[sym]['entry'],float(f['trail20'].loc[dt])); reason='20d_trail_plus_breakeven'
                elif mode=='time':
                    held=(dt-pos[sym]['date']).days
                    if held>=cfg['days'] and peak_r<cfg['min_r']:
                        proceeds,trade=_close_trade(pos[sym],px,dt,'time_stop_60d'); cash+=proceeds; trades.append(trade); del pos[sym]; continue
                if low <= effective_stop:
                    proceeds,trade=_close_trade(pos[sym],effective_stop,dt,reason); cash+=proceeds; trades.append(trade); del pos[sym]; continue
                if np.isfinite(ma50) and px<ma50:
                    proceeds,trade=_close_trade(pos[sym],px,dt,'50dma_close'); cash+=proceeds; trades.append(trade); del pos[sym]; continue

            pos[sym]['peak_r']=max(peak_r,(high-pos[sym]['entry'])/r)
        for sdt,sym,e,stop in sigs:
            if sdt!=dt or sym in pos or len(pos)>=MAX_POSITIONS: continue
            risk=e-stop; equity=cash+sum(x['qty']*float(cache[k].loc[dt,'close']) for k,x in pos.items() if dt in cache[k].index)
            qty=min(int(equity*RISK_PCT/risk),int(cash/(e*(1+COST_BPS/10000)))) if risk>0 else 0
            if qty<=0: continue
            fee=qty*e*COST_BPS/10000; cost=qty*e+fee
            if cost>cash: continue
            cash-=cost; pos[sym]={'symbol':sym,'date':dt,'entry':e,'stop':stop,'risk':risk,'qty':qty,'cost':cost,'peak_r':0.0}
        curve.append((dt,cash+sum(x['qty']*float(cache[k].loc[dt,'close']) for k,x in pos.items() if dt in cache[k].index)))
    for sym,p in list(pos.items()):
        d=cache[sym]; final_dt=d.loc[:end].index[-1]; px=float(d.loc[final_dt,'close']); proceeds,trade=_close_trade(p,px,final_dt,'end_of_test'); cash+=proceeds; trades.append(trade)
    tr=pd.DataFrame(trades,columns=['entry_date','exit_date','symbol','entry','exit','qty','pnl','exit_reason'])
    eq=pd.Series({d:v for d,v in curve}).sort_index(); final=cash
    dd=float((eq/eq.cummax()-1).min()) if len(eq) else 0; years=max((eq.index[-1]-eq.index[0]).days/365.25,1/365.25) if len(eq) else 1
    cagr=(final/INITIAL)**(1/years)-1; daily=eq.pct_change().dropna(); sharpe=float(np.sqrt(252)*daily.mean()/daily.std()) if daily.std()>0 else 0
    wins=tr.loc[tr.pnl>0,'pnl']; losses=tr.loc[tr.pnl<0,'pnl']; pf=float(wins.sum()/abs(losses.sum())) if len(losses) else np.inf
    return {'start':str(start.date()),'end':str(end.date()),'near_high':near,'tighten':tighten,'vol_mult':vm,'exit_variant':variant,'universe':'WINNER_ONLY' if winner_symbols is not None else 'ALL_STOCKS','return_pct':(final/INITIAL-1)*100,'cagr_pct':cagr*100,'max_drawdown_pct':dd*100,'trades':len(tr),'win_rate_pct':len(wins)/len(tr)*100 if len(tr) else 0,'profit_factor':pf,'sharpe':sharpe,'net_pnl':float(tr.pnl.sum()) if len(tr) else 0},tr


def main():
    root=Path('data'); market=rb.load(root/'NIFTY50.csv'); rb.MARKET_CACHE=market
    files=[p for p in root.glob('*.csv') if p.name!='NIFTY50.csv']
    rb.get_cache(files,root/'NIFTY50.csv'); rb.build_rs_rank(files,market)
    cls_path=Path('results/stock_winner_loser_classification.csv')
    winner_symbols=None
    if cls_path.exists():
        cls=pd.read_csv(cls_path)
        winner_symbols=set(cls.loc[cls.historical_label=='WINNER','symbol'].astype(str))
    rows=[]; trades=[]
    # Target experiment is run both on the full universe and the historical-WINNER subset.
    for near,tighten,vm in CONFIGS:
        for a,b in WINDOWS:
            for variant in VARIANTS:
                for universe in [None, winner_symbols] if winner_symbols else [None]:
                    r,tr=run_variant(files,market,pd.Timestamp(a),pd.Timestamp(b),near,tighten,vm,variant,universe); rows.append(r)
                    tr['start']=a; tr['end']=b; tr['near_high']=near; tr['tighten']=tighten; tr['vol_mult']=vm; tr['exit_variant']=variant; tr['universe']=r['universe']; trades.append(tr)
    out=Path('results'); out.mkdir(exist_ok=True); df=pd.DataFrame(rows); df.to_csv(out/'exit_management_grid.csv',index=False); pd.concat(trades,ignore_index=True).to_csv(out/'exit_management_trades.csv',index=False)
    summary=df.groupby(['universe','exit_variant']).agg(return_mean=('return_pct','mean'),return_median=('return_pct','median'),cagr_mean=('cagr_pct','mean'),dd_mean=('max_drawdown_pct','mean'),pf_median=('profit_factor','median'),sharpe_mean=('sharpe','mean'),trades_mean=('trades','mean'),net_pnl_mean=('net_pnl','mean')).reset_index()
    summary.to_csv(out/'exit_management_summary.csv',index=False); print('EXIT MANAGEMENT SUMMARY'); print(summary.to_string(index=False))
    target=df[df.exit_variant.str.startswith('profit_target_')].copy()
    target.to_csv(out/'profit_target_30_50_results.csv',index=False)
    target_trades=pd.concat(trades,ignore_index=True); target_trades=target_trades[target_trades.exit_variant.str.startswith('profit_target_')]
    target_trades.to_csv(out/'profit_target_30_50_trades.csv',index=False)
    winner_target=target[target.universe=='WINNER_ONLY'].copy(); winner_target.to_csv(out/'winner_stock_profit_target_results.csv',index=False)
    winner_target_trades=target_trades[target_trades.universe=='WINNER_ONLY']; winner_target_trades.to_csv(out/'winner_stock_profit_target_trades.csv',index=False)
    print('\nPROFIT TARGET 30-50% — WINNER ONLY')
    print(winner_target[['start','end','near_high','tighten','vol_mult','exit_variant','return_pct','net_pnl','trades','win_rate_pct','profit_factor','max_drawdown_pct']].to_string(index=False))

if __name__=='__main__': main()
