from __future__ import annotations
from pathlib import Path
import itertools
import numpy as np
import pandas as pd

INITIAL=1_000_000.0; COST_BPS=10.0; RISK_PCT=0.005

DATA_CACHE={}
FEATURE_CACHE={}
SIGNAL_CACHE={}
MARKET_CACHE=None
MARKET_OK_CACHE={}


def load(p):
    d=pd.read_csv(p); d.timestamp=pd.to_datetime(d.timestamp); d=d.sort_values('timestamp').set_index('timestamp')
    for c in ['open','high','low','close','volume']: d[c]=pd.to_numeric(d[c],errors='coerce')
    return d.dropna(subset=['open','high','low','close','volume'])


def get_cache(files, market_path):
    global MARKET_CACHE
    if MARKET_CACHE is None:
        MARKET_CACHE=load(market_path)
    for p in files:
        if p.stem not in DATA_CACHE:
            DATA_CACHE[p.stem]=load(p)
    return DATA_CACHE, MARKET_CACHE


def build_features(sym,d,market):
    if sym in FEATURE_CACHE:
        return FEATURE_CACHE[sym]
    c,h,l,v=d.close,d.high,d.low,d.volume
    ma50=c.rolling(50).mean(); ma150=c.rolling(150).mean(); ma200=c.rolling(200).mean()
    high60=h.rolling(60).max()
    r5=(h.rolling(5).max()-l.rolling(5).min())/c
    r10=(h.rolling(10).max()-l.rolling(10).min())/c
    r20=(h.rolling(20).max()-l.rolling(20).min())/c
    r30=(h.rolling(30).max()-l.rolling(30).min())/c
    r40=(h.rolling(40).max()-l.rolling(40).min())/c
    vol20=v.rolling(20).mean(); pivot=h.rolling(20).max().shift(1); vol20_prev=vol20.shift(1)
    rs=c/market.close.reindex(d.index).ffill(); rsma=rs.rolling(50).mean(); atr14=(h-l).rolling(14).mean(); low10=l.rolling(10).min()
    peak=c.cummax(); trail20=l.rolling(20).min()
    f={'trend':(c>ma150)&(c>ma200)&(ma150>ma200)&(c>ma50),'near_base':c/high60,
       'r5':r5,'r10':r10,'r20':r20,'r30':r30,'r40':r40,'dry_volume_ratio':v/vol20,
       'pivot':pivot,'vol20_prev':vol20_prev,'rs':rs,'rsma':rsma,'atr14':atr14,
       'low10':low10,'ma50':ma50,'peak':peak,'trail20':trail20}
    FEATURE_CACHE[sym]=f
    return f


def signals(sym,d,market,near,tighten,vol_mult):
    key=(sym,float(near),float(tighten),float(vol_mult))
    if key in SIGNAL_CACHE:
        return SIGNAL_CACHE[key]
    f=build_features(sym,d,market)
    # Multi-contraction VCP quality gate: the short-term range must contract
    # versus the intermediate range, and the intermediate range must itself
    # contract versus the broader base. This is deliberately parameter-free;
    # the existing 27-point robustness grid remains the only parameter sweep.
    contraction=(f['r10'] < f['r20']*tighten) & (f['r20'] < f['r40']*0.90)
    # Keep the pivot area tight: avoid buying a breakout after an already-large
    # extension from the recent 20-day base.
    pivot_distance=(d.close/f['pivot']-1.0).clip(lower=-np.inf)
    setup=(f['trend']&(f['near_base']>=near)&contraction&(f['dry_volume_ratio']<0.75)&(pivot_distance<=0.03))
    breakout=(setup.shift(1,fill_value=False)&(d.close>f['pivot'])&(d.volume>=f['vol20_prev']*vol_mult)&(f['rs']>f['rsma']))
    entry_mask=breakout.shift(1,fill_value=False)
    idx=np.flatnonzero(entry_mask.to_numpy())
    if len(idx):
        entries=d.open.iloc[idx].to_numpy(dtype=float); bi=idx-1
        atr=f['atr14'].iloc[bi].to_numpy(dtype=float); lows=f['low10'].iloc[bi].to_numpy(dtype=float); closes=d.close.iloc[bi].to_numpy(dtype=float)
        stops=np.minimum(lows,closes-1.5*atr); dates=d.index[idx]
        valid=np.isfinite(entries)&np.isfinite(stops)&(stops<entries)
        out=[(dt,float(e),float(s)) for dt,e,s,ok in zip(dates,entries,stops,valid) if ok]
    else: out=[]
    SIGNAL_CACHE[key]=out
    return out


def run(files,market,start,end,near=.85,tighten=.65,vol_mult=1.5):
    cache,_=get_cache(files,None) if market is None else (DATA_CACHE, market)
    key=(str(start),str(end))
    if key not in MARKET_OK_CACHE:
        md=market.loc[:end]; MARKET_OK_CACHE[key]=(md.close>md.close.rolling(200).mean()).shift(1).fillna(False)
    market_ok=MARKET_OK_CACHE[key]
    sigs=[]
    for sym,d in cache.items():
        for dt,e,stop in signals(sym,d,market,near,tighten,vol_mult):
            if start<=dt<=end and bool(market_ok.reindex([dt]).fillna(False).iloc[0]): sigs.append((dt,sym,e,stop))
    sigs.sort(); cash=INITIAL; pos={}; trades=[]; curve=[]
    for dt in market.index[(market.index>=start)&(market.index<=end)]:
        for sym in list(pos):
            d=cache[sym]; px=float(d.loc[dt,'close']) if dt in d.index else np.nan
            if not np.isfinite(px): continue
            f=FEATURE_CACHE[sym]; ma50=float(f['ma50'].loc[dt]); stop=pos[sym]['stop']; peak=float(f['peak'].loc[dt]); trail=float(f['trail20'].loc[dt])
            r=pos[sym]['risk']; effective_stop=max(stop,trail) if peak>=pos[sym]['entry']+2*r else stop
            if px<=effective_stop or (np.isfinite(ma50) and px<ma50):
                proceeds=pos[sym]['qty']*px; fee=(pos[sym]['qty']*pos[sym]['entry']+proceeds)*COST_BPS/10000; pnl=proceeds-pos[sym]['cost']-fee; cash+=proceeds-fee
                trades.append((pos[sym]['date'],dt,sym,pos[sym]['entry'],px,pos[sym]['qty'],pnl)); del pos[sym]
        for sdt,sym,e,stop in sigs:
            if sdt!=dt or sym in pos or len(pos)>=10: continue
            risk=e-stop; equity=cash+sum(x['qty']*float(cache[k].loc[dt,'close']) for k,x in pos.items() if dt in cache[k].index)
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
    global MARKET_CACHE
    root=Path('data'); market_path=root/'NIFTY50.csv'; MARKET_CACHE=load(market_path)
    files=[p for p in root.glob('*.csv') if p.name!='NIFTY50.csv']
    if not files: raise RuntimeError('No stock CSV files found')
    cache,_=get_cache(files,market_path)
    rows=[]; configs=list(itertools.product([.80,.85,.90],[.55,.65,.75],[1.25,1.50,1.75])); windows=[(pd.Timestamp('2019-01-01'),pd.Timestamp('2022-12-31')),(pd.Timestamp('2023-01-01'),pd.Timestamp('2024-12-31')),(pd.Timestamp('2025-01-01'),pd.Timestamp('2026-03-31'))]
    for near,tighten,vm in configs:
        for sym,d in cache.items(): signals(sym,d,MARKET_CACHE,near,tighten,vm)
    for near,tighten,vm in configs:
        for start,end in windows: rows.append(run(files,MARKET_CACHE,start,end,near,tighten,vm)[0])
    out=Path('results'); out.mkdir(exist_ok=True); df=pd.DataFrame(rows); df.to_csv(out/'robustness_grid.csv',index=False)
    base,tr,eq=run(files,MARKET_CACHE,pd.Timestamp('2025-04-01'),pd.Timestamp('2026-03-31')); pd.DataFrame([base]).to_csv(out/'robustness_baseline.csv',index=False); tr.to_csv(out/'robustness_trades.csv',index=False); eq.rename('equity').to_csv(out/'equity_curve.csv')
    b=MARKET_CACHE.loc['2025-04-01':'2026-03-31','close']; bench=(b.iloc[-1]/b.iloc[0]-1)*100 if len(b)>1 else 0
    pd.DataFrame([{'strategy_return_pct':base['return_pct'],'nifty_buy_hold_return_pct':bench,'strategy_minus_benchmark_pct':base['return_pct']-bench}]).to_csv(out/'benchmark.csv',index=False)
    print(df.groupby(['start','end']).agg(return_mean=('return_pct','mean'),return_median=('return_pct','median'),trades_mean=('trades','mean'),pf_median=('profit_factor','median')).to_string()); print('BASELINE',base,'BENCHMARK',bench)
if __name__=='__main__': main()
