"""Short-term momentum backtest: fixed 15-day and 30-day holding limits."""
from pathlib import Path
import numpy as np
import pandas as pd

INITIAL=1_000_000.0; COST_BPS=10.0; RISK_PCT=0.005; MAX_POSITIONS=10
START=pd.Timestamp("2021-04-01"); END=pd.Timestamp("2026-03-31")

def load(p):
    d=pd.read_csv(p); d["timestamp"]=pd.to_datetime(d["timestamp"]); d=d.sort_values("timestamp").set_index("timestamp")
    for c in ["open","high","low","close","volume"]: d[c]=pd.to_numeric(d[c],errors="coerce")
    return d.dropna(subset=["open","high","low","close","volume"])

def make_signals(d, market, rs_rank):
    c,h,l,v=d.close,d.high,d.low,d.volume
    ma50=c.rolling(50).mean(); ma150=c.rolling(150).mean(); ma200=c.rolling(200).mean()
    high60=h.rolling(60).max(); r10=(h.rolling(10).max()-l.rolling(10).min())/c
    r20=(h.rolling(20).max()-l.rolling(20).min())/c; r40=(h.rolling(40).max()-l.rolling(40).min())/c
    vol20=v.rolling(20).mean(); pivot=h.rolling(20).max().shift(1); vol20_prev=vol20.shift(1)
    rs=c/market.close.reindex(d.index).ffill(); rsma=rs.rolling(50).mean(); atr=(h-l).rolling(14).mean(); low10=l.rolling(10).min()
    trend=(c>ma150)&(c>ma200)&(ma150>ma200)&(c>ma50)
    setup=trend&(c/high60>=.85)&(r10<r20*.65)&(r20<r40*.90)&(v/vol20<.75)&((c/pivot-1)<=.03)
    loc=(c-l)/(h-l).replace(0,np.nan)
    breakout=setup.shift(1,fill_value=False)&(c>pivot)&(v>=vol20_prev*1.5)&(rs>rsma)&(loc>=.70)&(rs_rank>=.70)
    idx=np.flatnonzero(breakout.shift(1,fill_value=False).to_numpy()); out=[]
    for j in idx:
        e=float(d.open.iloc[j]); stop=min(float(low10.iloc[j-1]),float(c.iloc[j-1]-1.5*atr.iloc[j-1]))
        if np.isfinite(e) and np.isfinite(stop) and stop<e: out.append((d.index[j],e,stop))
    return out

def simulate(cache,market,signals,max_days):
    market_ok=(market.close>market.close.rolling(200).mean()).shift(1).fillna(False)
    sigs=sorted((dt,sym,e,s) for sym,a in signals.items() for dt,e,s in a if START<=dt<=END and bool(market_ok.get(dt,False)))
    by={}; [by.setdefault(x[0],[]).append(x) for x in sigs]
    cash=INITIAL; pos={}; trades=[]; curve=[]; dates=market.index[(market.index>=START)&(market.index<=END)]
    for dt in dates:
        for sym in list(pos):
            p=pos[sym]; d=cache[sym]
            if dt not in d.index: continue
            row=d.loc[dt]; hold=d.index.get_loc(dt)-d.index.get_loc(p["entry_date"]); active=p["stop"] if p["stage"]==0 else p["locked_stop"]
            exit_px=None; reason=None
            if float(row.open)<=active: exit_px=float(row.open); reason="stop_gap"
            elif float(row.low)<=active: exit_px=active; reason="trailing_stop"
            elif hold>=max_days: exit_px=float(row.close); reason=f"time_{max_days}d"
            if exit_px is not None:
                proceeds=p["qty"]*exit_px; fee=(p["qty"]*p["entry"]+proceeds)*COST_BPS/10000; pnl=proceeds-p["cost"]-fee; cash+=proceeds-fee
                trades.append({**p["diag"],"exit_date":dt,"exit":exit_px,"pnl":pnl,"hold_days":hold,"exit_reason":reason,"r_multiple":pnl/(p["qty"]*p["risk"])})
                del pos[sym]; continue
        # Update stop only after close; it becomes active next session.
        for sym,p in pos.items():
            d=cache[sym]; close=float(d.loc[dt,"close"]); r=p["risk"]; progress=(close-p["entry"])/r
            if progress>=2:
                atr=float((d.high-d.low).rolling(14).mean().loc[dt]); low10=float(d.low.rolling(10).min().loc[dt])
                trail=max(low10,close-2*atr) if np.isfinite(atr) else low10
                p["stage"]=3; p["locked_stop"]=max(p["locked_stop"],p["entry"]+r,trail)
            elif progress>=1.5:
                p["stage"]=2; p["locked_stop"]=max(p["locked_stop"],p["entry"]+.5*r)
            elif progress>=1:
                p["stage"]=1; p["locked_stop"]=max(p["locked_stop"],p["entry"])
        for _,sym,e,stop in by.get(dt,[]):
            if sym in pos or len(pos)>=MAX_POSITIONS: continue
            risk=e-stop; equity=cash+sum(p["qty"]*float(cache[s].loc[dt,"close"]) for s,p in pos.items() if dt in cache[s].index)
            qty=min(int(equity*RISK_PCT/risk),int(cash/(e*(1+COST_BPS/10000)))) if risk>0 else 0
            if qty<=0: continue
            fee=qty*e*COST_BPS/10000; cost=qty*e+fee
            if cost>cash: continue
            cash-=cost; pos[sym]={"entry_date":dt,"entry":e,"stop":stop,"risk":risk,"qty":qty,"cost":cost,"stage":0,"locked_stop":stop,
                "diag":{"symbol":sym,"entry_date":dt,"entry":e,"initial_stop":stop,"initial_risk":risk}}
        curve.append((dt,cash+sum(p["qty"]*float(cache[s].loc[dt,"close"]) for s,p in pos.items() if dt in cache[s].index)))
    for sym,p in list(pos.items()):
        d=cache[sym]; dt=d.loc[:END].index[-1]; px=float(d.loc[dt,"close"]); proceeds=p["qty"]*px; fee=(p["qty"]*p["entry"]+proceeds)*COST_BPS/10000
        cash+=proceeds-fee; hold=d.index.get_loc(dt)-d.index.get_loc(p["entry_date"]); pnl=proceeds-p["cost"]-fee
        trades.append({**p["diag"],"exit_date":dt,"exit":px,"pnl":pnl,"hold_days":hold,"exit_reason":"end_of_test","r_multiple":pnl/(p["qty"]*p["risk"])})
    tr=pd.DataFrame(trades); eq=pd.Series(dict(curve)).sort_index(); wins=tr.loc[tr.pnl>0,"pnl"]; losses=tr.loc[tr.pnl<0,"pnl"]
    daily=eq.pct_change().dropna(); dd=float((eq/eq.cummax()-1).min()) if len(eq) else 0; pf=float(wins.sum()/abs(losses.sum())) if len(losses) else np.inf
    sharpe=float(np.sqrt(252)*daily.mean()/daily.std()) if len(daily)>1 and daily.std()>0 else 0
    return {"holding_limit_days":max_days,"final_capital":cash,"return_pct":(cash/INITIAL-1)*100,"max_drawdown_pct":dd*100,"trades":len(tr),"win_rate_pct":len(wins)/len(tr)*100 if len(tr) else 0,"profit_factor":pf,"sharpe":sharpe},tr

def main():
    root=Path("data"); market=load(root/"NIFTY50.csv"); files=[p for p in root.glob("*.csv") if p.name!="NIFTY50.csv"]
    cache={p.stem:load(p) for p in files}; cache={s:d for s,d in cache.items() if len(d)>=250}
    rs=pd.DataFrame({s:d.close/market.close.reindex(d.index).ffill() for s,d in cache.items()}); rank=rs.rank(axis=1,pct=True)
    signals={s:make_signals(d,market,rank[s].reindex(d.index)) for s,d in cache.items()}
    out=Path("results_short_term"); out.mkdir(exist_ok=True); rows=[]
    for days in (15,30):
        result,tr=simulate(cache,market,signals,days); result["period_start"]=str(START.date()); result["period_end"]=str(END.date()); rows.append(result); tr.to_csv(out/f"trades_{days}d.csv",index=False)
        if not tr.empty:
            tr.assign(entry_year=pd.to_datetime(tr.entry_date).dt.year).groupby("entry_year").agg(trades=("pnl","size"),pnl=("pnl","sum"),win_rate=("pnl",lambda x:(x>0).mean()*100),avg_r=("r_multiple","mean")).reset_index().to_csv(out/f"yearly_{days}d.csv",index=False)
        print(pd.DataFrame([result]).to_string(index=False))
    pd.DataFrame(rows).to_csv(out/"performance.csv",index=False)

if __name__=="__main__": main()
