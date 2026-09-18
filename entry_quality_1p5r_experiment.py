from pathlib import Path
import numpy as np
import pandas as pd
import robustness_backtest as rb

INITIAL=1_000_000.0; COST_BPS=10.0; RISK_PCT=0.005; MAX_POSITIONS=10
CONFIGS=[("baseline",0.70,0.03,1.50),("rs80",0.80,0.03,1.50),("rs85",0.85,0.03,1.50),("vol175",0.70,0.03,1.75),("vol200",0.70,0.03,2.00),("pivot1",0.70,0.01,1.50),("pivot2",0.70,0.02,1.50),("rs80_vol175",0.80,0.03,1.75)]
WINDOWS=[("2019-01-01","2022-12-31"),("2023-01-01","2024-12-31"),("2025-01-01","2026-03-31"),("2021-04-01","2026-03-31")]

def quality_signals(sym,d,market,rs_min,pivot_max,vol_mult):
    f=rb.build_features(sym,d,market); rank=rb.RS_RANK_CACHE[sym].reindex(d.index)
    contraction=(f["r10"]<f["r20"]*.65)&(f["r20"]<f["r40"]*.90)
    pivot_dist=(d.close/f["pivot"]-1).clip(lower=-np.inf)
    setup=f["trend"]&(f["near_base"]>=.85)&contraction&(f["dry_volume_ratio"]<.75)&(pivot_dist<=pivot_max)
    day_range=(d.high-d.low).replace(0,np.nan); loc=(d.close-d.low)/day_range
    breakout=(setup.shift(1,fill_value=False)&(d.close>f["pivot"])&(d.volume>=f["vol20_prev"]*vol_mult)&(f["rs"]>f["rsma"])&(loc>=.70)&(rank>=rs_min))
    mask=breakout.shift(1,fill_value=False); idx=np.flatnonzero(mask.to_numpy())
    out=[]
    if len(idx):
        entries=d.open.iloc[idx].to_numpy(float); bi=idx-1; atr=f["atr14"].iloc[bi].to_numpy(float); lows=f["low10"].iloc[bi].to_numpy(float); closes=d.close.iloc[bi].to_numpy(float)
        stops=np.minimum(lows,closes-1.5*atr)
        for dt,e,s in zip(d.index[idx],entries,stops):
            if np.isfinite(e) and np.isfinite(s) and s<e: out.append((dt,float(e),float(s)))
    return out

def run(files,market,start,end,rs_min,pivot_max,vol_mult):
    sigs=[]
    for sym,d in rb.DATA_CACHE.items():
        for dt,e,s in quality_signals(sym,d,market,rs_min,pivot_max,vol_mult):
            if start<=dt<=end: sigs.append((dt,sym,e,s))
    sigs.sort(); cash=INITIAL; pos={}; trades=[]; curve=[]
    md=market.loc[:end]; mok=(md.close>md.close.rolling(200).mean()).shift(1).fillna(False)
    for dt in market.index[(market.index>=start)&(market.index<=end)]:
        for sym in list(pos):
            p=pos[sym]; d=rb.DATA_CACHE[sym]; row=d.loc[dt] if dt in d.index else None
            if row is None: continue
            px=float(row.close); low=float(row.low); ma50=float(rb.FEATURE_CACHE[sym]["ma50"].loc[dt]); p["peak_r"]=max(p["peak_r"],(float(row.high)-p["entry"])/p["risk"])
            if low<=p["stop"] or (p["peak_r"]>=1.5 and np.isfinite(ma50) and px<ma50):
                proceeds=p["qty"]*(p["stop"] if low<=p["stop"] else px); fee=(p["qty"]*p["entry"]+proceeds)*COST_BPS/10000; pnl=proceeds-p["cost"]-fee; cash+=proceeds-fee; trades.append((p["date"],dt,sym,p["entry"],proceeds/p["qty"],p["qty"],pnl)); del pos[sym]
        for sdt,sym,e,stop in sigs:
            if sdt!=dt or sym in pos or len(pos)>=MAX_POSITIONS: continue
            risk=e-stop
            equity=cash+sum(x["qty"]*float(rb.DATA_CACHE[k].loc[dt,"close"]) for k,x in pos.items() if dt in rb.DATA_CACHE[k].index)
            qty=min(int(equity*RISK_PCT/risk),int(cash/(e*1.001))) if risk>0 else 0
            if qty<=0: continue
            fee=qty*e*COST_BPS/10000; cost=qty*e+fee; cash-=cost; pos[sym]={"date":dt,"entry":e,"stop":stop,"risk":risk,"qty":qty,"cost":cost,"peak_r":0.0}
        curve.append((dt,cash+sum(x["qty"]*float(rb.DATA_CACHE[k].loc[dt,"close"]) for k,x in pos.items() if dt in rb.DATA_CACHE[k].index)))
    for sym,p in list(pos.items()):
        d=rb.DATA_CACHE[sym]; fd=d.loc[:end].index[-1]; px=float(d.loc[fd,"close"]); proceeds=p["qty"]*px; fee=(p["qty"]*p["entry"]+proceeds)*COST_BPS/10000; cash+=proceeds-fee; trades.append((p["date"],fd,sym,p["entry"],px,p["qty"],proceeds-p["cost"]-fee))
    tr=pd.DataFrame(trades,columns=["entry_date","exit_date","symbol","entry","exit","qty","pnl"]); eq=pd.Series(dict(curve)).sort_index(); final=cash; dd=float((eq/eq.cummax()-1).min()) if len(eq) else 0; daily=eq.pct_change().dropna(); sharpe=float(np.sqrt(252)*daily.mean()/daily.std()) if daily.std()>0 else 0; wins=tr.loc[tr.pnl>0,"pnl"]; losses=tr.loc[tr.pnl<0,"pnl"]; pf=float(wins.sum()/abs(losses.sum())) if len(losses) else np.inf
    return {"start":str(start.date()),"end":str(end.date()),"return_pct":100*(final/INITIAL-1),"max_drawdown_pct":100*dd,"trades":len(tr),"win_rate_pct":100*len(wins)/len(tr) if len(tr) else 0,"profit_factor":pf,"sharpe":sharpe},tr

def main():
    root=Path("data"); market=rb.load(root/"NIFTY50.csv"); rb.MARKET_CACHE=market; files=[p for p in root.glob("*.csv") if p.name!="NIFTY50.csv"]; rb.get_cache(files,root/"NIFTY50.csv"); rb.build_rs_rank(files,market)
    rows=[]; alltr=[]
    for name,rs,pv,vm in CONFIGS:
        for a,b in WINDOWS:
            r,tr=run(files,market,pd.Timestamp(a),pd.Timestamp(b),rs,pv,vm); r.update({"config":name,"rs_min":rs,"pivot_max":pv,"vol_mult":vm}); rows.append(r); tr["config"]=name; alltr.append(tr)
    out=Path("results"); out.mkdir(exist_ok=True); df=pd.DataFrame(rows); df.to_csv(out/"entry_quality_1p5r_results.csv",index=False); pd.concat(alltr,ignore_index=True).to_csv(out/"entry_quality_1p5r_trades.csv",index=False)
    print(df.groupby("config").agg(return_mean=("return_pct","mean"),return_median=("return_pct","median"),pf_median=("profit_factor","median"),sharpe_mean=("sharpe","mean"),drawdown_mean=("max_drawdown_pct","mean"),trades_mean=("trades","mean")).to_string())
    print("\n2025-2026 OOS"); print(df[df.end=="2026-03-31"].groupby("config").agg(return_mean=("return_pct","mean"),pf_median=("profit_factor","median"),sharpe_mean=("sharpe","mean"),trades_mean=("trades","mean")).to_string())
if __name__=="__main__": main()
