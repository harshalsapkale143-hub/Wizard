from pathlib import Path
import numpy as np
import pandas as pd
import robustness_backtest as rb

INITIAL=1_000_000.0; COST_BPS=10.0; RISK_PCT=0.005; MAX_POSITIONS=10
WINDOWS=[("2019-01-01","2022-12-31"),("2023-01-01","2024-12-31"),("2025-01-01","2026-03-31"),("2021-04-01","2026-03-31")]
CONFIGS=[
 ("baseline",None,False,0,0,1.5),
 ("breadth40",.40,False,0,0,1.5),
 ("breadth50",.50,False,0,0,1.5),
 ("nifty50",None,True,0,0,1.5),
 ("breadth50_nifty50",.50,True,0,0,1.5),
 ("fail5",None,False,5,0,1.5),
 ("fail10",None,False,10,0,1.5),
 ("stall10",None,False,0,10,1.5),
 ("fail5_stall10",None,False,5,10,1.5),
 ("breadth50_fail10",.50,False,10,0,1.5),
 ("breadth50_stall10",.50,False,0,10,1.5),
 ("breadth50_fail10_stall10",.50,False,10,10,1.5),
 ("nifty50_fail10",None,True,10,0,1.5),
 ("breadth50_nifty50_fail10",.50,True,10,0,1.5),
 ("vol2",None,False,0,0,2.0),
]

def quality_signals(sym,d,market,vol_mult):
    f=rb.build_features(sym,d,market); rank=rb.RS_RANK_CACHE[sym].reindex(d.index)
    contraction=(f["r10"]<f["r20"]*.65)&(f["r20"]<f["r40"]*.90)
    setup=f["trend"]&(f["near_base"]>=.85)&contraction&(f["dry_volume_ratio"]<.75)&((d.close/f["pivot"]-1)<=.03)
    loc=(d.close-d.low)/(d.high-d.low).replace(0,np.nan)
    breakout=(setup.shift(1,fill_value=False)&(d.close>f["pivot"])&(d.volume>=f["vol20_prev"]*vol_mult)&(f["rs"]>f["rsma"])&(loc>=.70)&(rank>=.70))
    idx=np.flatnonzero(breakout.shift(1,fill_value=False).to_numpy()); out=[]
    if len(idx):
        bi=idx-1; e=d.open.iloc[idx].to_numpy(float); atr=f["atr14"].iloc[bi].to_numpy(float); lows=f["low10"].iloc[bi].to_numpy(float); closes=d.close.iloc[bi].to_numpy(float)
        for dt,en,st in zip(d.index[idx],e,np.minimum(lows,closes-1.5*atr)):
            if np.isfinite(en) and np.isfinite(st) and st<en: out.append((dt,float(en),float(st)))
    return out

def run(start,end,breadth_thr,nifty50,fail_days,stall_days,vol_mult):
    m=rb.MARKET_CACHE; dates=m.index[(m.index>=start)&(m.index<=end)]; sigs=[]
    for sym,d in rb.DATA_CACHE.items():
        for dt,e,s in quality_signals(sym,d,m,vol_mult):
            if start<=dt<=end: sigs.append((dt,sym,e,s))
    sigs.sort(); cash=INITIAL; pos={}; trades=[]; curve=[]
    bvals=[]
    for dt in dates:
        vals=[]
        for d in rb.DATA_CACHE.values():
            if dt in d.index: vals.append(float(d.close.loc[dt]>d.close.rolling(50).mean().loc[dt]))
        bvals.append(np.mean(vals) if vals else np.nan)
    breadth=pd.Series(bvals,index=dates).shift(1)
    n50=(m.close>m.close.rolling(50).mean()).shift(1).reindex(dates).fillna(False)
    n200=(m.close>m.close.rolling(200).mean()).shift(1).reindex(dates).fillna(False)
    for dt in dates:
        for sym in list(pos):
            p=pos[sym]; d=rb.DATA_CACHE[sym]
            if dt not in d.index: continue
            row=d.loc[dt]; px=float(row.close); low=float(row.low); p["age"]+=1; p["peak_r"]=max(p["peak_r"],(float(row.high)-p["entry"])/p["risk"])
            f=rb.FEATURE_CACHE[sym]; ma50=float(f["ma50"].loc[dt]); pivot=float(f["pivot"].loc[dt]); exit_px=None
            if low<=p["stop"]: exit_px=p["stop"]
            elif p["peak_r"]>=1.5 and np.isfinite(ma50) and px<ma50: exit_px=px
            elif fail_days and p["age"]<=fail_days and np.isfinite(pivot) and px<pivot: exit_px=px
            elif stall_days and p["age"]>=stall_days and p["peak_r"]<.5: exit_px=px
            if exit_px is not None:
                proceeds=p["qty"]*exit_px; fee=(p["qty"]*p["entry"]+proceeds)*COST_BPS/10000; pnl=proceeds-p["cost"]-fee; cash+=proceeds-fee
                trades.append((p["date"],dt,sym,p["entry"],exit_px,p["qty"],pnl,p["age"],p["peak_r"])); del pos[sym]
        ok=bool(n200.loc[dt]) and (breadth_thr is None or bool(breadth.loc[dt]>=breadth_thr)) and (not nifty50 or bool(n50.loc[dt]))
        if not ok: curve.append((dt,cash+sum(x["qty"]*float(rb.DATA_CACHE[k].loc[dt,"close"]) for k,x in pos.items() if dt in rb.DATA_CACHE[k].index)); continue
        for sdt,sym,e,stop in sigs:
            if sdt!=dt or sym in pos or len(pos)>=MAX_POSITIONS: continue
            risk=e-stop; equity=cash+sum(x["qty"]*float(rb.DATA_CACHE[k].loc[dt,"close"]) for k,x in pos.items() if dt in rb.DATA_CACHE[k].index); qty=min(int(equity*RISK_PCT/risk),int(cash/(e*1.001))) if risk>0 else 0
            if qty<=0: continue
            fee=qty*e*COST_BPS/10000; cost=qty*e+fee; cash-=cost; pos[sym]={"date":dt,"entry":e,"stop":stop,"risk":risk,"qty":qty,"cost":cost,"age":0,"peak_r":0.0}
        curve.append((dt,cash+sum(x["qty"]*float(rb.DATA_CACHE[k].loc[dt,"close"]) for k,x in pos.items() if dt in rb.DATA_CACHE[k].index)))
    for sym,p in list(pos.items()):
        d=rb.DATA_CACHE[sym]; fd=d.loc[:end].index[-1]; px=float(d.loc[fd,"close"]); proceeds=p["qty"]*px; fee=(p["qty"]*p["entry"]+proceeds)*COST_BPS/10000; cash+=proceeds-fee; trades.append((p["date"],fd,sym,p["entry"],px,p["qty"],proceeds-p["cost"]-fee,p["age"],p["peak_r"]))
    eq=pd.Series(dict(curve)).sort_index(); tr=pd.DataFrame(trades,columns=["entry_date","exit_date","symbol","entry","exit","qty","pnl","holding_days","peak_r"])
    final=cash; dd=float((eq/eq.cummax()-1).min()) if len(eq) else 0; daily=eq.pct_change().dropna(); sharpe=float(np.sqrt(252)*daily.mean()/daily.std()) if daily.std()>0 else 0; wins=tr.loc[tr.pnl>0,"pnl"]; losses=tr.loc[tr.pnl<0,"pnl"]; pf=float(wins.sum()/abs(losses.sum())) if len(losses) else np.inf
    return {"return_pct":100*(final/INITIAL-1),"max_drawdown_pct":100*dd,"trades":len(tr),"win_rate_pct":100*len(wins)/len(tr) if len(tr) else 0,"profit_factor":pf,"sharpe":sharpe},tr

def main():
    root=Path("data"); m=rb.load(root/"NIFTY50.csv"); rb.MARKET_CACHE=m; files=[p for p in root.glob("*.csv") if p.name!="NIFTY50.csv"]; rb.get_cache(files,root/"NIFTY50.csv"); rb.build_rs_rank(files,m)
    rows=[]; trs=[]
    for name,bt,n5,fd,sd,vm in CONFIGS:
        for a,b in WINDOWS:
            r,tr=run(pd.Timestamp(a),pd.Timestamp(b),bt,n5,fd,sd,vm); r.update({"config":name,"breadth":bt,"nifty50_gate":n5,"fail_days":fd,"stall_days":sd,"vol_mult":vm,"start":a,"end":b}); rows.append(r); tr["config"]=name; tr["window"]=a+"_"+b; trs.append(tr)
    out=Path("results"); out.mkdir(exist_ok=True); df=pd.DataFrame(rows); df.to_csv(out/"regime_rescue_results.csv",index=False); pd.concat(trs,ignore_index=True).to_csv(out/"regime_rescue_trades.csv",index=False)
    print("\nALL"); print(df.groupby("config").agg(return_mean=("return_pct","mean"),drawdown_mean=("max_drawdown_pct","mean"),pf_median=("profit_factor","median"),sharpe_mean=("sharpe","mean"),trades_mean=("trades","mean")).sort_values("return_mean",ascending=False).to_string())
    print("\n2025-2026 OOS"); print(df[df.end=="2026-03-31"].sort_values("return_pct",ascending=False)[["config","return_pct","max_drawdown_pct","profit_factor","sharpe","trades","win_rate_pct"]].to_string(index=False))
if __name__=="__main__": main()
