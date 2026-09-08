from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

COST_BPS = 10.0

def load(path):
    d = pd.read_csv(path)
    d['timestamp'] = pd.to_datetime(d['timestamp'])
    d = d.sort_values('timestamp').set_index('timestamp')
    for c in ['open','high','low','close','volume']:
        d[c] = pd.to_numeric(d[c], errors='coerce')
    return d.dropna(subset=['open','high','low','close','volume'])

def main():
    root = Path('data'); out = Path('results'); out.mkdir(exist_ok=True)
    trades = pd.read_csv(out/'trades.csv', parse_dates=['entry_date','exit_date'])
    if trades.empty:
        pd.DataFrame().to_csv(out/'trade_diagnostics.csv', index=False); return
    cache = {p.stem: load(p) for p in root.glob('*.csv') if p.name != 'NIFTY50.csv'}
    rows=[]
    for _, t in trades.iterrows():
        d=cache.get(str(t.symbol))
        if d is None: continue
        path=d.loc[t.entry_date:t.exit_date]
        if path.empty: continue
        entry=float(t.entry); risk=float(t.initial_risk)
        mfe=(float(path.high.max())/entry-1)*100
        mae=(float(path.low.min())/entry-1)*100
        mfe_r=(float(path.high.max())-entry)/risk if risk>0 else np.nan
        mae_r=(entry-float(path.low.min()))/risk if risk>0 else np.nan
        hold=int((t.exit_date-t.entry_date).days)
        exit_r=float(t.pnl)/(float(t.qty)*risk) if risk>0 else np.nan
        rows.append({**t.to_dict(),'holding_days':hold,'mfe_pct':mfe,'mae_pct':mae,'mfe_r':mfe_r,'mae_r':mae_r,'realized_r':exit_r,'entry_cost_bps':COST_BPS})
    diag=pd.DataFrame(rows)
    diag.to_csv(out/'trade_diagnostics.csv',index=False)
    if diag.empty: return
    def bucket(x):
        if x < 1.25: return '<1.25x'
        if x < 1.5: return '1.25-1.50x'
        if x < 2.0: return '1.50-2.00x'
        return '>=2.00x'
    diag['volume_bucket']=diag.breakout_volume_ratio.map(bucket)
    # rs_ma_ratio is already the relative-strength ratio to its 50-day mean
    # (RS / RS_MA). The previous formula divided the absolute RS level by this
    # ratio, producing nonsensical values such as -93%. Measure the intended
    # distance from the RS moving average directly.
    diag['rs_strength_pct']=(diag.rs_ma_ratio-1)*100
    diag['pivot_bucket']=pd.cut(diag.distance_from_pivot_pct,[-np.inf,0,1,2,3,np.inf],labels=['<=0','0-1%','1-2%','2-3%','>3%'])
    diag['outcome']=np.where(diag.pnl>0,'win','loss')
    summary=pd.DataFrame([{
        'trades':len(diag),'win_rate_pct':float((diag.pnl>0).mean()*100),
        'mean_pnl':float(diag.pnl.mean()),'median_pnl':float(diag.pnl.median()),
        'mean_mfe_pct':float(diag.mfe_pct.mean()),'median_mfe_pct':float(diag.mfe_pct.median()),
        'mean_mae_pct':float(diag.mae_pct.mean()),'median_mae_pct':float(diag.mae_pct.median()),
        'mean_mfe_r':float(diag.mfe_r.mean()),'mean_mae_r':float(diag.mae_r.mean()),
        'mean_holding_days':float(diag.holding_days.mean()),'median_holding_days':float(diag.holding_days.median()),
        'mean_realized_r':float(diag.realized_r.mean()),
        'gap_over_3pct_pct':float((diag.gap_pct>3).mean()*100),
        'breakout_volume_under_1_75_pct':float((diag.breakout_volume_ratio<1.75).mean()*100),
        'rs_strength_mean_pct':float(diag.rs_strength_pct.mean()),
    }])
    summary.to_csv(out/'trade_diagnostics_summary.csv',index=False)
    for col in ['volume_bucket','pivot_bucket','exit_reason']:
        g=diag.groupby(col,observed=False).agg(trades=('pnl','size'),avg_pnl=('pnl','mean'),win_rate_pct=('pnl',lambda s:(s>0).mean()*100),avg_mfe_r=('mfe_r','mean'),avg_mae_r=('mae_r','mean')).reset_index()
        g.to_csv(out/f'trade_diagnostics_by_{col}.csv',index=False)
    print('TRADE DIAGNOSTICS SUMMARY')
    print(summary.to_string(index=False))

if __name__=='__main__': main()
