from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

COST_BPS = 10.0
TARGETS=[0.30,0.35,0.40,0.45,0.50]

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
    labels=pd.DataFrame()
    cls=out/'stock_winner_loser_classification.csv'
    if cls.exists(): labels=pd.read_csv(cls,usecols=['symbol','historical_label'])
    rows=[]
    for _, t in trades.iterrows():
        d=cache.get(str(t.symbol))
        if d is None: continue
        path=d.loc[t.entry_date:t.exit_date]
        if path.empty: continue
        entry=float(t.entry); risk=float(t.initial_risk); exit_px=float(t.exit)
        mfe=(float(path.high.max())/entry-1)*100
        mae=(float(path.low.min())/entry-1)*100
        mfe_r=(float(path.high.max())-entry)/risk if risk>0 else np.nan
        mae_r=(entry-float(path.low.min()))/risk if risk>0 else np.nan
        realized_pct=(exit_px/entry-1)*100
        hold=int((t.exit_date-t.entry_date).days)
        exit_r=float(t.pnl)/(float(t.qty)*risk) if risk>0 else np.nan
        row={**t.to_dict(),'holding_days':hold,'mfe_pct':mfe,'mae_pct':mae,'mfe_r':mfe_r,'mae_r':mae_r,'realized_return_pct':realized_pct,'giveback_pct':mfe-realized_pct,'realized_r':exit_r,'entry_cost_bps':COST_BPS}
        for target in TARGETS: row[f'hit_{int(target*100)}pct']=int((path.high>=entry*(1+target)).any())
        rows.append(row)
    diag=pd.DataFrame(rows)
    if not labels.empty: diag=diag.merge(labels,on='symbol',how='left')
    diag.to_csv(out/'trade_diagnostics.csv',index=False)
    if diag.empty: return
    def bucket(x):
        if x < 1.25: return '<1.25x'
        if x < 1.5: return '1.25-1.50x'
        if x < 2.0: return '1.50-2.00x'
        return '>=2.00x'
    diag['volume_bucket']=diag.breakout_volume_ratio.map(bucket)
    diag['rs_strength_pct']=(diag.rs_ma_ratio-1)*100
    diag['pivot_bucket']=pd.cut(diag.distance_from_pivot_pct,[-np.inf,0,1,2,3,np.inf],labels=['<=0','0-1%','1-2%','2-3%','>3%'])
    diag['outcome']=np.where(diag.pnl>0,'win','loss')
    summary=pd.DataFrame([{
        'trades':len(diag),'win_rate_pct':float((diag.pnl>0).mean()*100),
        'mean_pnl':float(diag.pnl.mean()),'median_pnl':float(diag.pnl.median()),'total_pnl':float(diag.pnl.sum()),
        'mean_mfe_pct':float(diag.mfe_pct.mean()),'median_mfe_pct':float(diag.mfe_pct.median()),
        'mean_mae_pct':float(diag.mae_pct.mean()),'median_mae_pct':float(diag.mae_pct.median()),
        'mean_mfe_r':float(diag.mfe_r.mean()),'mean_mae_r':float(diag.mae_r.mean()),
        'mean_holding_days':float(diag.holding_days.mean()),'median_holding_days':float(diag.holding_days.median()),
        'mean_realized_r':float(diag.realized_r.mean()),'mean_giveback_pct':float(diag.giveback_pct.mean()),
        'target_30_hit_pct':float(diag.hit_30pct.mean()*100),'target_40_hit_pct':float(diag.hit_40pct.mean()*100),'target_50_hit_pct':float(diag.hit_50pct.mean()*100),
        'gap_over_3pct_pct':float((diag.gap_pct>3).mean()*100),
        'breakout_volume_under_1_75_pct':float((diag.breakout_volume_ratio<1.75).mean()*100),
        'rs_strength_mean_pct':float(diag.rs_strength_pct.mean()),
    }])
    summary.to_csv(out/'trade_diagnostics_summary.csv',index=False)
    for col in ['volume_bucket','pivot_bucket','exit_reason','historical_label','outcome']:
        if col not in diag: continue
        g=diag.groupby(col,observed=False).agg(trades=('pnl','size'),total_pnl=('pnl','sum'),avg_pnl=('pnl','mean'),win_rate_pct=('pnl',lambda s:(s>0).mean()*100),avg_mfe_pct=('mfe_pct','mean'),avg_giveback_pct=('giveback_pct','mean'),avg_mfe_r=('mfe_r','mean'),avg_mae_r=('mae_r','mean')).reset_index()
        g.to_csv(out/f'trade_diagnostics_by_{col}.csv',index=False)
    target_rows=[]
    for target in TARGETS:
        hit=diag[f'hit_{int(target*100)}pct']>0
        target_rows.append({'target_pct':target*100,'trades_that_reached_target':int(hit.sum()),'target_hit_rate_pct':float(hit.mean()*100),'mean_baseline_pnl_hit_trades':float(diag.loc[hit,'pnl'].mean()) if hit.any() else np.nan,'mean_realized_return_hit_trades_pct':float(diag.loc[hit,'realized_return_pct'].mean()) if hit.any() else np.nan})
    pd.DataFrame(target_rows).to_csv(out/'trade_target_hit_analysis.csv',index=False)
    print('TRADE DIAGNOSTICS SUMMARY'); print(summary.to_string(index=False))
    if 'historical_label' in diag:
        print('\nP&L BY HISTORICAL LABEL'); print(diag.groupby('historical_label').agg(trades=('pnl','size'),total_pnl=('pnl','sum'),mean_pnl=('pnl','mean'),win_rate_pct=('pnl',lambda s:(s>0).mean()*100),mean_mfe_pct=('mfe_pct','mean'),mean_giveback_pct=('giveback_pct','mean')).to_string())
    print('\nTARGET HIT ANALYSIS'); print(pd.DataFrame(target_rows).to_string(index=False))

if __name__=='__main__': main()
