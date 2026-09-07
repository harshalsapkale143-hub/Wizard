from __future__ import annotations
from pathlib import Path
import itertools
import pandas as pd
import numpy as np
import robustness_backtest as rb

CONFIGS=list(itertools.product([.80,.85,.90],[.55,.65,.75],[1.25,1.50,1.75]))
FOLDS=[
    (pd.Timestamp('2019-01-01'),pd.Timestamp('2020-12-31'),pd.Timestamp('2021-01-01'),pd.Timestamp('2021-06-30')),
    (pd.Timestamp('2019-07-01'),pd.Timestamp('2021-06-30'),pd.Timestamp('2021-07-01'),pd.Timestamp('2021-12-31')),
    (pd.Timestamp('2020-01-01'),pd.Timestamp('2021-12-31'),pd.Timestamp('2022-01-01'),pd.Timestamp('2022-06-30')),
    (pd.Timestamp('2020-07-01'),pd.Timestamp('2022-06-30'),pd.Timestamp('2022-07-01'),pd.Timestamp('2022-12-31')),
    (pd.Timestamp('2021-01-01'),pd.Timestamp('2022-12-31'),pd.Timestamp('2023-01-01'),pd.Timestamp('2023-06-30')),
    (pd.Timestamp('2021-07-01'),pd.Timestamp('2023-06-30'),pd.Timestamp('2023-07-01'),pd.Timestamp('2023-12-31')),
    (pd.Timestamp('2022-01-01'),pd.Timestamp('2023-12-31'),pd.Timestamp('2024-01-01'),pd.Timestamp('2024-06-30')),
    (pd.Timestamp('2022-07-01'),pd.Timestamp('2024-06-30'),pd.Timestamp('2024-07-01'),pd.Timestamp('2024-12-31')),
    (pd.Timestamp('2023-01-01'),pd.Timestamp('2024-12-31'),pd.Timestamp('2025-01-01'),pd.Timestamp('2025-06-30')),
    (pd.Timestamp('2023-07-01'),pd.Timestamp('2025-06-30'),pd.Timestamp('2025-07-01'),pd.Timestamp('2025-12-31')),
    (pd.Timestamp('2024-01-01'),pd.Timestamp('2025-12-31'),pd.Timestamp('2026-01-01'),pd.Timestamp('2026-03-31')),
]

def score(x):
    # Selection happens only on the training window. Prefer positive PF and
    # return, while penalizing fragile low-trade configurations.
    pf=min(float(x['profit_factor']),5.0) if np.isfinite(x['profit_factor']) else 5.0
    trades=float(x['trades'])
    return float(x['return_pct']) + 4.0*(pf-1.0) + min(trades,30.0)*0.03

def main():
    root=Path('data'); market=rb.load(root/'NIFTY50.csv'); files=[p for p in root.glob('*.csv') if p.name!='NIFTY50.csv']
    rb.get_cache(files,root/'NIFTY50.csv'); rb.build_rs_rank(files,market)
    rows=[]; selections=[]
    for i,(ts,te,vs,ve) in enumerate(FOLDS,1):
        train=[]
        for near,tighten,vm in CONFIGS:
            train.append(rb.run(files,market,ts,te,near,tighten,vm)[0])
        trdf=pd.DataFrame(train); trdf['score']=trdf.apply(score,axis=1); best=trdf.sort_values(['score','profit_factor','return_pct'],ascending=False).iloc[0]
        near,tighten,vm=float(best.near_high),float(best.tighten),float(best.vol_mult)
        test=rb.run(files,market,vs,ve,near,tighten,vm)[0]
        test.update({'fold':i,'train_start':str(ts.date()),'train_end':str(te.date()),'test_start':str(vs.date()),'test_end':str(ve.date()),'selected_near_high':near,'selected_tighten':tighten,'selected_vol_mult':vm,'train_return_pct':float(best.return_pct),'train_profit_factor':float(best.profit_factor),'train_trades':int(best.trades),'train_score':float(best.score)})
        rows.append(test)
        selections.append({'fold':i,'train_start':str(ts.date()),'train_end':str(te.date()),'test_start':str(vs.date()),'test_end':str(ve.date()),'near_high':near,'tighten':tighten,'vol_mult':vm,'train_return_pct':float(best.return_pct),'train_profit_factor':float(best.profit_factor),'train_trades':int(best.trades),'train_score':float(best.score)})
    out=Path('results'); out.mkdir(exist_ok=True)
    wf=pd.DataFrame(rows); wf.to_csv(out/'walk_forward_oos.csv',index=False); pd.DataFrame(selections).to_csv(out/'walk_forward_selections.csv',index=False)
    oos=wf[['return_pct','profit_factor','max_drawdown_pct','trades','win_rate_pct','sharpe']]
    summary=pd.DataFrame([{
        'folds':len(wf),'positive_oos_folds':int((wf.return_pct>0).sum()),'positive_oos_fold_pct':float((wf.return_pct>0).mean()*100),
        'oos_compounded_return_pct':float((np.prod(1+wf.return_pct/100)-1)*100),'mean_oos_return_pct':float(wf.return_pct.mean()),
        'median_oos_return_pct':float(wf.return_pct.median()),'median_oos_pf':float(wf.profit_factor.replace([np.inf],np.nan).median()),
        'mean_oos_trades':float(wf.trades.mean()),'worst_oos_return_pct':float(wf.return_pct.min()),'mean_oos_drawdown_pct':float(wf.max_drawdown_pct.mean()),
        'median_oos_sharpe':float(wf.sharpe.median()),'unique_configs_selected':int(wf[['selected_near_high','selected_tighten','selected_vol_mult']].drop_duplicates().shape[0])
    }]); summary.to_csv(out/'walk_forward_summary.csv',index=False)
    print('\nWALK-FORWARD OOS\n',wf.to_string(index=False)); print('\nSUMMARY\n',summary.to_string(index=False))

if __name__=='__main__': main()
