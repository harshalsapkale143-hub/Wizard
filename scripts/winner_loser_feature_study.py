from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

DATA=Path('data'); OUT=Path('results')
MIN_ROWS=500
WINDOW_YEARS=3


def load(path: Path) -> pd.DataFrame:
    d=pd.read_csv(path)
    d['timestamp']=pd.to_datetime(d['timestamp'],errors='coerce')
    for c in ['open','high','low','close','volume']:
        d[c]=pd.to_numeric(d[c],errors='coerce')
    return d.dropna(subset=['timestamp','open','high','low','close','volume']).sort_values('timestamp').drop_duplicates('timestamp').set_index('timestamp')


def safe_slope(s: pd.Series, n: int) -> float:
    x=s.dropna().iloc[-n:]
    if len(x)<n: return np.nan
    y=x.to_numpy(dtype=float); t=np.arange(n,dtype=float)
    return float(np.polyfit(t,y,1)[0] / abs(y.mean())) if abs(y.mean())>1e-12 else np.nan


def features_at(d: pd.DataFrame, asof: pd.Timestamp) -> dict:
    x=d.loc[:asof].copy()
    if len(x)<260: return {}
    c=x.close; h=x.high; l=x.low; v=x.volume
    ma50=c.rolling(50).mean(); ma150=c.rolling(150).mean(); ma200=c.rolling(200).mean()
    high252=h.rolling(252).max(); low252=l.rolling(252).min()
    ret=lambda n: float(c.iloc[-1]/c.iloc[-1-n]-1.0) if len(c)>n and c.iloc[-1-n]>0 else np.nan
    vol20=float(c.pct_change().rolling(20).std().iloc[-1]*np.sqrt(252))
    atr14=float((h-l).rolling(14).mean().iloc[-1]/c.iloc[-1])
    vol_ratio=float(v.iloc[-1]/v.rolling(20).mean().iloc[-1])
    dd252=float(c.iloc[-1]/high252.iloc[-1]-1.0)
    window=c.iloc[-252:]; peak=window.cummax(); dd=float((window/peak-1.0).min())
    return {
        'feature_date':x.index[-1].date().isoformat(),
        'momentum_1m_pct':ret(21)*100,
        'momentum_3m_pct':ret(63)*100,
        'momentum_6m_pct':ret(126)*100,
        'momentum_12m_pct':ret(252)*100,
        'distance_52w_high_pct':dd252*100,
        'distance_52w_low_pct':(c.iloc[-1]/low252.iloc[-1]-1.0)*100 if low252.iloc[-1]>0 else np.nan,
        'price_vs_ma50_pct':(c.iloc[-1]/ma50.iloc[-1]-1.0)*100,
        'price_vs_ma150_pct':(c.iloc[-1]/ma150.iloc[-1]-1.0)*100,
        'price_vs_ma200_pct':(c.iloc[-1]/ma200.iloc[-1]-1.0)*100,
        'ma50_vs_ma150_pct':(ma50.iloc[-1]/ma150.iloc[-1]-1.0)*100,
        'ma150_vs_ma200_pct':(ma150.iloc[-1]/ma200.iloc[-1]-1.0)*100,
        'ma50_slope_pct_per_day':safe_slope(ma50,20)*100,
        'ma150_slope_pct_per_day':safe_slope(ma150,50)*100,
        'ma200_slope_pct_per_day':safe_slope(ma200,50)*100,
        'annualized_vol20_pct':vol20*100,
        'atr14_pct':atr14*100,
        'volume_ratio_20d':vol_ratio,
        'drawdown_252d_pct':dd*100,
        'above_ma_alignment':int(c.iloc[-1]>ma50.iloc[-1]>ma150.iloc[-1]>ma200.iloc[-1]),
        'trend_alignment':int(c.iloc[-1]>ma150.iloc[-1] and c.iloc[-1]>ma200.iloc[-1] and ma150.iloc[-1]>ma200.iloc[-1]),
    }


def main():
    cls_path=OUT/'stock_winner_loser_classification.csv'
    if not cls_path.exists(): raise SystemExit('Run winner/loser classification first.')
    cls=pd.read_csv(cls_path)
    market=load(DATA/'NIFTY50.csv')
    end=market.index.max(); t0=end-pd.DateOffset(years=WINDOW_YEARS)
    rows=[]
    for path in sorted(DATA.glob('*.csv')):
        if path.name=='NIFTY50.csv': continue
        sym=path.stem.upper()
        label=cls.loc[cls.symbol.astype(str)==sym,'historical_label']
        if label.empty: continue
        d=load(path)
        if len(d)<MIN_ROWS: continue
        f=features_at(d,t0)
        if not f: continue
        rows.append({'symbol':sym,'historical_label':label.iloc[0],'outcome_start':t0.date().isoformat(),**f})
    result=pd.DataFrame(rows)
    OUT.mkdir(exist_ok=True); result.to_csv(OUT/'winner_loser_feature_stock.csv',index=False)
    feature_cols=[c for c in result.columns if c not in {'symbol','historical_label','outcome_start','feature_date'}]
    summaries=[]
    for label,g in result.groupby('historical_label'):
        for col in feature_cols:
            s=pd.to_numeric(g[col],errors='coerce').dropna()
            if len(s)==0: continue
            summaries.append({'historical_label':label,'feature':col,'stocks':len(s),'mean':float(s.mean()),'median':float(s.median()),'q25':float(s.quantile(.25)),'q75':float(s.quantile(.75))})
    summary=pd.DataFrame(summaries); summary.to_csv(OUT/'winner_loser_feature_summary.csv',index=False)
    # Independent descriptive separation: standardized mean difference between
    # WINNER and LOSER groups. Labels are forward outcomes; features are frozen at T0.
    w=result[result.historical_label=='WINNER']; l=result[result.historical_label=='LOSER']
    effects=[]
    for col in feature_cols:
        a=pd.to_numeric(w[col],errors='coerce').dropna(); b=pd.to_numeric(l[col],errors='coerce').dropna()
        if len(a)<5 or len(b)<5: continue
        pooled=np.sqrt(((len(a)-1)*a.var(ddof=1)+(len(b)-1)*b.var(ddof=1))/(len(a)+len(b)-2))
        smd=float((a.mean()-b.mean())/pooled) if pooled>0 else np.nan
        effects.append({'feature':col,'winner_n':len(a),'loser_n':len(b),'winner_mean':float(a.mean()),'loser_mean':float(b.mean()),'winner_median':float(a.median()),'loser_median':float(b.median()),'standardized_mean_difference':smd})
    effects=pd.DataFrame(effects).sort_values('standardized_mean_difference',key=lambda s:s.abs(),ascending=False)
    effects.to_csv(OUT/'winner_loser_feature_effects.csv',index=False)
    print(f'Feature snapshot date: {t0.date()}')
    print(f'Stocks with leakage-safe features: {len(result)}')
    print('\nWINNER vs LOSER feature effects (largest absolute standardized difference):')
    print(effects.head(20).to_string(index=False))

if __name__=='__main__': main()
