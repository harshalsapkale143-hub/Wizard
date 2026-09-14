from __future__ import annotations
from pathlib import Path
import itertools
import numpy as np
import pandas as pd
import robustness_backtest as rb

# Research only. This file tests technical components inspired by William O'Neil's
# CAN SLIM framework without changing the production strategy. Historical
# fundamentals/estimates are not currently available in the daily universe, so
# C/A/I are intentionally NOT fabricated or approximated here.
INITIAL = 1_000_000.0
COST_BPS = 10.0
RISK_PCT = 0.005
MAX_POSITIONS = 10
CONFIGS = list(itertools.product([0.80, 0.85, 0.90], [0.65], [1.25, 1.50, 1.75], [0.70, 0.80, 0.90], [0.75, 0.85, 0.95]))
FY_WINDOWS = [
    ('FY2021-22', '2021-04-01', '2022-03-31'),
    ('FY2022-23', '2022-04-01', '2023-03-31'),
    ('FY2023-24', '2023-04-01', '2024-03-31'),
    ('FY2024-25', '2024-04-01', '2025-03-31'),
    ('FY2025-26', '2025-04-01', '2026-03-31'),
]


def market_overlay(market: pd.DataFrame) -> pd.DataFrame:
    c = market.close
    v = market.volume
    ma200 = c.rolling(200).mean()
    # O'Neil-style distribution proxy: index down materially while volume is
    # above the prior session. Count only the most recent 25 sessions.
    dist = ((c.pct_change() <= -0.002) & (v > v.shift(1))).astype(int)
    dist_count = dist.rolling(25).sum()
    # Keep the existing trend gate, then penalize a clustered distribution tape.
    # This is a diagnostic overlay, not a claim that the original book's exact
    # market-timing rules can be reproduced from NIFTY daily OHLCV alone.
    ok = (c > ma200).shift(1).fillna(False) & (dist_count <= 4).shift(1).fillna(False)
    return pd.DataFrame({'ok': ok, 'distribution_25d': dist_count})


def run_one(files, market, start, end, near, tighten, vol_mult, rs_cut, near52_cut):
    rb.get_cache(files, None)
    rb.build_rs_rank(files, market)
    mo = market_overlay(market)
    sigs = []
    for sym, d in rb.DATA_CACHE.items():
        f = rb.build_features(sym, d, market)
        rs_rank = rb.RS_RANK_CACHE[sym].reindex(d.index)
        c, h, l, v = d.close, d.high, d.low, d.volume
        high252 = h.rolling(252).max()
        contraction = (f['r10'] < f['r20'] * tighten) & (f['r20'] < f['r40'] * 0.90)
        pivot_distance = (c / f['pivot'] - 1.0)
        setup = (f['trend'] & (f['near_base'] >= near) & contraction &
                 (f['dry_volume_ratio'] < 0.75) & (pivot_distance <= 0.03) &
                 (c / high252 >= near52_cut))
        day_range = (h - l).replace(0, np.nan)
        close_location = (c - l) / day_range
        breakout = (setup.shift(1, fill_value=False) & (c > f['pivot']) &
                    (v >= f['vol20_prev'] * vol_mult) & (f['rs'] > f['rsma']) &
                    (close_location >= 0.70) & (rs_rank >= rs_cut))
        entry_mask = breakout.shift(1, fill_value=False)
        idx = np.flatnonzero(entry_mask.to_numpy())
        if len(idx):
            entries = d.open.iloc[idx].to_numpy(float)
            bi = idx - 1
            atr = f['atr14'].iloc[bi].to_numpy(float)
            lows = f['low10'].iloc[bi].to_numpy(float)
            closes = c.iloc[bi].to_numpy(float)
            stops = np.minimum(lows, closes - 1.5 * atr)
            dates = d.index[idx]
            for dt, e, s in zip(dates, entries, stops):
                if np.isfinite(e) and np.isfinite(s) and s < e and bool(mo.ok.reindex([dt]).fillna(False).iloc[0]):
                    sigs.append((dt, sym, float(e), float(s)))
    sigs.sort()
    cash = INITIAL
    pos = {}
    trades = []
    curve = []
    for dt in market.index[(market.index >= start) & (market.index <= end)]:
        for sym in list(pos):
            d = rb.DATA_CACHE[sym]
            if dt not in d.index:
                continue
            px = float(d.loc[dt, 'close'])
            f = rb.FEATURE_CACHE[sym]
            ma50 = float(f['ma50'].loc[dt])
            stop = pos[sym]['stop']
            peak = float(f['peak'].loc[dt])
            trail = float(f['trail20'].loc[dt])
            r = pos[sym]['risk']
            effective_stop = max(stop, trail) if peak >= pos[sym]['entry'] + 2 * r else stop
            if px <= effective_stop or (np.isfinite(ma50) and px < ma50):
                proceeds = pos[sym]['qty'] * px
                fee = (pos[sym]['qty'] * pos[sym]['entry'] + proceeds) * COST_BPS / 10000
                pnl = proceeds - pos[sym]['cost'] - fee
                cash += proceeds - fee
                trades.append((pos[sym]['date'], dt, sym, pos[sym]['entry'], px, pos[sym]['qty'], pnl))
                del pos[sym]
        for sdt, sym, e, stop in sigs:
            if sdt != dt or sym in pos or len(pos) >= MAX_POSITIONS:
                continue
            risk = e - stop
            equity = cash + sum(x['qty'] * float(rb.DATA_CACHE[k].loc[dt, 'close'])
                                for k, x in pos.items() if dt in rb.DATA_CACHE[k].index)
            qty = min(int(equity * RISK_PCT / risk), int(cash / (e * (1 + COST_BPS / 10000)))) if risk > 0 else 0
            if qty <= 0:
                continue
            fee = qty * e * COST_BPS / 10000
            cost = qty * e + fee
            if cost > cash:
                continue
            cash -= cost
            pos[sym] = {'date': dt, 'entry': e, 'stop': stop, 'risk': risk, 'qty': qty, 'cost': cost}
        curve.append((dt, cash + sum(x['qty'] * float(rb.DATA_CACHE[k].loc[dt, 'close'])
                                    for k, x in pos.items() if dt in rb.DATA_CACHE[k].index)))
    for sym, p in list(pos.items()):
        d = rb.DATA_CACHE[sym]
        px = float(d.loc[:end, 'close'].iloc[-1])
        proceeds = p['qty'] * px
        fee = (p['qty'] * p['entry'] + proceeds) * COST_BPS / 10000
        cash += proceeds - fee
        trades.append((p['date'], d.loc[:end].index[-1], sym, p['entry'], px, p['qty'], proceeds - p['cost'] - fee))
    tr = pd.DataFrame(trades, columns=['entry_date','exit_date','symbol','entry','exit','qty','pnl'])
    eq = pd.Series({d: v for d, v in curve}).sort_index()
    final = cash
    dd = float((eq / eq.cummax() - 1).min()) if len(eq) else 0
    years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1 / 365.25) if len(eq) else 1
    cagr = (final / INITIAL) ** (1 / years) - 1
    daily = eq.pct_change().dropna()
    sharpe = float(np.sqrt(252) * daily.mean() / daily.std()) if daily.std() > 0 else 0
    wins = tr.loc[tr.pnl > 0, 'pnl']; losses = tr.loc[tr.pnl < 0, 'pnl']
    pf = float(wins.sum() / abs(losses.sum())) if len(losses) else np.inf
    return {'return_pct': (final / INITIAL - 1) * 100, 'cagr_pct': cagr * 100,
            'max_drawdown_pct': dd * 100, 'trades': len(tr),
            'win_rate_pct': len(wins) / len(tr) * 100 if len(tr) else 0,
            'profit_factor': pf, 'sharpe': sharpe}, tr


def main():
    root = Path('data'); market = rb.load(root / 'NIFTY50.csv')
    files = [p for p in root.glob('*.csv') if p.name != 'NIFTY50.csv']
    rb.get_cache(files, root / 'NIFTY50.csv')
    rb.build_rs_rank(files, market)
    rows = []
    for fy, a, b in FY_WINDOWS:
        nifty = market.loc[a:b, 'close'].dropna()
        bench = (nifty.iloc[-1] / nifty.iloc[0] - 1) * 100 if len(nifty) > 1 else 0
        for near, tighten, vm, rs_cut, near52_cut in CONFIGS:
            r, _ = run_one(files, market, pd.Timestamp(a), pd.Timestamp(b), near, tighten, vm, rs_cut, near52_cut)
            rows.append({'fy': fy, 'near_high': near, 'tighten': tighten, 'vol_mult': vm,
                          'rs_cut': rs_cut, 'near52_cut': near52_cut, 'nifty_return_pct': bench,
                          'excess_pct': r['return_pct'] - bench, **r})
    out = Path('results'); out.mkdir(exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / 'canslim_overlay_grid.csv', index=False)
    summary = (df.sort_values(['fy','return_pct'], ascending=[True,False])
                 .groupby('fy', as_index=False).head(5))
    summary.to_csv(out / 'canslim_overlay_top5.csv', index=False)
    baseline = df[(df.rs_cut == .80) & (df.near52_cut == .85) & (df.vol_mult == 1.5) & (df.near_high == .85)]
    baseline.to_csv(out / 'canslim_overlay_baseline.csv', index=False)
    print('\nCAN SLIM TECHNICAL OVERLAY — TOP 5 PER FY (research only)\n', summary[['fy','near_high','vol_mult','rs_cut','near52_cut','return_pct','excess_pct','max_drawdown_pct','trades','profit_factor','sharpe']].to_string(index=False))
    print('\nOVERLAY BASELINE\n', baseline[['fy','return_pct','excess_pct','max_drawdown_pct','trades','profit_factor','sharpe']].to_string(index=False))

if __name__ == '__main__':
    main()
