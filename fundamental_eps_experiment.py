from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

import robustness_backtest as rb

INITIAL = rb.INITIAL
COST_BPS = rb.COST_BPS
RISK_PCT = rb.RISK_PCT
BASE_CONFIG = (0.85, 0.65, 1.50)
FY_WINDOWS = [
    ('FY2021-22', '2021-04-01', '2022-03-31'),
    ('FY2022-23', '2022-04-01', '2023-03-31'),
    ('FY2023-24', '2023-04-01', '2024-03-31'),
    ('FY2024-25', '2024-04-01', '2025-03-31'),
    ('FY2025-26', '2025-04-01', '2026-03-31'),
]


def num(s):
    return pd.to_numeric(s, errors='coerce')


def consolidated_flag(v):
    s = str(v or '').lower()
    return 'non-consolidated' not in s and 'standalone' not in s and 'non consolidated' not in s


def prepare_reports(path):
    try:
        d = pd.read_csv(path, parse_dates=['period_end', 'filing_date'])
    except Exception:
        return pd.DataFrame()
    if d.empty or not {'period_end', 'filing_date', 'eps'}.issubset(d.columns):
        return pd.DataFrame()
    d['eps'] = num(d['eps'])
    d = d.dropna(subset=['period_end', 'filing_date', 'eps']).sort_values(['filing_date', 'period_end']).copy()
    if d.empty:
        return d
    d['_consolidated'] = d.get('consolidated', '').map(consolidated_flag)
    # Preserve filing chronology. A later filing can revise an earlier quarter,
    # but it must not become visible before its own filing date.
    rows = []
    for _, g in d.groupby('period_end', sort=True):
        g = g.sort_values('filing_date')
        rows.extend(g.to_dict('records'))
    d = pd.DataFrame(rows).sort_values(['filing_date', 'period_end']).reset_index(drop=True)

    # For every filing, calculate EPS growth against the latest available
    # comparable quarter in the filing history. Acceleration compares that
    # growth rate with the previous comparable quarter's growth, using only
    # information that had been filed by the current filing date.
    growth = []
    for i, row in d.iterrows():
        current_end = row['period_end']
        current_eps = row['eps']
        cutoff = row['filing_date']
        prior = d[(d['filing_date'] <= cutoff) & (d['period_end'] <= current_end - pd.DateOffset(years=1))]
        if prior.empty or current_eps <= 0:
            growth.append(np.nan)
            continue
        # Nearest comparable period, preferring the exact one-year offset.
        target = current_end - pd.DateOffset(years=1)
        distance = (prior['period_end'] - target).abs()
        p = prior.loc[distance.idxmin()]
        if p['eps'] <= 0:
            growth.append(np.nan)
        else:
            growth.append(current_eps / p['eps'] - 1.0)
    d['eps_yoy'] = growth

    acc = []
    for i, row in d.iterrows():
        if not np.isfinite(row['eps_yoy']):
            acc.append(np.nan)
            continue
        prev = d[(d['filing_date'] <= row['filing_date']) & (d['period_end'] < row['period_end'])]
        prev = prev[np.isfinite(prev['eps_yoy'])]
        if prev.empty:
            acc.append(np.nan)
        else:
            acc.append(row['eps_yoy'] - prev.iloc[-1]['eps_yoy'])
    d['eps_accel'] = acc
    return d


def latest_state(reports, asof):
    if reports.empty:
        return None
    d = reports[reports['filing_date'] <= asof]
    if d.empty:
        return None
    # For the latest reported period, prefer the latest consolidated filing
    # available by the as-of date. This avoids using a later standalone filing
    # when a consolidated record for the same period was already available.
    latest_period = d['period_end'].max()
    g = d[d['period_end'] == latest_period]
    cons = g[g['_consolidated']]
    if not cons.empty:
        q = cons.sort_values('filing_date').iloc[-1]
    else:
        q = g.sort_values('filing_date').iloc[-1]
    return {
        'filing_date': q['filing_date'],
        'period_end': q['period_end'],
        'eps_yoy': float(q['eps_yoy']) if np.isfinite(q['eps_yoy']) else np.nan,
        'eps_accel': float(q['eps_accel']) if np.isfinite(q['eps_accel']) else np.nan,
    }


def simulate(market, signals, start, end):
    sigs = [x for x in signals if start <= x[0] <= end]
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
        for sdt, sym, entry, stop in sigs:
            if sdt != dt or sym in pos or len(pos) >= 10:
                continue
            d = rb.DATA_CACHE[sym]
            risk = entry - stop
            equity = cash + sum(x['qty'] * float(rb.DATA_CACHE[k].loc[dt, 'close']) for k, x in pos.items() if dt in rb.DATA_CACHE[k].index)
            qty = min(int(equity * RISK_PCT / risk), int(cash / (entry * (1 + COST_BPS / 10000)))) if risk > 0 else 0
            if qty <= 0:
                continue
            fee = qty * entry * COST_BPS / 10000
            cost = qty * entry + fee
            if cost > cash:
                continue
            cash -= cost
            pos[sym] = {'date': dt, 'entry': entry, 'stop': stop, 'risk': risk, 'qty': qty, 'cost': cost}
        curve.append((dt, cash + sum(x['qty'] * float(rb.DATA_CACHE[k].loc[dt, 'close']) for k, x in pos.items() if dt in rb.DATA_CACHE[k].index)))
    for sym, p in list(pos.items()):
        d = rb.DATA_CACHE[sym]
        px = float(d.loc[:end, 'close'].iloc[-1])
        proceeds = p['qty'] * px
        fee = (p['qty'] * p['entry'] + proceeds) * COST_BPS / 10000
        cash += proceeds - fee
        trades.append((p['date'], d.loc[:end].index[-1], sym, p['entry'], px, p['qty'], proceeds - p['cost'] - fee))
    tr = pd.DataFrame(trades, columns=['entry_date', 'exit_date', 'symbol', 'entry', 'exit', 'qty', 'pnl'])
    eq = pd.Series({d: v for d, v in curve}).sort_index()
    final = cash
    dd = float((eq / eq.cummax() - 1).min()) if len(eq) else 0
    daily = eq.pct_change().dropna()
    sharpe = float(np.sqrt(252) * daily.mean() / daily.std()) if daily.std() > 0 else 0
    wins = tr.loc[tr.pnl > 0, 'pnl']
    losses = tr.loc[tr.pnl < 0, 'pnl']
    pf = float(wins.sum() / abs(losses.sum())) if len(losses) else np.inf
    return {
        'return_pct': (final / INITIAL - 1) * 100,
        'max_drawdown_pct': dd * 100,
        'trades': len(tr),
        'win_rate_pct': len(wins) / len(tr) * 100 if len(tr) else 0,
        'profit_factor': pf,
        'sharpe': sharpe,
    }, tr


def main():
    root = Path('data')
    market = rb.load(root / 'NIFTY50.csv')
    files = [p for p in root.glob('*.csv') if p.name != 'NIFTY50.csv']
    rb.get_cache(files, root / 'NIFTY50.csv')
    rb.build_rs_rank(files, market)
    near, tighten, vm = BASE_CONFIG

    reports = {}
    coverage = []
    for p in sorted(Path('fundamentals').glob('*.csv')):
        if p.name.startswith('_'):
            continue
        r = prepare_reports(p)
        if not r.empty:
            reports[p.stem] = r
            coverage.append({'symbol': p.stem, 'reports': len(r), 'first_filing': r.filing_date.min(), 'last_filing': r.filing_date.max(), 'eps_growth_coverage': r.eps_yoy.notna().mean(), 'eps_accel_coverage': r.eps_accel.notna().mean()})
    if not reports:
        raise SystemExit('No usable EPS fundamental histories found')

    technical = []
    for sym, d in rb.DATA_CACHE.items():
        for dt, entry, stop in rb.signals(sym, d, market, near, tighten, vm):
            # Signals enter at next day's open; only filings known by the
            # breakout/session before entry are allowed to affect eligibility.
            prior_days = market.index[market.index < dt]
            asof = prior_days[-1] if len(prior_days) else dt
            state = latest_state(reports.get(sym, pd.DataFrame()), asof)
            technical.append((dt, sym, entry, stop, state))
    technical.sort()

    # Four deliberately simple, pre-declared tests: technical baseline, EPS
    # growth only, EPS acceleration only, and both together. Thresholds are
    # chosen from the intended screening concept, not optimized on these years.
    modes = {
        'baseline': lambda s: True,
        'eps_growth_20': lambda s: s is not None and np.isfinite(s['eps_yoy']) and s['eps_yoy'] >= 0.20,
        'eps_accel_nonnegative': lambda s: s is not None and np.isfinite(s['eps_accel']) and s['eps_accel'] >= 0.0,
        'eps_growth20_and_accel': lambda s: s is not None and np.isfinite(s['eps_yoy']) and np.isfinite(s['eps_accel']) and s['eps_yoy'] >= 0.20 and s['eps_accel'] >= 0.0,
    }

    out = Path('results')
    out.mkdir(exist_ok=True)
    pd.DataFrame(coverage).to_csv(out / 'fundamental_eps_data_quality.csv', index=False)
    pd.DataFrame([{'symbol': sym, 'entry_date': dt, 'filing_date': state['filing_date'] if state else pd.NaT, 'period_end': state['period_end'] if state else pd.NaT, 'eps_yoy': state['eps_yoy'] if state else np.nan, 'eps_accel': state['eps_accel'] if state else np.nan} for dt, sym, _, _, state in technical]).to_csv(out / 'fundamental_eps_entry_states.csv', index=False)

    rows = []
    for mode, predicate in modes.items():
        filtered = [(dt, sym, e, st) for dt, sym, e, st, state in technical if predicate(state)]
        for fy, a, b in FY_WINDOWS:
            start, end = pd.Timestamp(a), pd.Timestamp(b)
            r, _ = simulate(market, filtered, start, end)
            bench_data = market.loc[start:end, 'close'].dropna()
            bench = (bench_data.iloc[-1] / bench_data.iloc[0] - 1) * 100 if len(bench_data) > 1 else 0
            rows.append({'mode': mode, 'fy': fy, 'nifty_return_pct': bench, 'excess_pct': r['return_pct'] - bench, **r})
    result = pd.DataFrame(rows)
    result.to_csv(out / 'fundamental_eps_experiment.csv', index=False)
    print('\nPOINT-IN-TIME EPS GROWTH / ACCELERATION EXPERIMENT')
    print(result.to_string(index=False))
    print(f'\nFundamental symbols: {len(reports)}; technical entries: {len(technical)}')


if __name__ == '__main__':
    main()
