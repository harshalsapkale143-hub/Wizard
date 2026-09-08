from pathlib import Path
import pandas as pd
import robustness_backtest as rb

# Diagnostic only: keep the production near-base and contraction settings fixed
# and vary breakout-volume confirmation. Do not select a production value from
# these same years; use this to identify whether the bull-market weakness is
# plausibly caused by an overly strict volume gate.
VOLUME_THRESHOLDS = [1.00, 1.25, 1.50, 1.75, 2.00]
FY_WINDOWS = [
    (pd.Timestamp('2021-04-01'), pd.Timestamp('2022-03-31'), 'FY2021-22'),
    (pd.Timestamp('2022-04-01'), pd.Timestamp('2023-03-31'), 'FY2022-23'),
    (pd.Timestamp('2023-04-01'), pd.Timestamp('2024-03-31'), 'FY2023-24'),
    (pd.Timestamp('2024-04-01'), pd.Timestamp('2025-03-31'), 'FY2024-25'),
    (pd.Timestamp('2025-04-01'), pd.Timestamp('2026-03-31'), 'FY2025-26'),
]


def main():
    root = Path('data')
    market = rb.load(root / 'NIFTY50.csv')
    rb.MARKET_CACHE = market
    files = [p for p in root.glob('*.csv') if p.name != 'NIFTY50.csv']
    rb.get_cache(files, root / 'NIFTY50.csv')
    rb.build_rs_rank(files, market)

    rows = []
    for vm in VOLUME_THRESHOLDS:
        for start, end, fy in FY_WINDOWS:
            result, _, _ = rb.run(
                files, market, start, end,
                near=0.85, tighten=0.65, vol_mult=vm,
            )
            result['fy'] = fy
            rows.append(result)

    df = pd.DataFrame(rows)
    regime = pd.read_csv('results/market_regime_classification.csv')
    if {'fy', 'regime'}.issubset(regime.columns):
        df = df.merge(regime[['fy', 'regime']], on='fy', how='left')
    else:
        df['regime'] = ''

    out = Path('results')
    out.mkdir(exist_ok=True)
    df.to_csv(out / 'bull_market_volume_diagnostics.csv', index=False)

    summary = (
        df.groupby(['regime', 'vol_mult'], dropna=False)
          .agg(
              fy_count=('fy', 'count'),
              return_median=('return_pct', 'median'),
              excess_median=('excess_pct', 'median') if 'excess_pct' in df else ('return_pct', 'median'),
              pf_median=('profit_factor', 'median'),
              sharpe_median=('sharpe', 'median'),
              trades_mean=('trades', 'mean'),
          )
          .reset_index()
    )
    summary.to_csv(out / 'bull_market_volume_summary.csv', index=False)

    # A compact report that makes the bull-regime comparison explicit.
    bull = df[df['regime'].eq('bull')].copy()
    if not bull.empty:
        print('\nBULL REGIME VOLUME DIAGNOSTIC')
        print(
            bull.groupby('vol_mult')
                .agg(return_median=('return_pct', 'median'),
                     trades_mean=('trades', 'mean'),
                     pf_median=('profit_factor', 'median'),
                     sharpe_median=('sharpe', 'median'))
                .round(3)
                .to_string()
        )
    print('\nFULL DIAGNOSTIC')
    print(df[['fy', 'regime', 'vol_mult', 'return_pct', 'profit_factor', 'sharpe', 'trades']].round(3).to_string(index=False))


if __name__ == '__main__':
    main()
