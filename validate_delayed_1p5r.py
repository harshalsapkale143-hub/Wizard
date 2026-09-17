from pathlib import Path
import pandas as pd

# Validation report for the selected delayed 50-DMA threshold.
# This intentionally does not alter the production backtest rules.
RESULTS = Path('results')
INPUT = RESULTS / 'delayed_50dma_threshold_results.csv'
OUT = RESULTS / 'delayed_50dma_1p5r_validation.csv'

if not INPUT.exists():
    raise FileNotFoundError(INPUT)

df = pd.read_csv(INPUT)
df['threshold_r'] = pd.to_numeric(df['threshold_r'], errors='coerce')
sel = df[df['threshold_r'].eq(1.5)].copy()

if sel.empty:
    raise ValueError('No 1.5R rows found in threshold-grid results')

# Preserve every tested window/config row and add simple robustness flags.
sel['positive_return'] = sel['return'].gt(0)
sel['pf_above_1'] = sel['profit_factor'].gt(1)
sel['sharpe_positive'] = sel['sharpe'].gt(0)
sel['recent_oos'] = sel['start'].astype(str).eq('2025-01-01')
sel.to_csv(OUT, index=False)

summary = sel.groupby('start', as_index=False).agg(
    return_mean=('return','mean'),
    pf_mean=('profit_factor','mean'),
    sharpe_mean=('sharpe','mean'),
    drawdown_mean=('max_drawdown','mean'),
    trades_mean=('trades','mean'),
    positive_configs=('positive_return','sum'),
    total_configs=('positive_return','size'),
)
summary.to_csv(RESULTS / 'delayed_50dma_1p5r_validation_summary.csv', index=False)

print(summary.to_string(index=False))
print('\n1.5R validation rows:', len(sel))
print('2025-2026 OOS rows:', int(sel['recent_oos'].sum()))
