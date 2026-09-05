# Minervini SEPA/VCP NSE Algo V3.1

Research/backtest implementation based on the project discussion of Minervini-style SEPA/VCP rules.

## Backtest target

2025-04-01 through 2026-03-31.

## Required data

Put NSE 5-minute OHLCV CSV files in `data/` with columns:

`timestamp,open,high,low,close,volume`

Also provide `data/NIFTY50.csv`.

For an unbiased historical test, provide:

- `universe_by_date.csv` with `symbol,available_from,available_to`
- point-in-time fundamentals where available
- sector mapping where available

Do not use today's constituents as a historical universe if avoiding survivorship bias is important.

## Run

```bash
python minervini_sepa_nse_algo_v3.py --data-dir data --market NIFTY50 --output results
```

With a point-in-time universe:

```bash
python minervini_sepa_nse_algo_v3.py --data-dir data --market NIFTY50 --universe universe_by_date.csv --output results
```

## Outputs

`results/trades.csv` and `results/equity_curve.csv` are produced by the engine.

The implementation is for research and backtesting, not a guarantee of live performance or financial advice.
