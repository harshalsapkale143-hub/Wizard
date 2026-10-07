# Wizard Phase 2

Phase 2 adds a fail-closed boundary between acquired market data and the existing Wizard research engine.

## Added

- phase2_data_validation.py validates OHLCV schema, price relationships, volume and timestamps.
- Missing benchmark or stock data blocks a run.
- A point-in-time universe file can be supplied to reduce survivorship bias.
- phase2_pipeline.py runs the existing robustness_backtest only after validation passes.
- Every run emits data_validation.json, phase2_summary.csv, phase2_trades.csv, phase2_equity.csv and phase2_manifest.json.
- No synthetic or placeholder performance values are generated.

## Point-in-time universe

Provide universe_by_date.csv with:

symbol,available_from,available_to

If it is omitted, the run is allowed but reports a survivorship-bias warning.

## Run

python phase2_pipeline.py --data-dir data --start 2025-04-01 --end 2026-03-31

A validation failure exits non-zero and prevents publication of backtest metrics.

## API integration

The Phase 1 API should expose the Phase 2 validation status before showing performance metrics:

PASS = validated
WARN = validated with research-quality caveats
FAIL = do not display performance metrics
