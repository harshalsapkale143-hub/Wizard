# Wizard Web Dashboard

Initial frontend for the Wizard SEPA research platform.

## Files
- `index.html` — application shell and views
- `styles.css` — responsive dark trading UI
- `app.js` — navigation, demo charts, scanner/trade tables and controls

## Current state
The frontend is intentionally connected only to demo/sample display values. It does **not** claim those values are live backtest results.

## Next integration
Connect the UI to the existing Python research engine (`daily_sepa_backtest.py`, `robustness_backtest.py`, and the fundamental experiments) through a small API/service layer. The API should return run IDs, status, metrics, equity series, candidates, and trade logs.
