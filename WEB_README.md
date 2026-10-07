# Wizard Web Dashboard

Initial web frontend and API bridge for the Wizard SEPA research platform.

## Run locally

From the repository root:

```bash
python web_api.py
```

Open `http://localhost:8000`.

The server serves the frontend and exposes:

- `GET /api/health` — data availability
- `GET /api/summary` — latest saved backtest summary
- `GET /api/equity` — saved equity curve
- `GET /api/trades` — saved trades
- `GET /api/candidates` — latest SEPA technical scan
- `POST /api/backtest` — runs the existing `robustness_backtest.py`

## Data requirement

The API expects the existing research format:

`data/*.csv` with `timestamp,open,high,low,close,volume` and `data/NIFTY50.csv`.

If the data is absent, the website reports that state instead of fabricating live results.

## Current architecture

`index.html + styles.css + app.js` → `web_api.py` → existing Wizard Python research engine → `results/`.

The frontend is intentionally not the source of strategy logic. The Python research engine remains authoritative.

## Important

The UI no longer treats placeholder performance numbers as live data. Saved results are displayed only when the API finds actual result files, and the Run Backtest button invokes the research engine.

Next integration work should add point-in-time universe/fundamental data and replace the synchronous API run with a background job/status model for long backtests.
