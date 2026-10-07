"""Production-safe Phase 2 orchestration around the existing robustness engine."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd
import phase2_data_validation as gate
import robustness_backtest as rb

def run(data_dir="data", start="2025-04-01", end="2026-03-31",
        market_name="NIFTY50.csv", universe=None, output="results"):
    root = Path(data_dir)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    validation = gate.validate_dataset(root, market_name, universe)
    gate.write_report(validation, out / "data_validation.json")
    gate.assert_backtest_safe(validation)

    market = rb.load(root / market_name)
    files = [p for p in root.glob("*.csv") if p.name != market_name]
    if not files:
        raise RuntimeError("No stock CSV files found after validation")
    rb.get_cache(files, root / market_name)
    rb.build_rs_rank(files, market)

    summary, trades, equity = rb.run(files, market, pd.Timestamp(start), pd.Timestamp(end))
    summary = dict(summary)
    summary.update({
        "schema_version": "wizard.phase2.backtest.v1",
        "data_validation_status": validation["status"],
        "data_validation_report": str(out / "data_validation.json"),
        "engine": "robustness_backtest",
        "market": market_name,
        "universe_file": universe,
    })

    pd.DataFrame([summary]).to_csv(out / "phase2_summary.csv", index=False)
    trades.to_csv(out / "phase2_trades.csv", index=False)
    equity.rename("equity").to_csv(out / "phase2_equity.csv", header=True)

    manifest = {
        "schema_version": "wizard.phase2.run.v1",
        "summary": summary,
        "artifacts": {
            "validation": str(out / "data_validation.json"),
            "summary": str(out / "phase2_summary.csv"),
            "trades": str(out / "phase2_trades.csv"),
            "equity": str(out / "phase2_equity.csv"),
        },
    }
    (out / "phase2_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    return manifest

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--start", default="2025-04-01")
    ap.add_argument("--end", default="2026-03-31")
    ap.add_argument("--market", default="NIFTY50.csv")
    ap.add_argument("--universe", default=None)
    ap.add_argument("--output", default="results")
    args = ap.parse_args()
    print(json.dumps(run(args.data_dir, args.start, args.end, args.market, args.universe, args.output), indent=2, default=str))
