import os
import pandas as pd
import numpy as np

INITIAL = 1_000_000
COST_BPS = 10
RISK_PCT = 0.005
MAX_POSITIONS = 10
CONFIGS = [(0.80,0.65,1.5),(0.85,0.65,1.5),(0.90,0.65,1.5),(0.85,0.65,1.75)]
WINDOWS = [
    ("2019_2022","2019-01-01","2022-12-31"),
    ("2023_2024","2023-01-01","2024-12-31"),
    ("2025_2026","2025-01-01","2026-03-31"),
    ("2021_2026","2021-04-01","2026-03-31"),
]
VARIANTS = [
    "baseline_50dma",
    "partial_50dma_then_atr",
    "partial_50dma_then_20dma",
    "breakeven_2r_50dma",
    "delayed_50dma",
]

# This experiment intentionally reuses the existing winner-preservation backtest engine.
# It applies structural exit changes while keeping entries, sizing, costs, universe,
# dates, and configuration grid unchanged.
try:
    from winner_preservation_fine_grid_experiment import run_backtest
except Exception:
    run_backtest = None


def transform_exit_logic(trades, variant):
    if trades is None or len(trades) == 0:
        return trades
    df = trades.copy()
    # The underlying fine-grid engine may expose raw trade columns. Preserve all
    # columns and add a research label so downstream aggregation can distinguish runs.
    df["structural_exit_variant"] = variant
    return df


def main():
    if run_backtest is None:
        raise RuntimeError("winner_preservation_fine_grid_experiment.py could not be imported")

    # Run the baseline engine for each existing window/configuration, then apply
    # structural exit rules in a dedicated research layer when compatible columns
    # are present. If the engine does not expose the required raw state, fail loudly
    # rather than silently producing invalid backtest numbers.
    rows = []
    trades_out = []
    for window_name, start, end in WINDOWS:
        for cfg in CONFIGS:
            result = run_backtest(start, end, cfg)
            if not isinstance(result, tuple) or len(result) < 2:
                raise RuntimeError("Unexpected backtest return format; refusing to fabricate structural exits")
            summary, trades = result[0], result[1]
            if not isinstance(summary, dict):
                raise RuntimeError("Unexpected summary format")
            # Baseline is directly measured. Structural variants are marked pending
            # unless the engine exposes enough state to model them exactly.
            for variant in VARIANTS:
                if variant == "baseline_50dma":
                    s = dict(summary)
                    s.update({"window":window_name,"variant":variant,"config":str(cfg)})
                    rows.append(s)
                    if isinstance(trades, pd.DataFrame):
                        t = trades.copy(); t["window"] = window_name; t["variant"] = variant; t["config"] = str(cfg); trades_out.append(t)
                else:
                    raise RuntimeError("Structural exit engine requires raw daily position state; implement variant-specific simulation before publishing results")

    os.makedirs("results", exist_ok=True)
    pd.DataFrame(rows).to_csv("results/exit_structural_experiment_results.csv", index=False)
    if trades_out:
        pd.concat(trades_out, ignore_index=True).to_csv("results/exit_structural_experiment_trades.csv", index=False)

if __name__ == "__main__":
    main()
