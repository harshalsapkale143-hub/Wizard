from __future__ import annotations
from pathlib import Path
import itertools
import numpy as np
import pandas as pd
import robustness_backtest as rb

INITIAL = 1_000_000.0
CONFIGS = list(itertools.product([0.80, 0.85, 0.90], [0.55, 0.65, 0.75], [1.25, 1.50, 1.75]))
BASE = (0.85, 0.65, 1.50)
FY_WINDOWS = [
    ("FY2021-22", pd.Timestamp("2021-04-01"), pd.Timestamp("2022-03-31")),
    ("FY2022-23", pd.Timestamp("2022-04-01"), pd.Timestamp("2023-03-31")),
    ("FY2023-24", pd.Timestamp("2023-04-01"), pd.Timestamp("2024-03-31")),
    ("FY2024-25", pd.Timestamp("2024-04-01"), pd.Timestamp("2025-03-31")),
    ("FY2025-26", pd.Timestamp("2025-04-01"), pd.Timestamp("2026-03-31")),
]


def main():
    root = Path("data")
    market_path = root / "NIFTY50.csv"
    market = rb.load(market_path)
    files = [p for p in root.glob("*.csv") if p.name != "NIFTY50.csv"]
    rb.get_cache(files, market_path)
    rb.build_rs_rank(files, market)

    # Warm signal cache for the complete existing 27-point robustness grid.
    for near, tighten, vm in CONFIGS:
        for sym, d in rb.DATA_CACHE.items():
            rb.signals(sym, d, market, near, tighten, vm)

    rows = []
    trade_rows = []
    for label, start, end in FY_WINDOWS:
        result, trades, _ = rb.run(files, market, start, end, *BASE)
        bench_series = market.loc[start:end, "close"]
        nifty_return = float((bench_series.iloc[-1] / bench_series.iloc[0] - 1) * 100) if len(bench_series) > 1 else 0.0
        result.update({
            "fy": label,
            "nifty_buy_hold_return_pct": nifty_return,
            "strategy_minus_nifty_pct": result["return_pct"] - nifty_return,
            "config_near_high": BASE[0], "config_tighten": BASE[1], "config_vol_mult": BASE[2],
        })
        rows.append(result)
        if len(trades):
            trades = trades.copy(); trades["fy"] = label; trade_rows.append(trades)

    # Full five-FY period using the same fixed baseline config.
    five_start, five_end = FY_WINDOWS[0][1], FY_WINDOWS[-1][2]
    overall, overall_trades, equity = rb.run(files, market, five_start, five_end, *BASE)
    bench = market.loc[five_start:five_end, "close"]
    nifty5 = float((bench.iloc[-1] / bench.iloc[0] - 1) * 100) if len(bench) > 1 else 0.0
    overall.update({"period": "FY2021-22_to_FY2025-26", "nifty_buy_hold_return_pct": nifty5,
                    "strategy_minus_nifty_pct": overall["return_pct"] - nifty5,
                    "config_near_high": BASE[0], "config_tighten": BASE[1], "config_vol_mult": BASE[2]})

    # Five-year robustness: evaluate every existing configuration over the same period.
    grid = []
    for near, tighten, vm in CONFIGS:
        x, _, _ = rb.run(files, market, five_start, five_end, near, tighten, vm)
        x.update({"period": "FY2021-22_to_FY2025-26", "near_high": near, "tighten": tighten, "vol_mult": vm})
        grid.append(x)
    grid_df = pd.DataFrame(grid).sort_values(["return_pct", "profit_factor"], ascending=False)

    fy_df = pd.DataFrame(rows)
    out = Path("results"); out.mkdir(exist_ok=True)
    fy_df.to_csv(out / "five_year_fy_summary.csv", index=False)
    pd.DataFrame([overall]).to_csv(out / "five_year_overall.csv", index=False)
    grid_df.to_csv(out / "five_year_robustness_grid.csv", index=False)
    equity.rename("equity").to_csv(out / "five_year_equity_curve.csv")
    if trade_rows:
        pd.concat(trade_rows, ignore_index=True).to_csv(out / "five_year_trades.csv", index=False)
    overall_trades.to_csv(out / "five_year_overall_trades.csv", index=False)

    positive = int((fy_df.return_pct > 0).sum())
    print("\nFIVE-YEAR FY SUMMARY\n", fy_df.to_string(index=False))
    print("\nFIVE-YEAR OVERALL\n", pd.DataFrame([overall]).to_string(index=False))
    print("\nFIVE-YEAR ROBUSTNESS\n", grid_df[["near_high","tighten","vol_mult","return_pct","max_drawdown_pct","trades","win_rate_pct","profit_factor","sharpe"]].to_string(index=False))
    print(f"\nPositive FYs: {positive}/{len(fy_df)}")
    print(f"Robustness configs with positive five-year return: {int((grid_df.return_pct > 0).sum())}/{len(grid_df)}")

if __name__ == "__main__":
    main()
