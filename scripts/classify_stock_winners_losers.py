from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

DATA = Path("data")
OUT = Path("results")
MIN_ROWS = 500
WINDOW_YEARS = 3


def load_close(path: Path) -> pd.Series:
    d = pd.read_csv(path, usecols=["timestamp", "close"])
    d["timestamp"] = pd.to_datetime(d["timestamp"], errors="coerce")
    d["close"] = pd.to_numeric(d["close"], errors="coerce")
    d = d.dropna().sort_values("timestamp").drop_duplicates("timestamp")
    return d.set_index("timestamp")["close"]


def cagr(first: float, last: float, years: float) -> float:
    if first <= 0 or last <= 0 or years <= 0:
        return np.nan
    return (last / first) ** (1.0 / years) - 1.0


def max_drawdown(s: pd.Series) -> float:
    if s.empty:
        return np.nan
    return float((s / s.cummax() - 1.0).min())


def period_metrics(close: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    x = close.loc[(close.index >= start) & (close.index <= end)]
    if len(x) < 2:
        return {"rows": len(x), "start_date": "", "end_date": "", "total_return_pct": np.nan,
                "cagr_pct": np.nan, "max_drawdown_pct": np.nan}
    years = max((x.index[-1] - x.index[0]).days / 365.25, 1 / 365.25)
    return {
        "rows": len(x),
        "start_date": x.index[0].date().isoformat(),
        "end_date": x.index[-1].date().isoformat(),
        "total_return_pct": float((x.iloc[-1] / x.iloc[0] - 1.0) * 100.0),
        "cagr_pct": float(cagr(float(x.iloc[0]), float(x.iloc[-1]), years) * 100.0),
        "max_drawdown_pct": float(max_drawdown(x) * 100.0),
    }


def main() -> None:
    files = sorted(p for p in DATA.glob("*.csv") if p.name != "NIFTY50.csv")
    if not files or not (DATA / "NIFTY50.csv").exists():
        raise SystemExit("Stock data and NIFTY50.csv are required.")

    market = load_close(DATA / "NIFTY50.csv")
    if market.empty:
        raise SystemExit("NIFTY50 data is empty.")

    # Use the latest common date so every eligible stock is evaluated against
    # the same market endpoint. A strict three-year window avoids mixing
    # incomparable horizons. Stocks without enough history remain explicit
    # rather than being forced into winner/loser labels.
    common_end = market.index.max()
    start_3y = common_end - pd.DateOffset(years=WINDOW_YEARS)
    m3 = period_metrics(market, start_3y, common_end)
    mfull = period_metrics(market, market.index.min(), common_end)

    # Strategy-specific realized outcome from the already validated baseline
    # trade log, when available. This is supplementary and never determines
    # the market winner/loser label for non-traded stocks.
    trade_stats = pd.DataFrame()
    trade_path = OUT / "trades.csv"
    if trade_path.exists():
        t = pd.read_csv(trade_path)
        if {"symbol", "pnl"}.issubset(t.columns):
            trade_stats = t.groupby("symbol").agg(
                sepa_trades=("pnl", "size"),
                sepa_net_pnl=("pnl", "sum"),
                sepa_win_rate_pct=("pnl", lambda s: float((s > 0).mean() * 100.0)),
            ).reset_index()

    rows = []
    for path in files:
        symbol = path.stem.upper()
        try:
            close = load_close(path)
            full = period_metrics(close, close.index.min(), common_end)
            three = period_metrics(close, start_3y, common_end)
            row = {
                "symbol": symbol,
                "data_rows": len(close),
                "history_start": close.index.min().date().isoformat() if not close.empty else "",
                "history_end": close.index.max().date().isoformat() if not close.empty else "",
                "full_return_pct": full["total_return_pct"],
                "full_cagr_pct": full["cagr_pct"],
                "full_max_drawdown_pct": full["max_drawdown_pct"],
                "three_year_return_pct": three["total_return_pct"],
                "three_year_cagr_pct": three["cagr_pct"],
                "three_year_max_drawdown_pct": three["max_drawdown_pct"],
                "nifty_three_year_return_pct": m3["total_return_pct"],
                "nifty_three_year_cagr_pct": m3["cagr_pct"],
                "excess_three_year_return_pct": three["total_return_pct"] - m3["total_return_pct"] if pd.notna(three["total_return_pct"]) else np.nan,
                "excess_three_year_cagr_pct": three["cagr_pct"] - m3["cagr_pct"] if pd.notna(three["cagr_pct"]) else np.nan,
                "nifty_full_cagr_pct": mfull["cagr_pct"],
            }
            # Clear historical labels only: winner requires positive absolute
            # CAGR and outperformance; loser requires negative absolute CAGR
            # and underperformance. Everything else is Mixed.
            if three["rows"] < MIN_ROWS or pd.isna(three["cagr_pct"]):
                label = "INSUFFICIENT_HISTORY"
            elif three["cagr_pct"] > 0 and three["cagr_pct"] > m3["cagr_pct"]:
                label = "WINNER"
            elif three["cagr_pct"] < 0 and three["cagr_pct"] < m3["cagr_pct"]:
                label = "LOSER"
            else:
                label = "MIXED"
            row["historical_label"] = label
            rows.append(row)
        except Exception as exc:
            rows.append({"symbol": symbol, "data_rows": 0, "historical_label": "DATA_ERROR", "error": str(exc)})

    result = pd.DataFrame(rows)
    if not trade_stats.empty:
        result = result.merge(trade_stats, on="symbol", how="left")
    else:
        result["sepa_trades"] = 0
        result["sepa_net_pnl"] = np.nan
        result["sepa_win_rate_pct"] = np.nan
    result["sepa_trades"] = result["sepa_trades"].fillna(0).astype(int)
    result["sepa_trade_label"] = np.select(
        [result["sepa_net_pnl"] > 0, result["sepa_net_pnl"] < 0],
        ["PROFITABLE_TRADES", "LOSING_TRADES"],
        default="NO_BASELINE_TRADE",
    )
    result = result.sort_values(["historical_label", "excess_three_year_cagr_pct"], ascending=[True, False])

    OUT.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT / "stock_winner_loser_classification.csv", index=False)

    summary = result["historical_label"].value_counts(dropna=False).rename_axis("label").reset_index(name="stocks")
    summary.to_csv(OUT / "stock_winner_loser_summary.csv", index=False)
    print(f"Universe files: {len(files)}")
    print(f"Common endpoint: {common_end.date()}")
    print(f"3-year window: {start_3y.date()} to {common_end.date()}")
    print(summary.to_string(index=False))
    print("\nTop historical winners:")
    print(result[result.historical_label == "WINNER"][['symbol','three_year_cagr_pct','excess_three_year_cagr_pct','three_year_max_drawdown_pct']].head(20).to_string(index=False))
    print("\nHistorical losers:")
    print(result[result.historical_label == "LOSER"][['symbol','three_year_cagr_pct','excess_three_year_cagr_pct','three_year_max_drawdown_pct']].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
