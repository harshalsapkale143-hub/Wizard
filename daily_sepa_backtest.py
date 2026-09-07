"""Daily-data SEPA/VCP backtest.

Signals are evaluated using information available at the prior close.
A confirmed breakout is entered at the following session open, with
position sizing based on the actual entry price. This avoids the previous
mismatch where the signal used the breakout-day close while the position
was entered on the next session.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

START = pd.Timestamp("2025-04-01")
END = pd.Timestamp("2026-03-31")
INITIAL = 1_000_000.0
RISK_PCT = 0.005
MAX_POSITIONS = 10
COST_BPS = 10.0


def load(path):
    d = pd.read_csv(path)
    d["timestamp"] = pd.to_datetime(d["timestamp"])
    d = d.sort_values("timestamp").set_index("timestamp")
    for c in ["open", "high", "low", "close", "volume"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d.dropna(subset=["open", "high", "low", "close", "volume"])


def trend_template(d):
    c = d.close
    ma50, ma150, ma200 = c.rolling(50).mean(), c.rolling(150).mean(), c.rolling(200).mean()
    return (c > ma150) & (c > ma200) & (ma150 > ma200) & (c > ma50)


def vcp_setup(d):
    c, h, l, v = d.close, d.high, d.low, d.volume
    ma50, ma150, ma200 = c.rolling(50).mean(), c.rolling(150).mean(), c.rolling(200).mean()
    vol20 = v.rolling(20).mean()
    high60 = h.rolling(60).max()
    near_high = c >= high60 * 0.85
    r10 = (h.rolling(10).max() - l.rolling(10).min()) / c
    r30 = (h.rolling(30).max() - l.rolling(30).min()) / c
    tightening = r10 < r30 * 0.65
    dry = v < vol20 * 0.75
    return trend_template(d) & (c > ma50) & near_high & tightening & dry


def build_signals(d, market):
    setup = vcp_setup(d)
    pivot = d.high.rolling(20).max().shift(1)
    vol20 = d.volume.rolling(20).mean().shift(1)
    rs = d.close / market.close.reindex(d.index).ffill()
    rs_ma = rs.rolling(50).mean()
    signals = []
    for i in range(1, len(d) - 1):
        signal_dt = d.index[i]
        entry_dt = d.index[i + 1]
        if signal_dt < START or signal_dt > END:
            continue
        if not bool(setup.iloc[i - 1]):
            continue
        p = pivot.iloc[i]
        if not np.isfinite(p) or d.close.iloc[i] <= p:
            continue
        if d.volume.iloc[i] < vol20.iloc[i] * 1.5:
            continue
        if rs.iloc[i] <= rs_ma.iloc[i]:
            continue
        atr = (d.high - d.low).rolling(14).mean().iloc[i - 1]
        stop = min(float(d.low.iloc[max(0, i - 10):i].min()), float(d.close.iloc[i] - 1.5 * atr))
        entry = float(d.open.iloc[i + 1])
        if np.isfinite(stop) and stop < entry:
            signals.append({
                "signal_date": signal_dt,
                "entry_date": entry_dt,
                "symbol": d.attrs.get("symbol", ""),
                "pivot": float(p),
                "signal_close": float(d.close.iloc[i]),
                "entry": entry,
                "gap_pct": (entry / float(d.close.iloc[i]) - 1.0) * 100.0,
                "stop": stop,
                "initial_risk": entry - stop,
                "breakout_volume_ratio": float(d.volume.iloc[i] / vol20.iloc[i]) if vol20.iloc[i] else np.nan,
                "rs_ratio": float(rs.iloc[i]),
                "rs_ma_ratio": float(rs.iloc[i] / rs_ma.iloc[i]) if rs_ma.iloc[i] else np.nan,
                "distance_from_pivot_pct": (float(d.close.iloc[i]) / float(p) - 1.0) * 100.0,
            })
    return signals


def backtest(files, market):
    md = load(market)
    md = md.loc[(md.index >= START - pd.Timedelta(days=400)) & (md.index <= END)]
    market_ma200 = md.close.rolling(200).mean()
    market_ok = (md.close > market_ma200).shift(1).fillna(False)

    signals = []
    stock_cache = {}
    for f in files:
        d = load(f)
        d.attrs["symbol"] = f.stem
        d = d.loc[(d.index >= START - pd.Timedelta(days=400)) & (d.index <= END)]
        if len(d) < 250:
            continue
        stock_cache[f.stem] = d
        signals.extend(build_signals(d, md))
    signals.sort(key=lambda x: (x["entry_date"], x["symbol"]))

    cash = INITIAL
    positions = {}
    trades = []
    signal_map = {}
    for s in signals:
        signal_map.setdefault(s["entry_date"], []).append(s)

    dates = sorted(set(md.index[(md.index >= START) & (md.index <= END)]))
    for dt in dates:
        # Existing positions are managed before new entries. Stops are based on
        # closes in this daily model; the entry day is therefore not treated as
        # an exit day unless its close breaches the stop/50DMA.
        for sym in list(positions):
            pos = positions[sym]
            d = stock_cache[sym]
            if dt not in d.index:
                continue
            px = float(d.loc[dt, "close"])
            ma50 = float(d.close.rolling(50).mean().loc[dt])
            if px <= pos["stop"] or (np.isfinite(ma50) and px < ma50):
                proceeds = pos["qty"] * px
                fee = (pos["qty"] * pos["entry"] + proceeds) * COST_BPS / 10000.0
                pnl = proceeds - pos["cost"] - fee
                cash += proceeds - fee
                trades.append({**pos["diagnostics"], "exit_date": dt, "exit": px, "qty": pos["qty"], "pnl": pnl, "exit_reason": "stop_or_50dma"})
                del positions[sym]

        for s in signal_map.get(dt, []):
            sym = s["symbol"]
            if sym in positions or len(positions) >= MAX_POSITIONS:
                continue
            d = stock_cache[sym]
            entry = float(d.loc[dt, "open"])
            if not np.isfinite(entry) or entry <= 0:
                continue
            risk = entry - s["stop"]
            if risk <= 0:
                continue
            equity = cash + sum(p["qty"] * float(stock_cache[x].loc[dt, "close"]) for x, p in positions.items() if dt in stock_cache[x].index)
            qty = int((equity * RISK_PCT) / risk)
            qty = min(qty, int(cash / (entry * (1 + COST_BPS / 10000.0))))
            if qty <= 0:
                continue
            fee = qty * entry * COST_BPS / 10000.0
            cost = qty * entry + fee
            cash -= cost
            diagnostics = {k: v for k, v in s.items() if k not in {"entry_date", "entry", "stop", "initial_risk"}}
            diagnostics.update({"entry_date": dt, "entry": entry, "stop": s["stop"], "initial_risk": risk})
            positions[sym] = {"date": dt, "entry": entry, "stop": s["stop"], "risk": risk, "qty": qty, "cost": cost, "diagnostics": diagnostics}

    for sym, pos in list(positions.items()):
        d = stock_cache[sym]
        final_dt = d.loc[:END].index[-1]
        px = float(d.loc[final_dt, "close"])
        proceeds = pos["qty"] * px
        fee = (pos["qty"] * pos["entry"] + proceeds) * COST_BPS / 10000.0
        cash += proceeds - fee
        trades.append({**pos["diagnostics"], "exit_date": final_dt, "exit": px, "qty": pos["qty"], "pnl": proceeds - pos["cost"] - fee, "exit_reason": "end_of_test"})

    return pd.DataFrame(trades), cash


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--market", default="NIFTY50.csv")
    ap.add_argument("--output", default="results")
    args = ap.parse_args()
    root = Path(args.data_dir)
    files = [p for p in root.glob("*.csv") if p.name != args.market]
    trades, final = backtest(files, root / args.market)
    out = Path(args.output)
    out.mkdir(exist_ok=True)
    trades.to_csv(out / "trades.csv", index=False)
    pnl = trades.pnl if not trades.empty else pd.Series(dtype=float)
    wins, losses = pnl[pnl > 0], pnl[pnl < 0]
    result = {
        "initial_capital": INITIAL,
        "final_capital": final,
        "total_return_pct": (final / INITIAL - 1) * 100,
        "trades": len(trades),
        "win_rate_pct": len(wins) / len(pnl) * 100 if len(pnl) else 0,
        "profit_factor": wins.sum() / abs(losses.sum()) if len(losses) else np.inf,
    }
    pd.DataFrame([result]).to_csv(out / "performance.csv", index=False)
    if not trades.empty:
        losers = trades[trades.pnl < 0].copy()
        losers["loss_r"] = losers.pnl / (losers.qty * losers.initial_risk)
        losers.sort_values("pnl").to_csv(out / "losing_trades_diagnostics.csv", index=False)
        print("LOSING TRADES")
        print(losers[["symbol", "entry_date", "exit_date", "pnl", "loss_r", "gap_pct", "breakout_volume_ratio", "distance_from_pivot_pct"]].to_string(index=False))
    print(pd.DataFrame([result]).to_string(index=False))


if __name__ == "__main__":
    main()
