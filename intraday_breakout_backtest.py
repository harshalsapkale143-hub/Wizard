"""Intraday breakout execution model.

Daily setup/market regime is confirmed at the prior close. On the next session,
the first 5-minute bar that trades through the pivot with sufficient cumulative
volume triggers an entry at the breakout level (or bar open if it gaps through).
This avoids the daily-close/next-open entry assumption while avoiding lookahead.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

START = pd.Timestamp("2026-07-10")
END = pd.Timestamp("2026-09-08")
INITIAL = 1_000_000.0
RISK_PCT = 0.005
MAX_POSITIONS = 10
COST_BPS = 10.0
SLIPPAGE_BPS = 5.0
VOL_MULT = 1.5


def load(path):
    d = pd.read_csv(path)
    d["timestamp"] = pd.to_datetime(d["timestamp"], errors="coerce")
    d = d.dropna(subset=["timestamp"]).sort_values("timestamp").set_index("timestamp")
    for c in ["open", "high", "low", "close", "volume"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d.dropna(subset=["open", "high", "low", "close", "volume"])


def daily_features(d, market):
    c, h, l, v = d.close, d.high, d.low, d.volume
    ma50 = c.rolling(50).mean()
    ma150 = c.rolling(150).mean()
    ma200 = c.rolling(200).mean()
    trend = (c > ma50) & (c > ma150) & (c > ma200) & (ma150 > ma200)

    high60 = h.rolling(60).max()
    near_high = c >= high60 * 0.85
    r10 = (h.rolling(10).max() - l.rolling(10).min()) / c
    r30 = (h.rolling(30).max() - l.rolling(30).min()) / c
    dry = v < v.rolling(20).mean() * 0.75
    setup = trend & near_high & (r10 < r30 * 0.65) & dry

    pivot = h.rolling(20).max().shift(1)
    vol20 = v.rolling(20).mean().shift(1)
    rs = c / market.close.reindex(d.index).ffill()
    rs_ma = rs.rolling(50).mean()
    atr = (h - l).rolling(14).mean()
    stop = pd.concat([
        l.rolling(10).min().shift(1),
        c - 1.5 * atr.shift(1)
    ], axis=1).min(axis=1)

    return pd.DataFrame({
        "setup": setup,
        "pivot": pivot,
        "vol20": vol20,
        "rs": rs,
        "rs_ma": rs_ma,
        "stop": stop,
    }, index=d.index)


def intraday_load(path):
    x = load(path)
    # Normalize to NSE session dates while retaining timestamps.
    x["session"] = x.index.normalize()
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--intraday-dir", default="data/intraday5m")
    ap.add_argument("--market", default="NIFTY50.csv")
    ap.add_argument("--output", default="results_intraday")
    args = ap.parse_args()

    root = Path(args.data_dir)
    intra_root = Path(args.intraday_dir)
    out = Path(args.output)
    out.mkdir(exist_ok=True)

    md = load(root / args.market)
    market_ma200 = md.close.rolling(200).mean()
    market_ok = (md.close > market_ma200).shift(1).fillna(False)

    daily = {}
    breadth_rows = []
    files = [p for p in root.glob("*.csv") if p.name != args.market]
    for f in files:
        d = load(f)
        if len(d) < 250:
            continue
        daily[f.stem] = d
        breadth_rows.append((d.close > d.close.rolling(50).mean()).rename(f.stem))

    breadth = pd.concat(breadth_rows, axis=1).mean(axis=1) if breadth_rows else pd.Series(dtype=float)

    candidates = []
    for sym, d in daily.items():
        feat = daily_features(d, md)
        # The setup and all regime/RS information are from the prior close.
        for i in range(1, len(d)):
            day = d.index[i].normalize()
            if day < START or day > END:
                continue
            prev = d.index[i - 1]
            if not bool(feat.at[prev, "setup"]):
                continue
            pivot = feat.at[day, "pivot"]
            vol20 = feat.at[day, "vol20"]
            if not np.isfinite(pivot) or not np.isfinite(vol20):
                continue
            if not bool(market_ok.get(prev, False)):
                continue
            if float(breadth.get(prev, np.nan)) < 0.40:
                continue
            if not (float(feat.at[prev, "rs"]) > float(feat.at[prev, "rs_ma"])):
                continue
            stop = float(feat.at[prev, "stop"])
            if not np.isfinite(stop) or stop >= float(pivot):
                continue
            candidates.append((day, sym, float(pivot), float(vol20), stop))

    candidates.sort()
    intra_cache = {}
    signals = []
    for day, sym, pivot, vol20, stop in candidates:
        p = intra_root / f"{sym}.csv"
        if not p.exists():
            continue
        if sym not in intra_cache:
            intra_cache[sym] = intraday_load(p)
        x = intra_cache[sym]
        bars = x[x.session == day]
        if bars.empty:
            continue
        cumvol = bars.volume.cumsum()
        hit = (bars.high >= pivot) & (cumvol >= vol20 * VOL_MULT)
        if not hit.any():
            continue
        ts = hit[hit].index[0]
        bar = bars.loc[ts]
        raw_entry = float(bar.open) if float(bar.open) > pivot else pivot
        entry = raw_entry * (1.0 + SLIPPAGE_BPS / 10000.0)
        if stop >= entry:
            continue
        signals.append({
            "symbol": sym, "signal_date": day, "entry_ts": ts,
            "pivot": pivot, "entry": entry, "stop": stop,
            "initial_risk": entry - stop,
            "breakout_volume_ratio": float(cumvol.loc[ts] / vol20),
        })

    # Portfolio simulation uses 5m bars for entries/exits.  The delayed
    # 1.5R 50-DMA rule is evaluated on daily closes; the hard stop is intraday.
    signal_map = {}
    for s in signals:
        signal_map.setdefault(s["entry_ts"], []).append(s)

    positions = {}
    cash = INITIAL
    trades = []
    all_ts = sorted({ts for x in intra_cache.values() for ts in x.index if START <= ts.normalize() <= END})

    for ts in all_ts:
        # Intraday hard stops first.
        for sym in list(positions):
            pos = positions[sym]
            x = intra_cache[sym]
            if ts not in x.index:
                continue
            bar = x.loc[ts]
            if float(bar.low) <= pos["stop"]:
                exit_px = min(float(bar.open), pos["stop"]) * (1.0 - SLIPPAGE_BPS / 10000.0) if float(bar.open) <= pos["stop"] else pos["stop"] * (1.0 - SLIPPAGE_BPS / 10000.0)
                proceeds = pos["qty"] * exit_px
                fee = (pos["qty"] * pos["entry"] + proceeds) * COST_BPS / 10000.0
                pnl = proceeds - pos["cost"] - fee
                cash += proceeds - fee
                trades.append({**pos["diag"], "exit_ts": ts, "exit": exit_px, "qty": pos["qty"], "pnl": pnl, "exit_reason": "intraday_stop"})
                del positions[sym]

        # New entries at exact breakout timestamps.
        for s in signal_map.get(ts, []):
            sym = s["symbol"]
            if sym in positions or len(positions) >= MAX_POSITIONS:
                continue
            entry = s["entry"]
            risk = entry - s["stop"]
            equity = cash + sum(p["qty"] * float(intra_cache[x].loc[ts, "close"]) for x, p in positions.items() if ts in intra_cache[x].index)
            qty = int((equity * RISK_PCT) / risk)
            qty = min(qty, int(cash / (entry * (1 + COST_BPS / 10000.0))))
            if qty <= 0:
                continue
            fee = qty * entry * COST_BPS / 10000.0
            cost = qty * entry + fee
            cash -= cost
            diag = {k: v for k, v in s.items() if k not in {"entry", "stop", "initial_risk"}}
            positions[sym] = {"entry": entry, "stop": s["stop"], "risk": risk, "qty": qty, "cost": cost, "diag": {**diag, "entry": entry, "stop": s["stop"], "initial_risk": risk}}

        # Daily close management: once peak R reaches 1.5, activate 50-DMA exit.
        # This is deliberately conservative: the daily close is the earliest
        # point at which that management signal is known.
        if ts.hour == 15 and ts.minute >= 25:
            day = ts.normalize()
            for sym in list(positions):
                pos = positions[sym]
                x = daily[sym]
                if day not in x.index:
                    continue
                close = float(x.loc[day, "close"])
                r = (close - pos["entry"]) / pos["risk"]
                pos["peak_r"] = max(pos.get("peak_r", -np.inf), r)
                if pos["peak_r"] >= 1.5:
                    ma50 = float(x.close.rolling(50).mean().loc[day])
                    if np.isfinite(ma50) and close < ma50:
                        proceeds = pos["qty"] * close
                        fee = (pos["qty"] * pos["entry"] + proceeds) * COST_BPS / 10000.0
                        pnl = proceeds - pos["cost"] - fee
                        cash += proceeds - fee
                        trades.append({**pos["diag"], "exit_ts": ts, "exit": close, "qty": pos["qty"], "pnl": pnl, "exit_reason": "delayed_50dma"})
                        del positions[sym]

    # Mark any remaining positions to their last available 5m close.
    for sym, pos in list(positions.items()):
        x = intra_cache[sym]
        x = x[x.index.normalize() <= END]
        if x.empty:
            continue
        ts = x.index[-1]
        px = float(x.iloc[-1].close)
        proceeds = pos["qty"] * px
        fee = (pos["qty"] * pos["entry"] + proceeds) * COST_BPS / 10000.0
        cash += proceeds - fee
        trades.append({**pos["diag"], "exit_ts": ts, "exit": px, "qty": pos["qty"], "pnl": proceeds - pos["cost"] - fee, "exit_reason": "end_of_test"})

    t = pd.DataFrame(trades)
    t.to_csv(out / "trades.csv", index=False)
    pnl = t.pnl if not t.empty else pd.Series(dtype=float)
    wins, losses = pnl[pnl > 0], pnl[pnl < 0]
    result = {
        "initial_capital": INITIAL, "final_capital": cash,
        "total_return_pct": (cash / INITIAL - 1) * 100,
        "trades": len(t),
        "win_rate_pct": len(wins) / len(pnl) * 100 if len(pnl) else 0,
        "profit_factor": wins.sum() / abs(losses.sum()) if len(losses) else np.inf,
        "slippage_bps": SLIPPAGE_BPS,
        "entry_model": "first_5m_pivot_cross_with_cumulative_volume",
    }
    pd.DataFrame([result]).to_csv(out / "performance.csv", index=False)
    print(pd.DataFrame([result]).to_string(index=False))


if __name__ == "__main__":
    main()
