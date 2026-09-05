"""Daily-data SEPA/VCP backtest.

CSV format: timestamp,open,high,low,close,volume
Market file: data/NIFTY50.csv
Stock files: data/<SYMBOL>.csv

This is a research backtest, not investment advice. It deliberately avoids
same-day look-ahead: indicators and the setup are known at the prior close;
entry is at the following session's close when the breakout conditions hold.
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


def load(path):
    d = pd.read_csv(path)
    d["timestamp"] = pd.to_datetime(d["timestamp"])
    d = d.sort_values("timestamp").set_index("timestamp")
    for c in ["open","high","low","close","volume"]: d[c] = pd.to_numeric(d[c], errors="coerce")
    return d.dropna(subset=["open","high","low","close","volume"])


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
    # Practical, testable proxy for a tightening base: recent range is
    # materially smaller than the preceding range and volume is drying up.
    r10 = (h.rolling(10).max() - l.rolling(10).min()) / c
    r30 = (h.rolling(30).max() - l.rolling(30).min()) / c
    tightening = r10 < r30 * 0.65
    dry = v < vol20 * 0.75
    return trend_template(d) & (c > ma50) & near_high & tightening & dry


def backtest(files, market):
    md = load(market)
    md = md.loc[(md.index >= START - pd.Timedelta(days=400)) & (md.index <= END)]
    market_ma200 = md.close.rolling(200).mean()
    market_ok = md.close > market_ma200

    signals = []
    for f in files:
        d = load(f)
        d = d.loc[(d.index >= START - pd.Timedelta(days=400)) & (d.index <= END)]
        if len(d) < 250: continue
        setup = vcp_setup(d)
        pivot = d.high.rolling(20).max().shift(1)
        vol20 = d.volume.rolling(20).mean().shift(1)
        rs = d.close / md.close.reindex(d.index).ffill()
        rs_ma = rs.rolling(50).mean()
        for i in range(1, len(d)):
            dt = d.index[i]
            if dt < START or dt > END or not setup.iloc[i-1]: continue
            if not bool(market_ok.reindex([dt]).fillna(False).iloc[0]): continue
            p = pivot.iloc[i]
            if not np.isfinite(p) or d.close.iloc[i] <= p: continue
            if d.volume.iloc[i] < vol20.iloc[i] * 1.5: continue
            if rs.iloc[i] <= rs_ma.iloc[i]: continue
            atr = (d.high - d.low).rolling(14).mean().iloc[i-1]
            stop = min(d.low.iloc[max(0,i-10):i].min(), d.close.iloc[i] - 1.5 * atr)
            if np.isfinite(stop) and stop < d.close.iloc[i]:
                signals.append((dt, f.stem, float(d.close.iloc[i]), float(stop)))
    signals.sort()

    cash = INITIAL
    positions = {}
    trades = []
    dates = sorted(set(md.index[(md.index >= START) & (md.index <= END)]))
    stock_cache = {f.stem: load(f) for f in files}
    for dt in dates:
        # exits at today's close; entry signals generated from today's close
        # are intentionally ignored until the next loop day.
        for sym in list(positions):
            pos = positions[sym]; d = stock_cache[sym]
            if dt not in d.index: continue
            px = float(d.loc[dt, "close"])
            if px <= pos["stop"] or px < float(d.close.rolling(50).mean().loc[dt]):
                proceeds = pos["qty"] * px
                pnl = proceeds - pos["cost"]
                cash += proceeds
                trades.append({"entry_date":pos["date"],"exit_date":dt,"symbol":sym,"entry":pos["entry"],"exit":px,"qty":pos["qty"],"pnl":pnl})
                del positions[sym]
        # entries only from signals strictly before today's date
        for sdt, sym, entry, stop in signals:
            if sdt >= dt or sym in positions or len(positions) >= MAX_POSITIONS: continue
            # use the next available trading day after the signal
            d = stock_cache[sym]
            future = d.index[d.index > sdt]
            if len(future)==0 or future[0] != dt: continue
            risk_per_share = entry - stop
            equity = cash + sum(p["qty"] * float(stock_cache[x].loc[dt,"close"]) for x,p in positions.items() if dt in stock_cache[x].index)
            qty = int((equity * RISK_PCT) / risk_per_share) if risk_per_share > 0 else 0
            qty = min(qty, int(cash / entry))
            if qty <= 0: continue
            cost = qty * entry
            cash -= cost
            positions[sym] = {"date":dt,"entry":entry,"stop":stop,"qty":qty,"cost":cost}

    # close remaining positions at final available close
    for sym,pos in list(positions.items()):
        d=stock_cache[sym]; px=float(d.loc[:END].close.iloc[-1]); proceeds=pos["qty"]*px
        trades.append({"entry_date":pos["date"],"exit_date":d.loc[:END].index[-1],"symbol":sym,"entry":pos["entry"],"exit":px,"qty":pos["qty"],"pnl":proceeds-pos["cost"]})
        cash += proceeds
    return pd.DataFrame(trades), cash


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--data-dir",default="data"); ap.add_argument("--market",default="NIFTY50.csv"); ap.add_argument("--output",default="results"); args=ap.parse_args()
    root=Path(args.data_dir); files=[p for p in root.glob("*.csv") if p.name != args.market]
    trades, final=backtest(files, root/args.market)
    out=Path(args.output); out.mkdir(exist_ok=True)
    trades.to_csv(out/"trades.csv",index=False)
    pnl=trades.pnl if not trades.empty else pd.Series(dtype=float)
    wins=pnl[pnl>0]; losses=pnl[pnl<0]
    result={"initial_capital":INITIAL,"final_capital":final,"total_return_pct":(final/INITIAL-1)*100,"trades":len(trades),"win_rate_pct":(len(wins)/len(pnl)*100 if len(pnl) else 0),"profit_factor":(wins.sum()/abs(losses.sum()) if len(losses) else np.inf)}
    pd.DataFrame([result]).to_csv(out/"performance.csv",index=False)
    print(pd.DataFrame([result]).to_string(index=False))

if __name__ == "__main__": main()
