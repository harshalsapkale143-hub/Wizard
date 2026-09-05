# Minervini SEPA/VCP NSE Algo V3.1
# Full research/backtest engine generated for the Wizard project.
# See README_minervini_v3.md for data requirements and execution instructions.

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, List

import numpy as np
import pandas as pd

# Strategy constants
BREAKOUT_VOLUME_MULTIPLIER = 1.5
ATR_STOP_MULT = 2.0
MAX_PORTFOLIO_RISK = 0.06
MAX_POSITIONS = 10

START = pd.Timestamp("2025-04-01", tz="Asia/Kolkata")
END = pd.Timestamp("2026-03-31 23:59:59", tz="Asia/Kolkata")


def load_ohlcv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = {str(c).lower(): c for c in df.columns}
    required = ["timestamp", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in cols]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    df = df.rename(columns={cols[c]: c for c in required})
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("Asia/Kolkata")
    else:
        df["timestamp"] = df["timestamp"].dt.tz_convert("Asia/Kolkata")
    for c in required[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=required).sort_values("timestamp").set_index("timestamp")


def daily_from_intraday(df: pd.DataFrame) -> pd.DataFrame:
    return df.resample("1D").agg({"open":"first", "high":"max", "low":"min", "close":"last", "volume":"sum"}).dropna()


def load_universe(path: Optional[str]) -> Optional[pd.DataFrame]:
    if not path:
        return None
    df = pd.read_csv(path)
    required = {"symbol", "available_from", "available_to"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"universe_by_date.csv needs {sorted(required)}; missing {sorted(missing)}")
    df["symbol"] = df["symbol"].astype(str).str.upper()
    df["available_from"] = pd.to_datetime(df["available_from"], errors="coerce")
    df["available_to"] = pd.to_datetime(df["available_to"], errors="coerce")
    for c in ["available_from", "available_to"]:
        if df[c].dt.tz is None:
            df[c] = df[c].dt.tz_localize("Asia/Kolkata")
        else:
            df[c] = df[c].dt.tz_convert("Asia/Kolkata")
    return df.dropna(subset=["available_from", "available_to"])


def universe_contains(universe: Optional[pd.DataFrame], symbol: str, date: pd.Timestamp) -> bool:
    if universe is None:
        return True
    return not universe[(universe.symbol == symbol) & (universe.available_from <= date) & (universe.available_to >= date)].empty


def rs_percentile(close: pd.Series, market_close: pd.Series) -> float:
    if len(close) < 252 or len(market_close) < 252:
        return np.nan
    s = close.iloc[-1] / close.iloc[-252] - 1
    m = market_close.iloc[-1] / market_close.iloc[-252] - 1
    return float(50 + 50 * np.tanh((s - m) * 4))


def trend_template(d: pd.DataFrame) -> bool:
    if len(d) < 220:
        return False
    c = d.close
    ma50 = c.rolling(50).mean().iloc[-1]
    ma150 = c.rolling(150).mean().iloc[-1]
    ma200 = c.rolling(200).mean().iloc[-1]
    ma200_20 = c.rolling(200).mean().iloc[-21]
    high52 = c.iloc[-252:].max() if len(c) >= 252 else c.max()
    low52 = c.iloc[-252:].min() if len(c) >= 252 else c.min()
    return bool(c.iloc[-1] > ma50 > ma150 > ma200 and ma200 > ma200_20 and c.iloc[-1] >= 0.75 * high52 and c.iloc[-1] >= 1.25 * low52)


def detect_vcp(d: pd.DataFrame) -> Optional[dict]:
    if len(d) < 80:
        return None
    x = d.tail(100).copy()
    high = x.high
    low = x.low
    close = x.close
    pivot = float(high.tail(20).max())
    recent = close.tail(20)
    if recent.max() <= 0:
        return None
    peak = float(recent.max())
    trough = float(recent.min())
    contraction = (peak - trough) / peak
    if contraction <= 0 or contraction > 0.25:
        return None
    prior = close.iloc[-40:-20]
    if len(prior) < 10:
        return None
    prior_contraction = (prior.max() - prior.min()) / prior.max()
    if prior_contraction <= contraction:
        return None
    vol = d.volume
    vol20 = vol.rolling(20).mean().iloc[-1]
    if not np.isfinite(vol20) or vol20 <= 0:
        return None
    dryup = float(vol.tail(10).mean() / vol20)
    if dryup > 0.9:
        return None
    tr = pd.concat([high-low, (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
    atr = float(tr.rolling(14).mean().iloc[-1])
    return {"pivot": pivot, "atr": atr, "contraction": contraction, "dryup": dryup}


def market_regime(market_daily: pd.DataFrame, date: pd.Timestamp) -> bool:
    d = market_daily.loc[market_daily.index <= date]
    if len(d) < 200:
        return False
    c = d.close
    return bool(c.iloc[-1] > c.rolling(200).mean().iloc[-1] and c.rolling(50).mean().iloc[-1] > c.rolling(200).mean().iloc[-1])


@dataclass
class Position:
    symbol: str
    qty: int
    entry_time: pd.Timestamp
    entry: float
    stop: float
    risk_per_share: float
    highest: float


class Backtester:
    def __init__(self, intraday: Dict[str,pd.DataFrame], market_daily: pd.DataFrame, fundamentals=None, sector_map=None, universe=None, initial_capital=1_000_000):
        self.intraday = intraday
        self.market_daily = market_daily
        self.fundamentals = fundamentals
        self.sector_map = sector_map or {}
        self.universe = universe
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str,Position] = {}
        self.trades = []
        self.equity = []

    def enter(self, symbol, ts, price, stop, atr):
        if len(self.positions) >= MAX_POSITIONS or symbol in self.positions:
            return
        risk_budget = self.initial_capital * MAX_PORTFOLIO_RISK / max(MAX_POSITIONS, 1)
        risk_per_share = max(price - stop, 0.01)
        qty = int(risk_budget / risk_per_share)
        if qty <= 0:
            return
        cost = qty * price
        if cost > self.cash:
            qty = int(self.cash / price)
        if qty <= 0:
            return
        self.cash -= qty * price
        self.positions[symbol] = Position(symbol, qty, ts, price, stop, risk_per_share, price)

    def exit(self, symbol, ts, price, reason):
        p = self.positions.pop(symbol)
        proceeds = p.qty * price
        self.cash += proceeds
        pnl = (price - p.entry) * p.qty
        self.trades.append({"symbol":symbol,"entry_time":p.entry_time,"exit_time":ts,"entry":p.entry,"exit":price,"qty":p.qty,"pnl":pnl,"return_pct":price/p.entry-1,"reason":reason})

    def run(self):
        daily = {s: daily_from_intraday(df) for s,df in self.intraday.items()}
        all_dates = sorted(set().union(*[set(x.index.normalize()) for x in daily.values()]))
        setups = []
        for day in all_dates:
            date = pd.Timestamp(day)
            if date < START.normalize() or date > END.normalize():
                continue
            if not market_regime(self.market_daily, date):
                continue
            for symbol, d in daily.items():
                if not universe_contains(self.universe, symbol, date):
                    continue
                hist = d.loc[d.index < date]
                if not trend_template(hist):
                    continue
                sig = detect_vcp(hist)
                if sig:
                    sig.update({"symbol":symbol,"date":date})
                    setups.append(sig)

        for day_date in all_dates:
            day_date = pd.Timestamp(day_date)
            if day_date < START.normalize() or day_date > END.normalize():
                continue
            candidates = [s for s in setups if s["date"] < day_date and s["date"] >= day_date - pd.Timedelta(days=10)]
            latest = {}
            for sig in candidates:
                if sig["symbol"] not in latest or sig["date"] > latest[sig["symbol"]]["date"]:
                    latest[sig["symbol"]] = sig
            for symbol, sig in latest.items():
                if symbol in self.positions or symbol not in self.intraday:
                    continue
                bars = self.intraday[symbol].loc[self.intraday[symbol].index.normalize() == day_date].copy()
                if bars.empty:
                    continue
                bars["vol_ma20"] = bars.volume.shift(1).rolling(20).mean()
                for ts, row in bars.iterrows():
                    if not (row.close > sig["pivot"] and row.high > sig["pivot"] and np.isfinite(row.vol_ma20) and row.volume >= row.vol_ma20 * BREAKOUT_VOLUME_MULTIPLIER):
                        continue
                    rng = row.high - row.low
                    if rng <= 0 or (row.close-row.low)/rng < 0.70:
                        continue
                    prior = bars.loc[bars.index < ts].tail(12)
                    if len(prior) < 5:
                        continue
                    prev_close = prior.close.shift()
                    tr = pd.concat([prior.high-prior.low,(prior.high-prev_close).abs(),(prior.low-prev_close).abs()],axis=1).max(axis=1)
                    atr5 = tr.rolling(14).mean().iloc[-1]
                    if not np.isfinite(atr5): atr5 = sig["atr"]
                    recent_low = float(prior.low.min())
                    entry = float(row.close)
                    stop = min(recent_low, entry - ATR_STOP_MULT*float(atr5))
                    self.enter(symbol, ts, entry, stop, float(atr5))
                    break

            # Manage open positions using each day's 5-minute bars.
            for symbol in list(self.positions):
                if symbol not in self.intraday:
                    continue
                bars = self.intraday[symbol].loc[self.intraday[symbol].index.normalize() == day_date]
                for ts,row in bars.iterrows():
                    p = self.positions.get(symbol)
                    if not p: break
                    p.highest = max(p.highest, float(row.high))
                    trail = p.highest - 3.0*p.risk_per_share
                    p.stop = max(p.stop, trail)
                    if row.low <= p.stop:
                        self.exit(symbol, ts, float(min(row.open,p.stop)), "stop")
                        break
            total = self.cash
            for symbol,p in self.positions.items():
                bars = self.intraday[symbol].loc[self.intraday[symbol].index.normalize() == day_date]
                if not bars.empty: total += p.qty * float(bars.close.iloc[-1])
            self.equity.append({"date":day_date,"equity":total})
        return pd.DataFrame(self.trades), pd.DataFrame(self.equity)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--market", default="NIFTY50")
    parser.add_argument("--fundamentals", default=None)
    parser.add_argument("--sector-map", default=None)
    parser.add_argument("--universe", default=None)
    parser.add_argument("--output", default="results")
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    files = sorted(data_dir.glob("*.csv"))
    intraday = {}
    for f in files:
        if f.name == Path(args.market + ".csv").name:
            continue
        try: intraday[f.stem.upper()] = load_ohlcv(f)
        except Exception as e: print("Skipping", f, e)
    market_path = data_dir / (args.market + ".csv")
    if not market_path.exists(): raise FileNotFoundError(market_path)
    market_daily = daily_from_intraday(load_ohlcv(market_path))
    universe = load_universe(args.universe)
    bt = Backtester(intraday, market_daily, universe=universe)
    trades, equity = bt.run()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    trades.to_csv(out/"trades.csv", index=False)
    equity.to_csv(out/"equity_curve.csv", index=False)
    print("Trades:", len(trades))
    if not equity.empty:
        ret = equity.equity.iloc[-1]/equity.equity.iloc[0]-1
        dd = equity.equity/equity.equity.cummax()-1
        print("Total return:", f"{ret:.2%}")
        print("Max drawdown:", f"{dd.min():.2%}")
    print("Results written to", out)

if __name__ == "__main__": main()
