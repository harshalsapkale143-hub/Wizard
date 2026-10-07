"""Phase 2 data-quality and backtest safety gate for Wizard."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable
import json
import pandas as pd

REQUIRED = ("timestamp", "open", "high", "low", "close", "volume")

@dataclass
class Check:
    name: str
    status: str
    message: str
    details: dict

def check(name: str, status: str, message: str, **details) -> Check:
    return Check(name, status, message, details)

def load_ohlcv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    cols = {str(c).strip().lower(): c for c in df.columns}
    missing = [c for c in REQUIRED if c not in cols]
    if missing:
        raise ValueError(f"{path}: missing required columns {missing}")
    df = df.rename(columns={cols[c]: c for c in REQUIRED})
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    for c in REQUIRED[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=REQUIRED).copy()
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("Asia/Kolkata")
    else:
        df["timestamp"] = df["timestamp"].dt.tz_convert("Asia/Kolkata")
    return df.sort_values("timestamp").drop_duplicates("timestamp", keep="last").set_index("timestamp")

def validate_ohlcv(df: pd.DataFrame, symbol: str) -> list[Check]:
    if df.empty:
        return [check("non_empty", "fail", f"{symbol}: no usable OHLCV rows")]
    bad_ohlc = (
        (df["low"] > df[["open", "close"]].min(axis=1)) |
        (df["high"] < df[["open", "close"]].max(axis=1)) |
        (df[["open", "high", "low", "close"]] <= 0).any(axis=1)
    )
    bad_volume = df["volume"] < 0
    duplicates = int(df.index.duplicated().sum())
    return [
        check("ohlc_consistency", "fail" if bad_ohlc.any() else "pass",
              "OHLC relationships are valid" if not bad_ohlc.any() else "Invalid OHLC rows found",
              invalid_rows=int(bad_ohlc.sum())),
        check("volume", "fail" if bad_volume.any() else "pass",
              "Volume is non-negative" if not bad_volume.any() else "Negative volume found",
              invalid_rows=int(bad_volume.sum())),
        check("duplicate_timestamps", "fail" if duplicates else "pass",
              "No duplicate timestamps" if not duplicates else "Duplicate timestamps found",
              duplicates=duplicates),
        check("timestamp_order", "pass", "Timestamps normalized and sorted",
              first=str(df.index[0]), last=str(df.index[-1]), rows=len(df)),
    ]

def validate_dataset(data_dir: str | Path, market: str = "NIFTY50.csv",
                     universe_file: str | Path | None = None) -> dict:
    root = Path(data_dir)
    checks: list[Check] = []
    market_path = root / market
    if not market_path.exists():
        checks.append(check("market_data", "fail", f"Missing market benchmark: {market_path}"))
        return report(checks, root)
    checks.extend(validate_ohlcv(load_ohlcv(market_path), "NIFTY50"))
    stocks = [p for p in sorted(root.glob("*.csv")) if p.name != market]
    if not stocks:
        checks.append(check("stock_universe", "fail", "No stock OHLCV files found"))
    else:
        failed = 0
        for p in stocks:
            result = validate_ohlcv(load_ohlcv(p), p.stem)
            failed += sum(x.status == "fail" for x in result)
            checks.extend(result)
        checks.append(check("stock_universe", "fail" if failed else "pass",
                            f"Validated {len(stocks)} stock files",
                            files=len(stocks), failed_checks=failed))
    if universe_file:
        up = Path(universe_file)
        if not up.exists():
            checks.append(check("point_in_time_universe", "fail", f"Missing universe file: {up}"))
        else:
            u = pd.read_csv(up)
            required = {"symbol", "available_from", "available_to"}
            missing = sorted(required - set(u.columns))
            if missing:
                checks.append(check("point_in_time_universe", "fail",
                                    "Universe file is missing required columns", missing=missing))
            else:
                for c in ("available_from", "available_to"):
                    u[c] = pd.to_datetime(u[c], errors="coerce")
                bad = u["available_from"].isna() | u["available_to"].isna() | (u["available_from"] > u["available_to"])
                checks.append(check("point_in_time_universe", "fail" if bad.any() else "pass",
                                    "Point-in-time universe intervals are valid" if not bad.any()
                                    else "Invalid listing interval found",
                                    invalid_rows=int(bad.sum()), rows=len(u)))
    else:
        checks.append(check("survivorship_bias_guard", "warn",
                            "No point-in-time universe supplied; results may contain survivorship bias"))
    return report(checks, root)

def report(checks: Iterable[Check], root: Path) -> dict:
    checks = list(checks)
    failures = [asdict(c) for c in checks if c.status == "fail"]
    warnings = [asdict(c) for c in checks if c.status == "warn"]
    return {
        "schema_version": "wizard.phase2.validation.v1",
        "data_dir": str(root),
        "status": "fail" if failures else ("warn" if warnings else "pass"),
        "can_backtest": not failures,
        "checks": [asdict(c) for c in checks],
        "failure_count": len(failures),
        "warning_count": len(warnings),
    }

def write_report(value: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")

def assert_backtest_safe(value: dict) -> None:
    if not value.get("can_backtest", False):
        messages = [x["message"] for x in value.get("checks", []) if x["status"] == "fail"]
        raise RuntimeError("Wizard Phase 2 data gate failed: " + " | ".join(messages))

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--market", default="NIFTY50.csv")
    ap.add_argument("--universe", default=None)
    ap.add_argument("--report", default="results/data_validation.json")
    args = ap.parse_args()
    value = validate_dataset(args.data_dir, args.market, args.universe)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    write_report(value, args.report)
    print(json.dumps(value, indent=2))
    if not value["can_backtest"]:
        raise SystemExit(2)
