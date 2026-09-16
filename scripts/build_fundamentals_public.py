from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pandas as pd
import requests

BASE = "https://bharatstockapi.com/v1/stocks/{symbol}"
OUT = Path("fundamentals_public")
MAX_SYMBOLS = int(os.environ.get("FUNDAMENTALS_MAX_SYMBOLS", "10"))
DAILY_LIMIT = int(os.environ.get("BHARATSTOCK_DAILY_LIMIT", "40"))
SLEEP_SECONDS = float(os.environ.get("BHARATSTOCK_SLEEP_SECONDS", "1.0"))


def main():
    meta = pd.read_csv("metadata/universe_symbols.csv")
    scol = next(c for c in meta.columns if c.lower() in {"symbol", "ticker", "symbols"})
    key = os.environ.get("BHARATSTOCK_API_KEY", "").strip()
    if not key:
        raise SystemExit("BHARATSTOCK_API_KEY secret is missing.")

    OUT.mkdir(parents=True, exist_ok=True)
    headers = {
        "Accept": "application/json",
        "X-API-Key": key,
        "User-Agent": "Wizard-fundamentals/1.0",
    }

    target = meta.dropna(subset=[scol]).head(min(MAX_SYMBOLS, DAILY_LIMIT))
    ok = 0
    auth_failed = False

    for _, row in target.iterrows():
        symbol = str(row[scol]).replace(".NS", "").strip()
        try:
            response = requests.get(
                BASE.format(symbol=symbol),
                headers=headers,
                timeout=20,
            )
            if response.status_code == 200:
                payload = response.json()
                (OUT / f"{symbol}.json").write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2)
                )
                ok += 1
                print(f"{symbol}: OK")
            elif response.status_code in {401, 403}:
                auth_failed = True
                print(f"{symbol}: authentication/permission HTTP {response.status_code}")
                break
            elif response.status_code == 429:
                print(f"{symbol}: rate limited (429); stopping seed run")
                break
            else:
                print(f"{symbol}: provider HTTP {response.status_code}")
        except requests.RequestException as exc:
            print(f"{symbol}: request error {type(exc).__name__}: {exc}")
        time.sleep(SLEEP_SECONDS)

    print(f"BharatStock usable files: {ok}/{len(target)}")
    if auth_failed:
        raise SystemExit(
            "BharatStock rejected the API key. Verify the secret value and API-plan access."
        )
    if ok == 0:
        raise SystemExit("BharatStock returned no usable data for the seed universe.")


if __name__ == "__main__":
    main()
