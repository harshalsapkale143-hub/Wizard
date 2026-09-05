"""Download NSE 5-minute historical candles from Angel One SmartAPI.

Credentials are read ONLY from environment variables:
  ANGEL_API_KEY
  ANGEL_CLIENT_CODE
  ANGEL_PIN
  ANGEL_TOTP_SECRET

Never commit these values to GitHub.

Angel One's Historical API currently supports FIVE_MINUTE and up to 100 days
per request. This script chunks the requested period into <=100-day windows
and throttles requests below the documented historical endpoint limit.
"""

import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pyotp
import requests

BASE = "https://apiconnect.angelone.in"
LOGIN_URL = BASE + "/rest/auth/angelbroking/user/v1/loginByPassword"
CANDLE_URL = BASE + "/rest/secure/angelbroking/historical/v1/getCandleData"
SCRIP_URL = "https://margincalculator.angelone.in/OpenAPI_File/files/OpenAPIScripMaster.json"

START = datetime(2024, 1, 1, 9, 15)
END = datetime(2026, 3, 31, 15, 30)
INTERVAL = "FIVE_MINUTE"
OUT = Path("data")
OUT.mkdir(exist_ok=True)

HEADERS_BASE = {
    "Accept": "application/json",
    "Content-Type": "application/json",
    "X-UserType": "USER",
    "X-SourceID": "WEB",
    "X-ClientLocalIP": os.getenv("ANGEL_CLIENT_LOCAL_IP", "127.0.0.1"),
    "X-ClientPublicIP": os.getenv("ANGEL_CLIENT_PUBLIC_IP", "127.0.0.1"),
    "X-MACAddress": os.getenv("ANGEL_MAC_ADDRESS", "00:00:00:00:00:00"),
}


def required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing GitHub secret/environment variable: {name}")
    return value


def login() -> str:
    api_key = required_env("ANGEL_API_KEY")
    client_code = required_env("ANGEL_CLIENT_CODE")
    pin = required_env("ANGEL_PIN")
    totp_secret = required_env("ANGEL_TOTP_SECRET")

    headers = dict(HEADERS_BASE)
    headers["X-PrivateKey"] = api_key
    payload = {
        "clientcode": client_code,
        "password": pin,
        "totp": pyotp.TOTP(totp_secret).now(),
    }

    r = requests.post(LOGIN_URL, headers=headers, json=payload, timeout=30)
    r.raise_for_status()
    body = r.json()
    if not body.get("status"):
        raise RuntimeError(f"Angel One login failed: {body}")
    return body["data"]["jwtToken"]


def load_tokens() -> dict[str, str]:
    r = requests.get(SCRIP_URL, timeout=60)
    r.raise_for_status()
    instruments = r.json()
    result = {}
    for x in instruments:
        if x.get("exch_seg") == "nse_cm" and str(x.get("symbol", "")).endswith("-EQ"):
            result[x["symbol"].upper()] = str(x["token"])
    # NIFTY 50 token is normally 99926000; use the master if present.
    for x in instruments:
        if str(x.get("name", "")).upper() == "NIFTY" and str(x.get("symbol", "")).upper() == "NIFTY":
            result["NIFTY50"] = str(x["token"])
    result.setdefault("NIFTY50", "99926000")
    return result


def fetch_symbol(jwt: str, symbol: str, token: str) -> pd.DataFrame:
    headers = dict(HEADERS_BASE)
    headers["X-PrivateKey"] = required_env("ANGEL_API_KEY")
    headers["Authorization"] = "Bearer " + jwt

    chunks = []
    cursor = START
    while cursor < END:
        chunk_end = min(cursor + timedelta(days=99), END)
        payload = {
            "exchange": "NSE",
            "symboltoken": token,
            "interval": INTERVAL,
            "fromdate": cursor.strftime("%Y-%m-%d %H:%M"),
            "todate": chunk_end.strftime("%Y-%m-%d %H:%M"),
        }

        for attempt in range(4):
            try:
                r = requests.post(CANDLE_URL, headers=headers, json=payload, timeout=60)
                if r.status_code == 403:
                    time.sleep(10 * (attempt + 1))
                    continue
                r.raise_for_status()
                body = r.json()
                if not body.get("status"):
                    raise RuntimeError(body)
                rows = body.get("data") or []
                if rows:
                    chunks.extend(rows)
                break
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(5 * (attempt + 1))

        # Angel documents 3 requests/sec for historical API; stay comfortably below it.
        time.sleep(0.45)
        cursor = chunk_end + timedelta(minutes=5)

    if not chunks:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    df = pd.DataFrame(chunks, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["timestamp", "open", "high", "low", "close"])
    return df.drop_duplicates("timestamp").sort_values("timestamp")


def main():
    jwt = login()
    tokens = load_tokens()
    symbols = [
        x.strip().upper() for x in Path("symbols.txt").read_text().splitlines()
        if x.strip() and not x.strip().startswith("#")
    ]

    for symbol in symbols:
        token = tokens.get(symbol)
        if not token:
            print(f"SKIP {symbol}: token not found in Angel One scrip master")
            continue
        print(f"Downloading {symbol} token={token}")
        df = fetch_symbol(jwt, symbol, token)
        if df.empty:
            print(f"  NO DATA for {symbol}")
            continue
        filename = "NIFTY50.csv" if symbol == "NIFTY50" else symbol.replace("-EQ", "") + ".csv"
        df.to_csv(OUT / filename, index=False)
        print(f"  saved {len(df):,} bars -> {OUT / filename}")


if __name__ == "__main__":
    main()
