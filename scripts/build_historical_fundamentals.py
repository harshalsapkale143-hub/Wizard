from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests

ROOT = Path('metadata')
OUT = Path('fundamentals')
START = '01-01-2019'
END = '31-03-2026'

ALIASES = {
    'revenue': ['revenueFromOperations', 'revenueFromOperation', 'revenue', 'totalRevenue', 'incomeFromOperations'],
    'pat': ['profitAfterTax', 'profitLossForThePeriod', 'profitLoss', 'netProfit', 'netProfitAfterTax'],
    'eps': ['basicEarningsPerShare', 'basicEPS', 'eps', 'dilutedEarningsPerShare'],
}


def num(x):
    try:
        if x in (None, '', '-', 'NA', 'N/A'):
            return None
        return float(str(x).replace(',', '').replace('%', ''))
    except Exception:
        return None


def flatten(obj: Any, out: dict[str, Any] | None = None):
    out = {} if out is None else out
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, (dict, list)):
                flatten(v, out)
            else:
                out[str(k)] = v
    elif isinstance(obj, list):
        for x in obj:
            flatten(x, out)
    return out


def pick(flat: dict[str, Any], aliases):
    low = {k.lower(): v for k, v in flat.items()}
    for a in aliases:
        if a.lower() in low:
            return num(low[a.lower()])
    for k, v in low.items():
        if any(a.lower() in k for a in aliases):
            n = num(v)
            if n is not None:
                return n
    return None


def get_json(session, symbol):
    url = 'https://www.nseindia.com/api/corporates-financial-results'
    params = {'index': 'equities', 'symbol': symbol, 'from_date': START, 'to_date': END}
    r = session.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def normalize(payload, symbol):
    rows = payload if isinstance(payload, list) else payload.get('data', payload.get('results', [])) if isinstance(payload, dict) else []
    out = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        flat = flatten(row)
        def first(keys):
            for k in keys:
                for fk, fv in flat.items():
                    if k.lower() == fk.lower() or k.lower() in fk.lower():
                        return fv
            return None
        period = first(['periodEnded', 'periodEndDate', 'period_end', 'period'])
        broadcast = first(['broadcastDate', 'broadcastDateTime', 'exchangeReceivedTime', 'broadcast'])
        if not period or not broadcast:
            continue
        pe = pd.to_datetime(period, errors='coerce', dayfirst=True)
        bd = pd.to_datetime(broadcast, errors='coerce', dayfirst=True)
        if pd.isna(pe) or pd.isna(bd):
            continue
        revenue = pick(flat, ALIASES['revenue'])
        pat = pick(flat, ALIASES['pat'])
        eps = pick(flat, ALIASES['eps'])
        if revenue is None and pat is None and eps is None:
            continue
        out.append({'symbol': symbol, 'period_end': pe.date().isoformat(), 'filing_date': bd.date().isoformat(), 'revenue': revenue, 'pat': pat, 'eps': eps})
    return out


def main():
    meta = pd.read_csv(ROOT / 'universe_symbols.csv')
    symbol_col = next(c for c in meta.columns if c.lower() in {'symbol', 'symbols', 'ticker'})
    symbols = sorted(set(meta[symbol_col].dropna().astype(str).str.replace('.NS', '', regex=False)))
    OUT.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124 Safari/537.36', 'Accept': 'application/json,text/plain,*/*', 'Referer': 'https://www.nseindia.com/'})
    session.get('https://www.nseindia.com/', timeout=30)
    ok = 0
    for i, symbol in enumerate(symbols, 1):
        path = OUT / f'{symbol}.csv'
        try:
            payload = get_json(session, symbol)
            rows = normalize(payload, symbol)
            if rows:
                pd.DataFrame(rows).drop_duplicates(['period_end', 'filing_date']).sort_values('filing_date').to_csv(path, index=False)
                ok += 1
            elif path.exists():
                path.unlink()
        except Exception as e:
            print(f'[{i}/{len(symbols)}] {symbol}: {type(e).__name__}: {e}')
        if i % 25 == 0:
            print(f'Processed {i}/{len(symbols)}; usable symbols={ok}')
        time.sleep(0.15)
    print(f'Historical fundamentals complete: {ok}/{len(symbols)} symbols with usable NSE filings')


if __name__ == '__main__':
    main()
