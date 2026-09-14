from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from bs4 import BeautifulSoup

ROOT = Path('metadata')
OUT = Path('fundamentals')
START = '01-01-2019'
END = '31-03-2026'


def num(x):
    if x is None:
        return None
    s = str(x).strip().replace(',', '').replace('₹', '').replace('%', '')
    if s in {'', '-', '—', 'NA', 'N/A', 'None', 'nan'}:
        return None
    neg = s.startswith('(') and s.endswith(')')
    s = s.strip('()')
    try:
        v = float(re.sub(r'[^0-9.\-]', '', s))
        return -v if neg else v
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


def first_value(row: dict[str, Any], names):
    flat = flatten(row)
    low = {str(k).lower(): v for k, v in flat.items()}
    for name in names:
        if name.lower() in low and low[name.lower()] not in (None, ''):
            return low[name.lower()]
    for k, v in low.items():
        if any(name.lower() in k for name in names) and v not in (None, ''):
            return v
    return None


def metadata_rows(payload):
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ('data', 'results', 'financialResults', 'corporateFinancialResults'):
            if isinstance(payload.get(key), list):
                return payload[key]
    return []


def get_metadata(session, symbol):
    r = session.get(
        'https://www.nseindia.com/api/corporates-financial-results',
        params={'index': 'equities', 'symbol': symbol, 'from_date': START, 'to_date': END},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def parse_date(value):
    if value is None:
        return pd.NaT
    return pd.to_datetime(str(value).strip(), errors='coerce', dayfirst=True)


def filing_meta(row, symbol):
    period_end = first_value(row, ['toDate', 'periodEnded', 'periodEndDate', 'period_end'])
    filing = first_value(row, ['broadcastDateTime', 'broadcastDate', 'exchangeReceivedTime', 'broadcast', 'filingDate'])
    xbrl = first_value(row, ['xbrl', 'xbrlLink', 'xbrlFile', 'xbrlUrl', 'xbrlURL'])
    period = str(first_value(row, ['period', 'periodType', 'relatingTo']) or '')
    consolidated = str(first_value(row, ['consolidated', 'consolidatedNonConsolidated']) or '')
    cumulative = str(first_value(row, ['cumulative', 'cumulativeNonCumulative']) or '')
    audited = str(first_value(row, ['audited', 'auditedUnaudited']) or '')
    pe = parse_date(period_end)
    bd = parse_date(filing)
    if pd.isna(pe) or pd.isna(bd):
        return None
    return {
        'symbol': symbol,
        'period_end': pe.date().isoformat(),
        'filing_date': bd.date().isoformat(),
        'period_type': period,
        'cumulative': cumulative,
        'consolidated': consolidated,
        'audited': audited,
        'xbrl': str(xbrl or '').strip(),
    }


def page_text(url, session):
    if not url or url.lower() in {'none', 'nan'}:
        return ''
    try:
        r = session.get(url, timeout=30)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, 'html.parser')
        return soup.get_text(' ', strip=True)
    except Exception:
        return ''


def extract_number_after(text, labels):
    # XBRL HTML renders the label followed by the reported number. Keep the
    # parser deliberately conservative: only accept a number close to the
    # target label and never infer values from unrelated sections.
    for label in labels:
        m = re.search(re.escape(label) + r'(.{0,180}?)', text, flags=re.I)
        if not m:
            continue
        tail = m.group(1)
        nums = re.findall(r'(?<![A-Za-z])\(?-?\d[\d,]*(?:\.\d+)?\)?', tail)
        if nums:
            return num(nums[0])
    return None


def extract_xbrl(url, session):
    text = page_text(url, session)
    if not text:
        return None, None, None
    revenue = extract_number_after(text, ['Revenue from operations', 'Revenue from Operations'])
    pat = extract_number_after(text, [
        'Total profit (loss) for period',
        'Net Profit Loss for the period from continuing operations',
        'Profit (loss) for the period',
    ])
    eps = extract_number_after(text, [
        'Basic earnings (loss) per share from continuing operations',
        'Basic earnings (loss) per share',
    ])
    return revenue, pat, eps


def normalize(payload, symbol, session):
    out = []
    for row in metadata_rows(payload):
        if not isinstance(row, dict):
            continue
        meta = filing_meta(row, symbol)
        if meta is None:
            continue
        # Only use quarter-level, non-cumulative filings. NSE exposes these
        # dimensions explicitly; using them avoids mixing YTD and quarterly
        # values when calculating YoY growth.
        pt = meta['period_type'].lower()
        if 'quarter' not in pt and 'third' not in pt and 'second' not in pt and 'first' not in pt:
            continue
        if 'cumulative' in meta['cumulative'].lower() and 'non' not in meta['cumulative'].lower():
            continue
        if 'consolidated' not in meta['consolidated'].lower():
            continue
        revenue, pat, eps = extract_xbrl(meta['xbrl'], session)
        if revenue is None and pat is None and eps is None:
            continue
        meta.update({'revenue': revenue, 'pat': pat, 'eps': eps})
        out.append(meta)
    return out


def main():
    meta = pd.read_csv(ROOT / 'universe_symbols.csv')
    symbol_col = next(c for c in meta.columns if c.lower() in {'symbol', 'symbols', 'ticker'})
    symbols = sorted(set(meta[symbol_col].dropna().astype(str).str.replace('.NS', '', regex=False)))
    OUT.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0',
        'Accept': 'application/json,text/plain,*/*',
        'Referer': 'https://www.nseindia.com/',
    })
    session.get('https://www.nseindia.com/', timeout=30)
    ok = 0
    for i, symbol in enumerate(symbols, 1):
        path = OUT / f'{symbol}.csv'
        try:
            rows = normalize(get_metadata(session, symbol), symbol, session)
            if rows:
                df = pd.DataFrame(rows).drop_duplicates(
                    ['period_end', 'filing_date', 'consolidated', 'cumulative']
                ).sort_values(['filing_date', 'period_end'])
                df.to_csv(path, index=False)
                ok += 1
            elif path.exists():
                path.unlink()
        except Exception as e:
            print(f'[{i}/{len(symbols)}] {symbol}: {type(e).__name__}: {e}')
        if i % 25 == 0:
            print(f'Processed {i}/{len(symbols)}; usable symbols={ok}')
        time.sleep(0.20)
    print(f'Historical fundamentals complete: {ok}/{len(symbols)} symbols with usable NSE filings')


if __name__ == '__main__':
    main()
