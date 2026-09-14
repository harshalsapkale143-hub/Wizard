from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

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
        # NSE has used several wrappers over time; retain only actual row lists.
        for key in ('data', 'results', 'financialResults', 'corporateFinancialResults', 'records'):
            value = payload.get(key)
            if isinstance(value, list):
                return value
            if isinstance(value, dict):
                for subkey in ('data', 'results', 'financialResults'):
                    if isinstance(value.get(subkey), list):
                        return value[subkey]
    return []


def get_json(session, url, params):
    r = session.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def get_metadata(session, symbol):
    # Current NSE API uses period=Quarterly. The old implementation supplied
    # from_date/to_date here; that is not the supported contract and can return
    # an empty response. The quarterly endpoint already provides the historical
    # filing series needed for FY2021-26.
    payload = get_json(
        session,
        'https://www.nseindia.com/api/corporates-financial-results',
        {'index': 'equities', 'symbol': symbol, 'period': 'Quarterly'},
    )
    return payload


def get_integrated_metadata(session, symbol):
    # NSE moved financial results for quarter ended March 2025 onward to
    # Integrated Filing - Financials. Keep this as a separate source so the
    # point-in-time availability date remains the exchange broadcast timestamp.
    try:
        return get_json(
            session,
            'https://www.nseindia.com/api/integrated-filing-results',
            {
                'type': 'Integrated Filing- Financials',
                'index': 'equities',
                'symbol': symbol,
                'period_ended': 'all',
                'page': 1,
                'size': 100,
            },
        )
    except Exception:
        return None


def parse_date(value):
    if value is None:
        return pd.NaT
    return pd.to_datetime(str(value).strip(), errors='coerce', dayfirst=True)


def filing_meta(row, symbol, source='financial-results'):
    period_end = first_value(row, [
        'toDate', 'periodEnded', 'periodEndDate', 'period_end', 'quarterEndDate',
        'quarterEnd', 'financialYearEnd',
    ])
    filing = first_value(row, [
        'broadcastDateTime', 'broadcastDate', 'broadCastDate',
        'exchangeReceivedTime', 'broadcast', 'filingDate', 'revisionDateTime',
    ])
    xbrl = first_value(row, [
        'xbrl', 'xbrlLink', 'xbrlFile', 'xbrlUrl', 'xbrlURL', 'xbrlFileLink',
    ])
    period = str(first_value(row, ['period', 'periodType', 'relatingTo', 'details']) or '')
    consolidated = str(first_value(row, [
        'consolidated', 'consolidatedNonConsolidated', 'consolidatedStandalone',
    ]) or '')
    cumulative = str(first_value(row, ['cumulative', 'cumulativeNonCumulative']) or '')
    audited = str(first_value(row, ['audited', 'auditedUnaudited']) or '')
    submission = str(first_value(row, ['typeOfSubmission', 'submissionType', 'type']) or '')
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
        'submission_type': submission,
        'source': source,
        'xbrl': urljoin('https://www.nseindia.com/', str(xbrl or '').strip()),
    }


def page_text(url, session):
    if not url or url.lower() in {'none', 'nan', 'https://www.nseindia.com/'}:
        return ''
    try:
        r = session.get(url, timeout=30)
        r.raise_for_status()
        content_type = r.headers.get('content-type', '').lower()
        if 'html' in content_type or not content_type:
            soup = BeautifulSoup(r.text, 'html.parser')
            return soup.get_text(' ', strip=True)
        return r.text
    except Exception:
        return ''


def extract_number_after(text, labels):
    for label in labels:
        m = re.search(re.escape(label) + r'(.{0,220}?)', text, flags=re.I)
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
    revenue = extract_number_after(text, [
        'Revenue from operations', 'Revenue from Operations',
        'Revenue from operations (in Rs.)',
    ])
    pat = extract_number_after(text, [
        'Total profit (loss) for period',
        'Net Profit Loss for the period from continuing operations',
        'Profit (loss) for the period',
        'Profit for the period',
    ])
    eps = extract_number_after(text, [
        'Basic earnings (loss) per share from continuing operations',
        'Basic earnings (loss) per share',
        'Basic EPS',
    ])
    return revenue, pat, eps


def is_quarter(row):
    text = ' '.join(str(row.get(k, '')) for k in ('period_type', 'period_end', 'source')).lower()
    return 'quarter' in text or any(x in text for x in ('first quarter', 'second quarter', 'third quarter', 'fourth quarter'))


def is_cumulative(value):
    s = str(value or '').strip().lower()
    return bool(s) and 'cumulative' in s and 'non-cumulative' not in s


def is_non_consolidated(value):
    s = str(value or '').strip().lower()
    return 'non-consolidated' in s or 'standalone' in s or 'non consolidated' in s


def normalize(payload, symbol, session, source):
    out = []
    for row in metadata_rows(payload):
        if not isinstance(row, dict):
            continue
        meta = filing_meta(row, symbol, source)
        if meta is None or not is_quarter(meta):
            continue
        if is_cumulative(meta['cumulative']):
            continue
        out.append(meta)

    if not out:
        return []

    # Keep one filing per period where possible. Prefer consolidated records,
    # then the latest broadcast/revision. If NSE only exposes standalone data,
    # retain it but mark it explicitly so downstream analysis can audit quality.
    df = pd.DataFrame(out)
    df['period_end'] = pd.to_datetime(df['period_end'])
    df['filing_date'] = pd.to_datetime(df['filing_date'])
    chosen = []
    for period_end, g in df.groupby('period_end'):
        cons = g[~g['consolidated'].map(is_non_consolidated)]
        if len(cons):
            g = cons
        chosen.append(g.sort_values('filing_date').iloc[-1].to_dict())

    out = []
    for meta in chosen:
        revenue, pat, eps = extract_xbrl(meta['xbrl'], session)
        if revenue is None and pat is None and eps is None:
            # Metadata-only rows are useful for diagnostics but must not enter
            # the research dataset as if financial numbers were available.
            continue
        meta.update({
            'period_end': pd.Timestamp(meta['period_end']).date().isoformat(),
            'filing_date': pd.Timestamp(meta['filing_date']).date().isoformat(),
            'revenue': revenue,
            'pat': pat,
            'eps': eps,
        })
        out.append(meta)
    return out


def main():
    meta = pd.read_csv(ROOT / 'universe_symbols.csv')
    symbol_col = next(c for c in meta.columns if c.lower() in {'symbol', 'symbols', 'ticker'})
    symbols = sorted(set(meta[symbol_col].dropna().astype(str).str.replace('.NS', '', regex=False)))
    OUT.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/134 Safari/537.36',
        'Accept': 'application/json,text/plain,*/*',
        'Accept-Language': 'en-US,en;q=0.9',
        'Referer': 'https://www.nseindia.com/companies-listing/corporate-filings-financial-results',
        'X-Requested-With': 'XMLHttpRequest',
    })
    # NSE's current API can work without priming the home page; try it but do
    # not make a 403 from the landing page fatal.
    try:
        session.get('https://www.nseindia.com/', timeout=30)
    except Exception:
        pass

    ok = 0
    diagnostics = []
    for i, symbol in enumerate(symbols, 1):
        path = OUT / f'{symbol}.csv'
        try:
            rows = normalize(get_metadata(session, symbol), symbol, session, 'financial-results')
            # Integrated Filing is the authoritative NSE location for results
            # from quarter ended March 2025 onward. Add it when available.
            integrated = get_integrated_metadata(session, symbol)
            if integrated is not None:
                rows += normalize(integrated, symbol, session, 'integrated-financials')

            if rows:
                df = pd.DataFrame(rows)
                df = df.drop_duplicates(
                    ['period_end', 'filing_date', 'consolidated', 'cumulative', 'source']
                ).sort_values(['filing_date', 'period_end'])
                df.to_csv(path, index=False)
                ok += 1
                diagnostics.append({
                    'symbol': symbol,
                    'rows': len(df),
                    'first_period': df.period_end.min(),
                    'last_period': df.period_end.max(),
                    'first_filing': df.filing_date.min(),
                    'last_filing': df.filing_date.max(),
                })
            elif path.exists():
                path.unlink()
        except Exception as e:
            print(f'[{i}/{len(symbols)}] {symbol}: {type(e).__name__}: {e}')
        if i % 25 == 0:
            print(f'Processed {i}/{len(symbols)}; usable symbols={ok}')
        time.sleep(0.20)

    if diagnostics:
        pd.DataFrame(diagnostics).to_csv(OUT / '_build_diagnostics.csv', index=False)
    print(f'Historical fundamentals complete: {ok}/{len(symbols)} symbols with usable NSE filings')


if __name__ == '__main__':
    main()
