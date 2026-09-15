from __future__ import annotations

import json
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
NSE_HOME = 'https://www.nseindia.com/'
NSE_RESULTS_PAGE = 'https://www.nseindia.com/companies-listing/corporate-filings-financial-results'
LEGACY_API = 'https://www.nseindia.com/api/corporates-financial-results'
INTEGRATED_API = 'https://www.nseindia.com/api/integrated-filing-results'


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
        v = low.get(name.lower())
        if v not in (None, ''):
            return v
    for k, v in low.items():
        if any(name.lower() in k for name in names) and v not in (None, ''):
            return v
    return None


def metadata_rows(payload):
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ('data', 'results', 'financialResults', 'corporateFinancialResults', 'records'):
            value = payload.get(key)
            if isinstance(value, list):
                return value
            if isinstance(value, dict):
                for subkey in ('data', 'results', 'financialResults'):
                    if isinstance(value.get(subkey), list):
                        return value[subkey]
    return []


def parse_date(value):
    if value is None:
        return pd.NaT
    s = str(value).strip()
    for dayfirst in (True, False):
        d = pd.to_datetime(s, errors='coerce', dayfirst=dayfirst)
        if not pd.isna(d):
            return d
    return pd.NaT


def filing_meta(row, symbol, source='financial-results'):
    period_end = first_value(row, [
        'toDate', 'to_date', 'periodEnded', 'periodEndDate', 'period_end',
        'quarterEndDate', 'quarterEnd', 'financialYearEnd', 'qe_Date', 'qeDate',
    ])
    filing = first_value(row, [
        'broadcastDateTime', 'broadcastDate', 'broadCastDate', 'broadcast_Date',
        'exchangeReceivedTime', 'broadcast', 'filingDate', 'revisionDateTime',
    ])
    xbrl = first_value(row, [
        'xbrl', 'xbrlLink', 'xbrlFile', 'xbrlUrl', 'xbrlURL', 'xbrlFileLink',
    ])
    period = str(first_value(row, ['period', 'periodType', 'relatingTo', 'details']) or '')
    if source == 'integrated-financials':
        period = period or 'Quarterly'
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


class NSEBrowser:
    """NSE HTTP client using Playwright's request stack, without page navigation."""
    def __init__(self):
        from playwright.sync_api import sync_playwright
        self._pw = sync_playwright().start()
        self.browser = self._pw.chromium.launch(headless=True, args=['--disable-http2'])
        self.context = self.browser.new_context(
            user_agent='Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/134 Safari/537.36',
            locale='en-US',
            extra_http_headers={
                'Accept-Language': 'en-US,en;q=0.9',
                'Referer': NSE_HOME,
                'Accept': 'application/json,text/plain,*/*',
            },
        )
        self.request = self.context.request
        self._bootstrap()

    def _bootstrap(self):
        last_error = None
        for attempt in range(5):
            try:
                response = self.request.get(NSE_HOME, timeout=30000, fail_on_status_code=False)
                print(f'NSE HTTP bootstrap status={response.status}')
                if response.status in (200, 301, 302, 403):
                    return
                last_error = RuntimeError(f'HTTP {response.status}')
            except Exception as exc:
                last_error = exc
                print(f'NSE HTTP bootstrap retry {attempt + 1}/5: {type(exc).__name__}: {exc}')
            time.sleep(2)
        print(f'NSE HTTP bootstrap did not return a normal response: {last_error}; continuing with direct API retries')

    def fetch_json(self, url, params):
        from urllib.parse import urlencode
        target = url + '?' + urlencode(params)
        last_error = None
        for attempt in range(5):
            try:
                response = self.request.get(
                    target,
                    timeout=30000,
                    fail_on_status_code=False,
                    headers={
                        'Accept': 'application/json, text/plain, */*',
                        'Referer': NSE_RESULTS_PAGE,
                        'X-Requested-With': 'XMLHttpRequest',
                    },
                )
                body = response.text()
                if response.status == 200:
                    return json.loads(body or '{}')
                last_error = RuntimeError(f'NSE API HTTP {response.status}: {body[:200]}')
            except Exception as exc:
                last_error = exc
                print(f'NSE API retry {attempt + 1}/5 for {target}: {type(exc).__name__}: {exc}')
            time.sleep(2)
        raise RuntimeError(f'NSE API failed after retries for {target}: {last_error}')

    def fetch_text(self, url):
        last_error = None
        for attempt in range(3):
            try:
                response = self.request.get(
                    url,
                    timeout=30000,
                    fail_on_status_code=False,
                    headers={
                        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                        'Referer': NSE_RESULTS_PAGE,
                    },
                )
                if response.status == 200:
                    return response.text()
                last_error = RuntimeError(f'HTTP {response.status}')
            except Exception as exc:
                last_error = exc
            time.sleep(1)
        print(f'NSE XBRL fetch failed: {url}: {last_error}')
        return ''

    def close(self):
        self.context.close()
        self.browser.close()
        self._pw.stop()


def get_metadata(browser, symbol):
    return browser.fetch_json(LEGACY_API, {'index': 'equities', 'symbol': symbol, 'period': 'Quarterly'})


def get_integrated_metadata(browser, symbol):
    return browser.fetch_json(INTEGRATED_API, {'index': 'equities', 'symbol': symbol, 'period': 'Quarterly'})


def page_text(url, session, browser=None):
    if not url or url.lower() in {'none', 'nan', 'https://www.nseindia.com/'}:
        return ''
    try:
        r = session.get(url, timeout=30, headers={
            'User-Agent': 'Mozilla/5.0',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        })
        r.raise_for_status()
        content_type = r.headers.get('content-type', '').lower()
        if 'html' in content_type or not content_type:
            soup = BeautifulSoup(r.text, 'html.parser')
            return soup.get_text(' ', strip=True)
        return r.text
    except Exception:
        if browser is not None:
            text = browser.fetch_text(url)
            if text:
                soup = BeautifulSoup(text, 'html.parser')
                return soup.get_text(' ', strip=True)
        return ''


def extract_number_after(text, labels):
    for label in labels:
        m = re.search(re.escape(label) + r'(.{0,260}?)', text, flags=re.I)
        if not m:
            continue
        nums = re.findall(r'(?<![A-Za-z])\(?-?\d[\d,]*(?:\.\d+)?\)?', m.group(1))
        if nums:
            return num(nums[0])
    return None


def extract_xbrl(url, session, browser=None):
    text = page_text(url, session, browser)
    if not text:
        return None, None, None
    revenue = extract_number_after(text, ['Revenue from operations'])
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
    s = str(row.get('period_type', '')).lower()
    return any(x in s for x in ('quarter', 'first quarter', 'second quarter', 'third quarter', 'fourth quarter'))


def is_cumulative(value):
    s = str(value or '').strip().lower()
    return bool(s) and 'cumulative' in s and 'non-cumulative' not in s


def is_non_consolidated(value):
    s = str(value or '').strip().lower()
    return 'non-consolidated' in s or 'standalone' in s or 'non consolidated' in s


def normalize(payload, symbol, session, source, browser=None):
    out = []
    for row in metadata_rows(payload):
        if not isinstance(row, dict):
            continue
        if source == 'integrated-financials':
            typ = str(row.get('type', '')).strip().lower()
            if typ != 'integrated filing- financials':
                continue
            row = dict(row)
            row.setdefault('toDate', row.get('qe_Date'))
            row.setdefault('broadcastDate', row.get('broadcast_Date'))
            row.setdefault('period', 'Quarterly')
        meta = filing_meta(row, symbol, source)
        if meta is None or not is_quarter(meta):
            continue
        if is_cumulative(meta['cumulative']):
            continue
        xbrl = str(meta.get('xbrl') or '').strip()
        if not xbrl or xbrl in {'-', '_', '—'} or xbrl.endswith('nseindia.com/'):
            continue
        out.append(meta)

    if not out:
        return []

    df = pd.DataFrame(out)
    df['period_end'] = pd.to_datetime(df['period_end'])
    df['filing_date'] = pd.to_datetime(df['filing_date'])
    chosen = []
    for _, g in df.groupby('period_end'):
        cons = g[~g['consolidated'].map(is_non_consolidated)]
        if len(cons):
            g = cons
        chosen.append(g.sort_values('filing_date').iloc[-1].to_dict())

    # Deduplicate expensive XBRL fetches within a symbol.
    rows = []
    text_cache = {}
    for meta in chosen:
        xbrl = str(meta.get('xbrl') or '').strip()
        if xbrl not in text_cache:
            text_cache[xbrl] = page_text(xbrl, session, browser)
        text = text_cache[xbrl]
        if text:
            revenue = extract_number_after(text, ['Revenue from operations'])
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
        else:
            revenue = pat = eps = None
        if revenue is None and pat is None and eps is None:
            continue
        meta.update({
            'period_end': pd.Timestamp(meta['period_end']).date().isoformat(),
            'filing_date': pd.Timestamp(meta['filing_date']).date().isoformat(),
            'revenue': revenue,
            'pat': pat,
            'eps': eps,
        })
        rows.append(meta)
    return rows


def main():
    meta = pd.read_csv(ROOT / 'universe_symbols.csv')
    symbol_col = next(c for c in meta.columns if c.lower() in {'symbol', 'symbols', 'ticker'})
    symbols = sorted(set(meta[symbol_col].dropna().astype(str).str.replace('.NS', '', regex=False)))
    OUT.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({'User-Agent': 'Mozilla/5.0', 'Accept-Language': 'en-US,en;q=0.9'})
    browser = NSEBrowser()
    ok = 0
    diagnostics = []
    api_failures = 0
    try:
        for i, symbol in enumerate(symbols, 1):
            path = OUT / f'{symbol}.csv'
            try:
                if path.exists():
                    try:
                        existing = pd.read_csv(path)
                        required = {'period_end','filing_date','eps','revenue','pat'}
                        if len(existing) >= 5 and required.issubset(existing.columns):
                            ok += 1
                            diagnostics.append({
                                'symbol': symbol,
                                'rows': len(existing),
                                'first_period': existing.period_end.min(),
                                'last_period': existing.period_end.max(),
                                'first_filing': existing.filing_date.min(),
                                'last_filing': existing.filing_date.max(),
                                'sources': '|'.join(sorted(set(existing.get('source', pd.Series(dtype=str)).dropna().astype(str)))) if 'source' in existing else '',
                            })
                            print(f'[{i}/{len(symbols)}] {symbol}: reusing cached fundamentals')
                            continue
                    except Exception:
                        pass
                rows = normalize(get_metadata(browser, symbol), symbol, session, 'financial-results', browser)
                try:
                    integrated = get_integrated_metadata(browser, symbol)
                    rows += normalize(integrated, symbol, session, 'integrated-financials', browser)
                except Exception as e:
                    print(f'{symbol}: integrated API warning: {e}')

                if rows:
                    df = pd.DataFrame(rows)
                    df = df.drop_duplicates(['period_end', 'filing_date', 'consolidated', 'cumulative', 'source']).sort_values(['filing_date', 'period_end'])
                    df.to_csv(path, index=False)
                    ok += 1
                    diagnostics.append({
                        'symbol': symbol,
                        'rows': len(df),
                        'first_period': df.period_end.min(),
                        'last_period': df.period_end.max(),
                        'first_filing': df.filing_date.min(),
                        'last_filing': df.filing_date.max(),
                        'sources': '|'.join(sorted(set(df.source))),
                    })
                elif path.exists():
                    path.unlink()
            except Exception as e:
                api_failures += 1
                print(f'[{i}/{len(symbols)}] {symbol}: {type(e).__name__}: {e}')
            if i % 25 == 0:
                print(f'Processed {i}/{len(symbols)}; usable symbols={ok}; API failures={api_failures}')
            time.sleep(0.10)
    finally:
        browser.close()

    if diagnostics:
        pd.DataFrame(diagnostics).to_csv(OUT / '_build_diagnostics.csv', index=False)
    print(f'Historical fundamentals complete: {ok}/{len(symbols)} symbols with usable NSE filings')
    print(f'NSE catalog/API failures: {api_failures}/{len(symbols)}')
    if ok == 0:
        raise SystemExit('No usable point-in-time NSE fundamental filings were produced.')


if __name__ == '__main__':
    main()
