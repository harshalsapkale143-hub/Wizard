from __future__ import annotations

"""Download historical NSE metadata sources used by the point-in-time experiment.

This script deliberately fails closed. It does not invent market caps or listing
 dates. NSE's historical market-cap page is parsed for downloadable files; the
security-master source is taken from NSE's reports page when a direct CSV/GZ
link is exposed. If NSE changes its page structure or blocks the request, the
workflow stops and reports the problem instead of substituting current data.
"""
from pathlib import Path
from urllib.parse import urljoin
import os, re, time
import requests
from bs4 import BeautifulSoup

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'metadata_sources'; OUT.mkdir(exist_ok=True)
CAP_PAGE='https://www.nseindia.com/static/regulations/listing-compliance/nse-market-capitalisation-all-companies'
REPORT_PAGE='https://www.nseindia.com/all-reports'
YEARS=['2019','2020','2021','2022','2023','2024','2025']
UA='Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124 Safari/537.36'


def session():
    s=requests.Session(); s.headers.update({'User-Agent':UA,'Accept':'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8','Referer':'https://www.nseindia.com/'})
    for u in ('https://www.nseindia.com/','https://www.nseindia.com/static/regulations/listing-compliance/nse-market-capitalisation-all-companies'):
        try: s.get(u,timeout=30)
        except requests.RequestException: pass
        time.sleep(.5)
    return s


def links(html,base):
    soup=BeautifulSoup(html,'html.parser'); out=[]
    for a in soup.find_all('a',href=True):
        href=urljoin(base,a['href']); text=' '.join(a.stripped_strings)
        if re.search(r'\.(xlsx?|csv|zip|gz)(?:\?|$)',href,re.I): out.append((text,href))
    return out


def download(s,url,path):
    r=s.get(url,timeout=60,stream=True); r.raise_for_status()
    data=b''.join(r.iter_content(1024*1024))
    if len(data)<100: raise RuntimeError(f'Downloaded file is unexpectedly small: {url}')
    path.write_bytes(data); print(path, len(data))


def main():
    s=session(); r=s.get(CAP_PAGE,timeout=60); r.raise_for_status(); cap_links=links(r.text,CAP_PAGE)
    chosen=[]
    for text,url in cap_links:
        if any(y in text or y in url for y in YEARS) and ('mcap' in url.lower() or 'market' in text.lower() or 'capital' in text.lower()): chosen.append((text,url))
    # De-duplicate and require historical coverage rather than silently using one file.
    seen=set(); chosen=[x for x in chosen if not (x[1] in seen or seen.add(x[1]))]
    if not chosen:
        raise RuntimeError('NSE market-cap page exposed no downloadable historical files to the runner. Page structure or anti-bot behavior may have changed.')
    for i,(text,url) in enumerate(chosen):
        suffix=Path(url.split('?',1)[0]).suffix or '.bin'; download(s,url,OUT/f'market_cap_{i:02d}{suffix}')

    rr=s.get(REPORT_PAGE,timeout=60); rr.raise_for_status(); report_links=links(rr.text,REPORT_PAGE)
    sec=[(t,u) for t,u in report_links if 'security' in (t+' '+u).lower() and re.search(r'\.(csv|gz|zip)(?:\?|$)',u,re.I)]
    if not sec:
        raise RuntimeError('NSE reports page exposed no downloadable security-master file to the runner.')
    # Prefer the latest security file exposed by NSE.
    text,url=sec[0]; suffix=Path(url.split('?',1)[0]).suffix or '.bin'; download(s,url,OUT/f'security_master{suffix}')
    print(f'Found {len(chosen)} market-cap source links and 1 security-master source.')

if __name__=='__main__': main()
