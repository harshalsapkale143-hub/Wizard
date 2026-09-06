from __future__ import annotations
"""Download historical NSE metadata sources; fail closed on missing/changed sources."""
from pathlib import Path
from urllib.parse import urljoin
import re,time,requests
from bs4 import BeautifulSoup
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'metadata_sources'; OUT.mkdir(exist_ok=True)
CAP_PAGE='https://www.nseindia.com/static/regulations/listing-compliance/nse-market-capitalisation-all-companies'; REPORT_PAGE='https://www.nseindia.com/all-reports'; YEARS=['2019','2020','2021','2022','2023','2024','2025']
UA='Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124 Safari/537.36'
def session():
 s=requests.Session(); s.headers.update({'User-Agent':UA,'Accept':'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8','Referer':'https://www.nseindia.com/'})
 for u in ('https://www.nseindia.com/',CAP_PAGE):
  try:s.get(u,timeout=30)
  except requests.RequestException:pass
  time.sleep(.5)
 return s
def links(html,base):
 soup=BeautifulSoup(html,'html.parser'); out=[]
 for a in soup.find_all('a',href=True):
  href=urljoin(base,a['href']); text=' '.join(a.stripped_strings)
  if re.search(r'\.(xlsx?|csv|zip|gz)(?:\?|$)',href,re.I):out.append((text,href))
 return out
def download(s,url,path):
 last=None
 for n in range(5):
  try:
   r=s.get(url,timeout=60); r.raise_for_status(); data=r.content
   if len(data)<100:raise RuntimeError('response too small')
   path.write_bytes(data); return
  except Exception as e:last=e;time.sleep(2*(n+1))
 raise RuntimeError(f'Failed downloading {url}: {last}')
def main():
 s=session(); r=s.get(CAP_PAGE,timeout=60); r.raise_for_status(); raw=links(r.text,CAP_PAGE)
 chosen=[]
 for text,url in raw:
  blob=(text+' '+url).lower()
  if any(y in blob for y in YEARS) and re.search(r'market|capital|mcap',blob):chosen.append((text,url))
 seen=set(); chosen=[x for x in chosen if not(x[1] in seen or seen.add(x[1]))]
 if not chosen:raise RuntimeError('No historical NSE market-cap download links discovered.')
 for i,(text,url) in enumerate(chosen):
  suffix=Path(url.split('?',1)[0]).suffix.lower() or '.bin'; download(s,url,OUT/f'market_cap_{i:02d}{suffix}')
 rr=s.get(REPORT_PAGE,timeout=60); rr.raise_for_status(); report=links(rr.text,REPORT_PAGE)
 sec=[(t,u) for t,u in report if 'security' in (t+' '+u).lower() and re.search(r'\.(csv|gz|zip)(?:\?|$)',u,re.I)]
 if not sec:raise RuntimeError('No NSE security-master download link discovered.')
 text,url=sec[0]; suffix=Path(url.split('?',1)[0]).suffix.lower() or '.bin'; download(s,url,OUT/f'security_master{suffix}')
 print(f'Downloaded {len(chosen)} historical market-cap files and one security-master source.')
if __name__=='__main__':main()
