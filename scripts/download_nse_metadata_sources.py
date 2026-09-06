from __future__ import annotations
"""Download historical NSE metadata sources; fail closed on missing/changed sources."""
from pathlib import Path
from urllib.parse import urljoin
import re,time,requests,json
from bs4 import BeautifulSoup
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'metadata_sources'; OUT.mkdir(exist_ok=True)
CAP_PAGE='https://www.nseindia.com/static/regulations/listing-compliance/nse-market-capitalisation-all-companies'; REPORT_PAGE='https://www.nseindia.com/all-reports'
SECURITY_API='https://www.nseindia.com/api/master-quote'
YEARS=['2019','2020','2021','2022','2023','2024','2025']
UA='Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124 Safari/537.36'
def session():
 s=requests.Session(); s.headers.update({'User-Agent':UA,'Accept':'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8','Referer':'https://www.nseindia.com/'})
 try:s.get('https://www.nseindia.com/',timeout=30)
 except requests.RequestException:pass
 time.sleep(1); return s
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
 # Prefer a directly linked security master; otherwise use the NSE public master endpoint.
 rr=s.get(REPORT_PAGE,timeout=60); rr.raise_for_status(); report=links(rr.text,REPORT_PAGE)
 sec=[(t,u) for t,u in report if re.search(r'security|master|listing',t+' '+u,re.I) and re.search(r'\.(csv|gz|zip|xlsx?)(?:\?|$)',u,re.I)]
 if sec:
  text,url=sec[0]; suffix=Path(url.split('?',1)[0]).suffix.lower() or '.bin'; download(s,url,OUT/f'security_master{suffix}')
 else:
  r=s.get(SECURITY_API,timeout=60); r.raise_for_status()
  try: payload=r.json()
  except Exception as e: raise RuntimeError(f'NSE security master endpoint returned non-JSON content: {e}')
  (OUT/'security_master.json').write_text(json.dumps(payload,indent=2))
  # Convert only if listing-date fields are present; otherwise fail closed.
  rows=payload if isinstance(payload,list) else payload.get('data',payload.get('symbols',[]))
  if not isinstance(rows,list) or not rows: raise RuntimeError('NSE security master endpoint returned no records')
  def pick(d,names):
   low={str(k).lower().replace('_','').replace(' ',''):v for k,v in d.items()}
   for n in names:
    if n in low:return low[n]
  out=[]
  for d in rows:
   if isinstance(d,dict):
    sym=pick(d,['symbol','tradingsymbol','securitysymbol']); dt=pick(d,['listingdate','listdate','dateoflisting'])
    if sym and dt:out.append({'symbol':sym,'listing_date':dt})
  if not out: raise RuntimeError('NSE security master response has no usable symbol/listing-date fields')
  import pandas as pd; pd.DataFrame(out).drop_duplicates('symbol').to_csv(OUT/'listing_dates.csv',index=False)
 print(f'Downloaded {len(chosen)} historical market-cap files and security-master data.')
if __name__=='__main__':main()
