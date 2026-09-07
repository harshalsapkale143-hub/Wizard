from __future__ import annotations
"""Download historical NSE metadata sources; fail closed on missing/changed sources."""
from pathlib import Path
from urllib.parse import urljoin
import re, time, requests
from bs4 import BeautifulSoup

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'metadata_sources'; OUT.mkdir(exist_ok=True)
CAP_PAGE='https://www.nseindia.com/static/regulations/listing-compliance/nse-market-capitalisation-all-companies'
HF_SECURITY_MASTER='https://huggingface.co/datasets/tickertruthorg/nse-india-security-master/resolve/main/data/nse_security_master.csv?download=true'
HF_REF='main (dataset currently published)'
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

def download(s,url,path,min_bytes=100):
 last=None
 for n in range(5):
  try:
   r=s.get(url,timeout=90); r.raise_for_status(); data=r.content
   if len(data)<min_bytes:raise RuntimeError(f'response too small: {len(data)} bytes')
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
 for i,(_,url) in enumerate(chosen):
  suffix=Path(url.split('?',1)[0]).suffix.lower() or '.bin'; download(s,url,OUT/f'market_cap_{i:02d}{suffix}',min_bytes=1000)
 listing_path=OUT/'listing_dates.csv'; download(s,HF_SECURITY_MASTER,listing_path,min_bytes=10000)
 import pandas as pd
 df=pd.read_csv(listing_path); required={'nse_symbol','listing_date'}
 if not required.issubset(df.columns):raise RuntimeError(f'Security-master schema changed; required={sorted(required)}, got={list(df.columns)}')
 df=df[['nse_symbol','listing_date']].rename(columns={'nse_symbol':'symbol'}); df['symbol']=df['symbol'].astype(str).str.strip().str.upper(); df['listing_date']=pd.to_datetime(df['listing_date'],errors='coerce')
 df=df.dropna(subset=['symbol','listing_date']).drop_duplicates('symbol').sort_values('symbol')
 if len(df)<2000:raise RuntimeError(f'Unexpectedly small security master: {len(df)} rows')
 df.to_csv(listing_path,index=False)
 (OUT/'security_master_source.txt').write_text(f'url={HF_SECURITY_MASTER}\nref={HF_REF}\nlicense=CC-BY-4.0\nstated_source=NSE India public equity master\n',encoding='utf-8')
 print(f'Downloaded {len(chosen)} historical market-cap files and {len(df)} listing-date records.')
if __name__=='__main__':main()
