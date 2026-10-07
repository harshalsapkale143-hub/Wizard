from __future__ import annotations
import json, subprocess, sys
from pathlib import Path
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse
import pandas as pd

ROOT=Path(__file__).resolve().parent
DATA=ROOT/"data"
RESULTS=ROOT/"results"
HOST="0.0.0.0"
PORT=8000

class Handler(SimpleHTTPRequestHandler):
    def _json(self, payload, status=200):
        raw=json.dumps(payload, default=str).encode()
        self.send_response(status); self.send_header("Content-Type","application/json"); self.send_header("Access-Control-Allow-Origin","*"); self.send_header("Content-Length",str(len(raw))); self.end_headers(); self.wfile.write(raw)
    def do_OPTIONS(self):
        self.send_response(204); self.send_header("Access-Control-Allow-Origin","*"); self.send_header("Access-Control-Allow-Headers","Content-Type"); self.send_header("Access-Control-Allow-Methods","GET,POST,OPTIONS"); self.end_headers()
    def do_GET(self):
        p=urlparse(self.path).path
        if p.startswith("/api/"): return self.api_get(p)
        return super().do_GET()
    def do_POST(self):
        p=urlparse(self.path).path
        if p=="/api/backtest": return self.run_backtest()
        self._json({"error":"Not found"},404)
    def api_get(self,p):
        if p=="/api/health":
            self._json({"ok":True,"data_dir":str(DATA),"data_available":DATA.exists(),"stock_files":len(list(DATA.glob("*.csv"))) if DATA.exists() else 0})
            return
        if p=="/api/summary":
            f=RESULTS/"robustness_baseline.csv"
            if not f.exists(): self._json({"ready":False,"message":"No backtest result is available yet. Run a backtest after loading the research data."}); return
            row=pd.read_csv(f).iloc[0].to_dict()
            self._json({"ready":True,"summary":row}); return
        if p=="/api/trades":
            f=RESULTS/"robustness_trades.csv"
            if not f.exists(): self._json({"ready":False,"trades":[]}); return
            df=pd.read_csv(f).tail(100).copy()
            self._json({"ready":True,"trades":df.to_dict(orient="records")}); return
        if p=="/api/equity":
            f=RESULTS/"equity_curve.csv"
            if not f.exists(): self._json({"ready":False,"points":[]}); return
            df=pd.read_csv(f); self._json({"ready":True,"points":df.to_dict(orient="records")}); return
        if p=="/api/candidates":
            try:
                out=self.scan_candidates()
                self._json({"ready":True,"candidates":out})
            except Exception as e:
                self._json({"ready":False,"candidates":[],"message":str(e)},503)
            return
        self._json({"error":"Not found"},404)
    def scan_candidates(self):
        if not DATA.exists(): raise RuntimeError("data/ directory is missing")
        market_path=DATA/"NIFTY50.csv"
        files=[x for x in DATA.glob("*.csv") if x.name!="NIFTY50.csv"]
        if not market_path.exists() or not files: raise RuntimeError("NIFTY50.csv or stock CSV files are missing")
        import robustness_backtest as rb
        rb.DATA_CACHE.clear(); rb.FEATURE_CACHE.clear(); rb.SIGNAL_CACHE.clear(); rb.MARKET_OK_CACHE.clear(); rb.RS_RANK_CACHE=None
        market=rb.load(market_path)
        rb.get_cache(files,market_path); rb.build_rs_rank(files,market)
        rows=[]
        for p in files:
            d=rb.DATA_CACHE[p.stem]
            if len(d)<250: continue
            f=rb.build_features(p.stem,d,market)
            rsrank=rb.RS_RANK_CACHE[p.stem]
            dt=d.index[-1]
            trend=bool(f["trend"].iloc[-1]); near=float(f["near_base"].iloc[-1]); dry=float(f["dry_volume_ratio"].iloc[-1])
            r10=float(f["r10"].iloc[-1]); r20=float(f["r20"].iloc[-1]); r40=float(f["r40"].iloc[-1])
            setup=trend and near>=.85 and r10<r20*.65 and r20<r40*.90 and dry<.75
            rows.append({"symbol":p.stem,"date":str(dt.date()),"close":float(d.close.iloc[-1]),"rs_rank":float(rsrank.iloc[-1]*100) if pd.notna(rsrank.iloc[-1]) else None,"near_high_pct":float(near*100),"volume_ratio":dry,"trend":trend,"setup":setup})
        rows.sort(key=lambda x:(x["setup"],x["rs_rank"] or -1),reverse=True)
        return rows[:100]
    def run_backtest(self):
        if not DATA.exists() or not (DATA/"NIFTY50.csv").exists():
            self._json({"ok":False,"message":"Research data is not installed. Add data/*.csv and data/NIFTY50.csv before running the backtest."},400); return
        try:
            r=subprocess.run([sys.executable,"robustness_backtest.py"],cwd=ROOT,capture_output=True,text=True,timeout=900)
            if r.returncode!=0: self._json({"ok":False,"message":r.stderr[-4000:]},500); return
            f=RESULTS/"robustness_baseline.csv"
            summary=pd.read_csv(f).iloc[0].to_dict() if f.exists() else {}
            self._json({"ok":True,"summary":summary,"stdout":r.stdout[-4000:]})
        except subprocess.TimeoutExpired:
            self._json({"ok":False,"message":"Backtest exceeded the 15-minute API timeout."},504)
        except Exception as e:
            self._json({"ok":False,"message":str(e)},500)

if __name__=="__main__":
    print(f"Wizard web server: http://localhost:{PORT}")
    ThreadingHTTPServer((HOST,PORT),Handler).serve_forever()
