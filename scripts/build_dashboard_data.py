from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

def clean(v):
    if pd.isna(v): return None
    if hasattr(v, "item"): return v.item()
    return v

def main():
    results=Path("results"); web=Path("web/data"); web.mkdir(parents=True,exist_ok=True)
    validation=json.loads((results/"data_validation.json").read_text()) if (results/"data_validation.json").exists() else {"status":"FAIL"}
    summary={}
    if (results/"phase2_summary.csv").exists():
        row=pd.read_csv(results/"phase2_summary.csv").iloc[0].to_dict()
        summary={k:clean(v) for k,v in row.items()}
    trades=[]
    if (results/"phase2_trades.csv").exists():
        df=pd.read_csv(results/"phase2_trades.csv")
        if not df.empty:
            df=df.sort_values("entry_date",ascending=False).head(12)
            trades=[{k:clean(v) for k,v in r.items()} for r in df.to_dict("records")]
    payload={
      "schema_version":"wizard.web.dashboard.v1",
      "generated_at":pd.Timestamp.now(tz="Asia/Kolkata").isoformat(),
      "validation":{"status":validation.get("status"),"checks":validation.get("checks",[])},
      "summary":summary,
      "recent_trades":trades
    }
    (web/"dashboard.json").write_text(json.dumps(payload,indent=2,default=str))
    print(json.dumps({"validation":payload["validation"]["status"],"trades":len(trades),"summary":bool(summary)}))

if __name__=="__main__": main()
