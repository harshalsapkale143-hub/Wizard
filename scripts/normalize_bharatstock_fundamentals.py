from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

SRC = Path("fundamentals_public")
OUT = Path("fundamentals_bharatstock_canonical")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    files = sorted(SRC.glob("*.json"))
    if not files:
        raise SystemExit("No BharatStock JSON files found.")

    total_rows = 0
    canonical_rows = 0
    for path in files:
        payload = json.loads(path.read_text())
        rows = payload.get("data", [])
        out = []
        for row in rows:
            period_end = pd.to_datetime(row.get("period_end_date"), errors="coerce")
            if pd.isna(period_end):
                continue
            out.append(
                {
                    "symbol": path.stem,
                    "period_end": period_end.date().isoformat(),
                    "filing_date": "",
                    "revenue": row.get("revenue"),
                    "pat": row.get("net_profit_attributable_to_owners", row.get("net_profit")),
                    "eps": row.get("eps"),
                    "consolidated": row.get("consolidation_type", ""),
                    "is_audited": row.get("is_audited"),
                    "source": "bharatstock",
                    "point_in_time_eligible": False,
                    "point_in_time_reason": "BharatStock financial response does not expose a filing/availability date in the retrieved schema.",
                }
            )
        if out:
            pd.DataFrame(out).to_csv(OUT / f"{path.stem}.csv", index=False)
            canonical_rows += len(out)
        total_rows += len(rows)

    manifest = {
        "source_files": len(files),
        "provider_rows": total_rows,
        "canonical_rows": canonical_rows,
        "point_in_time_eligible": False,
        "reason": "No filing/availability date was present in the retrieved BharatStock financial rows; period_end_date is not substituted for filing_date.",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
