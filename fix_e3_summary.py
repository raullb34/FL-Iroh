import csv
from pathlib import Path

BASE = Path("results")

def infer(duration_ms):
    return "direct" if float(duration_ms) < 5.0 else "relay"

for scenario, fname in [
    ("net_cgnat", "e3/client/e3_nat_net_cgnat_client.csv"),
    ("net_nat1",  "e3/client/e3_nat_net_nat1_client.csv"),
]:
    path = BASE / fname
    if not path.exists():
        print(f"SKIP {fname} — not found")
        continue
    rows = list(csv.DictReader(open(path)))
    total = len(rows)
    counts = {"direct": 0, "relay": 0, "failed": 0}
    for r in rows:
        if r["success"] != "True":
            counts["failed"] += 1
            continue
        ct = r["conn_type"]
        if ct == "unknown":
            ct = infer(r["duration_ms"])
        counts[ct] = counts.get(ct, 0) + 1
    out = BASE / "e3" / f"e3_summary_{scenario}.csv"
    with open(out, "w", newline="") as f:
        f.write("scenario,total,pct_direct,pct_relay,pct_unknown,pct_failed\n")
        f.write(f"{scenario},{total},{100*counts['direct']/total:.1f},{100*counts['relay']/total:.1f},0.0,{100*counts['failed']/total:.1f}\n")
    print(f"{scenario}: {counts}  total={total} → {out}")
