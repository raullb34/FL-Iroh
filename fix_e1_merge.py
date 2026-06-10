"""
Merge E1 distributed client results into the main e1_microbenchmark.csv.

- Renames existing 'iroh_unknown' rows → 'iroh_relay_loopback'  (in-process, loopback)
- Renames new WAN client rows        → 'iroh_relay_wan'         (RPi↔PC, 300km WAN)
- Regenerates e1_summary.csv

Run from repo root:
    .venv/bin/python fix_e1_merge.py
"""
from __future__ import annotations

import csv
import statistics
from pathlib import Path

RESULTS = Path("results/e1")
MAIN_CSV   = RESULTS / "e1_microbenchmark.csv"
CLIENT_CSV = RESULTS / "client" / "e1_dist_client.csv"
SUMMARY_CSV = RESULTS / "e1_summary.csv"


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def coerce(rows: list[dict]) -> list[dict]:
    result = []
    for r in rows:
        row = dict(r)
        for col in ("throughput_mbps", "conn_time_ms", "duration_ms"):
            v = row.get(col)
            if v in ("", "None", None):
                row[col] = None
            else:
                try:
                    row[col] = float(v)
                except ValueError:
                    row[col] = None
        if "payload_bytes" in row:
            try:
                row["payload_bytes"] = int(row["payload_bytes"])
            except (ValueError, TypeError):
                pass
        if "iter" in row:
            try:
                row["iter"] = int(row["iter"])
            except (ValueError, TypeError):
                pass
        row["success"] = str(row.get("success", "false")).lower() in ("true", "1")
        result.append(row)
    return result


def summarize(rows: list[dict]) -> list[dict]:
    from collections import defaultdict
    grouped: dict[tuple, list[float]] = defaultdict(list)
    for r in rows:
        if r["success"] and r.get("throughput_mbps") is not None:
            grouped[(r["transport"], r["payload_bytes"])].append(r["throughput_mbps"])

    summary = []
    for (transport, payload), vals in sorted(grouped.items()):
        vals_sorted = sorted(vals)
        n = len(vals_sorted)
        summary.append({
            "transport"  : transport,
            "payload_bytes": payload,
            "n"          : n,
            "mean_mbps"  : round(statistics.mean(vals), 4),
            "median_mbps": round(statistics.median(vals), 4),
            "p95_mbps"   : round(vals_sorted[int(0.95 * n)], 4) if n else None,
            "stdev_mbps" : round(statistics.stdev(vals), 4) if n > 1 else 0.0,
        })
    return summary


# --- Load and relabel ---
main_rows = coerce(read_csv(MAIN_CSV))
for r in main_rows:
    if r["transport"] in ("iroh_unknown", "iroh"):
        r["transport"] = "iroh_relay_loopback"

client_rows = coerce(read_csv(CLIENT_CSV))
for r in client_rows:
    r["transport"] = "iroh_relay_wan"

# Filter out failed loopback rows (payload_bytes=0) before merging
main_rows = [r for r in main_rows if r.get("payload_bytes", 0) != 0]

all_rows = main_rows + client_rows

# Sort by transport, payload, iter for readability
all_rows.sort(key=lambda r: (r["transport"], r.get("payload_bytes", 0), r.get("iter", 0)))

write_csv(MAIN_CSV, all_rows)
print(f"Merged {len(all_rows)} rows → {MAIN_CSV}")

summary = summarize(all_rows)
write_csv(SUMMARY_CSV, summary)
print(f"\ne1_summary.csv:")
print(f"  {'Transport':<22} {'Payload':>10}  {'Mean Mbit/s':>12}  {'P95':>8}  N")
print(f"  {'-'*22} {'-'*10}  {'-'*12}  {'-'*8}  -")
for s in summary:
    pb = s['payload_bytes']
    label = f"{pb/1e6:.1f}MB" if pb >= 1_000_000 else f"{pb/1e3:.0f}KB"
    print(f"  {s['transport']:<22} {label:>10}  {s['mean_mbps']:>12.2f}  {s['p95_mbps']:>8.2f}  {s['n']}")
