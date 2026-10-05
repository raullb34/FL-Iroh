"""
Re-classify the E3 cold-start campaign with a strict direct-path criterion.

The original classifier folded iroh's MIXED state into "direct".  MIXED means
the relay is carrying traffic while a direct candidate address is still being
validated; the per-address debug dump (FL_CONN_DEBUG_ADDRS=1) shows that in
those cases no direct address has a measured latency (no pong received), i.e.
no validated UDP path existed within the 5 s settle window.

For every cold-start run this script reads the last ``[conn-dbg]`` line of
``client.log`` and the per-run CSV, and labels the attempt:
  direct   iroh reported DIRECT and a direct address has a measured latency
  mixed    iroh reported MIXED (relay-carried, direct unvalidated)
  relay    iroh reported RELAY
  failed   the attempt failed (CSV) or no classification was logged

Usage:  python scripts/e3_reclassify.py [--root results/e3/establish]
"""
from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from pathlib import Path

KIND = re.compile(r"\[conn-dbg\] kind=(\w+) addrs=\[(.*)\]")
LAT = re.compile(r"lat=(?!None)")


def classify_run(run_dir: Path) -> dict:
    log_lines = (run_dir / "client.log").read_text(errors="replace").splitlines() \
        if (run_dir / "client.log").exists() else []
    dbg = [m for m in (KIND.search(l) for l in log_lines) if m]
    csvs = list(run_dir.glob("e3_nat_*_client.csv"))
    ok = False
    if csvs:
        rows = list(csv.DictReader(csvs[0].open()))
        ok = any(r.get("success") == "True" for r in rows)
    if not ok or not dbg:
        return {"run": run_dir.name, "label": "failed", "iroh_kind": dbg[-1].group(1) if dbg else ""}
    kind, addrs = dbg[-1].group(1), dbg[-1].group(2)
    validated = bool(LAT.search(addrs))
    if kind == "DIRECT":
        label = "direct" if validated else "direct-unvalidated"
    else:
        label = kind.lower()
    return {"run": run_dir.name, "label": label, "iroh_kind": kind, "direct_validated": validated}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/e3/establish")
    ap.add_argument("--out", default="results/e3/establish/reclassified.csv")
    a = ap.parse_args()
    rows = []
    for scen in sorted(p for p in Path(a.root).iterdir() if p.is_dir()):
        runs = sorted(p for p in scen.glob("run_*") if p.is_dir())
        for r in runs:
            rows.append({"scenario": scen.name, **classify_run(r)})
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["scenario", "run", "label", "iroh_kind", "direct_validated"])
        w.writeheader()
        w.writerows(rows)
    print(f"{'scenario':<11} {'n':>3} {'direct':>7} {'mixed':>6} {'relay':>6} {'failed':>7}  strict direct %")
    tot = Counter()
    for scen in sorted({r["scenario"] for r in rows}):
        c = Counter(r["label"] for r in rows if r["scenario"] == scen)
        n = sum(c.values())
        if scen != "net_lan":
            tot.update(c)
        print(f"{scen:<11} {n:>3} {c['direct']:>7} {c['mixed']:>6} {c['relay']:>6} {c['failed']:>7}"
              f"  {100 * c['direct'] / n:5.1f}")
    n = sum(tot.values())
    print(f"{'WAN total':<11} {n:>3} {tot['direct']:>7} {tot['mixed']:>6} {tot['relay']:>6} "
          f"{tot['failed']:>7}  {100 * tot['direct'] / n:5.1f}")
    if tot["direct-unvalidated"]:
        print(f"note: {tot['direct-unvalidated']} DIRECT without measured latency")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
