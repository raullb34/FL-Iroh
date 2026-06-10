#!/usr/bin/env python3
"""FL-Iroh — post-experiment result aggregator.

Reads all CSV outputs produced by experiments E1/E2/E3/E5/E6 and emits:
  • results/report_<TIMESTAMP>.json  — machine-readable full report
  • Prints a human-readable table to stdout

Usage::
    python slurm/collect_results.py
    python slurm/collect_results.py --results-dir results --out results/report.json
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def _load_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _floats(rows: list[dict], col: str) -> list[float]:
    out: list[float] = []
    for r in rows:
        v = r.get(col)
        if v not in (None, "", "None"):
            try:
                out.append(float(v))
            except ValueError:
                pass
    return out


def _maybe_float(v: Any) -> Optional[float]:
    if v in (None, "", "None"):
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def _maybe_int(v: Any) -> Optional[int]:
    if v in (None, "", "None"):
        return None
    try:
        return int(float(v))
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# E1: Communication microbenchmark
# ---------------------------------------------------------------------------

def collect_e1(results_dir: Path) -> dict:
    summary = _load_csv(results_dir / "e1" / "e1_summary.csv")
    if not summary:
        return {"status": "missing"}
    rows = []
    for row in summary:
        rows.append({
            "transport"   : row.get("transport"),
            "payload_bytes": _maybe_int(row.get("payload_bytes")),
            "n"           : _maybe_int(row.get("n")),
            "mean_mbps"   : _maybe_float(row.get("mean_mbps")),
            "median_mbps" : _maybe_float(row.get("median_mbps")),
            "p95_mbps"    : _maybe_float(row.get("p95_mbps")),
            "stdev_mbps"  : _maybe_float(row.get("stdev_mbps")),
        })
    return {"status": "ok", "rows": rows}


# ---------------------------------------------------------------------------
# E2: FL convergence
# ---------------------------------------------------------------------------

def collect_e2(results_dir: Path) -> dict:
    e2_dir = results_dir / "e2"
    if not e2_dir.exists():
        return {"status": "missing"}

    variants: dict[str, Any] = {}

    for csv_path in sorted(e2_dir.glob("*_round_events.csv")):
        rows = _load_csv(csv_path)
        if not rows:
            continue

        # Derive scenario label from the filename stem.
        # Stem examples: e2_B_iid_round_events, e2_B_noniid_0.1_round_events
        # The metrics tag set in run_arch_b is "e2_B_{scenario}", so the
        # actual file is "{tag}_round_events.csv".
        stem   = csv_path.stem                         # "e2_B_iid_round_events"
        parts  = stem.split("_round_events")[0]        # "e2_B_iid"
        # Drop leading "e2_B_" prefix if present
        label  = parts[len("e2_B_"):] if parts.startswith("e2_B_") else parts

        accs   = _floats(rows, "test_acc")
        losses = _floats(rows, "test_loss")
        n_ok   = sum(
            1 for r in rows
            if str(r.get("success", "")).lower() in ("true", "1")
        )

        # First round where accuracy reaches 40 %
        rounds_to_40: Optional[int] = None
        for r in rows:
            if (_maybe_float(r.get("test_acc")) or 0.0) >= 0.40:
                rounds_to_40 = _maybe_int(r.get("round"))
                break

        variants[label] = {
            "n_rounds"        : len(rows),
            "n_successful"    : n_ok,
            "acc_final"       : round(accs[-1], 4)                      if accs               else None,
            "acc_max"         : round(max(accs), 4)                     if accs               else None,
            "acc_mean_last10" : round(statistics.mean(accs[-10:]), 4)   if len(accs) >= 10    else None,
            "loss_final"      : round(losses[-1], 4)                    if losses             else None,
            "rounds_to_40pct" : rounds_to_40,
        }

    status = "ok" if variants else "empty"
    return {"status": status, "variants": variants}


# ---------------------------------------------------------------------------
# E3: NAT traversal (mock mode)
# ---------------------------------------------------------------------------

def collect_e3(results_dir: Path) -> dict:
    e3_dir = results_dir / "e3"
    if not e3_dir.exists():
        return {"status": "missing"}

    scenarios: dict[str, Any] = {}

    for csv_path in sorted(e3_dir.glob("e3_summary_*.csv")):
        for row in _load_csv(csv_path):
            scenario = row.get("scenario", csv_path.stem)
            scenarios[scenario] = {
                "total"      : _maybe_int(row.get("total")),
                "pct_direct" : _maybe_float(row.get("pct_direct")),
                "pct_relay"  : _maybe_float(row.get("pct_relay")),
                "pct_failed" : _maybe_float(row.get("pct_failed")),
            }

    status = "ok" if scenarios else "empty"
    return {"status": status, "scenarios": scenarios}


# ---------------------------------------------------------------------------
# E5: Churn resilience
# ---------------------------------------------------------------------------

def collect_e5(results_dir: Path) -> dict:
    e5_dir = results_dir / "e5"
    if not e5_dir.exists():
        return {"status": "missing"}

    churn_rates: dict[str, Any] = {}

    for csv_path in sorted(e5_dir.glob("*_round_events.csv")):
        rows = _load_csv(csv_path)
        if not rows:
            continue

        # Use the scenario field from the first row if available
        scenario = rows[0].get("scenario") or csv_path.stem.split("_round_events")[0]

        accs = _floats(rows, "test_acc")
        n_ok = sum(
            1 for r in rows
            if str(r.get("success", "")).lower() in ("true", "1")
        )

        rounds_to_40: Optional[int] = None
        for r in rows:
            if (_maybe_float(r.get("test_acc")) or 0.0) >= 0.40:
                rounds_to_40 = _maybe_int(r.get("round"))
                break

        churn_rates[scenario] = {
            "n_rounds"       : len(rows),
            "n_successful"   : n_ok,
            "acc_final"      : round(accs[-1], 4)    if accs else None,
            "acc_max"        : round(max(accs), 4)   if accs else None,
            "rounds_to_40pct": rounds_to_40,
        }

    status = "ok" if churn_rates else "empty"
    return {"status": status, "churn_rates": churn_rates}


# ---------------------------------------------------------------------------
# E6: CoAP discovery overhead
# ---------------------------------------------------------------------------

def collect_e6(results_dir: Path) -> dict:
    summary = _load_csv(results_dir / "e6" / "e6_summary.csv")
    if not summary:
        return {"status": "missing"}
    rows = []
    for row in summary:
        rows.append({
            "n_nodes"   : _maybe_int(row.get("n_nodes")),
            "filter"    : row.get("filter"),
            "n"         : _maybe_int(row.get("n")),
            "mean_ms"   : _maybe_float(row.get("mean_ms")),
            "p95_ms"    : _maybe_float(row.get("p95_ms")),
            "mean_bytes": _maybe_float(row.get("mean_bytes")),
        })
    return {"status": "ok", "rows": rows}


# ---------------------------------------------------------------------------
# SLURM job metadata
# ---------------------------------------------------------------------------

def collect_metadata(results_dir: Path) -> list[dict]:
    meta_dir = results_dir / "metadata"
    jobs: list[dict] = []
    for p in sorted(meta_dir.glob("job_*.json")):
        try:
            with p.open(encoding="utf-8") as fh:
                jobs.append(json.load(fh))
        except Exception:
            pass
    return jobs


# ---------------------------------------------------------------------------
# Console pretty-printer
# ---------------------------------------------------------------------------

_DIV = "─" * 72


def _header(title: str) -> None:
    print(f"\n{_DIV}")
    print(f"  {title}")
    print(_DIV)


def _print_e1(e1: dict) -> None:
    _header("E1 · Communication Microbenchmark  (iroh_relay_wan = PC↔RPi 300km WAN)")
    if e1.get("status") != "ok":
        print(f"  [no data — {e1.get('status')}]")
        return
    print(f"  {'Transport':<22} {'Payload':>9} {'Mean Mbps':>10} {'P95 Mbps':>10} {'Stdev':>8}  N")
    print(f"  {'-'*22} {'-'*9} {'-'*10} {'-'*10} {'-'*8}  -")
    for r in e1["rows"]:
        pb = r["payload_bytes"]
        pl = f"{pb/1e6:.1f}M" if pb and pb >= 1_000_000 else (f"{pb}B" if pb else "?")
        print(
            f"  {str(r['transport']):<22} {pl:>9}"
            f" {r['mean_mbps'] or 0:>10.2f} {r['p95_mbps'] or 0:>10.2f}"
            f" {r['stdev_mbps'] or 0:>8.2f}  {r['n'] or 0}"
        )


def _print_e2(e2: dict) -> None:
    _header("E2 · FL Convergence (Architecture B)")
    if e2.get("status") != "ok":
        print(f"  [no data — {e2.get('status')}]")
        return
    print(f"  {'Partition':<22} {'Acc Final':>10} {'Acc Max':>9} {'Loss Final':>11} {'→40%':>5}  Rounds")
    print(f"  {'-'*22} {'-'*10} {'-'*9} {'-'*11} {'-'*5}  ------")
    for scenario, v in sorted(e2["variants"].items()):
        r40 = str(v["rounds_to_40pct"]) if v["rounds_to_40pct"] else "—"
        print(
            f"  {scenario:<22} {v['acc_final'] or 0:>10.4f} {v['acc_max'] or 0:>9.4f}"
            f" {v['loss_final'] or 0:>11.4f} {r40:>5}  {v['n_rounds'] or 0}"
        )


def _print_e3(e3: dict) -> None:
    _header("E3 · NAT Traversal (real: PC↔RPi across NAT)")
    if e3.get("status") != "ok":
        print(f"  [no data — {e3.get('status')}]")
        return
    print(f"  {'Scenario':<22} {'Direct %':>9} {'Relay %':>8} {'Failed %':>9}  N")
    print(f"  {'-'*22} {'-'*9} {'-'*8} {'-'*9}  -")
    for scen, v in sorted(e3["scenarios"].items()):
        print(
            f"  {scen:<22} {v['pct_direct'] or 0:>9.1f} {v['pct_relay'] or 0:>8.1f}"
            f" {v['pct_failed'] or 0:>9.1f}  {v['total'] or 0}"
        )


def _print_e5(e5: dict) -> None:
    _header("E5 · Churn Resilience")
    if e5.get("status") != "ok":
        print(f"  [no data — {e5.get('status')}]")
        return
    print(f"  {'Scenario':<28} {'Acc Final':>10} {'Acc Max':>9} {'→40%':>5}  Rounds  OK")
    print(f"  {'-'*28} {'-'*10} {'-'*9} {'-'*5}  ------  --")
    for scenario, v in sorted(e5["churn_rates"].items()):
        r40 = str(v["rounds_to_40pct"]) if v["rounds_to_40pct"] else "—"
        print(
            f"  {scenario:<28} {v['acc_final'] or 0:>10.4f} {v['acc_max'] or 0:>9.4f}"
            f" {r40:>5}  {v['n_rounds'] or 0:>6}  {v['n_successful'] or 0}"
        )


def _print_e6(e6: dict) -> None:
    _header("E6 · CoAP Discovery Overhead")
    if e6.get("status") != "ok":
        print(f"  [no data — {e6.get('status')}]")
        return
    print(f"  {'N nodes':>8}  {'Filter':<12} {'Mean ms':>8} {'P95 ms':>7} {'Mean bytes':>11}  N")
    print(f"  {'-'*8}  {'-'*12} {'-'*8} {'-'*7} {'-'*11}  -")
    for r in e6["rows"]:
        print(
            f"  {r['n_nodes'] or 0:>8}  {str(r['filter']):<12}"
            f" {r['mean_ms'] or 0:>8.1f} {r['p95_ms'] or 0:>7.1f}"
            f" {r['mean_bytes'] or 0:>11.0f}  {r['n'] or 0}"
        )


def _print_jobs(jobs: list[dict]) -> None:
    if not jobs:
        return
    _header(f"SLURM Jobs ({len(jobs)})")
    print(f"  {'Task':>5}  {'Status':<12} {'Elapsed':>9}  {'Host':<16}  Git")
    print(f"  {'-'*5}  {'-'*12} {'-'*9}  {'-'*16}  ---")
    for j in sorted(jobs, key=lambda x: x.get("task_id", 0)):
        h = j.get("elapsed_sec", 0) / 3600
        print(
            f"  {j.get('task_id', '?'):>5}  {j.get('status', '?'):<12}"
            f" {h:>8.1f}h  {j.get('hostname', '?'):<16}  {j.get('git_commit', '?')}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect and report FL-Iroh experiment results"
    )
    parser.add_argument("--results-dir", default="results",
                        help="Root results directory (default: results)")
    parser.add_argument("--out", default=None,
                        help="Output JSON path (default: results/report_<ts>.json)")
    args = parser.parse_args()

    rdir = Path(args.results_dir)
    if not rdir.exists():
        print(f"[error] Results directory not found: {rdir}", file=sys.stderr)
        sys.exit(1)

    ts  = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = Path(args.out) if args.out else (rdir / f"report_{ts}.json")

    report: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "results_dir" : str(rdir.resolve()),
        "jobs"        : collect_metadata(rdir),
        "experiments" : {
            "e1": collect_e1(rdir),
            "e2": collect_e2(rdir),
            "e3": collect_e3(rdir),
            "e5": collect_e5(rdir),
            "e6": collect_e6(rdir),
        },
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    # ── Console summary ──────────────────────────────────────────────────────
    SEP = "═" * 72
    print(f"\n{SEP}")
    print(f"  FL-Iroh Experiment Results  ·  {ts}")
    print(SEP)

    exps = report["experiments"]
    _print_jobs(report["jobs"])
    _print_e1(exps["e1"])
    _print_e2(exps["e2"])
    _print_e3(exps["e3"])
    _print_e5(exps["e5"])
    _print_e6(exps["e6"])

    print(f"\n{SEP}")
    print(f"  Full report written to: {out}")
    print(f"{SEP}\n")


if __name__ == "__main__":
    main()
