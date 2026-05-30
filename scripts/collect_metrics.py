#!/usr/bin/env python3
"""
scripts/collect_metrics.py — Aggregate CSV files from multiple experiment runs.

Walks results/ (or a specified directory), finds all *_transfers.csv,
*_fl_metrics.csv, *_round_events.csv, and *_coap.csv files, concatenates
them, and produces:

  - results/aggregated_transfers.csv
  - results/aggregated_fl_metrics.csv
  - results/aggregated_round_events.csv
  - results/aggregated_coap.csv
  - results/summary.csv   (one row per architecture × scenario)

Usage::
    python scripts/collect_metrics.py --results-dir results/ [--output-dir results/]
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

log = logging.getLogger("collect_metrics")
logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)


def main() -> None:
    try:
        import pandas as pd
    except ImportError:
        log.error("pandas is required: pip install pandas")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Aggregate experiment CSVs")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output-dir",  default="results")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir  = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    categories = {
        "transfers":    "_transfers.csv",
        "fl_metrics":   "_fl_metrics.csv",
        "round_events": "_round_events.csv",
        "coap":         "_coap.csv",
    }

    dfs: dict[str, list] = {k: [] for k in categories}

    for suffix, pattern in [("_transfers.csv", "transfers"),
                             ("_fl_metrics.csv", "fl_metrics"),
                             ("_round_events.csv", "round_events"),
                             ("_coap.csv", "coap")]:
        for csv_file in sorted(results_dir.rglob(f"*{suffix}")):
            try:
                df = pd.read_csv(csv_file)
                df["_source_file"] = csv_file.name
                dfs[pattern].append(df)
                log.info("  %s → %d rows", csv_file, len(df))
            except Exception as exc:
                log.warning("Could not read %s: %s", csv_file, exc)

    # Write aggregated files
    written = []
    for cat, frames in dfs.items():
        if frames:
            agg = pd.concat(frames, ignore_index=True)
            out = output_dir / f"aggregated_{cat}.csv"
            agg.to_csv(out, index=False)
            written.append((cat, out, len(agg)))
            log.info("Wrote %s (%d rows)", out, len(agg))

    # Build summary
    if dfs["round_events"]:
        ev = pd.concat(dfs["round_events"], ignore_index=True)
        tr = pd.concat(dfs["transfers"], ignore_index=True) if dfs["transfers"] else pd.DataFrame()

        grp = ev.groupby(["architecture", "scenario"])

        rows = []
        for (arch, scen), sub in grp:
            accs   = sub["test_acc"].dropna()
            durs   = sub["duration_sec"].dropna()
            n_ok   = sub["success"].sum() if "success" in sub.columns else len(sub)
            n_tot  = len(sub)

            n_direct = n_relay = 0
            if not tr.empty and "architecture" in tr.columns:
                t_sub = tr[(tr["architecture"] == arch) & (tr["scenario"] == scen)]
                n_direct = (t_sub["conn_type"] == "direct").sum()
                n_relay  = (t_sub["conn_type"] == "relay").sum()
                n_t      = max(len(t_sub), 1)
            else:
                n_t = 1

            rows.append({
                "architecture"       : arch,
                "scenario"           : scen,
                "rounds_total"       : n_tot,
                "rounds_success"     : int(n_ok),
                "test_acc_final"     : float(accs.iloc[-1]) if len(accs) else None,
                "test_acc_max"       : float(accs.max())    if len(accs) else None,
                "duration_mean_sec"  : float(durs.mean())   if len(durs) else None,
                "pct_direct"         : 100 * n_direct / n_t,
                "pct_relay"          : 100 * n_relay  / n_t,
            })

        summary_df = pd.DataFrame(rows)
        summary_out = output_dir / "summary.csv"
        summary_df.to_csv(summary_out, index=False)
        log.info("Summary written: %s", summary_out)

    log.info("Done. Files written: %s", [str(o) for _, o, _ in written])


if __name__ == "__main__":
    main()
