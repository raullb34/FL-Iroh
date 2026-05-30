"""
E3 — NAT traversal success rate and connection type distribution.

For each network scenario in {net_lan, net_nat1, net_nat2, net_cgnat, net_fw443},
repeat n_iter=30 Iroh connection attempts and record:
  - conn_type  (direct | relay | failed)
  - conn_time_ms
  - first_byte_ms

This experiment is designed to run *inside* Docker with the appropriate
compose overlay applied.  When run on the host (development mode), it uses
two in-process mock transport nodes.

Outputs (results/e3/):
  e3_nat_traversal.csv   — raw per-iteration rows
  e3_summary.csv         — pct_direct / pct_relay / pct_failed per scenario

Usage::
    # Inside docker (with scenario env-var set by compose):
    FL_SCENARIO=net_nat1 python -m experiments.e3_nat_traversal \\
        --peer-host 172.20.0.10 --peer-iroh-id <nodeid> --n-iter 30

    # Local mock mode:
    python -m experiments.e3_nat_traversal --mock --n-iter 30
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import logging
import os
from pathlib import Path
from typing import Optional

import yaml

log = logging.getLogger("e3_nat_traversal")
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)

SCENARIOS = ["net_lan", "net_nat1", "net_nat2", "net_cgnat", "net_fw443"]


def _seed() -> int:
    try:
        with open("seeds.yaml") as f:
            return int(yaml.safe_load(f).get("experiment_e3", 202))
    except Exception:
        return 202


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


async def measure_connections(
    n_iter    : int,
    scenario  : str,
    peer_ep   = None,        # IrohEndpoint or None for mock
    mock      : bool = False,
) -> list[dict]:
    from fl_coap_iroh.transport.iroh_node import IrohTransportNode, ALPN_FL_MODEL
    import torch

    rows: list[dict] = []
    sender   = IrohTransportNode("e3-sender")
    receiver = IrohTransportNode("e3-receiver")
    sender_ep   = await sender.start()
    receiver_ep = await receiver.start()

    target_ep = receiver_ep if mock or peer_ep is None else peer_ep

    # Very small payload — we're measuring conn setup not throughput
    fake_tensors = {"ping": torch.zeros(1)}

    for i in range(n_iter):
        try:
            stats = await sender.send_tensors(
                target_ep, fake_tensors, round_num=i, alpn=ALPN_FL_MODEL
            )
            if not mock:
                await receiver.receive_tensors(ALPN_FL_MODEL, timeout=10.0)
            rows.append({
                "scenario"      : scenario,
                "iter"          : i,
                "conn_type"     : stats.conn_type.value,
                "conn_time_ms"  : round(stats.conn_time_ms, 3),
                "duration_ms"   : round(stats.transfer_duration_ms, 3),
                "success"       : True,
            })
        except Exception as exc:
            log.warning("iter %d failed: %s", i, exc)
            rows.append({
                "scenario": scenario, "iter": i,
                "conn_type": "failed", "conn_time_ms": None,
                "duration_ms": None, "success": False,
            })

    await sender.stop()
    await receiver.stop()
    return rows


def _summarize(rows: list[dict]) -> list[dict]:
    from collections import defaultdict
    buckets: dict[str, dict[str, int]] = defaultdict(lambda: {"direct": 0, "relay": 0, "failed": 0, "total": 0})
    for r in rows:
        s = r["scenario"]
        ct = r.get("conn_type", "failed")
        buckets[s]["total"] += 1
        if ct in buckets[s]:
            buckets[s][ct] += 1

    summary = []
    for scen, b in buckets.items():
        n = max(b["total"], 1)
        summary.append({
            "scenario"   : scen,
            "total"      : b["total"],
            "pct_direct" : round(100 * b["direct"] / n, 1),
            "pct_relay"  : round(100 * b["relay"]  / n, 1),
            "pct_failed" : round(100 * b["failed"] / n, 1),
        })
    return summary


async def main_async(args: argparse.Namespace) -> None:
    import random
    random.seed(_seed())

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    peer_ep = None
    if not args.mock and args.peer_iroh_id:
        from fl_coap_iroh.types import IrohEndpoint
        peer_ep = IrohEndpoint(
            node_id_iroh   = args.peer_iroh_id,
            addrs          = [f"{args.peer_host}:11204"],
            relay_url      = args.peer_relay or None,
            direct_capable = True,
        )

    scenario = os.environ.get("FL_SCENARIO", args.scenario)
    log.info("E3 NAT traversal — scenario: %s, n_iter: %d", scenario, args.n_iter)

    rows = await measure_connections(
        n_iter   = args.n_iter,
        scenario = scenario,
        peer_ep  = peer_ep,
        mock     = args.mock,
    )

    raw_out = results_dir / f"e3_nat_{scenario}.csv"
    _write_csv(raw_out, rows)
    log.info("Raw results: %s", raw_out)

    summary = _summarize(rows)
    sum_out = results_dir / f"e3_summary_{scenario}.csv"
    _write_csv(sum_out, summary)
    for row in summary:
        log.info(
            "  %s: direct=%.0f%% relay=%.0f%% failed=%.0f%%",
            row["scenario"], row["pct_direct"], row["pct_relay"], row["pct_failed"],
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="E3: NAT traversal measurement")
    parser.add_argument("--scenario",    default=os.environ.get("FL_SCENARIO", "net_lan"))
    parser.add_argument("--peer-host",   default="")
    parser.add_argument("--peer-iroh-id",default="")
    parser.add_argument("--peer-relay",  default="")
    parser.add_argument("--n-iter",      type=int, default=30)
    parser.add_argument("--mock",        action="store_true",
                        help="Use in-process mock transport (no Docker needed)")
    parser.add_argument("--results-dir", default="results/e3")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
