"""
E3 — NAT traversal success rate and connection type distribution.

Three operating modes:

1. Mock (--mock):
   Two in-process Iroh nodes.  No real network, used for HPC/SLURM runs.
   python -m experiments.e3_nat_traversal --mock --n-iter 30 --scenario net_lan

2. Distributed — server side (--role server):
   Start a real Iroh node, print its endpoint as JSON, wait for n_iter incoming
   connections from the client, then write results.
   python -m experiments.e3_nat_traversal --role server --n-iter 30 --scenario net_cgnat

3. Distributed — client side (--role client):
   Connect to the server endpoint (paste JSON printed by server) and send n_iter
   small tensors, recording conn_type / latency per attempt.
   python -m experiments.e3_nat_traversal --role client --n-iter 30 --scenario net_cgnat \\
       --server-endpoint '{"node_id":"<id>","addrs":["1.2.3.4:11204"],"relay_url":"https://..."}'

   Or load from a file saved on the server machine:
   python -m experiments.e3_nat_traversal --role client --n-iter 30 \\
       --server-endpoint @results/e3/server_endpoint.json

4. Docker mode (legacy):
   FL_SCENARIO=net_nat1 python -m experiments.e3_nat_traversal \\
       --peer-host 172.20.0.10 --peer-iroh-id <nodeid> --n-iter 30

Outputs (results/e3/):
  e3_nat_<scenario>.csv      — raw per-iteration rows
  e3_summary_<scenario>.csv  — pct_direct / pct_relay / pct_failed
  server_endpoint.json       — server endpoint (written by --role server)
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import os
import sys
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
    buckets: dict[str, dict[str, int]] = defaultdict(lambda: {"direct": 0, "relay": 0, "unknown": 0, "failed": 0, "total": 0})
    for r in rows:
        s = r["scenario"]
        ct = r.get("conn_type", "failed")
        buckets[s]["total"] += 1
        if ct in ("direct", "relay", "unknown", "failed"):
            buckets[s][ct] += 1
        else:
            buckets[s]["unknown"] += 1

    summary = []
    for scen, b in buckets.items():
        n = max(b["total"], 1)
        summary.append({
            "scenario"   : scen,
            "total"      : b["total"],
            "pct_direct" : round(100 * b["direct"] / n, 1),
            "pct_relay"  : round(100 * b["relay"]  / n, 1),
            "pct_unknown": round(100 * b["unknown"] / n, 1),
            "pct_failed" : round(100 * b["failed"] / n, 1),
        })
    return summary


async def run_server(n_iter: int, scenario: str, results_dir: Path) -> None:
    """
    Server role: start a real Iroh node, print endpoint JSON, wait for
    n_iter incoming tensors from the client, write results CSV.
    """
    from fl_coap_iroh.transport.iroh_node import IrohTransportNode, ALPN_FL_MODEL

    node = IrohTransportNode("e3-server")
    ep = await node.start()

    # Serialize endpoint for the client to paste / copy
    ep_dict = {
        "node_id" : ep.node_id_iroh,
        "addrs"   : ep.addrs,
        "relay_url": ep.relay_url or "",
    }
    ep_json = json.dumps(ep_dict, indent=2)
    ep_file = results_dir / "server_endpoint.json"
    ep_file.write_text(ep_json, encoding="utf-8")

    print("\n" + "="*60)
    print("SERVER ENDPOINT — paste this into --server-endpoint on the client:")
    print(ep_json)
    print(f"(also saved to {ep_file})")
    print("="*60)
    print(f"Waiting for {n_iter} connections from client…\n")

    rows: list[dict] = []
    for i in range(n_iter):
        try:
            # Use the raw-byte receive path so the server never imports torch
            # (lets this role run on hosts where torch is unavailable/broken).
            _, stats = await node._receive_bytes(ALPN_FL_MODEL, 120.0)
            rows.append({
                "scenario"    : scenario,
                "iter"        : i,
                "conn_type"   : stats.conn_type.value if stats else "relay",
                "conn_time_ms": round(stats.conn_time_ms, 3) if stats else None,
                "duration_ms" : round(stats.transfer_duration_ms, 3) if stats else None,
                "success"     : True,
            })
            log.info("[server] iter %d/%d received — conn_type=%s", i + 1, n_iter,
                     rows[-1]["conn_type"])
        except Exception as exc:
            log.warning("[server] iter %d receive failed: %s", i, exc)
            rows.append({"scenario": scenario, "iter": i, "conn_type": "failed",
                         "conn_time_ms": None, "duration_ms": None, "success": False})

    await node.stop()
    _write_csv(results_dir / f"e3_nat_{scenario}_server.csv", rows)
    summary = _summarize(rows)
    _write_csv(results_dir / f"e3_summary_{scenario}.csv", summary)
    for row in summary:
        log.info("  %s: direct=%.0f%% relay=%.0f%% unknown=%.0f%% failed=%.0f%%",
                 row["scenario"], row["pct_direct"], row["pct_relay"],
                 row["pct_unknown"], row["pct_failed"])


async def run_client(
    n_iter      : int,
    scenario    : str,
    server_ep_raw: str,
    results_dir : Path,
) -> None:
    """
    Client role: connect to server endpoint and send n_iter small tensors,
    recording conn_type / latency per attempt.
    """
    from fl_coap_iroh.transport.iroh_node import IrohTransportNode, ALPN_FL_MODEL
    from fl_coap_iroh.types import IrohEndpoint
    import torch

    # Parse endpoint — supports inline JSON string or @filename
    if server_ep_raw.startswith("@"):
        ep_dict = json.loads(Path(server_ep_raw[1:]).read_text(encoding="utf-8"))
    else:
        ep_dict = json.loads(server_ep_raw)

    server_ep = IrohEndpoint(
        node_id_iroh   = ep_dict["node_id"],
        addrs          = ep_dict.get("addrs", []),
        relay_url      = ep_dict.get("relay_url") or None,
        direct_capable = True,
    )
    log.info("Connecting to server node_id=%s…", server_ep.node_id_iroh[:16])

    node = IrohTransportNode("e3-client")
    await node.start()

    fake_tensors = {"ping": torch.zeros(1)}
    rows: list[dict] = []

    for i in range(n_iter):
        try:
            stats = await node.send_tensors(
                server_ep, fake_tensors, round_num=i, alpn=ALPN_FL_MODEL
            )
            rows.append({
                "scenario"    : scenario,
                "iter"        : i,
                "conn_type"   : stats.conn_type.value,
                "conn_time_ms": round(stats.conn_time_ms, 3),
                "duration_ms" : round(stats.transfer_duration_ms, 3),
                "success"     : True,
            })
            log.info("[client] iter %d/%d — conn_type=%s  conn_time=%.1fms",
                     i + 1, n_iter, stats.conn_type.value, stats.conn_time_ms)
        except Exception as exc:
            log.warning("[client] iter %d failed: %s", i, exc)
            rows.append({"scenario": scenario, "iter": i, "conn_type": "failed",
                         "conn_time_ms": None, "duration_ms": None, "success": False})

    await node.stop()
    _write_csv(results_dir / f"e3_nat_{scenario}_client.csv", rows)
    summary = _summarize(rows)
    for row in summary:
        log.info("  %s: direct=%.0f%% relay=%.0f%% unknown=%.0f%% failed=%.0f%%",
                 row["scenario"], row["pct_direct"], row["pct_relay"],
                 row["pct_unknown"], row["pct_failed"])


async def main_async(args: argparse.Namespace) -> None:
    import random
    random.seed(_seed())

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    scenario = os.environ.get("FL_SCENARIO", args.scenario)

    # ── Distributed mode ──────────────────────────────────────────────────────
    if args.role == "server":
        await run_server(args.n_iter, scenario, results_dir)
        return

    if args.role == "client":
        if not args.server_endpoint:
            log.error("--server-endpoint required in client mode")
            sys.exit(1)
        await run_client(args.n_iter, scenario, args.server_endpoint, results_dir)
        return

    # ── Mock / Docker mode ────────────────────────────────────────────────────
    peer_ep = None
    if not args.mock and args.peer_iroh_id:
        from fl_coap_iroh.types import IrohEndpoint
        peer_ep = IrohEndpoint(
            node_id_iroh   = args.peer_iroh_id,
            addrs          = [f"{args.peer_host}:11204"],
            relay_url      = args.peer_relay or None,
            direct_capable = True,
        )

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
            "  %s: direct=%.0f%% relay=%.0f%% unknown=%.0f%% failed=%.0f%%",
            row["scenario"], row["pct_direct"], row["pct_relay"],
            row["pct_unknown"], row["pct_failed"],
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="E3: NAT traversal measurement")
    parser.add_argument("--role",            choices=["server", "client"],
                        default=None,
                        help="Distributed mode: 'server' prints endpoint and waits, "
                             "'client' connects to server endpoint")
    parser.add_argument("--server-endpoint", default="",
                        help="Server endpoint JSON string or @path/to/file.json "
                             "(required for --role client)")
    parser.add_argument("--scenario",        default=os.environ.get("FL_SCENARIO", "net_lan"))
    parser.add_argument("--peer-host",       default="")
    parser.add_argument("--peer-iroh-id",    default="")
    parser.add_argument("--peer-relay",      default="")
    parser.add_argument("--n-iter",          type=int, default=30)
    parser.add_argument("--mock",            action="store_true",
                        help="Use in-process mock transport (no real network)")
    parser.add_argument("--results-dir",     default="results/e3")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
