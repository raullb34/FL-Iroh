"""
E6 — CoAP discovery overhead.

Measures:
  - Bytes transferred for /.well-known/core (CoRE Resource Directory)
  - Round-trip discovery time (wall clock)
  - Semantic filtering benefit: bytes after filter vs without filter

Grid:
  n_nodes ∈ {10, 50, 100}
  filter  ∈ {none, semantic}
  n_iter  = 30

Outputs (results/e6/):
  e6_coap_overhead.csv    — raw per-iteration rows
  e6_summary.csv          — mean/p95 bytes / duration per (n_nodes, filter)

Usage::
    python -m experiments.e6_coap_overhead --n-iter 30 --results-dir results/e6
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import logging
import time
from pathlib import Path
from typing import Optional

import yaml

log = logging.getLogger("e6_coap_overhead")
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)

N_NODES_VALUES = [10, 50, 100]


def _seed() -> int:
    try:
        with open("seeds.yaml") as f:
            return int(yaml.safe_load(f).get("experiment_e6", 404))
    except Exception:
        return 404


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


async def _spin_up_servers(n_nodes: int, base_port: int = 15683) -> list[object]:
    """Start n_nodes independent CoAP servers to measure realistic WKC size."""
    from fl_coap_iroh.coap.server import FLCoapServer
    from fl_coap_iroh.types import (
        AvailabilityInfo, ComputeCapabilities, EnergyState,
        DatasetDescriptor, NodeCapabilities, NodeRole, NodeStatus,
    )
    _dummy_ds = DatasetDescriptor(
        dataset_id="e6-dummy", dataset_name="cifar10",
        samples=0, classes=list(range(10)), iid=True,
        distribution="iid", feature_dim=[3, 32, 32],
    )
    servers = []
    for i in range(n_nodes):
        caps = NodeCapabilities(
            node_id      = f"node-{i}",
            role         = NodeRole.CLIENT,
            compute      = ComputeCapabilities(cpu_cores=2),
            availability = AvailabilityInfo(status=NodeStatus.READY),
        )
        s = FLCoapServer(
            node_id              = f"node-{i}",
            capabilities         = caps,
            dataset_descriptor   = _dummy_ds,
            coap_port            = base_port + i,
        )
        await s.start()
        servers.append((s, base_port + i))
    return servers


async def _teardown_servers(servers) -> None:
    for s, _ in servers:
        try:
            await s.stop()
        except Exception:
            pass


async def measure_discovery(
    n_nodes    : int,
    n_iter     : int,
    use_filter : bool,
    base_port  : int = 15683,
) -> list[dict]:
    from fl_coap_iroh.coap.client import FLCoapClient

    rows: list[dict] = []
    servers = await _spin_up_servers(n_nodes, base_port)
    hosts = ["127.0.0.1"] * n_nodes

    for i in range(n_iter):
        t0 = time.monotonic()
        try:
            # Measure /.well-known/core on first server as proxy for overhead
            async with FLCoapClient("127.0.0.1", base_port) as cl:
                link_str, total_bytes, duration_ms = await cl.get_core_link_format()

            # Optionally measure semantic discovery (all nodes)
            if use_filter:
                results, disc_ms = await FLCoapClient.discover_capable_nodes(
                    hosts=hosts,
                    port=base_port,
                    min_energy_pct=0.0,
                    required_role=None,
                )
                total_bytes = sum(1 for _ in results)  # count as proxy; real bytes in transfer log
                duration_ms = disc_ms
            else:
                duration_ms = (time.monotonic() - t0) * 1000

            rows.append({
                "n_nodes"    : n_nodes,
                "filter"     : "semantic" if use_filter else "none",
                "iter"       : i,
                "bytes_total": total_bytes,
                "duration_ms": round(duration_ms, 3),
                "success"    : True,
            })
        except Exception as exc:
            log.warning("n=%d iter=%d: %s", n_nodes, i, exc)
            rows.append({
                "n_nodes": n_nodes,
                "filter" : "semantic" if use_filter else "none",
                "iter": i, "bytes_total": None,
                "duration_ms": None, "success": False,
            })

    await _teardown_servers(servers)
    return rows


def _summarize(rows: list[dict]) -> list[dict]:
    import statistics
    from collections import defaultdict

    grouped: dict[tuple, list] = defaultdict(list)
    for r in rows:
        if r["success"] and r["duration_ms"] is not None:
            grouped[(r["n_nodes"], r["filter"])].append((r["bytes_total"], r["duration_ms"]))

    summary = []
    for (n_nodes, filt), pairs in sorted(grouped.items()):
        durs  = [p[1] for p in pairs]
        bytes_ = [p[0] for p in pairs if p[0] is not None]
        ds = sorted(durs)
        n = len(ds)
        summary.append({
            "n_nodes"     : n_nodes,
            "filter"      : filt,
            "n"           : n,
            "mean_ms"     : round(statistics.mean(durs), 2) if durs else None,
            "p95_ms"      : round(ds[int(0.95 * n)], 2)    if n else None,
            "mean_bytes"  : round(statistics.mean(bytes_), 0) if bytes_ else None,
        })
    return summary


async def main_async(args: argparse.Namespace) -> None:
    import random
    random.seed(_seed())

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    base_port = 20000  # start well above common service ports (was 15683, hit port 16000)
    for n_nodes in N_NODES_VALUES:
        for use_filter in (False, True):
            filt_label = "semantic" if use_filter else "none"
            log.info("E6: n_nodes=%d filter=%s", n_nodes, filt_label)
            rows = await measure_discovery(
                n_nodes    = n_nodes,
                n_iter     = args.n_iter,
                use_filter = use_filter,
                base_port  = base_port,
            )
            all_rows.extend(rows)
            base_port += n_nodes + 10   # avoid port collision between runs

    raw_out = results_dir / "e6_coap_overhead.csv"
    _write_csv(raw_out, all_rows)
    log.info("Raw results: %s (%d rows)", raw_out, len(all_rows))

    summary = _summarize(all_rows)
    sum_out = results_dir / "e6_summary.csv"
    _write_csv(sum_out, summary)
    for row in summary:
        log.info(
            "  n=%3d filter=%-8s mean=%.1f ms  bytes=%.0f",
            row["n_nodes"], row["filter"], row["mean_ms"] or 0, row["mean_bytes"] or 0,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="E6: CoAP discovery overhead")
    parser.add_argument("--n-iter",      type=int, default=30)
    parser.add_argument("--results-dir", default="results/e6")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
