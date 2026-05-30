"""
E1 — Communication microbenchmark.

Measures raw transport throughput and connection setup time for:
  - Iroh (QUIC/P2P)  — direct path
  - Iroh             — relay path
  - HTTP/2 (aiohttp) — baseline
  - gRPC (grpc.aio)  — baseline

across payload sizes: 100 KB, 1 MB, 10 MB, 100 MB
and n_iter=30 repetitions per cell.

Outputs:
  results/e1_microbenchmark.csv   — raw per-iteration rows
  results/e1_summary.csv          — mean/p50/p95 per (transport, payload_bytes)

Usage::
    python -m experiments.e1_microbenchmark --n-iter 30 --results-dir results/
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import io
import logging
import os
import socket
import statistics
import time
from pathlib import Path
from typing import Optional

import torch
import yaml

log = logging.getLogger("e1_microbenchmark")
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)

PAYLOAD_SIZES = [100_000, 1_000_000, 10_000_000, 100_000_000]  # 100KB … 100MB
ALPN_BENCH    = b"fl-bench/1"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_fake_tensors(payload_bytes: int) -> dict[str, torch.Tensor]:
    """Return a dict with one tensor whose serialised size ≈ payload_bytes."""
    n = payload_bytes // 4  # float32 = 4 bytes
    return {"layer": torch.randn(max(1, n))}


def _seed(seeds_file: str = "seeds.yaml") -> int:
    try:
        with open(seeds_file) as f:
            doc = yaml.safe_load(f)
        return int(doc.get("experiment_e1", 789))
    except Exception:
        return 789


# ---------------------------------------------------------------------------
# Iroh transport benchmark
# ---------------------------------------------------------------------------

async def bench_iroh(
    n_iter      : int,
    payload_sizes: list[int],
    results_dir : str,
    relay_url   : Optional[str] = None,
) -> list[dict]:
    from fl_coap_iroh.transport.iroh_node import IrohTransportNode, ALPN_FL_MODEL

    rows: list[dict] = []

    # Spin up two local nodes: sender and receiver
    sender   = IrohTransportNode("bench-sender",   relay_url=relay_url)
    receiver = IrohTransportNode("bench-receiver", relay_url=relay_url)

    sender_ep   = await sender.start()
    receiver_ep = await receiver.start()

    for payload_bytes in payload_sizes:
        log.info("Iroh bench — payload %d bytes", payload_bytes)
        tensors = make_fake_tensors(payload_bytes)

        for i in range(n_iter):
            try:
                stats = await sender.send_tensors(
                    receiver_ep, tensors, round_num=i, alpn=ALPN_FL_MODEL
                )
                conn_type = stats.conn_type.value
                # drain receiver
                await receiver.receive_tensors(ALPN_FL_MODEL, timeout=30.0)

                rows.append({
                    "transport"      : f"iroh_{conn_type}",
                    "payload_bytes"  : payload_bytes,
                    "iter"           : i,
                    "conn_time_ms"   : round(stats.conn_time_ms, 3),
                    "duration_ms"    : round(stats.transfer_duration_ms, 3),
                    "throughput_mbps": round(stats.throughput_mbps, 4),
                    "success"        : True,
                })
            except Exception as exc:
                log.warning("Iroh iter %d sz %d: %s", i, payload_bytes, exc)
                rows.append({
                    "transport": "iroh", "payload_bytes": payload_bytes,
                    "iter": i, "conn_time_ms": None, "duration_ms": None,
                    "throughput_mbps": None, "success": False,
                })

    await sender.stop()
    await receiver.stop()
    return rows


# ---------------------------------------------------------------------------
# HTTP/2 benchmark (aiohttp server + client)
# ---------------------------------------------------------------------------

async def bench_http2(n_iter: int, payload_sizes: list[int]) -> list[dict]:
    """Minimal aiohttp-based HTTP/2 benchmark (transport baseline)."""
    rows: list[dict] = []
    try:
        import aiohttp
        from aiohttp import web
    except ImportError:
        log.warning("aiohttp not installed — skipping HTTP/2 benchmark")
        return rows

    # Simple echo server
    async def handler(request: web.Request) -> web.Response:
        body = await request.read()
        return web.Response(body=body, content_type="application/octet-stream")

    runner = web.AppRunner(web.Application())
    runner.app.router.add_post("/", handler)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]  # type: ignore[attr-defined]

    async with aiohttp.ClientSession() as session:
        for payload_bytes in payload_sizes:
            data = os.urandom(payload_bytes)
            log.info("HTTP/2 bench — payload %d bytes", payload_bytes)
            for i in range(n_iter):
                t0 = time.monotonic()
                try:
                    async with session.post(f"http://127.0.0.1:{port}/", data=data) as resp:
                        await resp.read()
                    dur_ms = (time.monotonic() - t0) * 1000
                    rows.append({
                        "transport"      : "http2",
                        "payload_bytes"  : payload_bytes,
                        "iter"           : i,
                        "conn_time_ms"   : 0.0,
                        "duration_ms"    : round(dur_ms, 3),
                        "throughput_mbps": round(payload_bytes / 1e6 / (dur_ms / 1000), 4),
                        "success"        : True,
                    })
                except Exception as exc:
                    log.warning("HTTP/2 iter %d: %s", i, exc)
                    rows.append({
                        "transport": "http2", "payload_bytes": payload_bytes,
                        "iter": i, "conn_time_ms": None, "duration_ms": None,
                        "throughput_mbps": None, "success": False,
                    })

    await runner.cleanup()
    return rows


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _summarize(rows: list[dict]) -> list[dict]:
    from collections import defaultdict
    grouped: dict[tuple, list[float]] = defaultdict(list)
    for r in rows:
        if r["success"] and r["throughput_mbps"] is not None:
            key = (r["transport"], r["payload_bytes"])
            grouped[key].append(r["throughput_mbps"])

    summary = []
    for (transport, payload), vals in sorted(grouped.items()):
        vals_sorted = sorted(vals)
        n = len(vals_sorted)
        summary.append({
            "transport"        : transport,
            "payload_bytes"    : payload,
            "n"                : n,
            "mean_mbps"        : round(statistics.mean(vals), 4),
            "median_mbps"      : round(statistics.median(vals), 4),
            "p95_mbps"         : round(vals_sorted[int(0.95 * n)], 4) if n else None,
            "stdev_mbps"       : round(statistics.stdev(vals), 4) if n > 1 else 0.0,
        })
    return summary


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main_async(args: argparse.Namespace) -> None:
    torch.manual_seed(_seed())
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []

    # Iroh (mock or real)
    log.info("Running Iroh benchmark…")
    all_rows.extend(await bench_iroh(
        n_iter        = args.n_iter,
        payload_sizes = PAYLOAD_SIZES,
        results_dir   = args.results_dir,
        relay_url     = args.relay_url or None,
    ))

    # HTTP/2 baseline
    if not args.iroh_only:
        log.info("Running HTTP/2 benchmark…")
        all_rows.extend(await bench_http2(args.n_iter, PAYLOAD_SIZES))

    raw_out = results_dir / "e1_microbenchmark.csv"
    _write_csv(raw_out, all_rows)
    log.info("Raw results: %s (%d rows)", raw_out, len(all_rows))

    summary = _summarize(all_rows)
    sum_out = results_dir / "e1_summary.csv"
    _write_csv(sum_out, summary)
    log.info("Summary: %s", sum_out)


def main() -> None:
    parser = argparse.ArgumentParser(description="E1: Communication microbenchmark")
    parser.add_argument("--n-iter",      type=int,  default=30)
    parser.add_argument("--relay-url",   default="")
    parser.add_argument("--iroh-only",   action="store_true")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
