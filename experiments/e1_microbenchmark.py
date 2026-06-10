"""
E1 — Communication microbenchmark.

Measures raw transport throughput and connection setup time for:
  - Iroh (QUIC/P2P)  — direct path (LAN, distributed mode)
  - Iroh             — relay path  (WAN / SLURM, in-process mode)
  - HTTP/2 (aiohttp) — baseline

across payload sizes: 100 KB, 1 MB, 10 MB, 100 MB
and n_iter=30 repetitions per cell.

Operating modes
---------------
1. In-process (default) — two local iroh nodes, relay only (UDP-blocked envs):
   python -m experiments.e1_microbenchmark --n-iter 30

2. Distributed server — real iroh node, prints endpoint JSON, receives payloads:
   python -m experiments.e1_microbenchmark --role server --n-iter 30

3. Distributed client — connects to server, sends payloads, records conn_type:
   python -m experiments.e1_microbenchmark --role client --n-iter 30 \\
       --server-endpoint '{"node_id":"<id>","addrs":["IP:PORT"],"relay_url":"https://..."}'

   Or load from file saved by the server:
   python -m experiments.e1_microbenchmark --role client --n-iter 30 \\
       --server-endpoint @results/e1/server_endpoint.json

   In a LAN environment the client will record conn_type=direct → labelled iroh_direct.
   Client results are appended to e1_microbenchmark.csv and summary is regenerated.

Outputs (results/e1/):
  e1_microbenchmark.csv   — raw per-iteration rows (all transports, appended)
  e1_summary.csv          — mean/p50/p95 per (transport, payload_bytes)
  server_endpoint.json    — server endpoint written by --role server
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
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

    # Simple echo server — raise client_max_size so payloads up to 200 MB are accepted.
    # aiohttp's default (1 MB) returns 413 for larger payloads, making throughput look
    # artificially high (the 413 response arrives instantly).
    async def handler(request: web.Request) -> web.Response:
        body = await request.read()
        return web.Response(body=body, content_type="application/octet-stream")

    app = web.Application(client_max_size=200 * 1024 * 1024)  # 200 MB
    app.router.add_post("/", handler)
    runner = web.AppRunner(app)
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
                        body = await resp.read()
                        if resp.status != 200:
                            raise RuntimeError(f"HTTP {resp.status}: {body[:120]}")
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


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _append_and_save(path: Path, new_rows: list[dict]) -> None:
    """Append new_rows to an existing CSV (or create it), preserving all rows."""
    existing = _read_csv(path)
    # Cast numeric columns back from strings (DictReader reads everything as str)
    numeric_cols = {"payload_bytes", "iter", "conn_time_ms", "duration_ms",
                    "throughput_mbps"}
    coerced: list[dict] = []
    for r in existing:
        row = dict(r)
        for col in numeric_cols:
            if col in row and row[col] not in ("", "None", None):
                try:
                    row[col] = float(row[col]) if "." in str(row[col]) else int(row[col])
                except ValueError:
                    pass
            elif col in row and row[col] in ("None",):
                row[col] = None
        if "success" in row:
            row["success"] = str(row["success"]).lower() in ("true", "1")
        coerced.append(row)
    all_rows = coerced + new_rows
    _write_csv(path, all_rows)


# ---------------------------------------------------------------------------
# Distributed benchmarks
# ---------------------------------------------------------------------------

async def run_server_bench(
    n_iter      : int,
    payload_sizes: list[int],
    results_dir : Path,
) -> None:
    """
    Server role: start a real iroh node, print endpoint JSON, receive
    n_iter × len(payload_sizes) payloads from the client, write results.
    """
    from fl_coap_iroh.transport.iroh_node import IrohTransportNode, ALPN_FL_MODEL

    node = IrohTransportNode("e1-server")
    ep   = await node.start()

    ep_dict = {
        "node_id"  : ep.node_id_iroh,
        "addrs"    : list(ep.addrs),
        "relay_url": ep.relay_url or "",
    }
    ep_json  = json.dumps(ep_dict, indent=2)
    ep_file  = results_dir / "server_endpoint.json"
    ep_file.write_text(ep_json, encoding="utf-8")

    total = n_iter * len(payload_sizes)
    print("\n" + "=" * 60)
    print("E1 SERVER ENDPOINT — paste into --server-endpoint on client:")
    print(ep_json)
    print(f"(also saved to {ep_file})")
    print("=" * 60)
    print(f"Waiting for {total} transfers ({len(payload_sizes)} payload sizes × {n_iter} iters)…\n")

    rows: list[dict] = []
    for i in range(total):
        try:
            _tensors, stats = await node.receive_tensors(ALPN_FL_MODEL, timeout=300.0)
            rows.append({
                "transport"      : f"iroh_{stats.conn_type.value}",
                "payload_bytes"  : stats.bytes_payload,
                "iter"           : i,
                "conn_time_ms"   : round(stats.conn_time_ms, 3),
                "duration_ms"    : round(stats.transfer_duration_ms, 3),
                "throughput_mbps": round(stats.throughput_mbps, 4),
                "success"        : True,
            })
            log.info(
                "[server] recv %d/%d  %dB via %s  %.1f Mbit/s",
                i + 1, total,
                stats.bytes_payload, stats.conn_type.value, stats.throughput_mbps,
            )
        except Exception as exc:
            log.warning("[server] recv %d failed: %s", i, exc)
            rows.append({
                "transport": "iroh_unknown", "payload_bytes": 0, "iter": i,
                "conn_time_ms": None, "duration_ms": None,
                "throughput_mbps": None, "success": False,
            })

    await node.stop()

    raw_out = results_dir / "e1_dist_server.csv"
    _write_csv(raw_out, rows)
    log.info("Server raw results: %s (%d rows)", raw_out, len(rows))

    summary = _summarize(rows)
    _write_csv(results_dir / "e1_dist_server_summary.csv", summary)
    for s in summary:
        log.info("  %-18s %8dB  mean=%.1f Mbit/s  p95=%.1f",
                 s["transport"], s["payload_bytes"], s["mean_mbps"], s["p95_mbps"])


async def run_client_bench(
    n_iter        : int,
    payload_sizes : list[int],
    server_ep_raw : str,
    results_dir   : Path,
) -> None:
    """
    Client role: connect to server, send all payload sizes n_iter times,
    record conn_type + throughput.  Results are appended to e1_microbenchmark.csv
    so that collect_results.py sees all transports in one file.
    """
    import json as _json
    from fl_coap_iroh.transport.iroh_node import IrohTransportNode, ALPN_FL_MODEL
    from fl_coap_iroh.types import IrohEndpoint

    # Parse endpoint
    if server_ep_raw.startswith("@"):
        ep_dict = _json.loads(Path(server_ep_raw[1:]).read_text(encoding="utf-8"))
    else:
        ep_dict = _json.loads(server_ep_raw)

    server_ep = IrohEndpoint(
        node_id_iroh   = ep_dict["node_id"],
        addrs          = ep_dict.get("addrs", []),
        relay_url      = ep_dict.get("relay_url") or None,
        direct_capable = True,
    )
    log.info("Connecting to server node_id=%s…", server_ep.node_id_iroh[:16])

    node = IrohTransportNode("e1-client")
    await node.start()

    rows: list[dict] = []
    for payload_bytes in payload_sizes:
        tensors = make_fake_tensors(payload_bytes)
        log.info("Iroh dist bench — payload %d bytes", payload_bytes)
        for i in range(n_iter):
            try:
                stats = await node.send_tensors(
                    server_ep, tensors, round_num=i, alpn=ALPN_FL_MODEL
                )
                conn_label = f"iroh_{stats.conn_type.value}"
                rows.append({
                    "transport"      : conn_label,
                    "payload_bytes"  : payload_bytes,
                    "iter"           : i,
                    "conn_time_ms"   : round(stats.conn_time_ms, 3),
                    "duration_ms"    : round(stats.transfer_duration_ms, 3),
                    "throughput_mbps": round(stats.throughput_mbps, 4),
                    "success"        : True,
                })
                log.info(
                    "  [%s] iter %d/%d  %dB  %.1f Mbit/s  conn=%.0fms",
                    conn_label, i + 1, n_iter,
                    payload_bytes, stats.throughput_mbps, stats.conn_time_ms,
                )
            except Exception as exc:
                log.warning("Client iter %d sz %d: %s", i, payload_bytes, exc)
                rows.append({
                    "transport": "iroh_unknown", "payload_bytes": payload_bytes,
                    "iter": i, "conn_time_ms": None, "duration_ms": None,
                    "throughput_mbps": None, "success": False,
                })

    await node.stop()

    # Write client-specific raw file
    raw_out = results_dir / "e1_dist_client.csv"
    _write_csv(raw_out, rows)
    log.info("Client raw results: %s (%d rows)", raw_out, len(rows))

    # Append to main e1_microbenchmark.csv and regenerate summary
    main_csv = results_dir / "e1_microbenchmark.csv"
    _append_and_save(main_csv, rows)
    log.info("Appended %d rows to %s", len(rows), main_csv)

    all_rows = _read_csv(main_csv)
    # Re-coerce for _summarize (needs float throughput_mbps and bool success)
    coerced = []
    for r in all_rows:
        row = dict(r)
        for col in ("throughput_mbps", "conn_time_ms", "duration_ms"):
            if row.get(col) not in ("", "None", None):
                try:
                    row[col] = float(row[col])
                except ValueError:
                    row[col] = None
        if "payload_bytes" in row:
            try:
                row["payload_bytes"] = int(row["payload_bytes"])
            except (ValueError, TypeError):
                pass
        row["success"] = str(row.get("success", "false")).lower() in ("true", "1")
        coerced.append(row)

    summary = _summarize(coerced)
    sum_out = results_dir / "e1_summary.csv"
    _write_csv(sum_out, summary)
    log.info("Summary regenerated: %s", sum_out)
    for s in summary:
        log.info("  %-18s %8dB  mean=%.1f  p95=%.1f  N=%d",
                 s["transport"], s["payload_bytes"],
                 s["mean_mbps"], s["p95_mbps"], s["n"])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main_async(args: argparse.Namespace) -> None:
    torch.manual_seed(_seed())
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Distributed server mode
    if args.role == "server":
        await run_server_bench(
            n_iter       = args.n_iter,
            payload_sizes= PAYLOAD_SIZES,
            results_dir  = results_dir,
        )
        return

    # Distributed client mode
    if args.role == "client":
        if not args.server_endpoint:
            raise SystemExit("--server-endpoint is required with --role client")
        await run_client_bench(
            n_iter        = args.n_iter,
            payload_sizes = PAYLOAD_SIZES,
            server_ep_raw = args.server_endpoint,
            results_dir   = results_dir,
        )
        return

    # In-process mode (default)
    all_rows: list[dict] = []

    log.info("Running Iroh benchmark (in-process)…")
    all_rows.extend(await bench_iroh(
        n_iter        = args.n_iter,
        payload_sizes = PAYLOAD_SIZES,
        results_dir   = args.results_dir,
        relay_url     = args.relay_url or None,
    ))

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
    parser.add_argument("--results-dir", default="results/e1")
    parser.add_argument(
        "--role",
        choices=["server", "client"],
        default=None,
        help="Distributed mode: 'server' prints endpoint and waits; "
             "'client' connects and sends payloads.",
    )
    parser.add_argument(
        "--server-endpoint",
        default=None,
        metavar="JSON_OR_@FILE",
        help="Server endpoint JSON string or @path/to/file.json (required with --role client).",
    )
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
