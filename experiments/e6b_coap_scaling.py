"""
E6b — CoAP discovery scaling to thousands of endpoints (R1.9).

Replaces the E6 measurement, which probed the same server n times and
extrapolated bytes.  Two discovery designs are compared:

  probe  client-side scan: GET /fl/capabilities (+ /iroh/endpoint when the
         node matches) on every node — one real CoAP server per node, each on
         its own port, hosted by worker processes;
  rd     Resource Directory on the aggregator (RFC 9176-style): nodes POST
         /rd once; discovery is one GET /rd-lookup/ep with server-side
         filtering on multiple capability tags (optionally paged).

Measured per N: simultaneous registration storm (RD), lookup latency and
wire bytes for 1-, 2- and 3-tag queries of decreasing selectivity, and the
probe scan (up to --probe-max nodes, as each probe node is a full server).

Usage::
    python -m experiments.e6b_coap_scaling --sizes 10,100,500,1000,2000,5000 \
        --probe-max 1000 --n-iter 30 --results-dir results/e6b
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import multiprocessing as mp
import random
import statistics
import time
from pathlib import Path

log = logging.getLogger("e6b")

REGIONS = [f"r{i}" for i in range(10)]
SENSORS = ["soil_moisture", "air_temp", "humidity", "leaf_wetness",
           "pm10", "pm25", "no2", "o3"]
# Queries of decreasing selectivity (≈50 %, ≈5 %, ≈1-2 % of nodes)
QUERIES = {"1tag": ["agri"], "2tag": ["agri", "r3"], "3tag": ["agri", "r3", "soil_moisture"]}


def _tags(i: int, rng: random.Random) -> list[str]:
    domain = "agri" if i % 2 == 0 else "air"
    return [domain, rng.choice(REGIONS), *rng.sample(SENSORS, rng.randint(2, 3))]


def _fake_ep(i: int) -> dict:
    return {"node_id_iroh": f"{i:064x}", "addrs": [f"10.{i // 65536 % 256}.{i // 256 % 256}.{i % 256}:11204"],
            "relay_url": "https://euw1-1.relay.iroh.network./", "direct_capable": True}


def _raise_nofile() -> None:
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
    except Exception:  # noqa: BLE001
        pass


# ---------------------------------------------------------------------------
# Separate processes: aggregator (RD) and probe-node hosts
# ---------------------------------------------------------------------------

def _aggregator_proc(port: int, ready, stop) -> None:
    _raise_nofile()
    from fl_coap_iroh.coap.server import FLCoapServer
    from fl_coap_iroh.types import DatasetDescriptor, NodeCapabilities, NodeRole

    async def run() -> None:
        s = FLCoapServer("aggregator", NodeCapabilities(node_id="aggregator", role=NodeRole.AGGREGATOR),
                         DatasetDescriptor(dataset_id="x", dataset_name="x", samples=0, classes=[],
                                           iid=True, distribution="iid", feature_dim=[]),
                         coap_host="127.0.0.1", coap_port=port, enable_rd=True)
        await s.start()
        ready.set()
        while not stop.is_set():
            await asyncio.sleep(0.2)
        await s.stop()

    asyncio.run(run())


def _nodes_proc(first: int, count: int, base_port: int, seed: int, ready, stop) -> None:
    _raise_nofile()
    from fl_coap_iroh.coap.server import FLCoapServer
    from fl_coap_iroh.types import (DatasetDescriptor, IrohEndpoint, NodeCapabilities,
                                    NodeRole)

    async def run() -> None:
        servers = []
        for i in range(first, first + count):
            rng = random.Random(seed + i)
            s = FLCoapServer(f"node-{i}", NodeCapabilities(node_id=f"node-{i}", role=NodeRole.CLIENT,
                                                           tags=_tags(i, rng)),
                             DatasetDescriptor(dataset_id="x", dataset_name="x", samples=0, classes=[],
                                               iid=True, distribution="iid", feature_dim=[]),
                             coap_host="127.0.0.1", coap_port=base_port + i)
            s.update_iroh_endpoint(IrohEndpoint(**_fake_ep(i)))
            await s.start()
            servers.append(s)
        ready.set()
        while not stop.is_set():
            await asyncio.sleep(0.2)
        for s in servers:
            await s.stop()

    asyncio.run(run())


# ---------------------------------------------------------------------------
# Measurements
# ---------------------------------------------------------------------------

async def _registration_storm(n: int, port: int, seed: int, concurrency: int) -> list[dict]:
    import aiocoap
    from fl_coap_iroh.coap.client import FLCoapClient
    ctx = await aiocoap.Context.create_client_context()
    sem = asyncio.Semaphore(concurrency)
    rows: list[dict] = []

    async def reg(i: int) -> None:
        rng = random.Random(seed + i)
        tags = _tags(i, rng)          # same draw as the probe nodes → same match sets
        entry = {"node_id": f"node-{i}", "role": "client", "energy": 100.0,
                 "status": "ready", "tags": tags, "lt": 3600, "iroh_endpoint": _fake_ep(i)}
        async with sem:
            c = FLCoapClient("127.0.0.1", port)
            c._ctx = ctx
            t = time.monotonic()
            try:
                nbytes = await c.rd_register(entry)
                rows.append({"i": i, "ok": True, "ms": (time.monotonic() - t) * 1000, "bytes": nbytes})
            except Exception as exc:  # noqa: BLE001
                rows.append({"i": i, "ok": False, "ms": None, "bytes": 0, "error": str(exc)[:80]})

    t0 = time.monotonic()
    await asyncio.gather(*(reg(i) for i in range(n)))
    total_ms = (time.monotonic() - t0) * 1000
    await ctx.shutdown()
    for r in rows:
        r["storm_total_ms"] = total_ms
    return rows


async def _rd_lookups(n: int, port: int, n_iter: int) -> list[dict]:
    from fl_coap_iroh.coap.client import FLCoapClient
    rows = []
    async with FLCoapClient("127.0.0.1", port) as c:
        for qname, tags in QUERIES.items():
            for paged in (False, True):
                for it in range(n_iter):
                    t = time.monotonic()
                    hits, nbytes = await c.rd_lookup(tags, count=50 if paged else None)
                    rows.append({"design": "rd", "n_nodes": n, "query": qname,
                                 "paged": paged, "iter": it, "matches": len(hits),
                                 "duration_ms": (time.monotonic() - t) * 1000, "bytes": nbytes})
    return rows


async def _probe_scan(n: int, base_port: int, n_iter: int) -> list[dict]:
    from fl_coap_iroh.coap.client import probe_nodes
    targets = [("127.0.0.1", base_port + i) for i in range(n)]
    rows = []
    for qname, tags in QUERIES.items():
        for it in range(n_iter):
            hits, ms, nbytes = await probe_nodes(targets, tags)
            rows.append({"design": "probe", "n_nodes": n, "query": qname, "paged": False,
                         "iter": it, "matches": len(hits), "duration_ms": ms, "bytes": nbytes})
    return rows


def _start(target, args) -> tuple:
    ctx = mp.get_context("spawn")
    ready, stop = ctx.Event(), ctx.Event()
    p = ctx.Process(target=target, args=(*args, ready, stop), daemon=True)
    p.start()
    return p, ready, stop


def run(sizes: list[int], probe_max: int, n_iter: int, out: Path, seed: int,
        concurrency: int, per_worker: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    lookups, storms = [], []
    rd_port = 19683
    for n in sizes:
        # ---- Resource Directory -------------------------------------------
        p, ready, stop = _start(_aggregator_proc, (rd_port,))
        ready.wait(60)
        storm = asyncio.run(_registration_storm(n, rd_port, seed, concurrency))
        for r in storm:
            r["n_nodes"] = n
        storms += storm
        ok = sum(r["ok"] for r in storm)
        log.info("RD n=%d: %d/%d registered in %.0f ms", n, ok, n, storm[0]["storm_total_ms"])
        lookups += asyncio.run(_rd_lookups(n, rd_port, n_iter))
        stop.set(); p.join(10)
        rd_port += 1

        # ---- Probe scan ------------------------------------------------------
        if n <= probe_max:
            base = 30000
            procs = [_start(_nodes_proc, (f, min(per_worker, n - f), base, seed))
                     for f in range(0, n, per_worker)]
            for _, r, _ in procs:
                r.wait(600)
            lookups += asyncio.run(_probe_scan(n, base, max(3, n_iter // 3)))
            for pp, _, st in procs:
                st.set(); pp.join(30)
        _summarise(lookups, storms, out)


def _summarise(lookups: list[dict], storms: list[dict], out: Path) -> None:
    for name, rows in (("e6b_lookups.csv", lookups), ("e6b_registrations.csv", storms)):
        if rows:
            keys = sorted({k for r in rows for k in r})
            with (out / name).open("w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=keys)
                w.writeheader()
                w.writerows(rows)
    groups: dict[tuple, list[dict]] = {}
    for r in lookups:
        groups.setdefault((r["design"], r["n_nodes"], r["query"], r["paged"]), []).append(r)
    summary = []
    for (design, n, q, paged), rs in sorted(groups.items()):
        d = sorted(r["duration_ms"] for r in rs)
        summary.append({"design": design, "n_nodes": n, "query": q, "paged": paged,
                        "matches": rs[0]["matches"], "median_ms": round(statistics.median(d), 2),
                        "p95_ms": round(d[int(0.95 * (len(d) - 1))], 2),
                        "bytes_median": int(statistics.median(r["bytes"] for r in rs)),
                        "n": len(rs)})
    sgroups: dict[int, list[dict]] = {}
    for r in storms:
        sgroups.setdefault(r["n_nodes"], []).append(r)
    for n, rs in sorted(sgroups.items()):
        ms = sorted(r["ms"] for r in rs if r["ok"])
        summary.append({"design": "rd-register", "n_nodes": n, "query": "storm", "paged": False,
                        "matches": sum(r["ok"] for r in rs),
                        "median_ms": round(statistics.median(ms), 2) if ms else None,
                        "p95_ms": round(ms[int(0.95 * (len(ms) - 1))], 2) if ms else None,
                        "bytes_median": int(statistics.median(r["bytes"] for r in rs)),
                        "n": len(rs), "storm_total_ms": round(rs[0]["storm_total_ms"], 1)})
    keys = sorted({k for r in summary for k in r})
    with (out / "e6b_summary.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(summary)
    for r in summary:
        log.info("%s", json.dumps(r))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", default="10,100,500,1000,2000,5000")
    ap.add_argument("--probe-max", type=int, default=1000)
    ap.add_argument("--n-iter", type=int, default=30)
    ap.add_argument("--concurrency", type=int, default=64, help="in-flight registrations")
    ap.add_argument("--per-worker", type=int, default=250, help="probe nodes per worker process")
    ap.add_argument("--seed", type=int, default=404)
    ap.add_argument("--results-dir", default="results/e6b")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    _raise_nofile()
    run([int(s) for s in a.sizes.split(",")], a.probe_max, a.n_iter, Path(a.results_dir),
        a.seed, a.concurrency, a.per_worker)


if __name__ == "__main__":
    main()
