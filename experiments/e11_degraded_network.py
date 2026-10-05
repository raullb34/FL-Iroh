"""
E11 — FL-Iroh under degraded networks (R1.8): latency, jitter, loss, bandwidth.

Runs ONE network condition (the impairment itself is applied by
``scripts/run_netem_sweep.sh`` with tc-netem inside an isolated network
namespace, so this script needs no privileges):

  1. transport microbenchmark: 100 kB / 1 MB / 10 MB transfers over real
     Iroh/QUIC — completion time, goodput, failures;
  2. federated training: AirMLP + FedAvg on the air-quality task (10 clients,
     geographic partition) over real Iroh — per-round duration, final
     accuracy, macro-F1 and balanced accuracy.

All peers share the impaired namespace, so the netem qdisc on ``lo`` acts as
a shared bottleneck: each packet is delayed/lost once per direction, and a
rate limit is shared by all concurrent flows.

Usage (normally invoked by the sweep script)::
    python -m experiments.e11_degraded_network --condition loss_5 --rounds 20
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import os
import statistics
import time
from pathlib import Path

log = logging.getLogger("e11")

SIZES = [100_000, 1_000_000, 10_000_000]


async def microbench(n_iter: int, timeout_s: float) -> list[dict]:
    from fl_coap_iroh.transport.iroh_node import ALPN_FL_UPDATE, IrohTransportNode
    a, b = IrohTransportNode("bench-a"), IrohTransportNode("bench-b")
    await a.start()
    eb = await b.start()
    rows = []
    for size in SIZES:
        payload = os.urandom(size)
        for i in range(n_iter):
            t0 = time.monotonic()
            try:
                st = await asyncio.wait_for(a._send_bytes(eb, payload, 0, ALPN_FL_UPDATE), timeout_s)
                await asyncio.wait_for(b._receive_bytes(ALPN_FL_UPDATE, timeout_s), timeout_s)
                ms = (time.monotonic() - t0) * 1000
                rows.append({"size": size, "iter": i, "ok": True, "ms": round(ms, 2),
                             "goodput_mbps": round(size * 8 / ms / 1000, 4),
                             "conn_type": st.conn_type.value})
            except Exception as exc:  # noqa: BLE001
                rows.append({"size": size, "iter": i, "ok": False, "ms": None,
                             "goodput_mbps": None, "conn_type": f"failed:{type(exc).__name__}"})
    await a.stop()
    await b.stop()
    return rows


async def fl_run(out: Path, rounds: int) -> dict:
    from experiments.e7_air_quality_fl import _seed, run_airmlp
    return await run_airmlp("geographic", out, rounds, _seed(), model_kind="airmlp")


def _write(path: Path, rows: list[dict]) -> None:
    if rows:
        with path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--condition", required=True)
    ap.add_argument("--netem", default="", help="netem spec, recorded for provenance")
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--bench-iter", type=int, default=10)
    ap.add_argument("--bench-timeout", type=float, default=180)
    ap.add_argument("--results-dir", default="results/e11")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    os.environ["FL_MOCK_IROH"] = "0"
    os.environ.setdefault("FL_CONN_CLASSIFY_SETTLE_S", "0.5")

    out = Path(a.results_dir) / a.condition
    out.mkdir(parents=True, exist_ok=True)
    bench = asyncio.run(microbench(a.bench_iter, a.bench_timeout))
    _write(out / "e11_transfers.csv", bench)

    t0 = time.monotonic()
    fl = asyncio.run(fl_run(out, a.rounds))
    fl_wall = time.monotonic() - t0

    rounds_csv = out / "e7_airmlp_geographic_round_events.csv"
    durs = []
    if rounds_csv.exists():
        durs = [float(r["duration_sec"]) for r in csv.DictReader(rounds_csv.open())
                if r.get("success") == "True"]
    summary = {"condition": a.condition, "netem": a.netem, "fl_wall_s": round(fl_wall, 1),
               "rounds_ok": len(durs), "rounds": a.rounds,
               "round_s_median": round(statistics.median(durs), 3) if durs else None,
               "round_s_max": round(max(durs), 3) if durs else None,
               **{k: fl.get(k) for k in ("test_acc_final", "macro_f1", "balanced_accuracy")}}
    for size in SIZES:
        rs = [r for r in bench if r["size"] == size]
        ok = [r["ms"] for r in rs if r["ok"]]
        summary[f"xfer_{size // 1000}kB_ok"] = f"{len(ok)}/{len(rs)}"
        summary[f"xfer_{size // 1000}kB_ms_median"] = round(statistics.median(ok), 1) if ok else None
    (out / "e11_summary.json").write_text(json.dumps(summary, indent=2))
    log.info("E11 %s: %s", a.condition, json.dumps(summary))


if __name__ == "__main__":
    main()
