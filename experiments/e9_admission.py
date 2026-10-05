"""
E9 — Federation admission control via Ed25519 NodeId allow-list (R1.10).

Measures, for an aggregator that enforces an allow-list:
  * unauthorised attempts: whether any payload is accepted (must be 0) and the
    time until the unauthorised peer observes the connection close;
  * authorised transfers: latency with enforcement on vs. off (overhead);
  * spoofed registration: a peer registering another node's NodeId over
    fl-ctrl/1 is rejected (NodeId ≠ authenticated QUIC peer).

Roles:
  local   all nodes in one process over loopback (quick, reproducible)
  server  run the aggregator only; write its endpoint JSON; count outcomes
  client  run authorised/unauthorised peers against a remote server
          (e.g. PC ↔ Raspberry Pi over the WAN)

Examples:
  python -m experiments.e9_admission --role local --n 50
  # RPi (server):  python -m experiments.e9_admission --role server \
  #                    --allowlist allow.txt --endpoint-out server_ep.json --duration 900
  # PC  (client):  python -m experiments.e9_admission --role client \
  #                    --server-endpoint server_ep.json --key good.key --n 50
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

from fl_coap_iroh.transport.admission import AllowList
from fl_coap_iroh.transport.iroh_node import ALPN_FL_UPDATE, IrohTransportNode
from fl_coap_iroh.types import IrohEndpoint

log = logging.getLogger("e9")

PAYLOAD = os.urandom(64 * 1024)   # ~ size of a small AgriMLP/AirMLP update


async def _attempt(node: IrohTransportNode, server_ep: IrohEndpoint) -> dict:
    """One raw connection attempt; returns timings and the close reason."""
    import iroh  # type: ignore[import]
    pk = iroh.PublicKey.from_string(server_ep.node_id_iroh)
    addr = iroh.NodeAddr(pk, server_ep.relay_url or None, list(server_ep.addrs))
    t0 = time.monotonic()
    row = {"connect_ms": None, "closed_ms": None, "close_reason": "", "error": ""}
    try:
        conn = await asyncio.wait_for(
            node._iroh_node.node().endpoint().connect(addr, ALPN_FL_UPDATE), timeout=30)
        row["connect_ms"] = (time.monotonic() - t0) * 1000
        try:
            s = await conn.open_uni()
            await s.write(len(PAYLOAD).to_bytes(4, "big") + PAYLOAD + b"\0" * 32)
            await s.finish()
        except Exception as exc:  # noqa: BLE001 — closed while writing is expected
            row["error"] = type(exc).__name__
        reason = await asyncio.wait_for(conn.closed(), timeout=15)
        row["closed_ms"] = (time.monotonic() - t0) * 1000
        row["close_reason"] = str(reason)[:120]
    except Exception as exc:  # noqa: BLE001
        row["error"] = f"{type(exc).__name__}: {exc}"[:120]
    return row


async def _authorised_latency(node, server, server_ep, n) -> list[float]:
    out = []
    for _ in range(n):
        t0 = time.monotonic()
        await node._send_bytes(server_ep, PAYLOAD, 0, ALPN_FL_UPDATE)
        if server is not None:
            await server._receive_bytes(ALPN_FL_UPDATE, timeout=30)
        out.append((time.monotonic() - t0) * 1000)
    return out


async def run_local(n: int, out_dir: Path) -> None:
    os.environ.setdefault("FL_CONN_CLASSIFY_SETTLE_S", "0")
    good, bad = IrohTransportNode("good"), IrohTransportNode("bad")
    good_ep = await good.start()
    bad_ep = await bad.start()
    rows: list[dict] = []

    for enforce in (False, True):
        server = IrohTransportNode(
            "server", allowlist=AllowList({good_ep.node_id_iroh}, enforce=enforce))
        server_ep = await server.start()
        await _authorised_latency(good, server, server_ep, 3)          # warm-up
        lat = await _authorised_latency(good, server, server_ep, n)
        rows += [{"test": "authorised", "enforce": enforce, "iter": i, "latency_ms": round(v, 3)}
                 for i, v in enumerate(lat)]
        if enforce:
            for i in range(n):
                r = await _attempt(bad, server_ep)
                rows.append({"test": "unauthorised", "enforce": True, "iter": i, **r})
            accepted = 0
            try:
                while True:
                    await server._receive_bytes(ALPN_FL_UPDATE, timeout=2)
                    accepted += 1
            except asyncio.TimeoutError:
                pass
            # Spoofed registration: bad peer claims good's NodeId over fl-ctrl/1.
            spoof_ok = await _spoofed_registration(bad, good_ep, server_ep, server)
            summary = {
                "n": n,
                "unauthorised_payloads_accepted": accepted,
                "unauthorised_rejections_logged": sum(
                    1 for x in server.allowlist.rejections if x.channel.startswith("iroh:fl-update")),
                "spoofed_registration_rejected": spoof_ok,
            }
        await server.stop()

    for n_ in (good, bad):
        await n_.stop()
    _write(out_dir, rows, summary)


async def _spoofed_registration(bad, good_ep, server_ep, server) -> bool:
    """Allow-list the bad peer too, so only the NodeId-binding check can stop it."""
    server.allowlist.add(bad.node_id_str)
    await bad.send_control(server_ep, {
        "type": "register", "client_id": "spoof", "iroh_endpoint": good_ep.model_dump()})
    msg, stats = await server.receive_control(timeout=20)
    return msg["iroh_endpoint"]["node_id_iroh"] != stats.peer_node_id


def _write(out_dir: Path, rows: list[dict], summary: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r})
    with (out_dir / "e9_admission_events.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    def stats(sel):
        v = [r[sel[1]] for r in rows if r["test"] == sel[0] and r.get("enforce") == sel[2]
             and r.get(sel[1]) is not None]
        if not v:
            return {}
        v = sorted(v)
        return {"median": round(statistics.median(v), 2),
                "p95": round(v[int(0.95 * (len(v) - 1))], 2), "n": len(v)}

    summary.update({
        "authorised_latency_ms_open": stats(("authorised", "latency_ms", False)),
        "authorised_latency_ms_enforced": stats(("authorised", "latency_ms", True)),
        "unauthorised_time_to_close_ms": stats(("unauthorised", "closed_ms", True)),
    })
    (out_dir / "e9_admission_summary.json").write_text(json.dumps(summary, indent=2))
    log.info("E9 summary: %s", json.dumps(summary, indent=2))


async def run_server(allowlist: str, endpoint_out: str, key: str, duration: float,
                     out_dir: Path) -> None:
    server = IrohTransportNode("server", secret_key_file=key or None,
                               allowlist=AllowList.from_file(allowlist))
    ep = await server.start()
    Path(endpoint_out).write_text(json.dumps(ep.model_dump(), indent=2))
    log.info("server endpoint → %s", endpoint_out)
    accepted: list[dict] = []
    deadline = time.monotonic() + duration
    while time.monotonic() < deadline:
        try:
            _, st = await server._receive_bytes(ALPN_FL_UPDATE, timeout=5)
            accepted.append({"t": time.time(), "peer": st.peer_node_id,
                             "conn_type": st.conn_type.value})
        except asyncio.TimeoutError:
            continue
    await server.stop()
    _write(out_dir, [{"test": "server_accepted", **a} for a in accepted] +
           [{"test": "server_rejected", **r.__dict__} for r in server.allowlist.rejections],
           {"accepted": len(accepted), "rejected": len(server.allowlist.rejections)})


async def run_client(server_endpoint: str, key: str, n: int, out_dir: Path) -> None:
    server_ep = IrohEndpoint(**json.loads(Path(server_endpoint).read_text()))
    node = IrohTransportNode("client", secret_key_file=key or None)
    await node.start()
    rows = []
    if key:
        # Authorised: the server keeps the connection open, so time full transfers.
        for i, v in enumerate(await _authorised_latency(node, None, server_ep, n)):
            rows.append({"test": "authorised", "enforce": True, "iter": i, "latency_ms": round(v, 3)})
    else:
        for i in range(n):
            rows.append({"test": "unauthorised", "enforce": True, "iter": i,
                         **await _attempt(node, server_ep)})
    await node.stop()
    _write(out_dir, rows, {"n": n, "key": key or "(ephemeral)"})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--role", choices=["local", "server", "client"], default="local")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--allowlist", default="")
    ap.add_argument("--endpoint-out", default="server_endpoint.json")
    ap.add_argument("--server-endpoint", default="server_endpoint.json")
    ap.add_argument("--key", default="", help="secret key file (client: authorised identity)")
    ap.add_argument("--duration", type=float, default=900)
    ap.add_argument("--out-dir", default="results/e9")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    out = Path(a.out_dir) / a.role
    if a.role == "local":
        asyncio.run(run_local(a.n, out))
    elif a.role == "server":
        asyncio.run(run_server(a.allowlist, a.endpoint_out, a.key, a.duration, out))
    else:
        asyncio.run(run_client(a.server_endpoint, a.key, a.n, out))


if __name__ == "__main__":
    main()
