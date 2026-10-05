"""
E12 — Connectivity campaign on real hardware: forced relay, path recovery,
real disconnections and IP/NAT-mapping changes (R1.2, R1.7; with
scripts/nat_classify.py for R1.1).

A long-lived client sends a transfer every ``--interval`` seconds to a
long-lived server (both keep one Iroh endpoint; every transfer re-dials the
server by NodeId, as the FL data plane does).  Each transfer logs the path
Iroh actually used (direct / relay), the active address, connect and transfer
times, and the client's current local IPs.  Scheduled *events* perturb the
network mid-run:

  block_udp            drop all outgoing UDP except DNS → hole punching and
                       direct QUIC are impossible; only the relay (HTTPS/TCP
                       443) remains  (forced-relay condition, R1.2)
  unblock_udp          remove the rule → measures upgrade back to direct
  link_down:IFACE:SEC  take IFACE down for SEC seconds (real disconnection)
  net_switch:CONN      `nmcli connection up CONN` (new IP + NAT mapping)
  mark:TEXT            just log a marker (e.g. before a manual Wi-Fi→4G switch)

Events that change firewall/links need root (run with sudo).  A manual event
can also be logged at any time with ``pkill -USR1 -f e12_connectivity``.

Derived metrics (``e12_<label>_events.csv``): after each event, the time to
the first successful transfer and the time to the first transfer on each path
type — i.e. relay-fallback latency, direct-upgrade latency and reconnection
time — plus the number of failed transfers in between.

Examples
--------
  # server (PC or RPi), persistent identity:
  python -m experiments.e12_connectivity --role server --key keys/server.key \
      --endpoint-out results/e12/server_ep.json

  # client (RPi): 30 min, forced relay between minutes 5 and 15, then a 60 s
  # Wi-Fi outage at minute 20; 1 MB payload also measures relay throughput
  sudo .venv/bin/python -m experiments.e12_connectivity --role client \
      --server-endpoint results/e12/server_ep.json --label rpi-fibre-isp2 \
      --duration 1800 --interval 5 --payload 1000000 \
      --events "300:block_udp,900:unblock_udp,1200:link_down:wlan0:60"
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import os
import signal
import socket
import subprocess
import time
from pathlib import Path

from fl_coap_iroh.transport.iroh_node import ALPN_FL_UPDATE, IrohTransportNode
from fl_coap_iroh.types import IrohEndpoint

log = logging.getLogger("e12")

_RULE = ["OUTPUT", "-p", "udp", "!", "--dport", "53", "-m", "comment",
         "--comment", "fl-e12", "-j", "DROP"]


def _fw(action: str) -> None:
    """Block/unblock all outgoing UDP except DNS (nftables preferred, else iptables)."""
    import shutil
    if shutil.which("nft"):
        if action == "block":
            script = ("table inet fle12 {\n  chain out {\n"
                      "    type filter hook output priority 0; policy accept;\n"
                      # Tailscale's own WireGuard traffic stays up, so an SSH
                      # session over Tailscale survives the forced-relay window.
                      "    udp sport 41641 accept\n"
                      "    udp dport != 53 drop\n  }\n}\n")
            r = subprocess.run(["nft", "-f", "-"], input=script, capture_output=True, text=True)
        else:
            r = subprocess.run(["nft", "delete", "table", "inet", "fle12"],
                               capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"nft {action} failed: {r.stderr.strip()} (run as root)")
        return
    for tool in ("iptables", "ip6tables"):
        flag = "-I" if action == "block" else "-D"
        cmd = [tool, flag, *_RULE] if flag == "-D" else [tool, "-I", _RULE[0], "1", *_RULE[1:]]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0 and tool == "iptables":
            raise RuntimeError(f"{' '.join(cmd)} failed: {r.stderr.strip()} (run as root)")


_OVERLAY_TABLE = ("table inet fle12_overlay {\n  chain out {\n"
                  "    type filter hook output priority 0; policy accept;\n"
                  "    ip daddr 100.64.0.0/10 udp dport != 53 drop\n"
                  "    ip6 daddr fd7a:115c:a1e0::/48 udp dport != 53 drop\n"
                  "    oifname \"tailscale0\" udp dport != 53 drop\n  }\n}\n")


def _overlay_guard(on: bool) -> bool:
    """Keep Iroh off VPN overlays (Tailscale 100.64.0.0/10, tailscale0) for the
    whole run, so 'direct' always means an Iroh hole-punched path, never a path
    tunnelled through Tailscale.  TCP (e.g. SSH over Tailscale) and DNS are
    unaffected.  Returns whether the guard is active."""
    import shutil
    if not shutil.which("nft"):
        log.warning("nft not found — overlay guard disabled; audit active_addr for 100.x paths")
        return False
    if on:
        r = subprocess.run(["nft", "-f", "-"], input=_OVERLAY_TABLE, capture_output=True, text=True)
    else:
        r = subprocess.run(["nft", "delete", "table", "inet", "fle12_overlay"],
                           capture_output=True, text=True)
    if r.returncode != 0:
        if on:
            log.warning("overlay guard not applied (%s) — run as root", r.stderr.strip())
        return False
    return on


def _is_overlay(addr: str) -> bool:
    import ipaddress
    host = addr.rsplit(":", 1)[0].strip("[]")
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    return ip in ipaddress.ip_network("100.64.0.0/10") or ip in ipaddress.ip_network("fd7a:115c:a1e0::/48")


def _local_ips() -> str:
    try:
        out = subprocess.run(["hostname", "-I"], capture_output=True, text=True, timeout=2).stdout
        return " ".join(out.split())
    except Exception:  # noqa: BLE001
        try:
            return socket.gethostbyname(socket.gethostname())
        except OSError:
            return ""


class Campaign:
    def __init__(self, a: argparse.Namespace) -> None:
        self.a = a
        self.events: list[dict] = []
        self.transfers: list[dict] = []
        self.t0 = time.monotonic()
        self._udp_blocked = False

    def now(self) -> float:
        return round(time.monotonic() - self.t0, 3)

    def mark(self, kind: str, detail: str = "") -> None:
        self.events.append({"t": self.now(), "wall": time.time(), "event": kind, "detail": detail})
        log.info("EVENT %s %s at t=%.1fs", kind, detail, self.now())

    async def run_event(self, spec: str) -> None:
        kind, *rest = spec.split(":")
        if kind == "block_udp":
            _fw("block"); self._udp_blocked = True
            self.mark("block_udp")
        elif kind == "unblock_udp":
            _fw("unblock"); self._udp_blocked = False
            self.mark("unblock_udp")
        elif kind == "link_down":
            iface, secs = rest[0], float(rest[1])
            subprocess.run(["ip", "link", "set", iface, "down"], check=True)
            self.mark("link_down", f"{iface} {secs:.0f}s")
            await asyncio.sleep(secs)
            subprocess.run(["ip", "link", "set", iface, "up"], check=True)
            self.mark("link_up", iface)
        elif kind == "net_switch":
            # Switch to another saved NetworkManager connection (new IP and NAT
            # mapping), e.g. 4G hotspot → office Wi-Fi.
            name = ":".join(rest)
            self.mark("net_switch", name)
            r = subprocess.run(["nmcli", "connection", "up", name],
                               capture_output=True, text=True, timeout=120)
            self.mark("net_switch_done" if r.returncode == 0 else "net_switch_error",
                      (r.stdout or r.stderr).strip()[:100])
        elif kind == "mark":
            self.mark("mark", ":".join(rest))
        else:
            raise ValueError(f"unknown event {spec!r}")

    async def schedule(self, specs: list[tuple[float, str]]) -> None:
        for at, spec in specs:
            await asyncio.sleep(max(0.0, at - (time.monotonic() - self.t0)))
            try:
                await self.run_event(spec)
            except Exception as exc:  # noqa: BLE001
                self.mark("event_error", f"{spec}: {exc}")

    async def client(self) -> None:
        a = self.a
        server_ep = IrohEndpoint(**json.loads(Path(a.server_endpoint).read_text()))
        guard = _overlay_guard(not a.allow_overlay)
        self.mark("overlay_guard", "on" if guard else "off")
        node = IrohTransportNode("e12-client", secret_key_file=a.key or None)
        await node.start()
        payload = os.urandom(a.payload)
        loop = asyncio.get_running_loop()
        loop.add_signal_handler(signal.SIGUSR1, lambda: self.mark("manual"))
        events = [(float(t), s) for t, s in (e.split(":", 1) for e in a.events.split(",") if e)]
        sched = asyncio.create_task(self.schedule(events))
        i = 0
        try:
            while self.now() < a.duration:
                t_iter = time.monotonic()
                row = {"i": i, "t": self.now(), "wall": time.time(), "udp_blocked": self._udp_blocked,
                       "local_ips": _local_ips()}
                try:
                    st = await asyncio.wait_for(
                        node._send_bytes(server_ep, payload, i, ALPN_FL_UPDATE), a.timeout)
                    row.update(ok=True, conn_type=st.conn_type.value, active_addr=st.active_addr,
                               overlay_path=_is_overlay(st.active_addr),
                               send_attempts=st.send_attempts,
                               connect_ms=round(st.conn_time_ms, 1),
                               transfer_ms=round(st.transfer_duration_ms, 1),
                               goodput_mbps=round(st.throughput_mbps, 4), error="")
                except Exception as exc:  # noqa: BLE001
                    row.update(ok=False, conn_type="failed", active_addr="", overlay_path=False, send_attempts=None,
                               connect_ms=None,
                               transfer_ms=None, goodput_mbps=None,
                               error=f"{type(exc).__name__}: {exc}"[:100])
                self.transfers.append(row)
                log.info("#%d t=%.0fs %s %s %s", i, row["t"], row["conn_type"],
                         row["active_addr"], row.get("transfer_ms"))
                i += 1
                await asyncio.sleep(max(0.0, a.interval - (time.monotonic() - t_iter)))
        finally:
            sched.cancel()
            if self._udp_blocked:
                _fw("unblock")
            if guard:
                _overlay_guard(False)
            await node.stop()

    async def server(self) -> None:
        a = self.a
        node = IrohTransportNode("e12-server", secret_key_file=a.key or None)
        ep = await node.start()
        Path(a.endpoint_out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.endpoint_out).write_text(json.dumps(ep.model_dump(), indent=2))
        log.info("server NodeId %s → %s", ep.node_id_iroh[:16], a.endpoint_out)
        while self.now() < a.duration:
            try:
                _, st = await node._receive_bytes(ALPN_FL_UPDATE, timeout=10)
            except asyncio.TimeoutError:
                continue
            self.transfers.append({"t": self.now(), "wall": time.time(), "peer": st.peer_node_id[:16],
                                   "conn_type": st.conn_type.value, "active_addr": st.active_addr,
                                   "bytes": st.bytes_payload,
                                   "transfer_ms": round(st.transfer_duration_ms, 1)})
        await node.stop()

    def export(self) -> None:
        out = Path(self.a.results_dir)
        out.mkdir(parents=True, exist_ok=True)
        tag = f"e12_{self.a.label}_{self.a.role}"
        for name, rows in (("transfers", self.transfers), ("markers", self.events)):
            if rows:
                keys = list(dict.fromkeys(k for r in rows for k in r))
                with (out / f"{tag}_{name}.csv").open("w", newline="") as fh:
                    w = csv.DictWriter(fh, fieldnames=keys)
                    w.writeheader()
                    w.writerows(rows)
        if self.a.role == "client":
            derived = derive(self.events, self.transfers)
            if derived:
                keys = list(dict.fromkeys(k for r in derived for k in r))
                with (out / f"{tag}_events.csv").open("w", newline="") as fh:
                    w = csv.DictWriter(fh, fieldnames=keys)
                    w.writeheader()
                    w.writerows(derived)
            summary = summarise(self.transfers)
            (out / f"{tag}_summary.json").write_text(json.dumps(
                {"label": self.a.label, **summary, "events": derived}, indent=2))
            log.info("summary: %s", json.dumps(summary))


def derive(events: list[dict], transfers: list[dict]) -> list[dict]:
    """Per event: time to first success / first direct / first relay, failures in between."""
    out = []
    for k, ev in enumerate(events):
        t_ev = ev["t"]
        t_next = events[k + 1]["t"] if k + 1 < len(events) else float("inf")
        after = [r for r in transfers if t_ev <= r["t"] < t_next]
        def first(pred):
            hit = next((r for r in after if pred(r)), None)
            return round(hit["t"] - t_ev, 2) if hit else None
        ips_before = next((r["local_ips"] for r in reversed(transfers) if r["t"] < t_ev), "")
        ips_after = next((r["local_ips"] for r in after if r["ok"]), "")
        out.append({
            **ev,
            "transfers_until_next_event": len(after),
            "failed_until_first_ok": next((j for j, r in enumerate(after) if r["ok"]), len(after)),
            "s_to_first_ok": first(lambda r: r["ok"]),
            "s_to_first_direct": first(lambda r: r["conn_type"] == "direct"),
            "s_to_first_relay": first(lambda r: r["conn_type"] in ("relay", "mixed")),
            "local_ip_changed": bool(ips_before and ips_after and ips_before != ips_after),
        })
    return out


def summarise(rows: list[dict]) -> dict:
    import statistics
    res: dict = {"transfers": len(rows)}
    for blocked in (False, True):
        sub = [r for r in rows if r["udp_blocked"] == blocked]
        ok = [r for r in sub if r["ok"]]
        key = "udp_blocked" if blocked else "normal"
        res[key] = {
            "n": len(sub), "ok": len(ok),
            "direct": sum(r["conn_type"] == "direct" for r in ok),
            "direct_via_overlay": sum(bool(r["conn_type"] == "direct" and r.get("overlay_path"))
                                      for r in ok),
            "mixed": sum(r["conn_type"] == "mixed" for r in ok),
            "relay": sum(r["conn_type"] == "relay" for r in ok),
            "retried": sum(1 for r in ok if (r.get("send_attempts") or 1) > 1),
            "transfer_ms_median": round(statistics.median(r["transfer_ms"] for r in ok), 1) if ok else None,
            "goodput_mbps_median": round(statistics.median(r["goodput_mbps"] for r in ok), 3) if ok else None,
        }
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--role", choices=["client", "server"], required=True)
    ap.add_argument("--label", default="run", help="network/scenario label for file names")
    ap.add_argument("--key", default="", help="persistent secret key file")
    ap.add_argument("--endpoint-out", default="results/e12/server_ep.json")
    ap.add_argument("--server-endpoint", default="results/e12/server_ep.json")
    ap.add_argument("--duration", type=float, default=1800)
    ap.add_argument("--interval", type=float, default=5)
    ap.add_argument("--payload", type=int, default=65536, help="bytes per transfer")
    ap.add_argument("--timeout", type=float, default=60, help="per-transfer timeout (s)")
    ap.add_argument("--events", default="", help='e.g. "300:block_udp,900:unblock_udp"')
    ap.add_argument("--allow-overlay", action="store_true",
                    help="do not block Iroh from Tailscale/VPN overlay addresses")
    ap.add_argument("--results-dir", default="results/e12")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    os.environ.setdefault("FL_CONN_CLASSIFY_SETTLE_S", "3")
    c = Campaign(a)
    try:
        asyncio.run(c.client() if a.role == "client" else c.server())
    except KeyboardInterrupt:
        pass
    finally:
        c.export()


if __name__ == "__main__":
    main()
