"""
RFC 4787 NAT behaviour classification with STUN (RFC 5389 / RFC 5780) — R1.1.

Run on each client network *before* an E3/E12 campaign and store the JSON
next to the results, so NAT labels are measured rather than assumed.

Tests (all from ONE local UDP port, as a QUIC endpoint would use):
  mapping    Binding requests to several STUN servers on different IPs.
             Same mapped (IP, port) everywhere → endpoint-independent (EIM);
             port varies by destination IP only → address-dependent (ADM);
             varies with destination port too   → address-and-port-dependent
             (APDM, the classic "symmetric" NAT that defeats hole punching).
  filtering  RFC 5780 CHANGE-REQUEST to a server that supports it:
             reply from other IP+port received → endpoint-independent filtering;
             only from other port              → address-dependent filtering;
             neither                            → address-and-port-dependent.
  extras     port preservation, whether the host has a public IP (no NAT),
             and whether the local address is in 100.64.0.0/10 (CGNAT space).

Usage:
    python scripts/nat_classify.py --label home-fibre-movistar --out results/nat/
"""
from __future__ import annotations

import argparse
import ipaddress
import json
import os
import socket
import struct
import time
from pathlib import Path

MAGIC = 0x2112A442
BINDING_REQUEST = 0x0001
ATTR_MAPPED = 0x0001
ATTR_CHANGE_REQUEST = 0x0003
ATTR_XOR_MAPPED = 0x0020
ATTR_OTHER_ADDRESS = 0x802C
ATTR_RESPONSE_ORIGIN = 0x802B

# Mapping servers: distinct operators/IPs.  Filtering needs RFC 5780 support.
MAPPING_SERVERS = [
    ("stun.l.google.com", 19302),
    ("stun1.l.google.com", 19302),
    ("stun.cloudflare.com", 3478),
    ("stun.nextcloud.com", 3478),
    ("stun.sipgate.net", 3478),
]
RFC5780_SERVERS = [("stun.stunprotocol.org", 3478), ("stun.voipgate.com", 3478)]


def _request(change: int = 0) -> tuple[bytes, bytes]:
    tid = os.urandom(12)
    attrs = b""
    if change:
        attrs = struct.pack(">HHI", ATTR_CHANGE_REQUEST, 4, change)
    return struct.pack(">HHI", BINDING_REQUEST, len(attrs), MAGIC) + tid + attrs, tid


def _parse_addr(value: bytes, xor: bool, tid: bytes) -> tuple[str, int]:
    family, port = value[1], struct.unpack(">H", value[2:4])[0]
    raw = value[4:8] if family == 1 else value[4:20]
    if xor:
        port ^= MAGIC >> 16
        key = struct.pack(">I", MAGIC) + (tid if family == 2 else b"")
        raw = bytes(b ^ k for b, k in zip(raw, key))
    return str(ipaddress.ip_address(raw)), port


def _parse(resp: bytes, tid: bytes) -> dict:
    out: dict = {}
    if len(resp) < 20 or resp[8:20] != tid:
        return out
    length = struct.unpack(">H", resp[2:4])[0]
    pos = 20
    while pos + 4 <= 20 + length:
        t, ln = struct.unpack(">HH", resp[pos:pos + 4])
        val = resp[pos + 4:pos + 4 + ln]
        if t == ATTR_XOR_MAPPED:
            out["mapped"] = _parse_addr(val, True, tid)
        elif t == ATTR_MAPPED and "mapped" not in out:
            out["mapped"] = _parse_addr(val, False, tid)
        elif t == ATTR_OTHER_ADDRESS:
            out["other"] = _parse_addr(val, False, tid)
        elif t == ATTR_RESPONSE_ORIGIN:
            out["origin"] = _parse_addr(val, False, tid)
        pos += 4 + ln + ((4 - ln % 4) % 4)
    return out


def _binding(sock: socket.socket, addr: tuple[str, int], change: int = 0,
             timeout: float = 1.5, retries: int = 3) -> dict | None:
    for _ in range(retries):
        msg, tid = _request(change)
        sock.sendto(msg, addr)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            sock.settimeout(max(0.05, deadline - time.monotonic()))
            try:
                data, src = sock.recvfrom(2048)
            except (socket.timeout, OSError):
                break
            parsed = _parse(data, tid)
            if parsed:
                parsed["from"] = src
                return parsed
    return None


def _resolve(host: str, port: int) -> tuple[str, int] | None:
    try:
        return socket.getaddrinfo(host, port, socket.AF_INET, socket.SOCK_DGRAM)[0][4]
    except OSError:
        return None


def _local_ip() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect(("8.8.8.8", 53))
        return s.getsockname()[0]


def classify(label: str) -> dict:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", 0))
    local_port = sock.getsockname()[1]
    local_ip = _local_ip()

    probes = []
    for host, port in MAPPING_SERVERS:
        addr = _resolve(host, port)
        if addr is None:
            continue
        r = _binding(sock, addr)
        probes.append({"server": f"{host}:{port}", "server_ip": addr[0], "dst_port": addr[1],
                       "mapped": list(r["mapped"]) if r and "mapped" in r else None})
    ok = [p for p in probes if p["mapped"]]

    # Same server IP, different port (needed to tell ADM from APDM) and the
    # filtering test both require an RFC 5780 server (one advertising OTHER-ADDRESS).
    alt_port = None
    filt = {"filtering": "unknown (no RFC 5780 server reachable)"}
    for host, port in RFC5780_SERVERS:
        addr = _resolve(host, port)
        if addr is None:
            continue
        r = _binding(sock, addr)
        if not (r and "other" in r and "mapped" in r):
            continue
        r2 = _binding(sock, (addr[0], r["other"][1]))
        if r2 and "mapped" in r2:
            alt_port = {"server": host, "port_a": addr[1], "mapped_a": list(r["mapped"]),
                        "port_b": r["other"][1], "mapped_b": list(r2["mapped"])}
        filt = _filtering(sock, addr)
        filt["filtering_server"] = host
        break

    mapped = {tuple(p["mapped"]) for p in ok}
    if not ok:
        mapping = "unknown (no STUN response — UDP blocked?)"
    elif len(mapped) == 1:
        mapping = "endpoint-independent"
    elif alt_port is None:
        mapping = "address-dependent or address-and-port-dependent (undetermined)"
    elif alt_port["mapped_a"] == alt_port["mapped_b"]:
        mapping = "address-dependent"
    else:
        mapping = "address-and-port-dependent"

    public_ip = ok[0]["mapped"][0] if ok else None
    lip = ipaddress.ip_address(local_ip)
    result = {
        "label": label,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "local_addr": f"{local_ip}:{local_port}",
        "local_is_rfc1918": lip.is_private and lip not in ipaddress.ip_network("100.64.0.0/10"),
        "local_in_cgnat_space": lip in ipaddress.ip_network("100.64.0.0/10"),
        "behind_nat": public_ip is not None and public_ip != local_ip,
        "public_ip": public_ip,
        "mapping": mapping,
        "port_preservation": bool(ok) and all(p["mapped"][1] == local_port for p in ok),
        **filt,
        "probes": probes,
        "same_ip_other_port": alt_port,
    }
    sock.close()
    return result


def _filtering(sock: socket.socket, addr: tuple[str, int]) -> dict:
    """RFC 5780 §4.4 filtering test (CHANGE-REQUEST flags: 0x4 IP, 0x2 port)."""
    r_both = _binding(sock, addr, change=0x06, retries=2)
    if r_both:
        return {"filtering": "endpoint-independent"}
    r_port = _binding(sock, addr, change=0x02, retries=2)
    if r_port:
        return {"filtering": "address-dependent"}
    return {"filtering": "address-and-port-dependent"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", required=True, help="network label, e.g. rpi-home-fibre-isp1")
    ap.add_argument("--out", default="results/nat")
    a = ap.parse_args()
    res = classify(a.label)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    path = Path(a.out) / f"nat_{a.label}.json"
    path.write_text(json.dumps(res, indent=2))
    print(json.dumps({k: res[k] for k in ("label", "behind_nat", "local_in_cgnat_space",
                                          "mapping", "filtering", "port_preservation")}, indent=2))
    print(f"-> {path}")


if __name__ == "__main__":
    main()
