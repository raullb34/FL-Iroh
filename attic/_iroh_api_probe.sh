#!/usr/bin/env bash
cd "$(dirname "$0")/.." || exit 1
./.venv/bin/python -u - <<'PY'
import iroh
print("iroh version:", getattr(iroh, "__version__", "?"))
print()
print("=== iroh module attrs (conn/remote/path related) ===")
for n in dir(iroh):
    nl = n.lower()
    if any(k in nl for k in ("conn", "remote", "path", "info", "type", "addr", "endpoint")):
        print(" ", n)
print()
print("=== Connection methods ===")
C = getattr(iroh, "Connection", None)
print("Connection:", C)
if C:
    print([m for m in dir(C) if not m.startswith("__")])
print()
print("=== RemoteInfo / ConnectionType if present ===")
for name in ("RemoteInfo", "ConnectionType", "ConnType", "PathInfo", "DirectAddrInfo", "RemoteInfoOptions"):
    obj = getattr(iroh, name, None)
    if obj is not None:
        print(name, "->", [m for m in dir(obj) if not m.startswith("__")])
print()
print("=== Net methods ===")
N = getattr(iroh, "Net", None)
if N:
    print([m for m in dir(N) if not m.startswith("__")])
PY
