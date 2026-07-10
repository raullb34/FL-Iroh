#!/usr/bin/env bash
cd "$(dirname "$0")/.." || exit 1
./.venv/bin/python -u - > /tmp/e3_timing.txt 2>/dev/null <<'PY'
import asyncio, iroh, struct, hashlib

ALPN = b"fl-model/1"

class H(iroh.ProtocolHandler):
    async def accept(self, conn):
        try:
            rs = await conn.accept_uni()
            ln = struct.unpack(">I", await rs.read_exact(4))[0]
            await rs.read_exact(ln); await rs.read_exact(32)
        except Exception: pass
    async def shutdown(self): pass

class C(iroh.ProtocolCreator):
    def __init__(self, h): self.h=h
    def create(self, ep): return self.h

async def main():
    out=[]
    opts = iroh.NodeOptions(protocols={ALPN: C(H())})
    srv = await iroh.Iroh.memory_with_options(opts)
    cli = await iroh.Iroh.memory()
    sid = await srv.net().node_id()
    saddr = await srv.net().node_addr()
    pk = iroh.PublicKey.from_string(sid)
    ep = cli.node().endpoint()
    na = iroh.NodeAddr(pk, saddr.relay_url(), list(saddr.direct_addresses()))
    conn = await ep.connect(na, ALPN)
    payload = b"x"*1577
    frame = struct.pack(">I", len(payload)) + payload + hashlib.sha256(payload).digest()
    for t in range(12):
        s = await conn.open_uni(); await s.write(frame); await s.finish()
        try:
            info = await cli.net().remote_info(pk)
            ct = str(info.conn_type.type()).rsplit(".",1)[-1] if info else "None"
            out.append(f"RESULT t={t} conn_type={ct} rtt={conn.rtt()}")
        except Exception as e:
            out.append(f"RESULT t={t} ERR {type(e).__name__}: {e}")
        await asyncio.sleep(0.7)
    await srv.node().shutdown(); await cli.node().shutdown()
    import sys
    print("\n".join(out), file=sys.stderr)

asyncio.run(main())
PY
./.venv/bin/python -u - <<'PY'
# nothing
PY
echo "===RESULTS==="
grep -a "RESULT" /tmp/e3_timing.txt 2>/dev/null || cat /tmp/e3_timing.txt
