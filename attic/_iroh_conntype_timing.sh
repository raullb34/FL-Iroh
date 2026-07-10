#!/usr/bin/env bash
cd "$(dirname "$0")/.." || exit 1
./.venv/bin/python -u - <<'PY'
import asyncio, iroh, struct, hashlib, time

ALPN = b"fl-model/1"

class H(iroh.ProtocolHandler):
    def __init__(self): self.got = 0
    async def accept(self, conn):
        try:
            rs = await conn.accept_uni()
            ln = struct.unpack(">I", await rs.read_exact(4))[0]
            await rs.read_exact(ln)
            await rs.read_exact(32)
            self.got += 1
        except Exception as e:
            print("accept err", e)
    async def shutdown(self): pass

class C(iroh.ProtocolCreator):
    def __init__(self, h): self.h=h
    def create(self, ep): return self.h

async def main():
    h = H()
    opts = iroh.NodeOptions(protocols={ALPN: C(h)})
    srv = await iroh.Iroh.memory_with_options(opts)
    cli = await iroh.Iroh.memory()
    sid = await srv.net().node_id()
    saddr = await srv.net().node_addr()
    print("server addrs:", saddr.direct_addresses(), "relay:", saddr.relay_url())

    pk = iroh.PublicKey.from_string(sid)
    ep = cli.node().endpoint()
    na = iroh.NodeAddr(pk, saddr.relay_url(), list(saddr.direct_addresses()))

    # ONE persistent connection, poll conn_type over time while sending pings
    conn = await ep.connect(na, ALPN)
    payload = b"x" * 1577
    frame = struct.pack(">I", len(payload)) + payload + hashlib.sha256(payload).digest()
    for t in range(12):
        s = await conn.open_uni()
        await s.write(frame); await s.finish()
        try:
            info = await cli.net().remote_info(pk)
            ct = info.conn_type.type() if info else None
            cts = str(ct).rsplit(".",1)[-1] if ct else "None"
            addrs = []
            if info:
                for d in info.addrs:
                    addrs.append((str(d.addr), str(d.latency), str(d.last_payload)))
            print(f"t={t} conn_type={cts} rtt={conn.rtt()} addrs={addrs}")
        except Exception as e:
            print(f"t={t} remote_info err {type(e).__name__}: {e}")
        await asyncio.sleep(0.7)

    await srv.node().shutdown()
    await cli.node().shutdown()

asyncio.run(main())
PY
