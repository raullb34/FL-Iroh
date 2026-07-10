#!/usr/bin/env bash
cd "$(dirname "$0")/.." || exit 1
./.venv/bin/python -u - <<'PY'
import asyncio, iroh

async def main():
    a = await iroh.Iroh.memory()
    b = await iroh.Iroh.memory()
    neta = a.net(); netb = b.net()
    aid = await neta.node_id()
    bid = await netb.node_id()
    print("a:", aid[:16], " b:", bid[:16])

    # Inspect remote_info signature/return
    import inspect
    print("remote_info is coroutine fn:", asyncio.iscoroutinefunction(neta.remote_info))

    # add b's addr to a so a knows about b
    baddr = await netb.node_addr()
    await neta.add_node_addr(baddr)

    try:
        info = await neta.remote_info(iroh.PublicKey.from_string(bid))
    except Exception as e:
        print("remote_info error:", type(e).__name__, e)
        info = None
    print("RemoteInfo:", info)
    if info is not None:
        for attr in dir(info):
            if attr.startswith("_"): continue
            try:
                val = getattr(info, attr)
                print("  ", attr, "=", val)
            except Exception as e:
                print("  ", attr, "ERR", e)
        ct = getattr(info, "conn_type", None)
        print("conn_type obj:", ct, type(ct))
        if ct is not None:
            try:
                print("  ct.type() =", ct.type())
            except Exception as e:
                print("  ct.type() ERR", e)

    await a.node().shutdown()
    await b.node().shutdown()

asyncio.run(main())
PY
