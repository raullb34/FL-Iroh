import iroh, asyncio

# Test: can we directly subclass ProtocolHandler and ProtocolCreator?
class MyHandler(iroh.ProtocolHandler):
    async def accept(self, connecting):
        pass
    async def shutdown(self):
        pass

class MyCreator(iroh.ProtocolCreator):
    def __init__(self):
        self._h = MyHandler()
    def create(self, ep):
        return self._h

async def main():
    options = iroh.NodeOptions(protocols={b"test/1": MyCreator()})
    node = await iroh.Iroh.memory_with_options(options)
    nid = await node.net().node_id()
    print("Success! node_id:", nid[:16])
    await node.node().shutdown()

asyncio.run(main())
