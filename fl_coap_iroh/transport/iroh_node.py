"""
Iroh/QUIC transport layer for the FL data plane.

Responsibilities:
  - Start an Iroh node (or a mock node for unit tests)
  - Publish the resulting NodeId + addresses back to CoAP via IrohEndpoint
  - Connect to a remote peer (direct hole-punch → relay fallback)
  - Send / receive length-prefixed byte blobs over QUIC streams
  - Collect per-transfer metrics (conn_type, latency, throughput)

ALPN protocols used:
  fl-model/1   — global model distribution (server → client)
  fl-update/1  — local gradient upload (client → server)
  fl-sync/1    — P2P sync in decentralised topology

Wire format (over each QUIC stream):
  [4 bytes big-endian uint32: payload_len][payload_bytes][32 bytes: SHA-256]

Notes on the iroh Python API (iroh 0.35.x):
  iroh.Iroh.memory()                      → create in-memory node (no protocol accept)
  iroh.Iroh.memory_with_options(opts)     → node with registered ProtocolHandlers
  await node.net().node_id()              → str  (hex-encoded PublicKey)
  await node.net().node_addr()            → NodeAddr
    node_addr.relay_url()                 → str | None
    node_addr.direct_addresses()          → list[str]
  iroh.NodeAddr(PublicKey, relay_url_str, addrs_list)  → NodeAddr
  iroh.PublicKey.from_string(hex_str)     → PublicKey
  node.node().endpoint().connect(addr, alpn) → Connection
  Connection.open_uni()  / accept_uni()   → SendStream / RecvStream
  Accepting connections via ProtocolHandler subclass + NodeOptions(protocols={alpn: creator})
  node.node().shutdown()                  → stops the node

If the iroh package is unavailable the class falls back to a mock transport
that simulates realistic delays so unit tests and CI can run without Iroh.
"""
from __future__ import annotations

import asyncio
import hashlib
import io
import logging
import struct
import time
from dataclasses import dataclass, field
from typing import Optional

import torch

from fl_coap_iroh.types import ConnType, IrohEndpoint, TransferEvent

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ALPN labels
# ---------------------------------------------------------------------------
ALPN_FL_MODEL  = b"fl-model/1"
ALPN_FL_UPDATE = b"fl-update/1"
ALPN_FL_SYNC   = b"fl-sync/1"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_LEN_PREFIX_BYTES   = 4           # uint32 big-endian
_HASH_SUFFIX_BYTES  = 32          # SHA-256
_OVERHEAD_BYTES     = _LEN_PREFIX_BYTES + _HASH_SUFFIX_BYTES
MAX_PAYLOAD_BYTES   = 512 * 1024 * 1024   # 512 MB safety cap
DIRECT_TIMEOUT_SEC  = 5.0         # seconds before relay fallback is attempted


# ---------------------------------------------------------------------------
# Transfer statistics
# ---------------------------------------------------------------------------

@dataclass
class TransferStats:
    bytes_payload   : int      = 0
    bytes_on_wire   : int      = 0        # includes framing overhead
    conn_type       : ConnType = ConnType.UNKNOWN
    conn_time_ms    : float    = 0.0
    transfer_duration_ms: float= 0.0

    @property
    def throughput_mbps(self) -> float:
        if self.transfer_duration_ms <= 0:
            return 0.0
        return (self.bytes_on_wire * 8) / (self.transfer_duration_ms * 1000)


# ---------------------------------------------------------------------------
# Mock transport  (no Iroh dependency)
# ---------------------------------------------------------------------------

class _MockTransport:
    """
    In-process mock transport for testing without iroh installed.

    Two mock nodes can exchange bytes via an asyncio.Queue.
    Create paired nodes with _MockTransport.pair().
    """

    _registry: dict[str, "_MockTransport"] = {}

    def __init__(self, node_id: str, bandwidth_mbps: float = 100.0) -> None:
        self._node_id     = node_id
        self._bandwidth   = bandwidth_mbps
        self._queue: asyncio.Queue[bytes] = asyncio.Queue()
        _MockTransport._registry[node_id] = self

    def endpoint_info(self) -> IrohEndpoint:
        return IrohEndpoint(
            node_id_iroh   = f"mock-{self._node_id}",
            addrs          = ["127.0.0.1:0"],
            relay_url      = None,
            direct_capable = True,
        )

    async def send(self, peer_iroh_id: str, payload: bytes) -> TransferStats:
        peer_key = peer_iroh_id.removeprefix("mock-")
        peer = _MockTransport._registry.get(peer_key)
        if peer is None:
            raise RuntimeError(f"Mock peer not found: {peer_iroh_id!r}")
        delay_sec = (len(payload) * 8) / (self._bandwidth * 1_000_000)
        await asyncio.sleep(delay_sec)
        await peer._queue.put(payload)
        return TransferStats(
            bytes_payload      = len(payload),
            bytes_on_wire      = len(payload) + _OVERHEAD_BYTES,
            conn_type          = ConnType.DIRECT,
            conn_time_ms       = 2.0,
            transfer_duration_ms = delay_sec * 1000,
        )

    async def receive(self, timeout: float = 60.0) -> tuple[bytes, TransferStats]:
        payload = await asyncio.wait_for(self._queue.get(), timeout=timeout)
        return payload, TransferStats(
            bytes_payload      = len(payload),
            bytes_on_wire      = len(payload) + _OVERHEAD_BYTES,
            conn_type          = ConnType.DIRECT,
            conn_time_ms       = 2.0,
            transfer_duration_ms = 0.0,
        )

    @classmethod
    def clear_registry(cls) -> None:
        cls._registry.clear()


# ---------------------------------------------------------------------------
# iroh 0.35 accept-side helpers (ProtocolHandler pattern)
# ---------------------------------------------------------------------------

class _IrohAcceptQueue:
    """Collects incoming iroh Connections (from the ProtocolHandler callback)
    and exposes them to async consumers via an asyncio.Queue.

    One instance is shared across all registered ALPNs so that
    ``get_for_alpn`` can skip-and-requeue connections that don't match.
    """

    def __init__(self) -> None:
        self._queue: asyncio.Queue = asyncio.Queue()

    async def enqueue(self, alpn: bytes, conn: object) -> None:
        await self._queue.put((alpn, conn))

    async def get_for_alpn(self, alpn: bytes, timeout: float) -> object:
        """Return the next Connection whose ALPN matches *alpn*.

        Connections with a different ALPN are put back into the queue so
        they can be consumed by a later call.
        """
        loop = asyncio.get_event_loop()
        deadline = loop.time() + timeout
        deferred: list = []
        try:
            while True:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    raise asyncio.TimeoutError()
                got_alpn, conn = await asyncio.wait_for(
                    self._queue.get(), timeout=remaining
                )
                if got_alpn == alpn:
                    return conn
                log.warning(
                    "Queued ALPN %r while waiting for %r — re-queuing",
                    got_alpn, alpn,
                )
                deferred.append((got_alpn, conn))
        finally:
            for item in deferred:
                self._queue.put_nowait(item)


def _make_iroh_adapters(
    iroh_module: object,
    queue: "_IrohAcceptQueue",
    alpns: list,
) -> dict:  # type: ignore[return]
    """Create per-ALPN ProtocolCreator instances at runtime.

    In iroh 0.35 the ProtocolHandler.accept() callback receives an already-
    established Connection object directly (no .connect() call needed).
    Each ALPN gets its own creator so the handler knows which ALPN to enqueue.
    """
    iroh = iroh_module  # type: ignore[assignment]

    class _Handler(iroh.ProtocolHandler):  # type: ignore[name-defined]
        def __init__(self, q: "_IrohAcceptQueue", alpn: bytes) -> None:
            self._q    = q
            self._alpn = alpn

        async def accept(self, conn: object) -> None:
            # conn is an iroh.Connection — already established, no .connect() needed
            try:
                await self._q.enqueue(self._alpn, conn)
            except Exception as exc:
                log.warning("iroh accept error [%r]: %s", self._alpn, exc)

        async def shutdown(self) -> None:
            pass

    class _Creator(iroh.ProtocolCreator):  # type: ignore[name-defined]
        def __init__(self, alpn: bytes) -> None:
            self._alpn = alpn

        def create(self, _endpoint: object) -> "_Handler":
            return _Handler(queue, self._alpn)

    return {alpn: _Creator(alpn) for alpn in alpns}


# ---------------------------------------------------------------------------
# Main transport node
# ---------------------------------------------------------------------------

class IrohTransportNode:
    """
    Wraps an Iroh node for FL data-plane operations.

    If the `iroh` package is not installed the node operates in mock mode,
    which is useful for unit tests and dry-run experiments.

    Typical server lifecycle::
        node = IrohTransportNode("server", relay_url="https://relay.example")
        ep   = await node.start()          # ep published via CoAP
        ...
        params, stats = await node.receive_tensors(ALPN_FL_UPDATE)
        await node.stop()

    Typical client lifecycle::
        node = IrohTransportNode("client-01")
        ep   = await node.start()
        stats = await node.send_tensors(server_ep, model.state_dict(), round=1)
        await node.stop()
    """

    def __init__(
        self,
        node_id: str,
        relay_url: Optional[str] = None,
        mock_bandwidth_mbps: float = 100.0,
    ) -> None:
        self._node_id          = node_id
        self._relay_url        = relay_url
        self._mock_bandwidth   = mock_bandwidth_mbps
        self._iroh_node        = None        # iroh.Iroh instance (or None in mock)
        self._accept_queue     : Optional[_IrohAcceptQueue] = None
        self._mock             : Optional[_MockTransport] = None
        self._iroh_node_id_str : Optional[str] = None
        self._transfer_log     : list[TransferEvent] = []
        self._mock_mode        : bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> IrohEndpoint:
        """Initialise the transport and return endpoint descriptor."""
        try:
            import iroh  # type: ignore[import]

            # iroh's Rust/Tokio runtime calls ProtocolHandler callbacks from native
            # threads where asyncio.get_running_loop() raises RuntimeError.
            # Patch iroh_ffi._uniffi_get_event_loop to fall back to the main loop.
            import iroh.iroh_ffi as _iroh_ffi  # type: ignore[import]
            _main_loop = asyncio.get_running_loop()
            def _patched_get_loop() -> asyncio.AbstractEventLoop:
                try:
                    return asyncio.get_running_loop()
                except RuntimeError:
                    return _main_loop
            _iroh_ffi._uniffi_get_event_loop = _patched_get_loop

            # Build shared accept queue and register per-ALPN handlers.
            # Adapter classes must be defined after iroh is imported (uniffi requirement).
            self._accept_queue = _IrohAcceptQueue()
            creators = _make_iroh_adapters(iroh, self._accept_queue, [ALPN_FL_MODEL, ALPN_FL_UPDATE, ALPN_FL_SYNC])

            options = iroh.NodeOptions(protocols=creators)

            self._iroh_node = await iroh.Iroh.memory_with_options(options)
            net = self._iroh_node.net()

            raw_id    = await net.node_id()           # str (hex PublicKey)
            node_addr = await net.node_addr()
            relay     = node_addr.relay_url() or ""   # method → str
            addrs     = node_addr.direct_addresses()  # method → list[str]

            self._iroh_node_id_str = raw_id

            ep = IrohEndpoint(
                node_id_iroh   = raw_id,
                addrs          = addrs,
                relay_url      = self._relay_url or relay,
                direct_capable = len(addrs) > 0,
            )
            log.info("Iroh node started: %s  addrs=%s", raw_id[:16], addrs)
            return ep

        except ImportError:
            log.warning("iroh package not available — using mock transport")
            self._mock_mode = True
            self._mock = _MockTransport(self._node_id, self._mock_bandwidth)
            self._iroh_node_id_str = f"mock-{self._node_id}"
            return self._mock.endpoint_info()

    async def stop(self) -> None:
        if self._iroh_node is not None:
            try:
                await self._iroh_node.node().shutdown()
            except Exception as exc:
                log.warning("Iroh shutdown error: %s", exc)
        _MockTransport.clear_registry()

    # ------------------------------------------------------------------
    # High-level tensor transfer
    # ------------------------------------------------------------------

    async def send_tensors(
        self,
        peer_ep  : IrohEndpoint,
        tensors  : dict[str, torch.Tensor],
        round_num: int,
        alpn     : bytes = ALPN_FL_MODEL,
    ) -> TransferStats:
        """Serialise *tensors* and send to *peer_ep*."""
        buf = io.BytesIO()
        torch.save({k: v.cpu() for k, v in tensors.items()}, buf)
        payload = buf.getvalue()
        log.debug("send_tensors: %d bytes → %s", len(payload), peer_ep.node_id_iroh[:16])
        stats = await self._send_bytes(peer_ep, payload, round_num, alpn)
        self._log_transfer(peer_ep.node_id_iroh, "send", stats, round_num)
        return stats

    async def receive_tensors(
        self,
        alpn     : bytes = ALPN_FL_MODEL,
        timeout  : float = 120.0,
    ) -> tuple[dict[str, torch.Tensor], TransferStats]:
        """Accept one incoming tensor transfer matching *alpn*."""
        payload, stats = await self._receive_bytes(alpn, timeout)
        buf     = io.BytesIO(payload)
        tensors = torch.load(buf, weights_only=True)
        return tensors, stats

    # ------------------------------------------------------------------
    # Low-level byte transfer
    # ------------------------------------------------------------------

    async def _send_bytes(
        self,
        peer_ep  : IrohEndpoint,
        payload  : bytes,
        round_num: int,
        alpn     : bytes,
    ) -> TransferStats:
        if self._mock_mode:
            return await self._mock.send(peer_ep.node_id_iroh, payload)  # type: ignore[union-attr]

        try:
            import iroh  # type: ignore[import]
        except ImportError:
            raise RuntimeError("iroh not available")

        t_start = time.monotonic()

        # Build NodeAddr using iroh 0.35 API:
        #   NodeAddr(PublicKey, relay_url_str | None, list[str])
        pk        = iroh.PublicKey.from_string(peer_ep.node_id_iroh)
        relay_url = peer_ep.relay_url if peer_ep.relay_url else None
        node_addr = iroh.NodeAddr(pk, relay_url, list(peer_ep.addrs))

        # Connect via the Endpoint obtained from node.node()
        # Retry up to 3 times to handle transient iroh connection failures
        # (e.g., remote node not yet fully initialised when the first round begins).
        t_conn0  = time.monotonic()
        endpoint = self._iroh_node.node().endpoint()
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                connection = await asyncio.wait_for(
                    endpoint.connect(node_addr, alpn),
                    timeout=DIRECT_TIMEOUT_SEC + 30,
                )
                break
            except Exception as exc:
                if attempt == max_attempts:
                    raise
                wait = 5 * attempt
                log.warning(
                    "_send_bytes connect attempt %d/%d failed (%s: %s) — retrying in %ds",
                    attempt, max_attempts, type(exc).__name__, exc, wait,
                )
                await asyncio.sleep(wait)
        conn_time_ms = (time.monotonic() - t_conn0) * 1000

        # Detect connection type
        conn_type = _detect_conn_type(connection)

        # Frame: [len(4)] + [payload] + [sha256(32)]
        sha = hashlib.sha256(payload).digest()
        frame = struct.pack(">I", len(payload)) + payload + sha

        send_stream = await connection.open_uni()
        await send_stream.write(frame)
        await send_stream.finish()

        stats = TransferStats(
            bytes_payload        = len(payload),
            bytes_on_wire        = len(frame),
            conn_type            = conn_type,
            conn_time_ms         = conn_time_ms,
            transfer_duration_ms = (time.monotonic() - t_start) * 1000,
        )
        log.info(
            "send %dB via %s in %.0fms (%.2f Mbit/s) to %s",
            len(payload), conn_type.value, stats.transfer_duration_ms,
            stats.throughput_mbps, peer_ep.node_id_iroh[:16],
        )
        return stats

    async def _receive_bytes(self, alpn: bytes, timeout: float) -> tuple[bytes, TransferStats]:
        if self._mock_mode:
            return await self._mock.receive(timeout)  # type: ignore[union-attr]

        try:
            import iroh  # type: ignore[import]
        except ImportError:
            raise RuntimeError("iroh not available")

        t_start = time.monotonic()

        # Accept next incoming connection matching *alpn* from the queue
        # that is populated by the ProtocolHandler registered at start().
        if self._accept_queue is None:
            raise RuntimeError("IrohTransportNode not started (accept_queue is None)")
        connection = await self._accept_queue.get_for_alpn(alpn, timeout)
        conn_type  = _detect_conn_type(connection)
        conn_time_ms = (time.monotonic() - t_start) * 1000

        # Accept unidirectional stream
        recv_stream = await connection.accept_uni()

        # Read length prefix
        raw_len = await recv_stream.read_exact(_LEN_PREFIX_BYTES)
        payload_len = struct.unpack(">I", raw_len)[0]
        if payload_len > MAX_PAYLOAD_BYTES:
            raise ValueError(f"Payload too large: {payload_len} > {MAX_PAYLOAD_BYTES}")

        payload = await recv_stream.read_exact(payload_len)

        # Verify SHA-256
        received_hash = await recv_stream.read_exact(_HASH_SUFFIX_BYTES)
        expected_hash = hashlib.sha256(payload).digest()
        if received_hash != expected_hash:
            raise ValueError("SHA-256 integrity check failed — payload corrupted")

        stats = TransferStats(
            bytes_payload        = payload_len,
            bytes_on_wire        = payload_len + _OVERHEAD_BYTES,
            conn_type            = conn_type,
            conn_time_ms         = conn_time_ms,
            transfer_duration_ms = (time.monotonic() - t_start) * 1000,
        )
        log.info(
            "recv %dB via %s in %.0fms",
            payload_len, conn_type.value, stats.transfer_duration_ms,
        )
        return payload, stats

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def _log_transfer(
        self, peer_id: str, direction: str, stats: TransferStats, round_num: int
    ) -> None:
        self._transfer_log.append(
            TransferEvent(
                peer_node_id      = peer_id,
                direction         = direction,
                bytes_transferred = stats.bytes_on_wire,
                duration_ms       = stats.transfer_duration_ms,
                throughput_mbps   = stats.throughput_mbps,
                conn_type         = stats.conn_type,
                round             = round_num,
            )
        )

    def drain_transfer_log(self) -> list[TransferEvent]:
        """Return and clear the accumulated transfer events."""
        events, self._transfer_log = self._transfer_log, []
        return events

    @property
    def node_id_str(self) -> Optional[str]:
        return self._iroh_node_id_str

    @property
    def mock_mode(self) -> bool:
        return self._mock_mode


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _detect_conn_type(connection: object) -> ConnType:
    """Extract ConnType from an iroh Connection object."""
    try:
        info = connection.conn_type()  # type: ignore[attr-defined]
        s = str(info).lower()
        if "direct" in s:
            return ConnType.DIRECT
        if "relay" in s:
            return ConnType.RELAY
    except Exception:
        pass
    return ConnType.UNKNOWN
