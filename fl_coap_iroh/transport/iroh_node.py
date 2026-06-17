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

# NOTE: torch is imported lazily inside send_tensors / receive_tensors so that
# nodes which only need the raw-byte transport (e.g. the E3 NAT-traversal
# server) can run on hosts where torch is unavailable or broken.
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
    active_addr     : str      = ""       # socket addr of direct path (audit)

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
# iroh accept-side helpers (ProtocolHandler pattern)
# ---------------------------------------------------------------------------

@dataclass
class _AcceptedTransfer:
    """Pre-read transfer placed in the accept queue by the ProtocolHandler.

    Reading the stream *inside* the handler avoids cross-thread use of iroh
    Connection objects, which is the root cause of accept_uni() failures when
    called from the main asyncio event loop after the handler returns.
    """
    alpn             : bytes
    payload          : bytes
    conn_type        : "ConnType"
    recv_start_ms    : float      # monotonic ms when accept() was called
    recv_done_ms     : float      # monotonic ms when last byte was read
    active_addr      : str = ""   # socket addr of direct path (audit)


class _IrohAcceptQueue:
    """Collects pre-read _AcceptedTransfer objects (from the ProtocolHandler
    callback) and exposes them to async consumers via an asyncio.Queue.

    All iroh API calls (accept_uni, read_exact) happen inside the handler so
    no Connection object crosses the thread boundary.
    """

    def __init__(self) -> None:
        self._queue: asyncio.Queue[_AcceptedTransfer] = asyncio.Queue()

    async def enqueue(self, transfer: "_AcceptedTransfer") -> None:
        await self._queue.put(transfer)

    async def get_for_alpn(self, alpn: bytes, timeout: float) -> "_AcceptedTransfer":
        """Return the next transfer whose ALPN matches *alpn*.

        Transfers with a different ALPN are put back so they can be consumed
        by another caller waiting for that ALPN.
        """
        loop = asyncio.get_event_loop()
        deadline = loop.time() + timeout
        deferred: list[_AcceptedTransfer] = []
        try:
            while True:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    raise asyncio.TimeoutError()
                transfer = await asyncio.wait_for(
                    self._queue.get(), timeout=remaining
                )
                if transfer.alpn == alpn:
                    return transfer
                log.warning(
                    "Queued ALPN %r while waiting for %r — re-queuing",
                    transfer.alpn, alpn,
                )
                deferred.append(transfer)
        finally:
            for item in deferred:
                self._queue.put_nowait(item)


def _make_iroh_adapters(
    iroh_module: object,
    queue: "_IrohAcceptQueue",
    alpns: list,
    main_loop: asyncio.AbstractEventLoop,
    node_ref: "IrohTransportNode",
) -> dict:  # type: ignore[return]
    """Create per-ALPN ProtocolCreator instances at runtime.

    Key design: ProtocolHandler.accept() runs in a Rust/Tokio thread (uniffi).
    We must NOT await iroh stream methods there — they require the main asyncio
    event loop.  Instead we schedule the actual stream reading via
    run_coroutine_threadsafe and return from accept() immediately.
    The closure reference to `conn` keeps the Connection alive in Python's
    reference count while the scheduled coroutine is pending.
    """
    iroh = iroh_module  # type: ignore[assignment]

    async def _read_and_enqueue(conn: object, alpn: bytes) -> None:
        """Run on main asyncio event loop: read stream and put transfer in queue."""
        t_start = time.monotonic() * 1000
        try:
            recv_stream = await conn.accept_uni()  # type: ignore[attr-defined]

            raw_len = await recv_stream.read_exact(_LEN_PREFIX_BYTES)
            payload_len = struct.unpack(">I", raw_len)[0]
            if payload_len > MAX_PAYLOAD_BYTES:
                raise ValueError(f"Payload too large: {payload_len}")

            payload = await recv_stream.read_exact(payload_len)

            received_hash = await recv_stream.read_exact(_HASH_SUFFIX_BYTES)
            if hashlib.sha256(payload).digest() != received_hash:
                raise ValueError("SHA-256 integrity check failed")

            t_done = time.monotonic() * 1000
            # Authoritative path classification via iroh's own remote_info API.
            conn_type, active_addr = await _classify_remote_conn(node_ref, conn)
            await queue.enqueue(_AcceptedTransfer(
                alpn          = alpn,
                payload       = payload,
                conn_type     = conn_type,
                recv_start_ms = t_start,
                recv_done_ms  = t_done,
                active_addr   = active_addr,
            ))
            log.debug(
                "iroh recv [%r]: %dB conn_type=%s addr=%s in %.0fms",
                alpn, payload_len, conn_type.value, active_addr, t_done - t_start,
            )
        except Exception as exc:
            log.warning(
                "iroh accept/read error [%r]: %s: %s",
                alpn, type(exc).__name__, exc,
            )

    class _Handler(iroh.ProtocolHandler):  # type: ignore[name-defined]
        def __init__(self, q: "_IrohAcceptQueue", alpn: bytes) -> None:
            self._q    = q
            self._alpn = alpn

        async def accept(self, conn: object) -> None:
            # Schedule stream reading on main asyncio event loop and return
            # immediately.  conn is kept alive by the coroutine closure.
            asyncio.run_coroutine_threadsafe(
                _read_and_enqueue(conn, self._alpn),
                main_loop,
            )

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
        import os
        if os.environ.get("FL_MOCK_IROH", "0") == "1":
            log.info("FL_MOCK_IROH=1 — forcing mock transport (iroh bypassed)")
            self._mock_mode = True
            self._mock = _MockTransport(self._node_id, self._mock_bandwidth)
            self._iroh_node_id_str = f"mock-{self._node_id}"
            return self._mock.endpoint_info()
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
            creators = _make_iroh_adapters(iroh, self._accept_queue, [ALPN_FL_MODEL, ALPN_FL_UPDATE, ALPN_FL_SYNC], _main_loop, self)

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
        import torch
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
        import torch
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

        # Frame: [len(4)] + [payload] + [sha256(32)]
        sha = hashlib.sha256(payload).digest()
        frame = struct.pack(">I", len(payload)) + payload + sha

        send_stream = await connection.open_uni()
        await send_stream.write(frame)
        await send_stream.finish()

        # Classify the path actually used, via the real iroh remote_info API
        # (authoritative: direct hole-punch vs DERP relay).  pk is the remote
        # PublicKey we connected to.
        conn_type, active_addr = await _classify_remote_conn(self, connection, pk)
        if active_addr:
            log.debug("path addr=%s", active_addr)

        stats = TransferStats(
            bytes_payload        = len(payload),
            bytes_on_wire        = len(frame),
            conn_type            = conn_type,
            conn_time_ms         = conn_time_ms,
            transfer_duration_ms = (time.monotonic() - t_start) * 1000,
            active_addr          = active_addr,
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

        if self._accept_queue is None:
            raise RuntimeError("IrohTransportNode not started (accept_queue is None)")

        t_wait_start = time.monotonic()
        # All iroh stream I/O was done inside the ProtocolHandler.accept() callback.
        # We just unpack the pre-read transfer from the queue.
        transfer = await self._accept_queue.get_for_alpn(alpn, timeout)
        wait_ms = (time.monotonic() - t_wait_start) * 1000

        conn_type = transfer.conn_type
        transfer_ms = transfer.recv_done_ms - transfer.recv_start_ms

        # conn_type is the authoritative value from iroh's remote_info API
        # (set in the accept handler). We intentionally do NOT fall back to a
        # latency heuristic here: an unknown path stays 'unknown' rather than
        # being fabricated from transfer duration.

        stats = TransferStats(
            bytes_payload        = len(transfer.payload),
            bytes_on_wire        = len(transfer.payload) + _OVERHEAD_BYTES,
            conn_type            = conn_type,
            conn_time_ms         = wait_ms,
            transfer_duration_ms = transfer_ms,
            active_addr          = transfer.active_addr,
        )
        log.info(
            "recv %dB via %s in %.0fms",
            len(transfer.payload), conn_type.value, transfer_ms,
        )
        return transfer.payload, stats

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

async def _classify_remote_conn(
    node_ref: "IrohTransportNode",
    conn: object,
    remote_pubkey: object = None,
) -> tuple[ConnType, str]:
    """Classify a live connection's path using the real iroh ``remote_info`` API.

    Returns ``(ConnType, active_addr)`` where *active_addr* is the socket
    address of the direct path when one is in use (empty string for relay).
    This is the authoritative classification: it reflects iroh's own view of
    whether QUIC traffic flows over a hole-punched direct path or the DERP
    relay.  The active address is returned so overlay paths (e.g. Tailscale
    100.64.0.0/10) can be audited out of "direct" counts.
    """
    try:
        node = node_ref._iroh_node
        if node is None:
            return ConnType.UNKNOWN, ""
        net = node.net()
        if remote_pubkey is None:
            remote_pubkey = conn.remote_node_id()  # type: ignore[attr-defined]
            if asyncio.iscoroutine(remote_pubkey):
                remote_pubkey = await remote_pubkey
        info = await net.remote_info(remote_pubkey)
    except Exception as exc:  # noqa: BLE001
        log.debug("remote_info classification failed: %s", exc)
        return ConnType.UNKNOWN, ""

    if info is None:
        return ConnType.UNKNOWN, ""
    try:
        ct_obj = info.conn_type
        kind = str(ct_obj.type()).rsplit(".", 1)[-1].upper()  # DIRECT/RELAY/MIXED/NONE
    except Exception as exc:  # noqa: BLE001
        log.debug("conn_type read failed: %s", exc)
        return ConnType.UNKNOWN, ""

    if kind == "DIRECT":
        addr = ""
        try:
            addr = str(ct_obj.as_direct())
        except Exception:
            pass
        return ConnType.DIRECT, addr
    if kind == "MIXED":
        # A direct path exists alongside relay; count as direct but keep the
        # address so overlay (Tailscale) paths can be audited out.
        addr = ""
        try:
            addr = str(ct_obj.as_mixed())
        except Exception:
            pass
        return ConnType.DIRECT, addr
    if kind == "RELAY":
        return ConnType.RELAY, ""
    return ConnType.UNKNOWN, ""


def _detect_conn_type(connection: object) -> ConnType:
    """Extract ConnType from an iroh Connection object.

    Tries multiple APIs in order of preference:
    1. iroh 1.0+ ``paths()`` → PathWatcher → selected PathInfo addr
    2. iroh 0.35 ``conn_type()`` → ConnType enum string
    Falls back to UNKNOWN if neither API is available.
    """
    # --- iroh 1.0+ paths() API ---
    try:
        watcher = connection.paths()  # type: ignore[attr-defined]
        # PathWatcher.get() or direct iteration depending on binding version
        path_list = watcher.get() if hasattr(watcher, "get") else list(watcher)
        for path_info in path_list:
            if getattr(path_info, "is_selected", False) and not getattr(path_info, "is_closed", False):
                addr = str(getattr(path_info, "addr", "")).lower()
                if "relay" in addr:
                    return ConnType.RELAY
                if "direct" in addr or "socket" in addr or ":" in addr:
                    return ConnType.DIRECT
    except Exception:
        pass

    # --- iroh 0.35 conn_type() API ---
    try:
        info = connection.conn_type()  # type: ignore[attr-defined]
        s = str(info).lower()
        r = repr(info).lower()
        if "direct" in s or "direct" in r:
            return ConnType.DIRECT
        if "relay" in s or "relay" in r or "mixed" in r:
            return ConnType.RELAY
    except Exception:
        pass

    return ConnType.UNKNOWN


def infer_conn_type_from_latency(duration_ms: float) -> ConnType:
    """
    Heuristic: infer connection type from transfer latency when iroh API
    does not expose conn_type directly.

    Thresholds (empirical, European relay ~50-100ms RTT):
      < 5ms  → likely direct (same LAN)
      5-200ms → relay (intercontinental relay adds 40-150ms)
      > 200ms → relay or congested
    """
    if duration_ms < 5.0:
        return ConnType.DIRECT
    return ConnType.RELAY
