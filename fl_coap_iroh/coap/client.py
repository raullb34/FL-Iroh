"""
CoAP client for querying FL control plane resources.

Used by:
  - Coordinator: discovers node capabilities and Iroh endpoints before a round
  - Nodes: query policy, round state, and post metrics back
  - E6 experiment: measures discovery overhead

Provides both single-node GET helpers and multi-node discovery with
semantic filtering via CoAP capability attributes.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Optional

import aiocoap

from fl_coap_iroh.types import (
    DatasetDescriptor,
    IrohEndpoint,
    NodeCapabilities,
    NodeStatus,
    RoundMetrics,
    RoundState,
    TrainingPolicy,
)

log = logging.getLogger(__name__)

# Per-request CoAP timeout (seconds)
REQUEST_TIMEOUT = 5.0


class FLCoapClient:
    """
    Async CoAP client for the FL control plane.

    Usage::
        async with FLCoapClient("192.168.1.10") as client:
            ep = await client.get_iroh_endpoint()
            caps = await client.get_capabilities()
    """

    def __init__(self, host: str, port: int = 5683) -> None:
        self.host = host
        self.port = port
        self._base = f"coap://{host}:{port}"
        self._ctx: Optional[aiocoap.Context] = None

    async def __aenter__(self) -> "FLCoapClient":
        self._ctx = await aiocoap.Context.create_client_context()
        return self

    async def __aexit__(self, *_args: object) -> None:
        if self._ctx is not None:
            await self._ctx.shutdown()
            self._ctx = None

    # ------------------------------------------------------------------
    # Low-level helpers
    # ------------------------------------------------------------------

    async def get(self, path: str) -> dict:
        """GET a CoAP resource and return decoded JSON payload."""
        assert self._ctx is not None, "Use as async context manager"
        uri = f"{self._base}{path}"
        t0 = time.monotonic()
        req = aiocoap.Message(code=aiocoap.GET, uri=uri)
        resp = await asyncio.wait_for(
            self._ctx.request(req).response, timeout=REQUEST_TIMEOUT
        )
        elapsed_ms = (time.monotonic() - t0) * 1000
        if not resp.code.is_successful():
            raise RuntimeError(f"CoAP GET {path} => {resp.code}")
        log.debug("GET %s  %dB  %.1fms", path, len(resp.payload), elapsed_ms)
        return json.loads(resp.payload.decode())

    async def put(self, path: str, data: dict) -> None:
        assert self._ctx is not None
        uri = f"{self._base}{path}"
        payload = json.dumps(data, default=str).encode()
        req = aiocoap.Message(
            code=aiocoap.PUT, uri=uri, payload=payload, content_format=50
        )
        resp = await asyncio.wait_for(
            self._ctx.request(req).response, timeout=REQUEST_TIMEOUT
        )
        if not resp.code.is_successful():
            raise RuntimeError(f"CoAP PUT {path} => {resp.code}")

    async def post(self, path: str, data: dict) -> None:
        assert self._ctx is not None
        uri = f"{self._base}{path}"
        payload = json.dumps(data, default=str).encode()
        req = aiocoap.Message(
            code=aiocoap.POST, uri=uri, payload=payload, content_format=50
        )
        resp = await asyncio.wait_for(
            self._ctx.request(req).response, timeout=REQUEST_TIMEOUT
        )
        if not resp.code.is_successful():
            raise RuntimeError(f"CoAP POST {path} => {resp.code}")

    # ------------------------------------------------------------------
    # Typed resource accessors
    # ------------------------------------------------------------------

    async def get_capabilities(self) -> NodeCapabilities:
        return NodeCapabilities(**await self.get("/fl/capabilities"))

    async def get_dataset_descriptor(self) -> DatasetDescriptor:
        return DatasetDescriptor(**await self.get("/fl/dataset"))

    async def get_iroh_endpoint(self) -> IrohEndpoint:
        return IrohEndpoint(**await self.get("/iroh/endpoint"))

    async def get_round_state(self) -> RoundState:
        return RoundState(**await self.get("/fl/round"))

    async def get_policy(self) -> TrainingPolicy:
        return TrainingPolicy(**await self.get("/fl/policy"))

    async def get_metrics(self) -> RoundMetrics:
        return RoundMetrics(**await self.get("/fl/metrics"))

    async def post_metrics(self, metrics: RoundMetrics) -> None:
        await self.post("/fl/metrics", metrics.model_dump())

    async def register_with_server(self, node_id: str, iroh_endpoint: IrohEndpoint) -> None:
        """POST client's iroh endpoint to server /fl/register."""
        await self.post("/fl/register", {
            "node_id": node_id,
            "iroh_endpoint": iroh_endpoint.model_dump(),
        })

    async def request_raw(self, code, path: str, payload: bytes = b"",
                          query: tuple[str, ...] = (), content_format: Optional[int] = None,
                          accept: Optional[int] = None) -> tuple[aiocoap.Message, int]:
        """Send one request; return (response, bytes on the wire both ways).

        Wire bytes are the encoded CoAP messages (header + options + payload),
        summed over every block when block-wise transfer is used."""
        assert self._ctx is not None
        req = aiocoap.Message(code=code, uri=f"{self._base}{path}", payload=payload)
        if query:
            req.opt.uri_query = query
        if content_format is not None:
            req.opt.content_format = content_format
        if accept is not None:
            req.opt.accept = accept
        resp = await asyncio.wait_for(self._ctx.request(req).response, timeout=REQUEST_TIMEOUT * 6)
        if not resp.code.is_successful():
            raise RuntimeError(f"CoAP {code} {path} => {resp.code}")
        req_bytes = len(req.encode()) if hasattr(req, "encode") else len(payload)
        n_blocks = max(1, -(-len(resp.payload) // 1024)) if len(resp.payload) > 1024 else 1
        # Each block response repeats the header/options (~ len(encode()) - payload)
        overhead = _encoded_len(resp) - len(resp.payload)
        resp_bytes = len(resp.payload) + overhead * n_blocks
        return resp, req_bytes * n_blocks + resp_bytes

    async def rd_register(self, entry: dict) -> int:
        """POST /rd — register this node in the aggregator's Resource Directory."""
        _, nbytes = await self.request_raw(
            aiocoap.POST, "/rd", json.dumps(entry, default=str).encode(), content_format=50)
        return nbytes

    async def rd_lookup(self, tags: list[str] = (), role: Optional[str] = None,
                        min_energy: float = 0.0, count: Optional[int] = None,
                        page: int = 0, cbor: bool = True) -> tuple[list[dict], int]:
        """GET /rd-lookup/ep with server-side filtering; returns (hits, wire bytes)."""
        q = [f"min_energy={min_energy}"]
        if tags:
            q.append("tag=" + ",".join(tags))
        if role:
            q.append(f"role={role}")
        if count is not None:
            q += [f"count={count}", f"page={page}"]
        resp, nbytes = await self.request_raw(
            aiocoap.GET, "/rd-lookup/ep", query=tuple(q), accept=60 if cbor else 50)
        if resp.opt.content_format == 60:
            import cbor2
            return cbor2.loads(resp.payload), nbytes
        return json.loads(resp.payload.decode()), nbytes

    async def get_core_link_format(self) -> tuple[str, int, float]:
        """
        GET /.well-known/core.

        Returns:
            (link_format_string, bytes_count, duration_ms)
        """
        assert self._ctx is not None
        uri = f"{self._base}/.well-known/core"
        t0 = time.monotonic()
        req = aiocoap.Message(code=aiocoap.GET, uri=uri)
        resp = await asyncio.wait_for(
            self._ctx.request(req).response, timeout=REQUEST_TIMEOUT
        )
        elapsed_ms = (time.monotonic() - t0) * 1000
        if not resp.code.is_successful():
            raise RuntimeError(f"CoRE discovery failed: {resp.code}")
        raw = resp.payload.decode()
        log.debug("/.well-known/core  %dB  %.1fms", len(resp.payload), elapsed_ms)
        return raw, len(resp.payload), elapsed_ms

    # ------------------------------------------------------------------
    # Semantic multi-node discovery  (used by coordinator before each round)
    # ------------------------------------------------------------------

    @staticmethod
    async def discover_capable_nodes(
        hosts: list[str],
        port: int = 5683,
        required_role: Optional[str] = None,
        min_energy_pct: float = 20.0,
        max_concurrent: int = 20,
    ) -> tuple[list[tuple[str, NodeCapabilities, IrohEndpoint]], float]:
        """
        Probe a list of hosts concurrently via CoAP.

        Applies semantic filtering:
          - Skip nodes with energy below *min_energy_pct*
          - Skip nodes whose role != *required_role* (if set)
          - Skip nodes with status == UNAVAILABLE

        Returns:
            (capable_nodes, total_discovery_ms)
            where capable_nodes is [(host, capabilities, iroh_endpoint), ...]
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        t_start = time.monotonic()

        async def probe_one(host: str) -> Optional[tuple[str, NodeCapabilities, IrohEndpoint]]:
            async with semaphore:
                try:
                    async with FLCoapClient(host, port) as c:
                        caps = await c.get_capabilities()
                        # --- semantic filtering ---
                        if required_role and caps.role.value != required_role:
                            return None
                        if caps.energy.level_pct < min_energy_pct:
                            log.debug("Exclude %s: energy %.1f%%", host, caps.energy.level_pct)
                            return None
                        if caps.availability.status == NodeStatus.UNAVAILABLE:
                            return None
                        ep = await c.get_iroh_endpoint()
                        return host, caps, ep
                except Exception as exc:
                    log.debug("Probe %s failed: %s", host, exc)
                    return None

        results_raw = await asyncio.gather(*(probe_one(h) for h in hosts))
        results = [r for r in results_raw if r is not None]
        discovery_ms = (time.monotonic() - t_start) * 1000

        log.info(
            "Discovery: %d/%d nodes capable  %.1fms",
            len(results), len(hosts), discovery_ms,
        )
        return results, discovery_ms


async def probe_nodes(
    targets: list[tuple[str, int]],
    required_tags: list[str] = (),
    min_energy_pct: float = 0.0,
    max_concurrent: int = 20,
) -> tuple[list[tuple[str, int, IrohEndpoint]], float, int]:
    """Client-side discovery by probing every node (no Resource Directory).

    Each target gets GET /fl/capabilities and, if it passes the semantic
    filter, GET /iroh/endpoint.  All probes share one client context.
    Returns (matches, discovery_ms, wire_bytes)."""
    ctx = await aiocoap.Context.create_client_context()
    sem = asyncio.Semaphore(max_concurrent)
    want = set(required_tags)
    total = 0
    t0 = time.monotonic()

    async def one(host: str, port: int):
        nonlocal total
        async with sem:
            c = FLCoapClient(host, port)
            c._ctx = ctx
            try:
                resp, n = await c.request_raw(aiocoap.GET, "/fl/capabilities")
                total += n
                caps = NodeCapabilities(**json.loads(resp.payload.decode()))
                if not want.issubset(caps.tags) or caps.energy.level_pct < min_energy_pct \
                        or caps.availability.status == NodeStatus.UNAVAILABLE:
                    return None
                resp, n = await c.request_raw(aiocoap.GET, "/iroh/endpoint")
                total += n
                return host, port, IrohEndpoint(**json.loads(resp.payload.decode()))
            except Exception as exc:  # noqa: BLE001
                log.debug("probe %s:%d failed: %s", host, port, exc)
                return None

    try:
        raw = await asyncio.gather(*(one(h, p) for h, p in targets))
    finally:
        await ctx.shutdown()
    return [r for r in raw if r is not None], (time.monotonic() - t0) * 1000, total


def _encoded_len(msg: aiocoap.Message) -> int:
    """Encoded CoAP message length (header + token + options + payload)."""
    try:
        return len(msg.encode())
    except Exception:  # noqa: BLE001 — e.g. unset message id on a received copy
        # option value + 1-byte delta/length header (+1 extended byte, conservatively)
        opts = sum(len(o.encode()) + 2 for o in msg.opt.option_list())
        return 4 + len(msg.token) + opts + (1 + len(msg.payload) if msg.payload else 0)
