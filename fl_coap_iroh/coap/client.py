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
