"""
CoAP resource classes for the FL control plane.

Each resource maps to a CoRE Link Format entry under /.well-known/core.
Resources use JSON (ct=50) by default; CBOR (ct=60) when requested via Accept.

Link Format overview:
  </fl/capabilities>;rt="fl.capabilities";ct=50;obs,
  </fl/dataset>;rt="fl.dataset";ct=50,
  </fl/model>;rt="fl.model";ct=50;obs,
  </fl/update>;rt="fl.update";ct=50,
  </fl/metrics>;rt="fl.metrics";ct=50;obs,
  </fl/policy>;rt="fl.policy";ct=50,
  </fl/round>;rt="fl.round";ct=50;obs,
  </iroh/endpoint>;rt="iroh.endpoint";ct=50
"""
from __future__ import annotations

import json
import logging
from typing import Any, Optional

import aiocoap
import aiocoap.resource as resource
from cbor2 import dumps as cbor_dumps, loads as cbor_loads

from fl_coap_iroh.types import (
    DatasetDescriptor,
    IrohEndpoint,
    ModelDescriptor,
    NodeCapabilities,
    RoundMetrics,
    RoundState,
    TrainingPolicy,
    UpdateDescriptor,
)

log = logging.getLogger(__name__)

# CoAP content-format numbers
CT_JSON = 50
CT_CBOR = 60


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------

def _accept_ct(request: aiocoap.Message) -> int:
    """Extract Accept content-format from a CoAP request (default JSON)."""
    for opt in request.opt.option_list():
        if opt.number == aiocoap.numbers.optionnumbers.OptionNumber.ACCEPT:
            return int.from_bytes(opt.value, "big")
    return CT_JSON


def _encode(data: Any, ct: int = CT_JSON) -> tuple[bytes, int]:
    if ct == CT_CBOR:
        return cbor_dumps(data), CT_CBOR
    return json.dumps(data, default=str).encode(), CT_JSON


def _decode(payload: bytes, ct: int = CT_JSON) -> Any:
    if ct == CT_CBOR:
        return cbor_loads(payload)
    return json.loads(payload.decode())


# ---------------------------------------------------------------------------
# /fl/capabilities
# ---------------------------------------------------------------------------

class CapabilitiesResource(resource.ObservableResource):
    """GET/PUT /fl/capabilities — node hardware and software capabilities."""

    def __init__(self, capabilities: NodeCapabilities) -> None:
        super().__init__()
        self._capabilities = capabilities

    async def render_get(self, request: aiocoap.Message) -> aiocoap.Message:
        ct = _accept_ct(request)
        payload, ct_out = _encode(self._capabilities.model_dump(), ct)
        return aiocoap.Message(payload=payload, content_format=ct_out)

    async def render_put(self, request: aiocoap.Message) -> aiocoap.Message:
        try:
            ct = request.opt.content_format or CT_JSON
            data = _decode(request.payload, ct)
            self._capabilities = NodeCapabilities(**data)
            self.updated_state()
            log.info("Capabilities updated for node %s", self._capabilities.node_id)
            return aiocoap.Message(code=aiocoap.CHANGED)
        except Exception as exc:
            log.error("PUT /fl/capabilities failed: %s", exc)
            return aiocoap.Message(code=aiocoap.BAD_REQUEST)

    def update(self, capabilities: NodeCapabilities) -> None:
        self._capabilities = capabilities
        self.updated_state()

    @property
    def current(self) -> NodeCapabilities:
        return self._capabilities


# ---------------------------------------------------------------------------
# /fl/dataset
# ---------------------------------------------------------------------------

class DatasetResource(resource.Resource):
    """GET /fl/dataset — dataset descriptor (metadata only, no raw data)."""

    def __init__(self, descriptor: DatasetDescriptor) -> None:
        super().__init__()
        self._descriptor = descriptor

    async def render_get(self, request: aiocoap.Message) -> aiocoap.Message:
        payload = json.dumps(self._descriptor.model_dump(), default=str).encode()
        return aiocoap.Message(payload=payload, content_format=CT_JSON)

    def update(self, descriptor: DatasetDescriptor) -> None:
        self._descriptor = descriptor


# ---------------------------------------------------------------------------
# /fl/model
# ---------------------------------------------------------------------------

class ModelResource(resource.ObservableResource):
    """GET /fl/model — global model metadata, observable."""

    def __init__(self) -> None:
        super().__init__()
        self._descriptor: Optional[ModelDescriptor] = None

    async def render_get(self, request: aiocoap.Message) -> aiocoap.Message:
        if self._descriptor is None:
            return aiocoap.Message(code=aiocoap.NOT_FOUND, payload=b"No model available yet")
        payload = json.dumps(self._descriptor.model_dump(), default=str).encode()
        return aiocoap.Message(payload=payload, content_format=CT_JSON)

    def update(self, descriptor: ModelDescriptor) -> None:
        self._descriptor = descriptor
        self.updated_state()


# ---------------------------------------------------------------------------
# /fl/update
# ---------------------------------------------------------------------------

class UpdateResource(resource.Resource):
    """GET/PUT /fl/update — update status reported by this node."""

    def __init__(self, node_id: str) -> None:
        super().__init__()
        self._update = UpdateDescriptor(node_id=node_id, round=0, samples_used=0)

    async def render_get(self, request: aiocoap.Message) -> aiocoap.Message:
        payload = json.dumps(self._update.model_dump(), default=str).encode()
        return aiocoap.Message(payload=payload, content_format=CT_JSON)

    async def render_put(self, request: aiocoap.Message) -> aiocoap.Message:
        try:
            data = _decode(request.payload, request.opt.content_format or CT_JSON)
            self._update = UpdateDescriptor(**data)
            return aiocoap.Message(code=aiocoap.CHANGED)
        except Exception as exc:
            log.error("PUT /fl/update failed: %s", exc)
            return aiocoap.Message(code=aiocoap.BAD_REQUEST)

    def update(self, descriptor: UpdateDescriptor) -> None:
        self._update = descriptor


# ---------------------------------------------------------------------------
# /fl/metrics
# ---------------------------------------------------------------------------

class MetricsResource(resource.ObservableResource):
    """GET/POST /fl/metrics — last round training metrics, observable."""

    def __init__(self) -> None:
        super().__init__()
        self._metrics: Optional[RoundMetrics] = None

    async def render_get(self, request: aiocoap.Message) -> aiocoap.Message:
        if self._metrics is None:
            return aiocoap.Message(code=aiocoap.NOT_FOUND, payload=b"No metrics yet")
        payload = json.dumps(self._metrics.model_dump(), default=str).encode()
        return aiocoap.Message(payload=payload, content_format=CT_JSON)

    async def render_post(self, request: aiocoap.Message) -> aiocoap.Message:
        try:
            data = _decode(request.payload, request.opt.content_format or CT_JSON)
            self._metrics = RoundMetrics(**data)
            self.updated_state()
            log.debug("Metrics POST: round=%d acc=%.4f", self._metrics.round, self._metrics.train_acc)
            return aiocoap.Message(code=aiocoap.CHANGED)
        except Exception as exc:
            log.error("POST /fl/metrics failed: %s", exc)
            return aiocoap.Message(code=aiocoap.BAD_REQUEST)

    def update(self, metrics: RoundMetrics) -> None:
        self._metrics = metrics
        self.updated_state()


# ---------------------------------------------------------------------------
# /fl/policy
# ---------------------------------------------------------------------------

class PolicyResource(resource.Resource):
    """GET/PUT /fl/policy — training hyper-parameters pushed by coordinator."""

    def __init__(self, policy: Optional[TrainingPolicy] = None) -> None:
        super().__init__()
        self._policy = policy or TrainingPolicy()

    async def render_get(self, request: aiocoap.Message) -> aiocoap.Message:
        payload = json.dumps(self._policy.model_dump()).encode()
        return aiocoap.Message(payload=payload, content_format=CT_JSON)

    async def render_put(self, request: aiocoap.Message) -> aiocoap.Message:
        try:
            data = _decode(request.payload, request.opt.content_format or CT_JSON)
            self._policy = TrainingPolicy(**data)
            log.info("Policy updated: lr=%.4f epochs=%d", self._policy.learning_rate, self._policy.local_epochs)
            return aiocoap.Message(code=aiocoap.CHANGED)
        except Exception as exc:
            log.error("PUT /fl/policy failed: %s", exc)
            return aiocoap.Message(code=aiocoap.BAD_REQUEST)

    @property
    def policy(self) -> TrainingPolicy:
        return self._policy

    def update(self, policy: TrainingPolicy) -> None:
        self._policy = policy


# ---------------------------------------------------------------------------
# /fl/round
# ---------------------------------------------------------------------------

class RoundResource(resource.ObservableResource):
    """GET/PUT /fl/round — current FL round state, observable by clients."""

    def __init__(self) -> None:
        super().__init__()
        self._state = RoundState(round=0, status="waiting", participants_expected=0)

    async def render_get(self, request: aiocoap.Message) -> aiocoap.Message:
        payload = json.dumps(self._state.model_dump(), default=str).encode()
        return aiocoap.Message(payload=payload, content_format=CT_JSON)

    async def render_put(self, request: aiocoap.Message) -> aiocoap.Message:
        try:
            data = _decode(request.payload, request.opt.content_format or CT_JSON)
            self._state = RoundState(**data)
            self.updated_state()
            return aiocoap.Message(code=aiocoap.CHANGED)
        except Exception as exc:
            log.error("PUT /fl/round failed: %s", exc)
            return aiocoap.Message(code=aiocoap.BAD_REQUEST)

    def update(self, state: RoundState) -> None:
        self._state = state
        self.updated_state()

    @property
    def state(self) -> RoundState:
        return self._state


# ---------------------------------------------------------------------------
# /iroh/endpoint
# ---------------------------------------------------------------------------

class IrohEndpointResource(resource.Resource):
    """GET/PUT /iroh/endpoint — Iroh NodeId and direct/relay connection info."""

    def __init__(self, endpoint: Optional[IrohEndpoint] = None) -> None:
        super().__init__()
        self._endpoint = endpoint

    async def render_get(self, request: aiocoap.Message) -> aiocoap.Message:
        if self._endpoint is None:
            return aiocoap.Message(
                code=aiocoap.NOT_FOUND, payload=b"Iroh endpoint not initialised"
            )
        payload = json.dumps(self._endpoint.model_dump(), default=str).encode()
        return aiocoap.Message(payload=payload, content_format=CT_JSON)

    async def render_put(self, request: aiocoap.Message) -> aiocoap.Message:
        try:
            data = _decode(request.payload, request.opt.content_format or CT_JSON)
            self._endpoint = IrohEndpoint(**data)
            log.info("Iroh endpoint updated: %s", self._endpoint.node_id_iroh[:16])
            return aiocoap.Message(code=aiocoap.CHANGED)
        except Exception as exc:
            log.error("PUT /iroh/endpoint failed: %s", exc)
            return aiocoap.Message(code=aiocoap.BAD_REQUEST)

    def update(self, endpoint: IrohEndpoint) -> None:
        self._endpoint = endpoint

    @property
    def endpoint(self) -> Optional[IrohEndpoint]:
        return self._endpoint


# ---------------------------------------------------------------------------
# /fl/register
# ---------------------------------------------------------------------------

class ClientRegistrationResource(resource.Resource):
    """POST /fl/register — clients self-register their Iroh endpoint with the server."""

    def __init__(self) -> None:
        super().__init__()
        self.on_register = None  # Optional[Callable[[str, IrohEndpoint], None]]

    async def render_post(self, request: aiocoap.Message) -> aiocoap.Message:
        try:
            ct = request.opt.content_format or CT_JSON
            data = _decode(request.payload, ct)
            client_id = data.get("node_id") or data.get("client_id", "unknown")
            ep = IrohEndpoint(**data["iroh_endpoint"])
            if self.on_register is not None:
                self.on_register(client_id, ep)
            log.info("Client registered: %s  iroh=%s", client_id, ep.node_id_iroh[:16])
            return aiocoap.Message(code=aiocoap.CREATED)
        except Exception as exc:
            log.error("POST /fl/register failed: %s", exc)
            return aiocoap.Message(code=aiocoap.BAD_REQUEST)
