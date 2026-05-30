"""
CoAP server hosting all FL control plane resources.

Mounts the following resource tree:
  /.well-known/core        — auto-generated CoRE Link Format (aiocoap WKCResource)
  /fl/capabilities         — node HW/SW capabilities (observable)
  /fl/dataset              — dataset descriptor
  /fl/model                — global model metadata (observable)
  /fl/update               — per-node update status
  /fl/metrics              — last round metrics (observable)
  /fl/policy               — training policy / hyper-parameters
  /fl/round                — FL round state (observable)
  /iroh/endpoint           — Iroh NodeId + addresses
"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

import aiocoap
import aiocoap.resource as resource

from fl_coap_iroh.coap.resources import (
    CapabilitiesResource,
    ClientRegistrationResource,
    DatasetResource,
    IrohEndpointResource,
    MetricsResource,
    ModelResource,
    PolicyResource,
    RoundResource,
    UpdateResource,
)
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


class FLCoapServer:
    """
    CoAP server for the FL control plane.

    Usage::
        server = FLCoapServer(node_id="node-01", capabilities=caps,
                              dataset_descriptor=ds_desc)
        await server.start()
        ...
        await server.stop()
    """

    def __init__(
        self,
        node_id: str,
        capabilities: NodeCapabilities,
        dataset_descriptor: DatasetDescriptor,
        coap_host: str = "0.0.0.0",
        coap_port: int = 5683,
    ) -> None:
        self.node_id    = node_id
        self.coap_host  = coap_host
        self.coap_port  = coap_port

        # Instantiate all resources
        self._r_capabilities  = CapabilitiesResource(capabilities)
        self._r_dataset        = DatasetResource(dataset_descriptor)
        self._r_model          = ModelResource()
        self._r_update         = UpdateResource(node_id)
        self._r_metrics        = MetricsResource()
        self._r_policy         = PolicyResource()
        self._r_round          = RoundResource()
        self._r_iroh_endpoint  = IrohEndpointResource()
        self._r_registration   = ClientRegistrationResource()

        self._context: Optional[aiocoap.Context] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Build the resource tree and start listening."""
        root = resource.Site()

        root.add_resource(["fl", "capabilities"],  self._r_capabilities)
        root.add_resource(["fl", "dataset"],        self._r_dataset)
        root.add_resource(["fl", "model"],          self._r_model)
        root.add_resource(["fl", "update"],         self._r_update)
        root.add_resource(["fl", "metrics"],        self._r_metrics)
        root.add_resource(["fl", "policy"],         self._r_policy)
        root.add_resource(["fl", "round"],          self._r_round)
        root.add_resource(["iroh", "endpoint"],     self._r_iroh_endpoint)
        root.add_resource(["fl", "register"],       self._r_registration)

        # Auto-generated /.well-known/core (CoRE Link Format)
        root.add_resource(
            [".well-known", "core"],
            resource.WKCResource(root.get_resources_as_linkheader),
        )

        self._context = await aiocoap.Context.create_server_context(
            root, bind=(self.coap_host, self.coap_port)
        )
        log.info("CoAP server [%s] listening on %s:%d", self.node_id, self.coap_host, self.coap_port)

    async def stop(self) -> None:
        if self._context is not None:
            await self._context.shutdown()
            self._context = None
            log.info("CoAP server [%s] stopped", self.node_id)

    # ------------------------------------------------------------------
    # Resource update helpers (called by FL logic)
    # ------------------------------------------------------------------

    def update_iroh_endpoint(self, ep: IrohEndpoint) -> None:
        self._r_iroh_endpoint.update(ep)

    def update_model_descriptor(self, desc: ModelDescriptor) -> None:
        self._r_model.update(desc)

    def update_metrics(self, metrics: RoundMetrics) -> None:
        self._r_metrics.update(metrics)

    def update_update_status(self, upd: UpdateDescriptor) -> None:
        self._r_update.update(upd)

    def update_round_state(self, state: RoundState) -> None:
        self._r_round.update(state)

    def set_policy(self, policy: TrainingPolicy) -> None:
        self._r_policy.update(policy)

    def update_capabilities(self, caps: NodeCapabilities) -> None:
        self._r_capabilities.update(caps)

    def set_registration_callback(self, callback) -> None:
        """Set callback(client_id, IrohEndpoint) invoked when a client POSTs to /fl/register."""
        self._r_registration.on_register = callback

    # ------------------------------------------------------------------
    # Read-only accessors
    # ------------------------------------------------------------------

    @property
    def policy(self) -> TrainingPolicy:
        return self._r_policy.policy

    @property
    def round_state(self) -> RoundState:
        return self._r_round.state

    @property
    def iroh_endpoint(self) -> Optional[IrohEndpoint]:
        return self._r_iroh_endpoint.endpoint
