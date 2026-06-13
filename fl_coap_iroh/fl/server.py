"""
FL server / aggregator: orchestrates rounds, distributes global model,
collects and aggregates client updates via Iroh, evaluates global model.

Supported topologies:
  B — Centralised FL over Iroh (all clients connect directly to this server)
  C — Gateway node in hierarchical FL (aggregates a local cluster, then
      forwards to a global aggregator — re-uses FLServer with n_clients = n_gateways)
  D — Hub in hybrid P2P FL (some connections direct, some relay)

The algorithm (FedAvg / FedProx) is kept constant across topologies;
only the communication layer changes.
"""
from __future__ import annotations

import asyncio
import logging
import random
import time
from typing import Callable, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from fl_coap_iroh.coap.server import FLCoapServer
from fl_coap_iroh.fl.fedavg import fedavg_aggregate
from fl_coap_iroh.metrics.collector import MetricsCollector
from fl_coap_iroh.transport.iroh_node import (
    ALPN_FL_MODEL,
    ALPN_FL_UPDATE,
    IrohTransportNode,
)
from fl_coap_iroh.types import (
    DatasetDescriptor,
    IrohEndpoint,
    ModelDescriptor,
    NodeCapabilities,
    RoundEvent,
    RoundState,
    TrainingPolicy,
)

log = logging.getLogger(__name__)

# Seconds to wait for all client updates before timing out a round
ROUND_TIMEOUT_SEC = 900.0  # 15 min — CPU training takes ~6-10 min per round


class FLServer:
    """
    FL server / aggregator node.

    Usage::
        server = FLServer(node_id="server", model=SimpleCNN(),
                          test_dataset=test_ds, capabilities=caps,
                          policy=policy)
        server_ep = await server.start()
        # share server_ep with clients via CoAP or out-of-band

        server.register_client("client-01", client_01_iroh_ep)
        server.register_client("client-02", client_02_iroh_ep)

        results = await server.run_rounds(n_rounds=50)
        await server.stop()
        server.metrics.export_csv()
    """

    def __init__(
        self,
        node_id      : str,
        model        : nn.Module,
        test_dataset : Dataset,
        capabilities : NodeCapabilities,
        policy       : TrainingPolicy,
        coap_port    : int           = 5683,
        relay_url    : Optional[str] = None,
        scenario     : str           = "net_lan",
        architecture : str           = "B",
        aggregator_fn: Optional[Callable] = None,
    ) -> None:
        self.node_id      = node_id
        self.model        = model
        self.test_ds      = test_dataset
        self.policy       = policy
        self._aggregator_fn = aggregator_fn if aggregator_fn is not None else fedavg_aggregate

        # CoAP server (control plane)
        ds_desc = DatasetDescriptor(
            dataset_id   = f"{node_id}-eval",
            dataset_name = "eval",
            samples      = len(test_dataset),
            classes      = list(range(10)),
            iid          = True,
            distribution = "full",
            feature_dim  = [],
        )
        self._coap = FLCoapServer(
            node_id            = node_id,
            capabilities       = capabilities,
            dataset_descriptor = ds_desc,
            coap_port          = coap_port,
        )
        self._coap.set_policy(policy)

        # Iroh transport (data plane)
        self._transport = IrohTransportNode(node_id=node_id, relay_url=relay_url)
        self.metrics    = MetricsCollector(
            node_id=node_id, scenario=scenario, architecture=architecture
        )

        self._clients: dict[str, IrohEndpoint] = {}
        self._round_results: list[dict] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> IrohEndpoint:
        """Start transport + CoAP server. Returns Iroh endpoint for client discovery."""
        ep = await self._transport.start()
        self._coap.update_iroh_endpoint(ep)
        self._coap.set_registration_callback(self.register_client)
        await self._coap.start()
        log.info("FLServer %s started (mock=%s)", self.node_id, self._transport.mock_mode)
        return ep

    async def stop(self) -> None:
        await self._transport.stop()
        await self._coap.stop()

    # ------------------------------------------------------------------
    # Client registration
    # ------------------------------------------------------------------

    def register_client(self, client_id: str, ep: IrohEndpoint) -> None:
        self._clients[client_id] = ep
        log.info("Registered client %s  iroh=%s", client_id, ep.node_id_iroh[:16])

    def unregister_client(self, client_id: str) -> None:
        self._clients.pop(client_id, None)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    async def run_rounds(self, n_rounds: Optional[int] = None) -> list[dict]:
        """Run FL training rounds. Returns list of per-round result dicts."""
        max_rounds = n_rounds or self.policy.max_rounds
        for r in range(1, max_rounds + 1):
            try:
                result = await self._run_round(r)
            except Exception as exc:
                log.error("Round %d failed: %s", r, exc)
                result = {
                    "round": r, "success": False,
                    "test_acc": None, "test_loss": None, "error": str(exc),
                }
            self._round_results.append(result)
            if result["success"]:
                log.info(
                    "Round %d/%d  acc=%.4f  loss=%.4f  %.1fs  clients=%d",
                    r, max_rounds,
                    result["test_acc"], result["test_loss"],
                    result["duration_sec"], result["clients_participated"],
                )
        return self._round_results

    async def _run_round(self, r: int) -> dict:
        t_start = time.monotonic()

        selected = self._select_clients()
        if len(selected) < self.policy.min_clients:
            raise RuntimeError(
                f"Only {len(selected)} clients available, need {self.policy.min_clients}"
            )

        # Update CoAP round state
        self._coap.update_round_state(RoundState(
            round                 = r,
            status                = "training",
            participants_expected = len(selected),
            start_time            = t_start,
        ))

        global_params = {k: v.cpu() for k, v in self.model.state_dict().items()}

        # --- Send global model to all selected clients (concurrent) ---
        log.info("Round %d — sending model to %d clients…", r, len(selected))
        send_coros = [
            self._transport.send_tensors(self._clients[cid], global_params, r, ALPN_FL_MODEL)
            for cid in selected
        ]
        send_results = await asyncio.gather(*send_coros, return_exceptions=True)

        # Log and filter failed sends
        active_clients = []
        for cid, res in zip(selected, send_results):
            if isinstance(res, Exception):
                log.error("send_tensors → %s FAILED: %s: %s", cid, type(res).__name__, res)
            else:
                active_clients.append(cid)
        if len(active_clients) < self.policy.min_clients:
            raise RuntimeError(
                f"Only {len(active_clients)} sends succeeded (need {self.policy.min_clients})"
            )

        # --- Receive updates (concurrent, with per-round timeout) ---
        log.info("Round %d — waiting for %d updates…", r, len(active_clients))
        recv_coros = [
            self._transport.receive_tensors(ALPN_FL_UPDATE, timeout=ROUND_TIMEOUT_SEC)
            for _ in active_clients
        ]
        recv_results = await asyncio.gather(*recv_coros, return_exceptions=True)

        updates: list[tuple[dict, float]] = []
        bytes_recv = 0
        bytes_direct = 0
        bytes_relay  = 0
        for res in recv_results:
            if isinstance(res, Exception):
                log.warning("Update receive failed: %s", res)
                continue
            params, stats = res
            # Use equal weighting (sample counts are not available server-side here;
            # extend by sending sample count in a header for weighted FedAvg)
            updates.append((params, 1.0))
            bytes_recv += stats.bytes_on_wire
            if stats.conn_type.value == "direct":
                bytes_direct += stats.bytes_on_wire
            else:
                bytes_relay  += stats.bytes_on_wire

        if not updates:
            raise RuntimeError("No updates received this round")

        # --- Aggregate ---
        aggregated = self._aggregator_fn(updates)
        self.model.load_state_dict(aggregated)

        # Update CoAP model descriptor
        import hashlib, io
        buf = io.BytesIO()
        torch.save(aggregated, buf)
        model_bytes = buf.getvalue()
        self._coap.update_model_descriptor(ModelDescriptor(
            model_id     = f"round-{r}",
            model_name   = type(self.model).__name__,
            round        = r,
            params_count = sum(p.numel() for p in self.model.parameters()),
            size_bytes   = len(model_bytes),
            sha256       = hashlib.sha256(model_bytes).hexdigest(),
        ))

        # --- Evaluate ---
        test_loss, test_acc = self._evaluate()

        duration_sec = time.monotonic() - t_start

        self._coap.update_round_state(RoundState(
            round                 = r,
            status                = "done",
            participants_expected = len(selected),
            participants_done     = active_clients,
            start_time            = t_start,
            end_time              = time.monotonic(),
        ))

        event = RoundEvent(
            round                = r,
            architecture         = self.metrics.architecture,
            scenario             = self.metrics.scenario,
            n_clients            = len(self._clients),
            clients_participated = len(updates),
            duration_sec         = duration_sec,
            success              = True,
            test_acc             = test_acc,
            test_loss            = test_loss,
            bytes_to_aggregator  = bytes_recv,
            bytes_p2p_direct     = bytes_direct,
            bytes_relay          = bytes_relay,
        )
        self.metrics.record_round_event(event)

        return {
            "round"               : r,
            "success"             : True,
            "test_acc"            : test_acc,
            "test_loss"           : test_loss,
            "duration_sec"        : duration_sec,
            "clients_participated": len(updates),
            "bytes_to_aggregator" : bytes_recv,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _select_clients(self) -> list[str]:
        available = list(self._clients)
        n = max(self.policy.min_clients, int(len(available) * self.policy.fraction_fit))
        return random.sample(available, min(n, len(available)))

    def _evaluate(self) -> tuple[float, float]:
        self.model.eval()
        device    = next(self.model.parameters()).device
        loader    = DataLoader(self.test_ds, batch_size=256, shuffle=False, num_workers=0)
        criterion = nn.CrossEntropyLoss()
        total_loss = correct = total = 0
        with torch.no_grad():
            for bx, by in loader:
                bx, by = bx.to(device), by.to(device)
                logits  = self.model(bx)
                total_loss += criterion(logits, by).item() * bx.size(0)
                correct    += (logits.argmax(1) == by).sum().item()
                total      += bx.size(0)
        return total_loss / max(total, 1), correct / max(total, 1)
