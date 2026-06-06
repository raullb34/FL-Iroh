"""
FL client node: local training + Iroh data transfer + CoAP control plane.

One FLClient represents a single federation participant.  Its responsibilities:
  1. Host a CoAP server advertising capabilities, dataset metadata, Iroh endpoint.
  2. Maintain an Iroh transport node (direct + relay-fallback).
  3. On each FL round:
       a) Receive global model via Iroh  (from server / gateway).
       b) Train locally with FedAvg/FedProx.
       c) Send local update (state dict) back via Iroh.
       d) Post round metrics to its own /fl/metrics CoAP resource.
"""
from __future__ import annotations

import logging
import time
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from fl_coap_iroh.coap.server import FLCoapServer
from fl_coap_iroh.metrics.collector import MetricsCollector
from fl_coap_iroh.transport.iroh_node import (
    ALPN_FL_MODEL,
    ALPN_FL_UPDATE,
    IrohTransportNode,
)
from fl_coap_iroh.types import (
    DatasetDescriptor,
    IrohEndpoint,
    NodeCapabilities,
    RoundMetrics,
    TrainingPolicy,
    UpdateDescriptor,
)

log = logging.getLogger(__name__)


class FLClient:
    """
    Federated Learning client.

    Usage::
        client = FLClient(node_id="client-01", model=SimpleCNN(),
                          train_dataset=train_sub, val_dataset=test_ds,
                          capabilities=caps, dataset_descriptor=ds_desc)
        await client.start()
        client.set_server_endpoint(server_iroh_ep)
        for r in range(1, n_rounds + 1):
            metrics = await client.run_round(r)
        await client.stop()
        client.metrics.export_csv()
    """

    def __init__(
        self,
        node_id            : str,
        model              : nn.Module,
        train_dataset      : Dataset,
        capabilities       : NodeCapabilities,
        dataset_descriptor : DatasetDescriptor,
        val_dataset        : Optional[Dataset]    = None,
        coap_port          : int                  = 5683,
        relay_url          : Optional[str]        = None,
        scenario           : str                  = "net_lan",
        architecture       : str                  = "B",
    ) -> None:
        self.node_id       = node_id
        self.model         = model
        self.train_dataset = train_dataset
        self.val_dataset   = val_dataset
        self._policy       = TrainingPolicy()
        self._server_ep    : Optional[IrohEndpoint] = None

        # Infrastructure
        self._coap = FLCoapServer(
            node_id            = node_id,
            capabilities       = capabilities,
            dataset_descriptor = dataset_descriptor,
            coap_port          = coap_port,
        )
        self._transport = IrohTransportNode(node_id=node_id, relay_url=relay_url)
        self.metrics    = MetricsCollector(
            node_id=node_id, scenario=scenario, architecture=architecture
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        ep = await self._transport.start()
        self._coap.update_iroh_endpoint(ep)
        await self._coap.start()
        log.info("FLClient %s started (mock=%s)", self.node_id, self._transport.mock_mode)

    async def stop(self) -> None:
        await self._transport.stop()
        await self._coap.stop()

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def set_server_endpoint(self, ep: IrohEndpoint) -> None:
        """Provide the server's Iroh endpoint (obtained via CoAP discovery)."""
        self._server_ep = ep

    def set_policy(self, policy: TrainingPolicy) -> None:
        self._policy = policy
        self._coap.set_policy(policy)

    @property
    def iroh_endpoint(self) -> Optional[IrohEndpoint]:
        return self._coap.iroh_endpoint

    # ------------------------------------------------------------------
    # FL round
    # ------------------------------------------------------------------

    async def run_round(self, round_num: int) -> RoundMetrics:
        """Execute one FL round and return training metrics."""
        if self._server_ep is None:
            raise RuntimeError("Server endpoint not set — call set_server_endpoint() first")

        t_round = time.monotonic()

        # 1. Receive global model
        # Timeout must exceed server's ROUND_TIMEOUT_SEC (900s) so a desync'd client
        # can wait for the server to finish the current round and send the next model,
        # self-correcting within one round instead of drifting permanently.
        log.info("[%s] Round %d — receiving model…", self.node_id, round_num)
        global_params, recv_stats = await self._transport.receive_tensors(
            ALPN_FL_MODEL, timeout=1800.0
        )
        self.model.load_state_dict(global_params)
        self.metrics.record_transfer(recv_stats, round_num, "recv")

        # 2. Local training
        log.info("[%s] Round %d — training locally…", self.node_id, round_num)
        train_result = self._train(round_num)

        # 3. Send update
        log.info("[%s] Round %d — sending update…", self.node_id, round_num)
        local_params = {k: v.cpu() for k, v in self.model.state_dict().items()}
        send_stats = await self._transport.send_tensors(
            self._server_ep, local_params, round_num, ALPN_FL_UPDATE
        )
        self.metrics.record_transfer(send_stats, round_num, "send")

        # 4. Optional validation
        val_loss = val_acc = None
        if self.val_dataset is not None:
            val_loss, val_acc = self._evaluate()

        duration_sec = time.monotonic() - t_round

        round_metrics = RoundMetrics(
            node_id      = self.node_id,
            round        = round_num,
            train_loss   = train_result["loss"],
            train_acc    = train_result["acc"],
            val_loss     = val_loss,
            val_acc      = val_acc,
            samples_used = train_result["samples"],
            duration_sec = duration_sec,
        )

        # Publish to CoAP
        self._coap.update_metrics(round_metrics)
        self._coap.update_update_status(
            UpdateDescriptor(
                node_id      = self.node_id,
                round        = round_num,
                samples_used = train_result["samples"],
                status       = "sent",
            )
        )
        self.metrics.record_round_metrics(round_metrics)

        log.info(
            "[%s] Round %d done — loss=%.4f acc=%.4f  %.1fs",
            self.node_id, round_num,
            round_metrics.train_loss, round_metrics.train_acc, duration_sec,
        )
        return round_metrics

    # ------------------------------------------------------------------
    # Training & evaluation internals
    # ------------------------------------------------------------------

    def _train(self, round_num: int) -> dict:
        self.model.train()
        device = next(self.model.parameters()).device
        loader = DataLoader(
            self.train_dataset,
            batch_size=self._policy.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=0,
        )
        optimizer = optim.SGD(
            self.model.parameters(),
            lr=self._policy.learning_rate,
            momentum=0.9,
            weight_decay=1e-4,
        )
        criterion = nn.CrossEntropyLoss()

        # FedProx: snapshot global params for proximal term
        global_snapshot: dict[str, torch.Tensor] = {}
        if self._policy.proximal_mu > 0:
            global_snapshot = {
                k: v.clone().detach().to(device)
                for k, v in self.model.state_dict().items()
            }

        total_loss = correct = total = 0
        for _epoch in range(self._policy.local_epochs):
            for bx, by in loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                logits = self.model(bx)
                loss   = criterion(logits, by)

                if self._policy.proximal_mu > 0:
                    prox = sum(
                        ((p - global_snapshot[n].to(device)) ** 2).sum()
                        for n, p in self.model.named_parameters()
                        if n in global_snapshot
                    )
                    loss = loss + (self._policy.proximal_mu / 2) * prox

                loss.backward()
                optimizer.step()

                total_loss += loss.item() * bx.size(0)
                correct    += (logits.argmax(1) == by).sum().item()
                total      += bx.size(0)

        return {
            "loss"   : total_loss / max(total, 1),
            "acc"    : correct    / max(total, 1),
            "samples": total,
        }

    def _evaluate(self) -> tuple[float, float]:
        self.model.eval()
        device    = next(self.model.parameters()).device
        loader    = DataLoader(self.val_dataset, batch_size=256, shuffle=False, num_workers=0)
        criterion = nn.CrossEntropyLoss()
        total_loss = correct = total = 0
        with torch.no_grad():
            for bx, by in loader:
                bx, by = bx.to(device), by.to(device)
                logits  = self.model(bx)
                loss    = criterion(logits, by)
                total_loss += loss.item() * bx.size(0)
                correct    += (logits.argmax(1) == by).sum().item()
                total      += bx.size(0)
        return total_loss / max(total, 1), correct / max(total, 1)
