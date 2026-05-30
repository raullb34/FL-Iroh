"""
Command-line interface for fl_coap_iroh.

Commands:
  fl-server    — start FL server / aggregator
  fl-node      — start FL client node
  fl-bench     — run E1 communication microbenchmark
  fl-discover  — perform CoAP discovery on a list of hosts
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import click
import yaml

log = logging.getLogger("fl_coap_iroh")


def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
        level=getattr(logging, level.upper(), logging.INFO),
    )


def _load_scenario(scenario_file: str) -> dict:
    p = Path(scenario_file)
    if not p.exists():
        return {}
    with p.open() as f:
        return yaml.safe_load(f) or {}


# ---------------------------------------------------------------------------
# fl-server
# ---------------------------------------------------------------------------

@click.command("fl-server")
@click.option("--node-id",        default=lambda: os.environ.get("FL_NODE_ID", "server"))
@click.option("--coap-port",      default=lambda: int(os.environ.get("FL_COAP_PORT", 5683)), type=int)
@click.option("--rounds",         default=lambda: int(os.environ.get("FL_ROUNDS", 50)),      type=int)
@click.option("--dataset",        default=lambda: os.environ.get("FL_DATASET", "cifar10"))
@click.option("--partition",      default=lambda: os.environ.get("FL_PARTITION", "iid"))
@click.option("--alpha",          default=lambda: float(os.environ.get("FL_ALPHA", 0.5)),    type=float)
@click.option("--min-clients",    default=2, type=int)
@click.option("--local-epochs",   default=1, type=int)
@click.option("--lr",             default=0.01, type=float)
@click.option("--relay-url",      default=lambda: os.environ.get("FL_RELAY_URL", ""),        type=str)
@click.option("--scenario",       default=lambda: os.environ.get("FL_SCENARIO", "net_lan"))
@click.option("--architecture",   default=lambda: os.environ.get("FL_ARCHITECTURE", "B"))
@click.option("--seed",           default=lambda: int(os.environ.get("FL_SEED", 42)),         type=int)
@click.option("--data-dir",       default="./data")
@click.option("--results-dir",    default="./results")
@click.option("--log-level",      default="INFO")
def server_cli(**kw: object) -> None:
    """Start an FL server / aggregator node."""
    _setup_logging(str(kw["log_level"]))
    asyncio.run(_run_server(**kw))


async def _run_server(
    node_id, coap_port, rounds, dataset, partition, alpha,
    min_clients, local_epochs, lr, relay_url, scenario, architecture,
    seed, data_dir, results_dir, log_level,
) -> None:
    import torch
    from fl_coap_iroh.data.partition import load_dataset
    from fl_coap_iroh.fl.server import FLServer
    from fl_coap_iroh.models.cnn import SimpleCNN
    from fl_coap_iroh.types import (
        AvailabilityInfo, ComputeCapabilities, EnergyState,
        NodeCapabilities, NodeRole, NodeStatus, TrainingPolicy,
    )

    torch.manual_seed(seed)

    _, test_ds = load_dataset(dataset, data_dir)
    model = SimpleCNN() if dataset == "cifar10" else SimpleCNN(in_channels=1)

    caps = NodeCapabilities(
        node_id      = node_id,
        role         = NodeRole.AGGREGATOR,
        compute      = ComputeCapabilities(cpu_cores=os.cpu_count() or 2),
        availability = AvailabilityInfo(status=NodeStatus.READY),
    )
    policy = TrainingPolicy(
        min_clients  = min_clients,
        local_epochs = local_epochs,
        learning_rate= lr,
        max_rounds   = rounds,
    )

    server = FLServer(
        node_id      = node_id,
        model        = model,
        test_dataset = test_ds,
        capabilities = caps,
        policy       = policy,
        coap_port    = coap_port,
        relay_url    = relay_url or None,
        scenario     = scenario,
        architecture = architecture,
    )
    server.metrics = __import__(
        "fl_coap_iroh.metrics.collector", fromlist=["MetricsCollector"]
    ).MetricsCollector(node_id, scenario, architecture, results_dir)

    server_ep = await server.start()
    log.info("Server endpoint: %s", server_ep.model_dump())

    # Write endpoint to file so client containers can discover it
    ep_file = Path(results_dir) / "server_endpoint.json"
    ep_file.parent.mkdir(parents=True, exist_ok=True)
    ep_file.write_text(json.dumps(server_ep.model_dump(), indent=2))

    # Wait until at least min_clients have registered (up to 120 s)
    min_needed = server.policy.min_clients
    deadline = asyncio.get_event_loop().time() + 120
    log.info("Waiting for %d clients to register (timeout 120 s)…", min_needed)
    while len(server._clients) < min_needed:
        if asyncio.get_event_loop().time() > deadline:
            log.warning("Timeout waiting for clients — proceeding with %d", len(server._clients))
            break
        await asyncio.sleep(1)
    log.info("%d client(s) registered — starting rounds", len(server._clients))

    await server.run_rounds(n_rounds=rounds)
    await server.stop()

    exported = server.metrics.export_csv()
    log.info("Results: %s", {k: str(v) for k, v in exported.items()})
    summary = server.metrics.summary()
    log.info("Summary: %s", json.dumps(summary, indent=2))


# ---------------------------------------------------------------------------
# fl-node (client)
# ---------------------------------------------------------------------------

@click.command("fl-node")
@click.option("--node-id",         default=lambda: os.environ.get("FL_NODE_ID", "client-01"))
@click.option("--coap-port",       default=lambda: int(os.environ.get("FL_COAP_PORT", 5684)), type=int)
@click.option("--server-host",     default=lambda: os.environ.get("FL_SERVER_HOST", "127.0.0.1"))
@click.option("--server-coap-port",default=lambda: int(os.environ.get("FL_SERVER_COAP_PORT", 5683)), type=int)
@click.option("--partition-idx",   default=lambda: int(os.environ.get("FL_PARTITION_IDX", 0)), type=int)
@click.option("--n-clients",       default=10, type=int)
@click.option("--dataset",         default=lambda: os.environ.get("FL_DATASET", "cifar10"))
@click.option("--partition",       default=lambda: os.environ.get("FL_PARTITION", "iid"))
@click.option("--alpha",           default=lambda: float(os.environ.get("FL_ALPHA", 0.5)), type=float)
@click.option("--rounds",          default=50, type=int)
@click.option("--relay-url",       default=lambda: os.environ.get("FL_RELAY_URL", ""), type=str)
@click.option("--scenario",        default=lambda: os.environ.get("FL_SCENARIO", "net_lan"))
@click.option("--architecture",    default=lambda: os.environ.get("FL_ARCHITECTURE", "B"))
@click.option("--seed",            default=lambda: int(os.environ.get("FL_SEED", 42)), type=int)
@click.option("--data-dir",        default="./data")
@click.option("--results-dir",     default="./results")
@click.option("--log-level",       default="INFO")
def node_cli(**kw: object) -> None:
    """Start an FL client node."""
    _setup_logging(str(kw["log_level"]))
    asyncio.run(_run_client(**kw))


async def _run_client(
    node_id, coap_port, server_host, server_coap_port, partition_idx,
    n_clients, dataset, partition, alpha, rounds, relay_url, scenario,
    architecture, seed, data_dir, results_dir, log_level,
) -> None:
    import torch
    from fl_coap_iroh.coap.client import FLCoapClient
    from fl_coap_iroh.data.partition import load_dataset, partition_dataset
    from fl_coap_iroh.fl.client import FLClient
    from fl_coap_iroh.models.cnn import SimpleCNN
    from fl_coap_iroh.types import (
        AvailabilityInfo, ComputeCapabilities, DatasetDescriptor,
        NodeCapabilities, NodeRole, NodeStatus,
    )

    torch.manual_seed(seed + partition_idx)

    train_ds, test_ds = load_dataset(dataset, data_dir)
    partitions = partition_dataset(train_ds, n_clients, partition, alpha, seed)
    my_partition = partitions[partition_idx % len(partitions)]

    model = SimpleCNN() if dataset == "cifar10" else SimpleCNN(in_channels=1)

    caps = NodeCapabilities(
        node_id      = node_id,
        role         = NodeRole.CLIENT,
        compute      = ComputeCapabilities(cpu_cores=os.cpu_count() or 1),
        availability = AvailabilityInfo(status=NodeStatus.READY),
    )
    ds_desc = DatasetDescriptor(
        dataset_id   = f"{node_id}-{dataset}-{partition_idx}",
        dataset_name = dataset,
        samples      = len(my_partition),
        classes      = list(range(10)),
        iid          = (partition == "iid"),
        distribution = "iid" if partition == "iid" else f"dirichlet-alpha-{alpha}",
        feature_dim  = [32, 32, 3] if dataset == "cifar10" else [28, 28, 1],
    )

    client = FLClient(
        node_id            = node_id,
        model              = model,
        train_dataset      = my_partition,
        val_dataset        = test_ds,
        capabilities       = caps,
        dataset_descriptor = ds_desc,
        coap_port          = coap_port,
        relay_url          = relay_url or None,
        scenario           = scenario,
        architecture       = architecture,
    )
    client.metrics = __import__(
        "fl_coap_iroh.metrics.collector", fromlist=["MetricsCollector"]
    ).MetricsCollector(node_id, scenario, architecture, results_dir)

    await client.start()

    # Discover server endpoint via CoAP, then register this client with server
    log.info("Discovering server endpoint at %s:%d…", server_host, server_coap_port)
    async with FLCoapClient(server_host, server_coap_port) as coap_cl:
        server_ep = await coap_cl.get_iroh_endpoint()
        client.set_server_endpoint(server_ep)
        log.info("Server Iroh endpoint: %s", server_ep.node_id_iroh[:16])
        # Register this client's iroh endpoint with the server
        client_ep = client.iroh_endpoint
        if client_ep is not None:
            await coap_cl.register_with_server(node_id, client_ep)
            log.info("Registered with server as %s", node_id)

    for r in range(1, rounds + 1):
        try:
            await client.run_round(r)
        except Exception as exc:
            log.error("Round %d failed: %s", r, exc)

    await client.stop()
    exported = client.metrics.export_csv()
    log.info("Results: %s", {k: str(v) for k, v in exported.items()})


# ---------------------------------------------------------------------------
# fl-bench  (E1 microbenchmark)
# ---------------------------------------------------------------------------

@click.command("fl-bench")
@click.option("--peer-host",    required=True, help="Peer IP / hostname")
@click.option("--peer-iroh-id", required=True, help="Peer Iroh NodeId string")
@click.option("--peer-relay",   default="",    help="Peer relay URL (optional)")
@click.option("--sizes",        default="100000,1000000,10000000", help="Payload sizes (bytes, comma-sep)")
@click.option("--n-iter",       default=30,    type=int)
@click.option("--scenario",     default="net_lan")
@click.option("--results-dir",  default="./results")
@click.option("--log-level",    default="INFO")
def benchmark_cli(**kw: object) -> None:
    """E1: raw Iroh transport microbenchmark (no FL)."""
    _setup_logging(str(kw["log_level"]))
    asyncio.run(_run_bench(**kw))


async def _run_bench(peer_host, peer_iroh_id, peer_relay, sizes, n_iter, scenario, results_dir, log_level) -> None:
    from fl_coap_iroh.transport.iroh_node import IrohTransportNode, ALPN_FL_MODEL
    from fl_coap_iroh.types import IrohEndpoint
    import os, random

    peer_ep = IrohEndpoint(
        node_id_iroh   = peer_iroh_id,
        addrs          = [f"{peer_host}:11204"],
        relay_url      = peer_relay or None,
        direct_capable = True,
    )

    node = IrohTransportNode("bench-sender")
    await node.start()

    payload_sizes = [int(s.strip()) for s in sizes.split(",")]
    rows = []
    for sz in payload_sizes:
        for i in range(n_iter):
            payload = os.urandom(sz)
            t0 = __import__("time").monotonic()
            try:
                import io, torch
                buf = io.BytesIO(payload)
                fake_tensor = {"data": torch.frombuffer(bytearray(payload), dtype=torch.uint8)}
                stats = await node.send_tensors(peer_ep, fake_tensor, round_num=i, alpn=ALPN_FL_MODEL)
                rows.append({
                    "scenario": scenario, "payload_bytes": sz, "iter": i,
                    "conn_type": stats.conn_type.value,
                    "conn_time_ms": stats.conn_time_ms,
                    "throughput_mbps": stats.throughput_mbps,
                    "duration_ms": stats.transfer_duration_ms,
                })
            except Exception as exc:
                log.warning("iter %d sz %d failed: %s", i, sz, exc)
                rows.append({"scenario": scenario, "payload_bytes": sz, "iter": i,
                             "conn_type": "failed", "conn_time_ms": None,
                             "throughput_mbps": None, "duration_ms": None})

    await node.stop()

    from fl_coap_iroh.metrics.collector import _write_csv
    from pathlib import Path
    out = Path(results_dir) / f"e1_bench_{scenario}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        _write_csv(out, rows)
        log.info("Benchmark results: %s", out)


# ---------------------------------------------------------------------------
# fl-discover
# ---------------------------------------------------------------------------

@click.command("fl-discover")
@click.argument("hosts", nargs=-1)
@click.option("--port",          default=5683, type=int)
@click.option("--min-energy",    default=20.0, type=float)
@click.option("--required-role", default=None)
@click.option("--log-level",     default="INFO")
def discover_cli(hosts, port, min_energy, required_role, log_level) -> None:
    """Query CoAP /fl/capabilities on a list of hosts and report capable nodes."""
    _setup_logging(log_level)
    asyncio.run(_run_discover(list(hosts), port, min_energy, required_role))


async def _run_discover(hosts, port, min_energy, required_role) -> None:
    from fl_coap_iroh.coap.client import FLCoapClient
    results, ms = await FLCoapClient.discover_capable_nodes(
        hosts, port=port, min_energy_pct=min_energy, required_role=required_role
    )
    log.info("Discovery completed in %.1f ms", ms)
    for host, caps, ep in results:
        log.info(
            "  %s  role=%s  energy=%.0f%%  iroh=%s",
            host, caps.role.value, caps.energy.level_pct, ep.node_id_iroh[:16],
        )
