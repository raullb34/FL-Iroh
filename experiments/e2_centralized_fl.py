"""
E2 — Centralized Federated Learning convergence.

Compares:
  Architecture A — Flower + gRPC (external baseline, run via subprocess)
  Architecture B — fl_coap_iroh (CoAP + Iroh)

Dataset: CIFAR-10 (IID + Dirichlet non-IID, α = 0.1 / 0.5 / 1.0)
Clients: 10
Rounds:  50
Hardware: CPU-only

For Architecture A, the script generates a minimal Flower config and
calls `flwr run` (if installed).  If Flower is not available, it logs a
warning and skips that column.

Outputs (results/e2/):
  e2_arch_B_iid.csv
  e2_arch_B_noniid_0.5.csv
  e2_arch_B_noniid_0.1.csv
  e2_arch_A_*.csv           (if Flower available)
  e2_convergence.csv        — merged, one row per (arch, partition, round)

Usage::
    python -m experiments.e2_centralized_fl \\
        --rounds 50 --n-clients 10 --dataset cifar10 \\
        --results-dir results/e2
"""
from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path

import torch
import yaml

log = logging.getLogger("e2_centralized_fl")
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)

ALPHA_VALUES = [0.1, 0.5, 1.0]


def _seed(seeds_file: str = "seeds.yaml") -> dict:
    try:
        with open(seeds_file) as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


async def run_arch_b(
    partition   : str,
    alpha       : float,
    n_clients   : int,
    rounds      : int,
    dataset     : str,
    results_dir : Path,
    seeds       : dict,
) -> None:
    """Run Architecture B (our system) in a single process."""
    import random
    import torch

    torch.manual_seed(seeds.get("model_init", 123))

    from fl_coap_iroh.data.partition import load_dataset, partition_dataset
    from fl_coap_iroh.fl.client import FLClient
    from fl_coap_iroh.fl.server import FLServer
    from fl_coap_iroh.metrics.collector import MetricsCollector
    from fl_coap_iroh.models.cnn import SimpleCNN
    from fl_coap_iroh.types import (
        AvailabilityInfo, ComputeCapabilities, DatasetDescriptor,
        NodeCapabilities, NodeRole, NodeStatus, TrainingPolicy,
    )

    scenario = f"{'iid' if partition == 'iid' else f'noniid_{alpha}'}"
    label    = f"B_{scenario}"
    log.info("=== Architecture B — %s ===", label)

    train_ds, test_ds = load_dataset(dataset, "./data")
    partitions = partition_dataset(
        train_ds, n_clients, partition, alpha,
        seed=seeds.get("data_partition", 42),
    )

    server_model = SimpleCNN()
    server_caps  = NodeCapabilities(
        node_id   = "server",
        role      = NodeRole.AGGREGATOR,
        compute   = ComputeCapabilities(cpu_cores=4),
        availability = AvailabilityInfo(status=NodeStatus.READY),
    )
    policy = TrainingPolicy(
        min_clients=max(2, n_clients // 2), local_epochs=1,
        learning_rate=0.01, max_rounds=rounds,
    )

    server = FLServer(
        node_id      = "server",
        model        = server_model,
        test_dataset = test_ds,
        capabilities = server_caps,
        policy       = policy,
        coap_port    = 5683,
        scenario     = scenario,
        architecture = "B",
    )
    server.metrics = MetricsCollector("server", scenario, "B", str(results_dir))

    server_ep = await server.start()

    # Start clients
    clients = []
    for i in range(n_clients):
        model = SimpleCNN()
        torch.manual_seed(seeds.get("model_init", 123) + i)
        caps = NodeCapabilities(
            node_id      = f"client-{i}",
            role         = NodeRole.CLIENT,
            compute      = ComputeCapabilities(cpu_cores=2),
            availability = AvailabilityInfo(status=NodeStatus.READY),
        )
        ds_desc = DatasetDescriptor(
            dataset_id   = f"client-{i}-{dataset}",
            dataset_name = dataset,
            samples      = len(partitions[i]),
            classes      = list(range(10)),
            iid          = (partition == "iid"),
            distribution = partition,
            feature_dim  = [32, 32, 3],
        )
        client = FLClient(
            node_id            = f"client-{i}",
            model              = model,
            train_dataset      = partitions[i],
            val_dataset        = test_ds,
            capabilities       = caps,
            dataset_descriptor = ds_desc,
            coap_port          = 5684 + i,
            scenario           = scenario,
            architecture       = "B",
        )
        client.metrics = MetricsCollector(f"client-{i}", scenario, "B", str(results_dir))
        ep = await client.start()
        client.set_server_endpoint(server_ep)
        clients.append(client)
        server.register_client(f"client-{i}", ep)

    # Run FL
    await server.run_rounds(n_rounds=rounds)

    # Stop all
    for c in clients:
        await c.stop()
    await server.stop()

    server.metrics.export_csv(tag=f"e2_{label}")
    log.info("Architecture B %s done. Summary: %s", label, server.metrics.summary())


async def main_async(args: argparse.Namespace) -> None:
    seeds = _seed()
    torch.manual_seed(seeds.get("experiment_e2", 101))
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    configs = [("iid", 0.5)]
    if args.noniid:
        configs += [("dirichlet", alpha) for alpha in ALPHA_VALUES]

    for partition, alpha in configs:
        await run_arch_b(
            partition   = partition,
            alpha       = alpha,
            n_clients   = args.n_clients,
            rounds      = args.rounds,
            dataset     = args.dataset,
            results_dir = results_dir,
            seeds       = seeds,
        )

    log.info("E2 complete. Results in: %s", results_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="E2: Centralized FL convergence")
    parser.add_argument("--rounds",      type=int,  default=50)
    parser.add_argument("--n-clients",   type=int,  default=10)
    parser.add_argument("--dataset",     default="cifar10")
    parser.add_argument("--noniid",      action="store_true",
                        help="Also run non-IID variants (α = 0.1 / 0.5 / 1.0)")
    parser.add_argument("--results-dir", default="results/e2")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
