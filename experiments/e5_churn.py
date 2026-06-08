"""
E5 — Churn resilience.

Runs FL (Architecture B) with simulated client churn at rates:
  0 % (baseline), 10 %, 30 %, 50 %

Churn is modelled by simply withholding a random fraction of clients
from each aggregation round (no Docker pause needed — purely in-process).

Metrics recorded per round:
  - test accuracy
  - participating clients
  - round duration

Outputs (results/e5/):
  e5_churn_{rate}.csv     — round-level metrics per churn rate
  e5_churn_summary.csv    — convergence statistics per churn rate

Usage::
    python -m experiments.e5_churn --rounds 50 --n-clients 10
"""
from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path

import torch
import yaml

log = logging.getLogger("e5_churn")
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)

CHURN_RATES = [0.0, 0.10, 0.30, 0.50]


def _seeds() -> dict:
    try:
        with open("seeds.yaml") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


async def run_churn_experiment(
    churn_rate  : float,
    n_clients   : int,
    rounds      : int,
    dataset     : str,
    results_dir : Path,
    seeds       : dict,
) -> None:
    import random
    from fl_coap_iroh.data.partition import load_dataset, partition_dataset
    from fl_coap_iroh.fl.client import FLClient
    from fl_coap_iroh.fl.server import FLServer
    from fl_coap_iroh.metrics.collector import MetricsCollector
    from fl_coap_iroh.models.cnn import SimpleCNN
    from fl_coap_iroh.models.agri_mlp import AgriMLP
    from fl_coap_iroh.types import (
        AvailabilityInfo, ComputeCapabilities, DatasetDescriptor,
        NodeCapabilities, NodeRole, NodeStatus, TrainingPolicy,
    )

    label = f"churn_{int(churn_rate * 100):02d}pct"
    scenario = f"net_churn{int(churn_rate * 100):02d}"
    log.info("=== E5: %s ===", label)

    rng = random.Random(seeds.get("churn_simulator", 456))
    torch.manual_seed(seeds.get("model_init", 123))

    train_ds, test_ds = load_dataset(dataset, "./data")
    partitions = partition_dataset(
        train_ds, n_clients, "iid", 0.5,
        seed=seeds.get("data_partition", 42),
    )

    def _make_model():
        if dataset == "crop":
            return AgriMLP()
        return SimpleCNN()

    server_model = _make_model()
    server_caps  = NodeCapabilities(
        node_id      = "server",
        role         = NodeRole.AGGREGATOR,
        compute      = ComputeCapabilities(cpu_cores=4),
        availability = AvailabilityInfo(status=NodeStatus.READY),
    )
    policy = TrainingPolicy(
        min_clients=max(2, int(n_clients * 0.4)), local_epochs=1,
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

    clients = []
    for i in range(n_clients):
        torch.manual_seed(seeds.get("model_init", 123) + i)
        model = _make_model()
        caps = NodeCapabilities(
            node_id      = f"client-{i}",
            role         = NodeRole.CLIENT,
            compute      = ComputeCapabilities(cpu_cores=2),
            availability = AvailabilityInfo(status=NodeStatus.READY),
        )
        ds_desc = DatasetDescriptor(
            dataset_id   = f"client-{i}",
            dataset_name = dataset,
            samples      = len(partitions[i]),
            classes      = list(range(22 if dataset == "crop" else 10)),
            iid          = True,
            distribution = "iid",
            feature_dim  = [7] if dataset == "crop" else [32, 32, 3],
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
        client._receive_timeout = 60.0  # mock transport; short timeout avoids 1800s stalls on churned rounds
        await client.start()
        ep = client.iroh_endpoint        # FLClient.start() returns None; endpoint stored internally
        client.set_server_endpoint(server_ep)
        clients.append(client)
        server.register_client(f"client-{i}", ep)

    # Override server client selection to inject churn.
    # _select_clients is stored as an instance attribute (plain function, not bound method),
    # so it is called as churned_select() with no arguments.
    original_select = server._select_clients

    def churned_select():
        selected = original_select()
        if churn_rate <= 0.0:
            return selected
        n_churn = max(0, round(len(selected) * churn_rate))
        to_drop = rng.sample(selected, min(n_churn, len(selected)))
        return [c for c in selected if c not in to_drop]

    server._select_clients = churned_select  # type: ignore[method-assign]

    # Run FL — server and clients must run concurrently
    async def _run_client(client, n_rounds: int) -> None:
        log.info("[%s] client task started (mock=%s)", client.node_id, client._transport.mock_mode)
        for r in range(1, n_rounds + 1):
            try:
                await client.run_round(r)
            except Exception as exc:
                log.error("[%s] round %d error: %s", client.node_id, r, exc)

    server_task = asyncio.create_task(server.run_rounds(n_rounds=rounds))
    client_tasks = [asyncio.create_task(_run_client(c, rounds)) for c in clients]
    await server_task
    for t in client_tasks:
        t.cancel()
    await asyncio.gather(*client_tasks, return_exceptions=True)

    for c in clients:
        await c.stop()
    await server.stop()

    server.metrics.export_csv(tag=f"e5_{label}")
    summary = server.metrics.summary()
    log.info("E5 %s → acc_final=%.4f, acc_max=%.4f",
             label,
             summary.get("test_acc_final") or 0,
             summary.get("test_acc_max")   or 0)


async def main_async(args: argparse.Namespace) -> None:
    seeds = _seeds()
    torch.manual_seed(seeds.get("experiment_e5", 303))
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    rates = [float(r) for r in args.churn_rates.split(",")]
    for rate in rates:
        await run_churn_experiment(
            churn_rate  = rate,
            n_clients   = args.n_clients,
            rounds      = args.rounds,
            dataset     = args.dataset,
            results_dir = results_dir,
            seeds       = seeds,
        )

    log.info("E5 complete. Results in: %s", results_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="E5: Churn resilience")
    parser.add_argument("--rounds",       type=int,  default=50)
    parser.add_argument("--n-clients",    type=int,  default=10)
    parser.add_argument("--dataset",      default="cifar10")
    parser.add_argument("--churn-rates",  default="0.0,0.1,0.3,0.5",
                        help="Comma-separated churn rates")
    parser.add_argument("--results-dir",  default="results/e5")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
