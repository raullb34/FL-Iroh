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
) -> dict:
    """Run Architecture B (our system) in a single process."""
    import random
    import torch

    torch.manual_seed(seeds.get("model_init", 123))

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

    scenario = f"{'iid' if partition == 'iid' else f'noniid_{alpha}'}"
    label    = f"B_{scenario}"
    log.info("=== Architecture B — %s ===", label)

    train_ds, test_ds = load_dataset(dataset, "./data")
    partitions = partition_dataset(
        train_ds, n_clients, partition, alpha,
        seed=seeds.get("data_partition", 42),
    )

    def _make_model():
        if dataset == "crop":
            return AgriMLP()
        return SimpleCNN()

    server_model = _make_model()
    server_caps  = NodeCapabilities(
        node_id   = "server",
        role      = NodeRole.AGGREGATOR,
        compute   = ComputeCapabilities(cpu_cores=4),
        availability = AvailabilityInfo(status=NodeStatus.READY),
    )
    policy = TrainingPolicy(
        # never require more clients than exist (K=1 = centralized upper bound)
        min_clients=min(n_clients, max(2, n_clients // 2)), local_epochs=1,
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
        model = _make_model()
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
            classes      = list(range(22 if dataset == "crop" else 10)),
            iid          = (partition == "iid"),
            distribution = partition,
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
        await client.start()
        ep = client.iroh_endpoint        # FLClient.start() returns None; endpoint stored internally
        client.set_server_endpoint(server_ep)
        clients.append(client)
        server.register_client(f"client-{i}", ep)

    # Run FL — server and clients must run concurrently:
    # server sends model → clients receive, train, send update → server aggregates
    async def _run_client(client: FLClient, n_rounds: int) -> None:
        log.info("[%s] client task started (mock=%s)", client.node_id, client._transport.mock_mode)
        for r in range(1, n_rounds + 1):
            try:
                await client.run_round(r)
            except Exception as exc:
                log.error("[%s] round %d error: %s", client.node_id, r, exc)

    await asyncio.gather(
        server.run_rounds(n_rounds=rounds),
        *[_run_client(c, rounds) for c in clients],
        return_exceptions=True,
    )
    for c in clients:
        await c.stop()
    await server.stop()

    server.metrics.export_csv(tag=f"e2_{label}")
    summary = server.metrics.summary()
    log.info("Architecture B %s done. Summary: %s", label, summary)
    return {
        "config": label, "partition": partition, "alpha": alpha,
        "dataset": dataset, **(summary if isinstance(summary, dict) else {}),
    }


async def main_async(args: argparse.Namespace) -> None:
    from experiments._replication import derive_seeds, load_replicate_seeds

    base_seeds = _seed()

    if args.noniid:
        # Legacy flag: run IID + all non-IID variants in sequence
        configs = [("iid", 0.5)] + [("dirichlet", alpha) for alpha in ALPHA_VALUES]
    else:
        configs = [(args.partition, args.alpha)]

    async def _sweep(results_dir: Path, seeds: dict) -> list[dict]:
        results_dir.mkdir(parents=True, exist_ok=True)
        torch.manual_seed(seeds.get("experiment_e2", 101))
        rows: list[dict] = []
        for partition, alpha in configs:
            row = await run_arch_b(
                partition=partition, alpha=alpha, n_clients=args.n_clients,
                rounds=args.rounds, dataset=args.dataset,
                results_dir=results_dir, seeds=seeds,
            )
            if isinstance(row, dict):
                rows.append(row)
        return rows

    def _write_summary(rows: list[dict], path: Path) -> None:
        if not rows:
            return
        import csv
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        log.info("E2 summary saved: %s", path)

    # Resolve replication master seeds (F2)
    if args.seeds:
        masters = list(args.seeds)
    elif args.all_seeds:
        masters = load_replicate_seeds()
        if not masters:
            log.warning("No replicate_seeds in seeds.yaml — falling back to single run")
    else:
        masters = []

    if not masters:
        rows = await _sweep(Path(args.results_dir), base_seeds)
        _write_summary(rows, Path(args.results_dir) / "e2_summary.csv")
        log.info("E2 complete. Results in: %s", args.results_dir)
        return

    log.info("E2 replication over %d master seeds: %s", len(masters), masters)
    seeds_root = Path(args.results_dir) / "seeds"
    for master in masters:
        derived = derive_seeds(base_seeds, master)
        seed_dir = seeds_root / f"seed{master}"
        log.info("=== E2 replicate master=%d (dir=%s) ===", master, seed_dir)
        rows = await _sweep(seed_dir, derived)
        for r in rows:
            r["seed"] = master
        _write_summary(rows, seeds_root / f"e2_summary_seed{master}.csv")
    log.info(
        "E2 replication complete (%d seeds). Aggregate with:\n"
        "  python scripts/aggregate_ci.py --glob '%s/e2_summary_seed*.csv' "
        "--group config --metric test_acc_final",
        len(masters), seeds_root,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="E2: Centralized FL convergence")
    parser.add_argument("--rounds",      type=int,  default=50)
    parser.add_argument("--n-clients",   type=int,  default=10)
    parser.add_argument("--dataset",     default="cifar10")
    parser.add_argument("--partition",   default="iid",
                        choices=["iid", "dirichlet"],
                        help="Data partition: 'iid' or 'dirichlet' (non-IID)")
    parser.add_argument("--alpha",       type=float, default=0.5,
                        help="Dirichlet α (only used with --partition dirichlet)")
    parser.add_argument("--noniid",      action="store_true",
                        help="Legacy: run IID + all non-IID variants sequentially")
    parser.add_argument("--results-dir", default="results/e2")
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=None,
        help="Master seeds for multi-seed replication (F2). Overrides --all-seeds.",
    )
    parser.add_argument(
        "--all-seeds", action="store_true",
        help="Replicate over every seed in seeds.yaml 'replicate_seeds'.",
    )
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
