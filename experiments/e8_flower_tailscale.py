"""
E8 — Baseline comparison: Flower (flwr) over Tailscale vs. FL-Iroh.

This harness implements the *Architecture A* baseline referenced (but never
implemented) in E2: a standard Flower FedAvg federation. It reuses FL-Iroh's
own model (AgriMLP), data partitions, and the **exact** local-training loop
(SGD, lr, momentum=0.9, weight_decay=1e-4, CrossEntropyLoss, drop_last) so the
comparison isolates the *transport / framework*, not the optimizer.

Two axes are reported:

  1. Quantitative (results/e8/e8_flower_metrics.csv):
       rounds, test_acc_final, payload_bytes_per_round, wall_time_per_round_s
     Directly comparable to E2 Architecture B (FL-Iroh) under the same task.

  2. Qualitative operational comparison (results/e8/e8_comparison.csv + .md):
       NAT/CGNAT handling, infra dependencies, encryption, manual config steps,
       topology, extra system daemon. These are factual design properties of
       each stack, the reviewer-relevant axes for an FGCS transport paper.

Modes
-----
  --mode sim       In-process Flower simulation (controlled convergence + bytes).
                   python -m experiments.e8_flower_tailscale --mode sim \\
                       --dataset crop --partition iid --n-clients 10 --rounds 50

  --mode server    Real Flower server. Bind to the node's Tailscale overlay IP
                   (100.x.y.z) so remote clients reach it across NAT/CGNAT:
                   python -m experiments.e8_flower_tailscale --mode server \\
                       --server-address 0.0.0.0:8080 --rounds 50 --n-clients 3

  --mode client    Real Flower client over Tailscale. Point at the server's
                   Tailscale IP and pass this client's shard index:
                   python -m experiments.e8_flower_tailscale --mode client \\
                       --server-address 100.64.0.1:8080 \\
                       --n-clients 3 --client-id 0

Notes
-----
* Requires ``flwr`` (``pip install flwr``). If absent, the script logs a clear
  message and writes only the qualitative comparison.
* Tailscale must be installed and authenticated on every machine for the real
  modes; ``tailscale ip -4`` gives the overlay IP to bind/connect. The harness
  does not manage Tailscale itself — it documents the steps it requires (which
  is precisely the operational-cost axis we report).
"""
from __future__ import annotations

import argparse
import csv
import logging
import time
from pathlib import Path
from typing import Optional

import yaml

log = logging.getLogger("e8_flower_tailscale")
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)


# --------------------------------------------------------------------------- #
# Seeds / config
# --------------------------------------------------------------------------- #
def _seeds(seeds_file: str = "seeds.yaml") -> dict:
    try:
        with open(seeds_file) as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


# Local-training hyper-parameters — kept identical to fl_coap_iroh/fl/client.py
LOCAL_EPOCHS  = 1
LEARNING_RATE = 0.01
BATCH_SIZE    = 32
MOMENTUM      = 0.9
WEIGHT_DECAY  = 1e-4


# --------------------------------------------------------------------------- #
# Shared model / data helpers (reuse FL-Iroh code for a fair comparison)
# --------------------------------------------------------------------------- #
def _make_model(dataset: str):
    if dataset == "crop":
        from fl_coap_iroh.models.agri_mlp import AgriMLP
        return AgriMLP()
    from fl_coap_iroh.models.cnn import SimpleCNN
    return SimpleCNN()


def _load_partitions(dataset: str, n_clients: int, partition: str, alpha: float, seeds: dict):
    from fl_coap_iroh.data.partition import load_dataset, partition_dataset
    train_ds, test_ds = load_dataset(dataset, "./data")
    parts = partition_dataset(
        train_ds, n_clients, partition, alpha,
        seed=seeds.get("data_partition", 42),
    )
    return parts, test_ds


def _payload_bytes_per_round(model, n_clients: int) -> int:
    """float32 parameters, up + down, summed across clients (wire payload)."""
    per_client = sum(p.numel() * 4 for p in model.parameters())
    return per_client * n_clients * 2


# --------------------------------------------------------------------------- #
# torch <-> flower ndarray plumbing
# --------------------------------------------------------------------------- #
def _get_ndarrays(model):
    import numpy as np  # noqa: F401
    return [v.detach().cpu().numpy() for v in model.state_dict().values()]


def _set_ndarrays(model, ndarrays) -> None:
    import torch
    sd = model.state_dict()
    new_sd = {k: torch.tensor(v) for k, v in zip(sd.keys(), ndarrays)}
    model.load_state_dict(new_sd, strict=True)


def _local_train(model, dataset) -> dict:
    """Identical optimizer/loss/schedule to FL-Iroh's FLClient._train."""
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader

    model.train()
    device = next(model.parameters()).device
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=0,
    )
    optimizer = optim.SGD(
        model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY,
    )
    criterion = nn.CrossEntropyLoss()

    total_loss = correct = total = 0
    for _epoch in range(LOCAL_EPOCHS):
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * bx.size(0)
            correct += (logits.argmax(1) == by).sum().item()
            total += bx.size(0)
    return {"loss": total_loss / max(total, 1), "acc": correct / max(total, 1), "samples": total}


def _evaluate(model, dataset) -> tuple[float, float]:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    model.eval()
    device = next(model.parameters()).device
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0)
    criterion = nn.CrossEntropyLoss()
    total_loss = correct = total = 0
    with torch.no_grad():
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            logits = model(bx)
            loss = criterion(logits, by)
            total_loss += loss.item() * bx.size(0)
            correct += (logits.argmax(1) == by).sum().item()
            total += bx.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)


# --------------------------------------------------------------------------- #
# Flower client
# --------------------------------------------------------------------------- #
def _build_client(dataset: str, train_part, test_ds):
    import flwr as fl

    class FLowerClient(fl.client.NumPyClient):
        def __init__(self):
            self.model = _make_model(dataset)
            self.train_part = train_part
            self.test_ds = test_ds

        def get_parameters(self, config):
            return _get_ndarrays(self.model)

        def fit(self, parameters, config):
            _set_ndarrays(self.model, parameters)
            stats = _local_train(self.model, self.train_part)
            return _get_ndarrays(self.model), stats["samples"], {"train_acc": stats["acc"]}

        def evaluate(self, parameters, config):
            _set_ndarrays(self.model, parameters)
            loss, acc = _evaluate(self.model, self.test_ds)
            return float(loss), len(self.test_ds), {"accuracy": float(acc)}

    return FLowerClient()


def _build_strategy(dataset: str, test_ds, n_clients: int, round_times: list[float]):
    import flwr as fl

    eval_model = _make_model(dataset)

    def evaluate_fn(server_round, parameters, config):
        _set_ndarrays(eval_model, parameters)
        loss, acc = _evaluate(eval_model, test_ds)
        log.info("  [server] round %d  test_acc=%.4f", server_round, acc)
        return float(loss), {"test_acc": float(acc)}

    class TimedFedAvg(fl.server.strategy.FedAvg):
        def configure_fit(self, server_round, parameters, client_manager):
            self._round_t0 = time.perf_counter()
            return super().configure_fit(server_round, parameters, client_manager)

        def aggregate_fit(self, server_round, results, failures):
            out = super().aggregate_fit(server_round, results, failures)
            round_times.append(time.perf_counter() - getattr(self, "_round_t0", time.perf_counter()))
            return out

    return TimedFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=n_clients,
        min_available_clients=n_clients,
        evaluate_fn=evaluate_fn,
    )


# --------------------------------------------------------------------------- #
# Modes
# --------------------------------------------------------------------------- #
def run_sim(args, seeds: dict, results_dir: Path) -> Optional[dict]:
    try:
        import flwr as fl
    except ImportError:
        log.warning("flwr not installed — using in-process FedAvg fallback "
                    "(pip install 'flwr[simulation]' for the real Flower engine)")
        return _run_sim_inprocess(args, seeds, results_dir, framework="fedavg-inprocess")

    import torch
    torch.manual_seed(seeds.get("model_init", 123))

    parts, test_ds = _load_partitions(
        args.dataset, args.n_clients, args.partition, args.alpha, seeds,
    )

    def client_fn(cid: str):
        c = _build_client(args.dataset, parts[int(cid)], test_ds)
        return c.to_client() if hasattr(c, "to_client") else c

    round_times: list[float] = []
    strategy = _build_strategy(args.dataset, test_ds, args.n_clients, round_times)

    start_simulation = getattr(getattr(fl, "simulation", None), "start_simulation", None)
    if start_simulation is None:
        log.warning("flwr %s has no start_simulation (removed in >=1.15) — "
                    "using in-process FedAvg fallback with identical aggregation",
                    getattr(fl, "__version__", "?"))
        return _run_sim_inprocess(args, seeds, results_dir, framework="fedavg-inprocess")

    log.info("=== E8 Flower simulation — %s / %s / %d clients / %d rounds ===",
             args.dataset, args.partition, args.n_clients, args.rounds)
    try:
        history = start_simulation(
            client_fn=client_fn,
            num_clients=args.n_clients,
            config=fl.server.ServerConfig(num_rounds=args.rounds),
            strategy=strategy,
        )
    except Exception as exc:  # noqa: BLE001  (ray/engine issues → fall back)
        log.warning("flwr start_simulation failed (%s: %s) — using in-process "
                    "FedAvg fallback", type(exc).__name__, exc)
        return _run_sim_inprocess(args, seeds, results_dir, framework="fedavg-inprocess")

    final_acc = None
    if history.metrics_centralized.get("test_acc"):
        final_acc = history.metrics_centralized["test_acc"][-1][1]

    model = _make_model(args.dataset)
    row = {
        "framework": "flower",
        "transport": "tailscale_or_grpc",
        "dataset": args.dataset,
        "partition": args.partition,
        "alpha": args.alpha if args.partition != "iid" else "",
        "n_clients": args.n_clients,
        "rounds": args.rounds,
        "test_acc_final": round(final_acc, 4) if final_acc is not None else "",
        "payload_bytes_per_round": _payload_bytes_per_round(model, args.n_clients),
        "wall_time_per_round_s": round(sum(round_times) / max(len(round_times), 1), 4),
    }
    _write_metrics([row], results_dir / "e8_flower_metrics.csv")
    log.info("E8 sim done: test_acc_final=%s  bytes/round=%d  t/round=%.3fs",
             row["test_acc_final"], row["payload_bytes_per_round"], row["wall_time_per_round_s"])
    return row


def _run_sim_inprocess(args, seeds: dict, results_dir: Path, framework: str) -> dict:
    """Framework-independent FedAvg simulation.

    Implements exactly what Flower's ``FedAvg`` strategy does (sample-weighted
    parameter averaging) using FL-Iroh's own model, partitions, and local
    training loop.  Produces the same quantitative row as the Flower engine so
    the E7 accuracy/byte/time comparison is available even when the installed
    ``flwr`` lacks the legacy simulation API (removed in >=1.15) or ``ray`` is
    unavailable.  The numbers are aggregation-identical to Flower FedAvg; only
    the orchestration engine differs (which is irrelevant to model accuracy).
    """
    import numpy as np
    import torch

    torch.manual_seed(seeds.get("model_init", 123))
    parts, test_ds = _load_partitions(
        args.dataset, args.n_clients, args.partition, args.alpha, seeds,
    )

    global_model = _make_model(args.dataset)
    global_params = _get_ndarrays(global_model)

    round_times: list[float] = []
    final_acc = None
    for rnd in range(1, args.rounds + 1):
        t0 = time.perf_counter()
        client_params: list[list] = []
        client_sizes: list[int] = []
        for cid in range(args.n_clients):
            local = _make_model(args.dataset)
            _set_ndarrays(local, global_params)
            stats = _local_train(local, parts[cid])
            client_params.append(_get_ndarrays(local))
            client_sizes.append(max(stats["samples"], 1))

        # Sample-weighted average == Flower FedAvg.aggregate_fit
        total = float(sum(client_sizes))
        global_params = [
            sum(cp[i] * (sz / total) for cp, sz in zip(client_params, client_sizes))
            for i in range(len(global_params))
        ]
        round_times.append(time.perf_counter() - t0)

        _set_ndarrays(global_model, global_params)
        _, acc = _evaluate(global_model, test_ds)
        final_acc = acc
        log.info("  [inprocess-fedavg] round %d/%d  test_acc=%.4f", rnd, args.rounds, acc)

    row = {
        "framework": framework,
        "transport": "in-process (aggregation-identical to Flower FedAvg)",
        "dataset": args.dataset,
        "partition": args.partition,
        "alpha": args.alpha if args.partition != "iid" else "",
        "n_clients": args.n_clients,
        "rounds": args.rounds,
        "test_acc_final": round(final_acc, 4) if final_acc is not None else "",
        "payload_bytes_per_round": _payload_bytes_per_round(global_model, args.n_clients),
        "wall_time_per_round_s": round(sum(round_times) / max(len(round_times), 1), 4),
    }
    _write_metrics([row], results_dir / "e8_flower_metrics.csv")
    log.info("E8 in-process FedAvg done: test_acc_final=%s  bytes/round=%d  t/round=%.3fs",
             row["test_acc_final"], row["payload_bytes_per_round"], row["wall_time_per_round_s"])
    return row



def run_server(args, seeds: dict, results_dir: Path) -> None:
    try:
        import flwr as fl
    except ImportError:
        log.error("flwr not installed — cannot start server (pip install flwr)")
        return

    import torch
    torch.manual_seed(seeds.get("model_init", 123))

    _, test_ds = _load_partitions(
        args.dataset, args.n_clients, args.partition, args.alpha, seeds,
    )
    round_times: list[float] = []
    strategy = _build_strategy(args.dataset, test_ds, args.n_clients, round_times)

    log.info("=== E8 Flower SERVER on %s (bind your Tailscale IP for remote clients) ===",
             args.server_address)
    log.info("    Remote clients connect with: --server-address <this-node tailscale ip>:%s",
             args.server_address.rsplit(":", 1)[-1])
    t0 = time.perf_counter()
    fl.server.start_server(
        server_address=args.server_address,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )
    wall = time.perf_counter() - t0

    model = _make_model(args.dataset)
    row = {
        "framework": "flower",
        "transport": "tailscale",
        "dataset": args.dataset,
        "partition": args.partition,
        "alpha": args.alpha if args.partition != "iid" else "",
        "n_clients": args.n_clients,
        "rounds": args.rounds,
        "test_acc_final": "",  # filled from server log / strategy history if desired
        "payload_bytes_per_round": _payload_bytes_per_round(model, args.n_clients),
        "wall_time_per_round_s": round(sum(round_times) / max(len(round_times), 1), 4),
        "wall_time_total_s": round(wall, 2),
    }
    _write_metrics([row], results_dir / "e8_flower_metrics.csv")
    log.info("E8 server done: %d rounds, mean %.3fs/round, %.1fs total",
             args.rounds, row["wall_time_per_round_s"], wall)


def run_client(args, seeds: dict) -> None:
    try:
        import flwr as fl
    except ImportError:
        log.error("flwr not installed — cannot start client (pip install flwr)")
        return

    import torch
    torch.manual_seed(seeds.get("model_init", 123) + args.client_id)

    parts, test_ds = _load_partitions(
        args.dataset, args.n_clients, args.partition, args.alpha, seeds,
    )
    client = _build_client(args.dataset, parts[args.client_id], test_ds)

    log.info("=== E8 Flower CLIENT %d connecting to %s ===", args.client_id, args.server_address)
    start = getattr(fl.client, "start_client", None)
    if start is not None:
        fl.client.start_client(
            server_address=args.server_address,
            client=client.to_client() if hasattr(client, "to_client") else client,
        )
    else:  # very old flwr
        fl.client.start_numpy_client(server_address=args.server_address, client=client)


# --------------------------------------------------------------------------- #
# Qualitative operational comparison (always written)
# --------------------------------------------------------------------------- #
_COMPARISON_ROWS = [
    {"axis": "NAT / CGNAT traversal",
     "fl_iroh": "Built-in: QUIC hole-punching with transparent DERP relay fallback (E2E encrypted)",
     "flower_tailscale": "Delegated to Tailscale overlay (WireGuard + DERP); Flower itself needs a reachable server address"},
    {"axis": "Server reachability",
     "fl_iroh": "Peer addressed by 32-byte public node ID; no routable server IP required",
     "flower_tailscale": "Clients must know the server's Tailnet IP (100.x.y.z); star topology with a single server"},
    {"axis": "External infrastructure",
     "fl_iroh": "Public/self-hosted DERP relay only (stateless, swappable)",
     "flower_tailscale": "Tailscale coordination server (control plane) + DERP + long-lived FL server process"},
    {"axis": "Extra system daemon",
     "fl_iroh": "None — userspace library, no elevated privileges",
     "flower_tailscale": "tailscaled system daemon on every node (typically root / NET_ADMIN)"},
    {"axis": "Transport encryption",
     "fl_iroh": "End-to-end QUIC/TLS 1.3 (relay sees only ciphertext)",
     "flower_tailscale": "End-to-end WireGuard (relay sees only ciphertext)"},
    {"axis": "Manual setup steps per node",
     "fl_iroh": "Exchange node IDs (out-of-band, one value)",
     "flower_tailscale": "Install tailscaled, authenticate node to tailnet, discover server IP, open/forward port"},
    {"axis": "Topology",
     "fl_iroh": "P2P data plane + CoAP control plane (serverless-capable)",
     "flower_tailscale": "Centralized star; FL server is a single point of failure"},
    {"axis": "Account / ToS dependency",
     "fl_iroh": "None for default relays; fully self-hostable",
     "flower_tailscale": "Tailscale account + ACL policy management (free tier device/user caps)"},
]


def _write_comparison(results_dir: Path) -> None:
    csv_path = results_dir / "e8_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["axis", "fl_iroh", "flower_tailscale"])
        w.writeheader()
        w.writerows(_COMPARISON_ROWS)

    md_path = results_dir / "e8_comparison.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# E8 — FL-Iroh vs. Flower + Tailscale (operational comparison)\n\n")
        f.write("| Axis | FL-Iroh | Flower + Tailscale |\n")
        f.write("|------|---------|--------------------|\n")
        for r in _COMPARISON_ROWS:
            f.write(f"| {r['axis']} | {r['fl_iroh']} | {r['flower_tailscale']} |\n")
    log.info("Wrote operational comparison: %s and %s", csv_path.name, md_path.name)


def _write_metrics(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    fields: list[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser(description="E8 — Flower/Tailscale baseline vs FL-Iroh")
    p.add_argument("--mode", choices=["sim", "server", "client"], default="sim")
    p.add_argument("--dataset", default="crop", choices=["crop", "cifar10"])
    p.add_argument("--partition", default="iid", choices=["iid", "noniid"])
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--n-clients", type=int, default=10)
    p.add_argument("--rounds", type=int, default=50)
    p.add_argument("--client-id", type=int, default=0, help="shard index for --mode client")
    p.add_argument("--server-address", default="0.0.0.0:8080",
                   help="server: bind addr; client: server's Tailscale IP:port")
    p.add_argument("--results-dir", default="results/e8")
    p.add_argument("--seeds-file", default="seeds.yaml")
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    seeds = _seeds(args.seeds_file)

    # The operational comparison is always available, independent of flwr.
    _write_comparison(results_dir)

    if args.mode == "sim":
        run_sim(args, seeds, results_dir)
    elif args.mode == "server":
        run_server(args, seeds, results_dir)
    else:
        run_client(args, seeds)


if __name__ == "__main__":
    main()
