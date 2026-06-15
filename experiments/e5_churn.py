"""
E5 — Churn resilience.

Runs FL (Architecture B) with client churn at rates:
  0 % (baseline), 10 %, 30 %, 50 %

Two churn mechanisms are available (``--churn-mode``):

  select  — (default, fast) a random fraction of clients is withheld from each
            aggregation round by overriding server-side client selection. No
            transport teardown happens; useful for mock-transport dry runs.

  real    — (faithful) the chosen victims are *actually disconnected* at the
            transport layer (``IrohTransportNode.stop()``) mid-run and later
            rejoin (``start()`` + re-registration with a fresh endpoint). The
            server therefore experiences genuine connection failures / receive
            timeouts and recovers through its real fault-tolerance path
            (failed-send filtering, ``min_clients`` quorum, per-round timeout).
            Requires real Iroh (mock mode shares a global registry that
            ``stop()`` would clear for every peer, so it is rejected).

Metrics recorded per round:
  - test accuracy
  - participating clients
  - round duration

Outputs (results/e5/):
  e5_churn_{rate}.csv          — round-level metrics per churn rate
  e5_churn_events_{rate}.csv   — real-mode disconnect/rejoin event log
  e5_churn_summary.csv         — convergence statistics per churn rate

Usage::
    python -m experiments.e5_churn --rounds 50 --n-clients 10
    python -m experiments.e5_churn --churn-mode real --churn-rates 0.0,0.3
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


async def _real_churn_controller(
    clients,
    server,
    server_ep,
    churn_rate  : float,
    rounds      : int,
    n_clients   : int,
    min_clients : int,
    rng,
    events      : list,
    downtime    : int = 1,
    poll_sec    : float = 0.5,
) -> None:
    """
    Drive REAL transport-level churn during the run.

    On each new round the controller:
      * rejoins clients whose downtime has elapsed (real ``start()`` + server
        re-registration with the freshly-minted Iroh endpoint), and
      * disconnects fresh victims (real ``stop()``) with per-client probability
        ``churn_rate``, never letting the online population drop below
        ``min_clients`` (so the round can still reach quorum).

    Every action is appended to ``events`` as a dict and later written to CSV.
    The set of victims is seeded (reproducible); the exact mid-round instant of
    each disconnect is best-effort, faithfully modelling unplanned departures.
    """
    by_id = {c.node_id: c for c in clients}
    # Keep the first `min_clients` clients as a stable core so the federation
    # always has a quorum; only the remainder are eligible to churn.
    eligible = [c.node_id for c in clients[min_clients:]]
    max_offline = max(0, n_clients - min_clients)

    online = set(by_id)
    return_round: dict[str, int] = {}
    last_round = 0

    async def _disconnect(cid: str, cur: int) -> None:
        try:
            await by_id[cid].stop()
        except Exception as exc:
            log.warning("[churn] disconnect %s failed: %s", cid, exc)
        online.discard(cid)
        events.append({"round": cur, "client": cid, "event": "disconnect",
                       "online_after": len(online)})
        log.info("[churn] round %d — %s DISCONNECTED (online=%d)", cur, cid, len(online))

    async def _rejoin(cid: str, cur: int) -> None:
        try:
            await by_id[cid].start()
            server.register_client(cid, by_id[cid].iroh_endpoint)
            by_id[cid].set_server_endpoint(server_ep)
        except Exception as exc:
            log.warning("[churn] rejoin %s failed: %s", cid, exc)
            return
        online.add(cid)
        events.append({"round": cur, "client": cid, "event": "rejoin",
                       "online_after": len(online)})
        log.info("[churn] round %d — %s REJOINED (online=%d)", cur, cid, len(online))

    try:
        while True:
            cur = len(server._round_results) + 1
            if cur != last_round and cur <= rounds:
                last_round = cur
                # 1) rejoin those whose downtime elapsed
                for cid in [c for c, rr in return_round.items() if rr <= cur]:
                    await _rejoin(cid, cur)
                    return_round.pop(cid, None)
                # 2) disconnect new victims (respecting the offline cap)
                for cid in eligible:
                    if cid not in online:
                        continue
                    if (n_clients - len(online)) >= max_offline:
                        break
                    if rng.random() < churn_rate:
                        await _disconnect(cid, cur)
                        return_round[cid] = cur + downtime
            await asyncio.sleep(poll_sec)
    except asyncio.CancelledError:
        return


async def run_churn_experiment(
    churn_rate  : float,
    n_clients   : int,
    rounds      : int,
    dataset     : str,
    results_dir : Path,
    seeds       : dict,
    churn_mode  : str = "select",
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

    # In "real" mode the transport is genuinely torn down/rejoined by a churn
    # controller, so we must NOT mask selection — the server has to face real
    # connection failures. In "select" mode we override selection as before.
    churn_events: list[dict] = []
    if churn_mode == "real" and churn_rate > 0.0:
        if any(getattr(c._transport, "mock_mode", False) for c in clients):
            raise RuntimeError(
                "--churn-mode real requires real Iroh transport. The mock "
                "transport shares a global registry that stop() clears for "
                "every peer. Run without FL_MOCK_IROH (Docker/HPC)."
            )
        log.info("E5 real churn: transport-level disconnect/rejoin enabled")
    else:
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

    churn_task = None
    if churn_mode == "real" and churn_rate > 0.0:
        churn_task = asyncio.create_task(_real_churn_controller(
            clients=clients, server=server, server_ep=server_ep,
            churn_rate=churn_rate, rounds=rounds, n_clients=n_clients,
            min_clients=policy.min_clients, rng=rng, events=churn_events,
        ))

    await server_task
    for t in client_tasks:
        t.cancel()
    if churn_task is not None:
        churn_task.cancel()
    await asyncio.gather(*client_tasks, return_exceptions=True)
    if churn_task is not None:
        await asyncio.gather(churn_task, return_exceptions=True)

    for c in clients:
        try:
            await c.stop()
        except Exception as exc:
            log.debug("client stop during teardown: %s", exc)
    await server.stop()

    server.metrics.export_csv(tag=f"e5_{label}")

    # Persist real-churn event log
    if churn_events:
        import csv
        ev_path = results_dir / f"e5_churn_events_{label}.csv"
        with open(ev_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["round", "client", "event", "online_after"])
            w.writeheader()
            w.writerows(churn_events)
        log.info("E5 churn events saved: %s (%d events)", ev_path, len(churn_events))

    summary = server.metrics.summary()
    log.info("E5 %s (mode=%s) → acc_final=%.4f, acc_max=%.4f",
             label, churn_mode,
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
            churn_mode  = args.churn_mode,
        )

    log.info("E5 complete. Results in: %s", results_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="E5: Churn resilience")
    parser.add_argument("--rounds",       type=int,  default=50)
    parser.add_argument("--n-clients",    type=int,  default=10)
    parser.add_argument("--dataset",      default="cifar10")
    parser.add_argument("--churn-rates",  default="0.0,0.1,0.3,0.5",
                        help="Comma-separated churn rates")
    parser.add_argument("--churn-mode",   default="select",
                        choices=["select", "real"],
                        help="select=withhold from aggregation (fast, mock-ok); "
                             "real=actual Iroh disconnect/rejoin (needs real Iroh)")
    parser.add_argument("--results-dir",  default="results/e5")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
