"""
E7 — Air Quality Federated Learning (Castilla y León).

Demonstrates FL-Iroh's algorithm-agnosticism by federating two qualitatively
different model families on the same infrastructure:

  Config A — AirMLP  + FedAvg    (standard neural network, mini-batch SGD)
  Config B — AirMLP  + FedAvg    (IID forced partition as baseline)
  Config C — ProphetWrapper + FedGAM  (GAM seasonality federating)
  Config D — ProphetWrapper + FedGAM  (IID forced partition as baseline)

Dataset  : CyL daily air-quality, 2011-2019, 10 provinces of Castilla y León
Clients  : 10  (one per province)
Task     : 3-class ICA classification (Bueno/Regular/Malo based on NO₂)

IID vs Geographic:
  geographic — natural partition: each client = one real province
  iid        — forced equal-size shuffle baseline (removes geographic non-IID)

Preprocessing prerequisite:
  python data/air-quailty/notebooks/preprocess_e7.py

Outputs (results/e7/):
  e7_airmlp_geographic_fl_metrics.csv
  e7_airmlp_iid_fl_metrics.csv
  e7_prophet_geographic_fl_metrics.csv
  e7_prophet_iid_fl_metrics.csv
  e7_summary.csv   — one row per config: (model, partition, accuracy, rounds)

Usage::
    python -m experiments.e7_air_quality_fl
    python -m experiments.e7_air_quality_fl --rounds 30 --configs airmlp_geo
"""
from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path
import torch
import yaml

log = logging.getLogger("e7_air_quality_fl")
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)

N_CLIENTS  = 10
N_ROUNDS   = 50   # AirMLP; Prophet uses fewer internal rounds but same outer loop
N_CLASSES  = 3
LSTM_WINDOW = 7   # AirLSTM sequence length (days of history per sample)
DATA_DIR   = "./data"
RESULTS_DIR_DEFAULT = "results/e7"

ConfigName = str

# Config registry. Each entry maps a config name to its runner spec.
# Prophet configs encode (aggregator, max_train_years) so the clean
# FedGAM-vs-FedAvg comparison (same model, same data, only aggregation differs)
# can be run both on full history and on the limited-history regime where
# federated seasonality is expected to help sparse provinces.
#   kind: "airmlp" | "prophet" | "airlstm"
PROPHET_SPECS: dict[str, dict] = {
    # Full-history clean comparison
    "prophet_fedgam_geographic": {"agg": "fedgam", "yrs": None, "part": "geographic"},
    "prophet_fedgam_iid":        {"agg": "fedgam", "yrs": None, "part": "iid"},
    "prophet_fedavg_geographic": {"agg": "fedavg", "yrs": None, "part": "geographic"},
    "prophet_fedavg_iid":        {"agg": "fedavg", "yrs": None, "part": "iid"},
    # Limited-history (1 year) clean comparison — the FL value-proposition regime
    "prophet_fedgam_1yr_geographic": {"agg": "fedgam", "yrs": 1, "part": "geographic"},
    "prophet_fedgam_1yr_iid":        {"agg": "fedgam", "yrs": 1, "part": "iid"},
    "prophet_fedavg_1yr_geographic": {"agg": "fedavg", "yrs": 1, "part": "geographic"},
    "prophet_fedavg_1yr_iid":        {"agg": "fedavg", "yrs": 1, "part": "iid"},
}

ALL_CONFIGS: list[ConfigName] = [
    "airmlp_geographic",
    "airmlp_iid",
    "airlstm_geographic",
    "airlstm_iid",
    *PROPHET_SPECS.keys(),
]


def _seed(seeds_file: str = "seeds.yaml") -> dict:
    try:
        with open(seeds_file) as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Neural runner: AirMLP (tabular) or AirLSTM (sequential) + FedAvg
# ---------------------------------------------------------------------------

async def run_airmlp(
    partition   : str,
    results_dir : Path,
    rounds      : int,
    seeds       : dict,
    model_kind  : str = "airmlp",
) -> dict:
    """Run E7 with a neural model + FedAvg on the air-quality dataset.

    model_kind:
      "airmlp"  — memoryless MLP over a single day's 6 features (cheap baseline).
      "airlstm" — LSTM over a ``LSTM_WINDOW``-day window of the same features
                  (sequence-aware accuracy ceiling). The recurrent tensors are
                  carried over the identical Iroh/QUIC transport, demonstrating
                  algorithm-agnosticism.
    """
    import random

    from fl_coap_iroh.data.partition import (
        load_air_quality_sequences, load_dataset, partition_dataset,
    )
    from fl_coap_iroh.fl.client import FLClient
    from fl_coap_iroh.fl.server import FLServer
    from fl_coap_iroh.metrics.collector import MetricsCollector
    from fl_coap_iroh.models.air_lstm import AirLSTM
    from fl_coap_iroh.models.air_mlp import AirMLP
    from fl_coap_iroh.types import (
        AvailabilityInfo, ComputeCapabilities, DatasetDescriptor,
        NodeCapabilities, NodeRole, NodeStatus, TrainingPolicy,
    )

    is_lstm  = (model_kind == "airlstm")
    model_name = "AirLSTM" if is_lstm else "AirMLP"
    label    = f"{model_kind}_{partition}"
    scenario = f"e7_{label}"
    log.info("=== E7 %s + FedAvg — %s ===", model_name, partition)

    torch.manual_seed(seeds.get("model_init", 123))

    if is_lstm:
        train_ds, test_ds = load_air_quality_sequences(DATA_DIR, window=LSTM_WINDOW)
        feature_dim: list[int] = [LSTM_WINDOW, 6]
    else:
        train_ds, test_ds = load_dataset("air_quality", DATA_DIR)
        feature_dim = [6]
    partitions = partition_dataset(
        train_ds, N_CLIENTS, partition,
        seed=seeds.get("data_partition", 42),
    )

    def _make_model() -> torch.nn.Module:
        if is_lstm:
            return AirLSTM(input_dim=6, hidden_dim=48, head_dim=24, num_classes=N_CLASSES)
        return AirMLP(input_dim=6, hidden1=32, hidden2=32, hidden3=16, num_classes=N_CLASSES)

    # Class weights from training set to handle geographic imbalance
    y_train = torch.tensor([int(train_ds[i][1]) for i in range(len(train_ds))])
    class_counts = torch.bincount(y_train, minlength=N_CLASSES).float()
    class_weights = (class_counts.sum() / (N_CLASSES * class_counts + 1e-8))
    log.info("%s class weights: %s", model_name, class_weights.tolist())

    server_caps = NodeCapabilities(
        node_id      = "server",
        role         = NodeRole.AGGREGATOR,
        compute      = ComputeCapabilities(cpu_cores=4),
        availability = AvailabilityInfo(status=NodeStatus.READY),
    )
    policy = TrainingPolicy(
        min_clients=max(2, N_CLIENTS // 2),
        local_epochs=1,
        learning_rate=0.01,
        max_rounds=rounds,
    )
    server = FLServer(
        node_id      = "server",
        model        = _make_model(),
        test_dataset = test_ds,
        capabilities = server_caps,
        policy       = policy,
        coap_port    = 5783,
        scenario     = scenario,
        architecture = "B",
        # FedAvg is the default aggregator_fn — no override needed
    )
    server.metrics = MetricsCollector("server", scenario, "B", str(results_dir))
    server_ep = await server.start()

    clients = []
    for i in range(N_CLIENTS):
        torch.manual_seed(seeds.get("model_init", 123) + i)
        caps = NodeCapabilities(
            node_id      = f"client-{i}",
            role         = NodeRole.CLIENT,
            compute      = ComputeCapabilities(cpu_cores=2),
            availability = AvailabilityInfo(status=NodeStatus.READY),
        )
        ds_desc = DatasetDescriptor(
            dataset_id   = f"client-{i}-air_quality",
            dataset_name = "air_quality",
            samples      = len(partitions[i]),
            classes      = list(range(N_CLASSES)),
            iid          = (partition == "iid"),
            distribution = partition,
            feature_dim  = feature_dim,
        )
        client = FLClient(
            node_id            = f"client-{i}",
            model              = _make_model(),
            train_dataset      = partitions[i],
            val_dataset        = test_ds,
            capabilities       = caps,
            dataset_descriptor = ds_desc,
            coap_port          = 5784 + i,
            scenario           = scenario,
            architecture       = "B",
        )
        client.metrics = MetricsCollector(f"client-{i}", scenario, "B", str(results_dir))
        await client.start()
        client.set_server_endpoint(server_ep)
        clients.append(client)
        server.register_client(f"client-{i}", client.iroh_endpoint)

    async def _run_client(c: FLClient, n: int) -> None:
        for r in range(1, n + 1):
            try:
                await c.run_round(r)
            except Exception as exc:
                log.error("[%s] round %d error: %s", c.node_id, r, exc)

    await asyncio.gather(
        server.run_rounds(n_rounds=rounds),
        *[_run_client(c, rounds) for c in clients],
        return_exceptions=True,
    )
    for c in clients:
        await c.stop()
    await server.stop()

    server.metrics.export_csv(tag=f"e7_{label}")
    summary = server.metrics.summary()
    log.info("%s %s done: %s", model_name, label, summary)
    return {"config": label, "model": model_name, "partition": partition, **summary}


# ---------------------------------------------------------------------------
# ProphetWrapper + FedGAM runner
# ---------------------------------------------------------------------------

async def run_prophet(
    partition       : str,
    results_dir     : Path,
    rounds          : int,
    seeds           : dict,
    aggregator      : str = "fedgam",
    max_train_years : int | None = None,
) -> dict:
    """
    Run E7 with ProphetWrapper, comparing FedGAM vs FedAvg aggregation.

    This is the clean apples-to-apples comparison requested to isolate the
    *aggregation algorithm* from the *model family*: the model (Prophet), the
    data, and the transport are identical; only the server-side aggregation
    differs.

      aggregator="fedgam":  federate ONLY shared Fourier seasonality; each
                            province keeps its own locally-fitted trend
                            (k, m, delta). Semantically correct for GAMs.
      aggregator="fedavg":  naively average ALL parameters including the trend,
                            mixing incompatible local trends (e.g. industrial
                            Ponferrada with rural Soria).

    max_train_years:  if set, truncate each client's training history to the
                      most recent N years. The limited-data regime is where
                      federated seasonality is expected to help sparse-history
                      provinces — i.e. the actual FL value proposition.

    Each 'round' consists of:
      1. Server distributes the global aggregate.
      2. Each client injects it as warm-start (seasonality only for FedGAM;
         seasonality + trend for FedAvg), then fits Prophet locally.
      3. Server aggregates the returned state dicts.
      4. Server evaluates on 2019 test dates.

    Because Prophet does not use mini-batch SGD, FLClient._train() is not
    used here.  We simulate the round manually within a single process.
    In a real deployment over Iroh, the existing Iroh transport layer handles
    the binary serialisation of the state_dict tensors without modification.
    """
    import csv

    from fl_coap_iroh.fl.fedavg import fedavg_aggregate
    from fl_coap_iroh.fl.fedgam import fedgam_aggregate
    from fl_coap_iroh.models.prophet_wrapper import ProphetWrapper, load_ica_thresholds

    if aggregator == "fedavg":
        aggregate_fn   = fedavg_aggregate
        federate_trend = True
    else:
        aggregate_fn   = fedgam_aggregate
        federate_trend = False

    suffix = aggregator
    if max_train_years is not None:
        suffix = f"{aggregator}_{max_train_years}yr"
    label   = f"prophet_{suffix}_{partition}"
    scenario = f"e7_{label}"
    log.info("=== E7 ProphetWrapper + %s — %s (max_train_years=%s) ===",
             aggregator.upper(), partition, max_train_years)

    torch.manual_seed(seeds.get("experiment_e7", 505))
    load_ica_thresholds(DATA_DIR)

    # Load timeseries CSV (unnormalised NO2 + wind speed for Prophet)
    ts_path = Path(DATA_DIR) / "air-quailty" / "datasets" / "air_quality_fl_timeseries.csv"
    if not ts_path.exists():
        raise FileNotFoundError(
            f"Timeseries CSV not found: {ts_path}\n"
            "Run: python data/air-quailty/notebooks/preprocess_e7.py"
        )

    # Read CSV into province-indexed dicts
    provinces_data: dict[str, dict] = {}
    with open(ts_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            prov  = row["provincia"]
            split = row["split"]
            fecha = row["fecha"]
            try:
                no2 = float(row["NO2"])
                vel = float(row["velmedia"])
            except (ValueError, KeyError):
                continue
            if prov not in provinces_data:
                provinces_data[prov] = {"train_dates": [], "train_no2": [], "train_vel": [],
                                         "test_dates":  [], "test_no2":  [], "test_vel":  []}
            d = provinces_data[prov]
            if split == "train":
                d["train_dates"].append(fecha)
                d["train_no2"].append(no2)
                d["train_vel"].append(vel)
            else:
                d["test_dates"].append(fecha)
                d["test_no2"].append(no2)
                d["test_vel"].append(vel)

    sorted_provs = sorted(provinces_data.keys())
    if len(sorted_provs) < 2:
        raise RuntimeError(
            "Not enough provinces found in timeseries CSV. "
            "Check: python data/air-quailty/notebooks/preprocess_e7.py"
        )
    log.info("Provinces loaded: %s", sorted_provs)

    # IID baseline: pool all data and distribute evenly
    if partition == "iid":
        all_train_dates: list[str] = []
        all_train_no2:   list[float] = []
        all_train_vel:   list[float] = []
        for d in provinces_data.values():
            all_train_dates.extend(d["train_dates"])
            all_train_no2.extend(d["train_no2"])
            all_train_vel.extend(d["train_vel"])
        rng = torch.Generator().manual_seed(seeds.get("data_partition", 42))
        perm = torch.randperm(len(all_train_dates), generator=rng).tolist()
        n_per_client = len(perm) // N_CLIENTS
        iid_splits: list[dict] = []
        for i in range(N_CLIENTS):
            start = i * n_per_client
            end   = start + n_per_client if i < N_CLIENTS - 1 else len(perm)
            idx   = perm[start:end]
            iid_splits.append({
                "train_dates": [all_train_dates[j] for j in idx],
                "train_no2"  : [all_train_no2[j]   for j in idx],
                "train_vel"  : [all_train_vel[j]    for j in idx],
                # IID test: use combined test from first province (or all)
                "test_dates" : sorted_provs and provinces_data[sorted_provs[0]]["test_dates"] or [],
                "test_no2"   : sorted_provs and provinces_data[sorted_provs[0]]["test_no2"]  or [],
                "test_vel"   : sorted_provs and provinces_data[sorted_provs[0]]["test_vel"]  or [],
            })
        client_data = iid_splits
    else:
        client_data = [provinces_data[p] for p in sorted_provs[:N_CLIENTS]]

    # Initialise one ProphetWrapper per client
    n_clients_actual = min(N_CLIENTS, len(client_data))

    # --- Limited-history scenario: keep only the most recent N training years ---
    # This is the regime where federated seasonality is expected to help sparse
    # provinces. Truncation is applied per client on the (date, no2, vel) triples.
    if max_train_years is not None:
        def _truncate(cd: dict) -> dict:
            dates = cd["train_dates"]
            if not dates:
                return cd
            years = [int(str(d)[:4]) for d in dates]
            cutoff = max(years) - max_train_years + 1
            keep = [j for j, y in enumerate(years) if y >= cutoff]
            return {
                **cd,
                "train_dates": [cd["train_dates"][j] for j in keep],
                "train_no2"  : [cd["train_no2"][j]   for j in keep],
                "train_vel"  : [cd["train_vel"][j]    for j in keep],
            }
        client_data = [_truncate(cd) for cd in client_data]
        log.info("Limited-history: kept last %d year(s) per client → sizes=%s",
                 max_train_years, [len(cd["train_dates"]) for cd in client_data])

    global_model = ProphetWrapper()
    clients_pw: list[ProphetWrapper] = [ProphetWrapper() for _ in range(n_clients_actual)]

    round_accuracies: list[float] = []

    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_rows: list[dict] = []

    for r in range(1, rounds + 1):
        log.info("ProphetWrapper %s round %d/%d", aggregator.upper(), r, rounds)

        # Distribute global model state to all clients
        global_sd = global_model.state_dict()

        updates: list[tuple[dict, float]] = []
        for i, (pw, cd) in enumerate(zip(clients_pw, client_data)):
            # Warm-start injection semantics differ by aggregator:
            #   FedAvg  → inject full global state (seasonality + averaged trend)
            #   FedGAM  → inject only seasonality; client keeps its own local trend
            # Round 1 has no global aggregate yet, so clients fit from scratch.
            warmstart = r > 1
            if warmstart:
                if federate_trend:
                    pw.load_state_dict(global_sd)
                else:
                    pw.inject_seasonality(global_sd)
            pw.train_prophet(
                dates        = cd["train_dates"],
                no2_values   = cd["train_no2"],
                velmedia     = cd["train_vel"],
                use_warmstart= warmstart,
            )
            n_samples = len(cd["train_dates"])
            updates.append((pw.state_dict(), float(n_samples)))

        # Aggregate: FedGAM averages only seasonality; FedAvg averages everything.
        aggregated_sd = aggregate_fn(updates)
        global_model.load_state_dict(aggregated_sd)

        # Evaluate on test set (each client predicts on its own province test split)
        total_correct = 0
        total_samples = 0
        for i, (pw, cd) in enumerate(zip(clients_pw, client_data)):
            if not cd["test_dates"]:
                continue
            preds = pw.predict_ica(cd["test_dates"], cd["test_vel"])
            # ICA labels from raw NO2
            from fl_coap_iroh.models.prophet_wrapper import _no2_to_ica
            true_labels = [_no2_to_ica(v) for v in cd["test_no2"]]
            correct = sum(p == t for p, t in zip(preds, true_labels))
            total_correct += correct
            total_samples += len(true_labels)

        accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        round_accuracies.append(accuracy)
        log.info("Round %d — test accuracy: %.3f  (%d samples)", r, accuracy, total_samples)
        metrics_rows.append({"round": r, "accuracy": accuracy, "samples": total_samples})

    # Save round metrics
    metrics_path = results_dir / f"e7_{label}_fl_metrics.csv"
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["round", "accuracy", "samples"])
        writer.writeheader()
        writer.writerows(metrics_rows)
    log.info("ProphetWrapper %s %s done. Final acc=%.3f  Saved: %s",
             aggregator.upper(), label, round_accuracies[-1] if round_accuracies else 0.0, metrics_path)

    final_acc = round_accuracies[-1] if round_accuracies else 0.0
    return {
        "config"     : label,
        "model"      : "ProphetWrapper",
        "partition"  : partition,
        "aggregator" : aggregator,
        "max_train_years": max_train_years if max_train_years is not None else "all",
        "accuracy"   : final_acc,
        "rounds"     : rounds,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def _run_configs(
    configs_to_run : list[ConfigName],
    results_dir    : Path,
    rounds         : int,
    seeds          : dict,
) -> list[dict]:
    """Run the requested configs once with the given seed dict; return summary rows."""
    summary_rows: list[dict] = []
    for config in configs_to_run:
        if config == "airmlp_geographic":
            row = await run_airmlp("geographic", results_dir, rounds, seeds)
        elif config == "airmlp_iid":
            row = await run_airmlp("iid",         results_dir, rounds, seeds)
        elif config == "airlstm_geographic":
            row = await run_airmlp("geographic", results_dir, rounds, seeds, model_kind="airlstm")
        elif config == "airlstm_iid":
            row = await run_airmlp("iid",         results_dir, rounds, seeds, model_kind="airlstm")
        elif config in PROPHET_SPECS:
            spec = PROPHET_SPECS[config]
            row = await run_prophet(
                spec["part"], results_dir, rounds, seeds,
                aggregator=spec["agg"], max_train_years=spec["yrs"],
            )
        else:
            log.warning("Unknown config '%s' — skipping", config)
            continue
        summary_rows.append(row)
    return summary_rows


def _write_summary(summary_rows: list[dict], summary_path: Path) -> None:
    if not summary_rows:
        return
    import csv
    fieldnames = list(summary_rows[0].keys())
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary_rows)
    log.info("E7 summary saved: %s", summary_path)

    print(f"\n=== E7 Results Summary ({summary_path.name}) ===")
    hdr = f"{'Config':<28}  {'Model':<18}  {'Partition':<12}  {'Accuracy':>8}"
    print(hdr)
    print("-" * len(hdr))
    for row in summary_rows:
        print(
            f"{row.get('config',''):<28}  "
            f"{row.get('model',''):<18}  "
            f"{row.get('partition',''):<12}  "
            f"{float(row.get('test_acc_final', row.get('accuracy', row.get('best_accuracy', 0.0)))):>8.3f}"
        )


async def main_async(args: argparse.Namespace) -> None:
    from experiments._replication import derive_seeds, load_replicate_seeds

    base_seeds = _seed()

    configs_to_run: list[ConfigName] = (
        args.configs if args.configs else ALL_CONFIGS
    )
    log.info("E7 configs to run: %s", configs_to_run)

    # Resolve replication master seeds (F2). --seeds overrides the yaml list;
    # --all-seeds uses the full replicate_seeds list; default = single canonical run.
    if args.seeds:
        masters = list(args.seeds)
    elif args.all_seeds:
        masters = load_replicate_seeds()
        if not masters:
            log.warning("No replicate_seeds in seeds.yaml — falling back to single run")
    else:
        masters = []

    if not masters:
        # Single canonical run (backward compatible)
        results_dir = Path(args.results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        torch.manual_seed(base_seeds.get("experiment_e7", 505))
        rows = await _run_configs(configs_to_run, results_dir, args.rounds, base_seeds)
        _write_summary(rows, results_dir / "e7_summary.csv")
        log.info("E7 complete. Results in: %s", results_dir)
        return

    # Multi-seed replication: one full config sweep per master seed
    log.info("E7 replication over %d master seeds: %s", len(masters), masters)
    seeds_root = Path(args.results_dir) / "seeds"
    seeds_root.mkdir(parents=True, exist_ok=True)
    for master in masters:
        derived = derive_seeds(base_seeds, master)
        seed_dir = seeds_root / f"seed{master}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        torch.manual_seed(derived.get("experiment_e7", 505))
        log.info("=== E7 replicate master=%d (dir=%s) ===", master, seed_dir)
        rows = await _run_configs(configs_to_run, seed_dir, args.rounds, derived)
        for r in rows:
            r["seed"] = master
        _write_summary(rows, seeds_root / f"e7_summary_seed{master}.csv")
    log.info(
        "E7 replication complete (%d seeds). Aggregate with:\n"
        "  python scripts/aggregate_ci.py --glob '%s/e7_summary_seed*.csv' "
        "--group config --metric test_acc_final",
        len(masters), seeds_root,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="E7: Air Quality FL experiment")
    parser.add_argument(
        "--rounds", type=int, default=N_ROUNDS,
        help=f"Number of FL rounds (default: {N_ROUNDS})",
    )
    parser.add_argument(
        "--results-dir", default=RESULTS_DIR_DEFAULT,
    )
    parser.add_argument(
        "--configs", nargs="+", choices=ALL_CONFIGS, default=None,
        help="Which configs to run (default: all four)",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=None,
        help="Master seeds for multi-seed replication (F2). Each runs a full "
             "config sweep into results/e7/seeds/seed<S>/ and writes a tagged "
             "summary. Overrides --all-seeds.",
    )
    parser.add_argument(
        "--all-seeds", action="store_true",
        help="Replicate over every seed in seeds.yaml 'replicate_seeds'.",
    )
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
