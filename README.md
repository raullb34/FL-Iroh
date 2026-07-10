# FL-Iroh — NAT-Transparent Federated Learning for Agricultural and Environmental IoT

FL-Iroh embeds NAT traversal in the federated-learning transport itself: an
[Iroh](https://iroh.computer) QUIC peer-to-peer data plane (public-key peer
addressing, transparent DERP relay fallback) coordinated by a lightweight
CoAP/CBOR control plane. This repository contains the full implementation,
the experiment harness, and the results backing every table in the paper.

## Paper ↔ code experiment mapping

The paper renumbers experiments; the code numbering is stable. Use this table
to locate the script and results for each experiment reported in the paper.

| Paper | Description                              | Module                              | Results dir      |
|-------|------------------------------------------|-------------------------------------|------------------|
| E1    | Transport throughput (HTTP/2 vs Iroh)    | `experiments/e1_microbenchmark.py`  | `results/e1/`    |
| E2    | FL convergence (CIFAR-10, IID/Dirichlet) | `experiments/e2_centralized_fl.py`  | `results/e2*/`   |
| E3    | NAT traversal (real PC↔RPi WAN)          | `experiments/e3_nat_traversal.py`   | `results/e3/`    |
| E4    | Churn resilience                         | `experiments/e5_churn.py`           | `results/e5/` (multi-seed: `results/e5/final-experiment/seeds/`) |
| E5    | CoAP discovery overhead                  | `experiments/e6_coap_overhead.py`   | `results/e6/`    |
| E6    | Air-quality FL (AirMLP / Prophet FedGAM) | `experiments/e7_air_quality_fl.py`  | `results/e7/`    |
| §4.8  | Flower-compatible parity harness (sim)   | `experiments/e8_flower_tailscale.py`| `results/e8/`    |

Additional result sets:

- `results/e3/establish/` — E3 repeated cold-start connection-establishment
  campaign (150 independent attempts, 30 per scenario; per-run CSV + client log
  + per-scenario `summary.csv`), produced by `scripts/e3_repeat_establish.sh`.
- `results/e2_central/` — centralized (K=1) upper-bound baselines, multi-seed,
  produced by `slurm/run_centralized_baselines.sh`.

Models: `fl_coap_iroh/models/` — AgriMLP (7,734 params), AirMLP (1,987 params),
SimpleCNN (94,762 params), ProphetWrapper (FedGAM).

## Setup

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e .            # uses pyproject.toml
# Prophet (E6) additionally requires cmdstan; see e7_air_quality_fl.py header.
```

Datasets:

- **Crop Recommendation** (2,200 rows, 22 classes): download from
  [Kaggle: atharvaingle/crop-recommendation-dataset](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset)
  and place `Crop_recommendation.csv` under `data/` (see
  `fl_coap_iroh/data/partition.py`).
- **Castilla y León air quality** (2011–2019, 10 monitoring zones): processed
  CSVs are included under `data/air-quailty/datasets/` (source: open-data
  portals of Junta de Castilla y León and AEMET). Class labels use
  training-set NO2 terciles (T1=7.5, T2=13.0 µg/m³, see
  `air_quality_ica_thresholds.json`).

## Running experiments

### Local (single machine)

```bash
python -m experiments.e2_centralized_fl --rounds 100 --n-clients 10 \
    --dataset cifar10 --partition iid --results-dir results/e2
```

Multi-seed replication: add `--all-seeds` (uses the master seed registry in
`seeds.yaml`; n=5 seeds, CI95 via Student-t; see `experiments/_replication.py`).

### HPC (Slurm)

Cluster partitions (July 2026 reorganization): all FL simulations here are
CPU-only workloads → `--partition=cpu` (max 2 days/job). Do not request GPUs
for these jobs.

```bash
bash slurm/submit.sh                    # main E1–E8 array (slurm/run_experiments.sh)
sbatch slurm/run_flower_ci.sh           # Flower parity harness, multi-seed CI
sbatch slurm/run_e4_churn_seeds.sh      # paper-E4 churn, multi-seed + non-IID
sbatch slurm/run_centralized_baselines.sh  # centralized upper bounds (crop + air)
```

Logs and per-job metadata land in `results/logs/` and `results/metadata/`.

### Real-hardware experiments (E1-WAN and E3) — not Slurm

E1 (WAN) and E3 require the physical PC ↔ Raspberry Pi 4B pair on separate
networks. See `scripts/` launchers:

- `scripts/_e3_server.sh` (PC) / `scripts/_e3_client.sh` (RPi) — classified E3
  runs per scenario (`net_lan`, `net_nat1`, `net_nat2`, `net_cgnat`, `net_fw443`).
- `scripts/e3_repeat_establish.sh` — repeated cold-start connection
  establishment (per-connection statistics; addresses the transfer-vs-connection
  distinction in the paper).

Trivial-baseline computation for E6 (majority class + 7-day persistence) is a
pure post-processing step: `python scripts/e6_trivial_baselines.py`.

## Results layout

Each experiment writes CSV/JSON metrics plus a `.log`, and `.ok`/`.failed`
sentinels under its results dir. `slurm/collect_results.py` aggregates them.
Multi-seed summaries are written per seed (`*_summary_seed<S>.csv`).

The per-run outputs backing every table in the paper are committed under
`results/` (see `.gitignore` for the few excluded scratch patterns).

## Paper

LaTeX sources under `paper/`:

- `paper/elsevier-journal/els-cas-templates/els-cas-templates/fl-iroh-cas-dc.tex`
  — Elsevier CAS (cas-dc) submission manuscript (canonical).
- `paper/paper.tex` — earlier IEEEtran draft (kept for reference).

## License / citation

Code is released under the MIT License (see `LICENSE`). If you use FL-Iroh,
please cite the paper (reference forthcoming upon publication).
