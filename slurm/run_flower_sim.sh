#!/usr/bin/env bash
# =============================================================================
# FL-Iroh — Flower (E8) quantitative sim on a COMPUTE node
#
# Runs the Flower / in-process-FedAvg simulation used for the E7 comparison
# (equivalent accuracy under the same FedAvg aggregation as FL-Iroh's E2).
# Runs two configs to mirror E2: IID and Dirichlet alpha=0.1 on CIFAR-10.
#
# Submit:
#   sbatch slurm/run_flower_sim.sh
#
# Override defaults via --export, e.g. crop dataset / more rounds:
#   sbatch --export=ALL,DATASET=crop,ROUNDS=50,NCLIENTS=10 slurm/run_flower_sim.sh
#
# Monitor:
#   squeue --me
#   tail -f results/logs/flower_<jobid>.out
# =============================================================================
#SBATCH --job-name=fl_flower_sim
#SBATCH --partition=cpu   # CPU-only FL simulations; 'all' partition removed 2026-07-15
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=results/logs/flower_%j.out
#SBATCH --error=results/logs/flower_%j.err

set -uo pipefail

# ── Locate repo root (sbatch sets SLURM_SUBMIT_DIR) ──────────────────────────
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "${REPO_ROOT}"
RESULTS_DIR="${REPO_ROOT}/results/e8"
VENV="${REPO_ROOT}/.venv"
mkdir -p "${REPO_ROOT}/results/logs" "${RESULTS_DIR}"

# ── Activate environment ─────────────────────────────────────────────────────
if [[ -f "${VENV}/bin/activate" ]]; then
    # shellcheck source=/dev/null
    source "${VENV}/bin/activate"
elif command -v conda &>/dev/null && conda env list 2>/dev/null | grep -q fl_iroh; then
    # shellcheck source=/dev/null
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate fl_iroh
else
    echo "[error] No .venv found and no conda env 'fl_iroh'. Run: bash slurm/setup_env.sh"
    exit 1
fi

# ── Thread counts (CPU training) ─────────────────────────────────────────────
NCPU="${SLURM_CPUS_PER_TASK:-16}"
export OMP_NUM_THREADS="${NCPU}"
export MKL_NUM_THREADS="${NCPU}"
export OPENBLAS_NUM_THREADS="${NCPU}"
export NUMEXPR_NUM_THREADS="${NCPU}"

# ── Config (override with --export) ──────────────────────────────────────────
DATASET="${DATASET:-cifar10}"
NCLIENTS="${NCLIENTS:-10}"
ROUNDS="${ROUNDS:-50}"

GIT_COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "[flower-sim] git=${GIT_COMMIT} host=$(hostname) cpus=${NCPU} $(date '+%F %T')"
echo "[flower-sim] dataset=${DATASET} n_clients=${NCLIENTS} rounds=${ROUNDS}"

# ── Run: IID then Dirichlet alpha=0.1 (mirror E2) ────────────────────────────
# Each run overwrites e8_flower_metrics.csv, so tag output by copying afterwards.
run_cfg () {
    local part="$1"; shift
    local alpha_flag="$1"; shift
    local tag="$1"; shift
    echo ""
    echo "=== [flower-sim] ${DATASET} / ${part}${alpha_flag:+ (alpha=$alpha_flag)} ==="
    python -m experiments.e8_flower_tailscale \
        --mode sim \
        --dataset "${DATASET}" \
        --partition "${part}" ${alpha_flag:+--alpha "${alpha_flag}"} \
        --n-clients "${NCLIENTS}" \
        --rounds "${ROUNDS}" \
        --results-dir "${RESULTS_DIR}"
    if [[ -f "${RESULTS_DIR}/e8_flower_metrics.csv" ]]; then
        cp "${RESULTS_DIR}/e8_flower_metrics.csv" "${RESULTS_DIR}/e8_flower_metrics_${tag}.csv"
        echo "[flower-sim] saved ${RESULTS_DIR}/e8_flower_metrics_${tag}.csv"
    fi
}

run_cfg iid    ""    "iid"
run_cfg noniid "0.1" "noniid_a0.1"

echo ""
echo "[flower-sim] done $(date '+%F %T')"
echo "[flower-sim] outputs:"
ls -1 "${RESULTS_DIR}"/e8_flower_metrics*.csv 2>/dev/null || true
