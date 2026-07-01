#!/usr/bin/env bash
# =============================================================================
# FL-Iroh — Flower (E8) quantitative baseline with MULTI-SEED CI on a COMPUTE node
#
# Runs the Flower / in-process-FedAvg simulation over the SAME replication
# master seeds as E2 (seeds.yaml: replicate_seeds), so the E7 accuracy row is
# reported as mean ± CI95 and is directly comparable to FL-Iroh's E2 numbers.
# Because the in-process fallback reuses FL-Iroh's own model, partitions and
# training loop, this demonstrates transport-agnostic accuracy parity.
#
# Submit (runs in the background on a compute node):
#   sbatch slurm/run_flower_ci.sh
#
# Override defaults via --export, e.g.:
#   sbatch --export=ALL,DATASET=cifar10,ROUNDS=100,NCLIENTS=10 slurm/run_flower_ci.sh
#
# Monitor:
#   squeue --me
#   tail -f results/logs/flower_ci_<jobid>.out
# =============================================================================
#SBATCH --job-name=fl_flower_ci
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=16:00:00
#SBATCH --output=results/logs/flower_ci_%j.out
#SBATCH --error=results/logs/flower_ci_%j.err

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
ROUNDS="${ROUNDS:-100}"   # match E2 (R=100) for a like-for-like accuracy comparison

GIT_COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "[flower-ci] git=${GIT_COMMIT} host=$(hostname) cpus=${NCPU} $(date '+%F %T')"
echo "[flower-ci] dataset=${DATASET} n_clients=${NCLIENTS} rounds=${ROUNDS}"

# ── Run each partition over ALL replicate seeds, then aggregate mean±CI95 ─────
# Per-partition results land in results/e8/<part>/seeds/e8_flower_summary_seed*.csv
run_cfg () {
    local part="$1"; shift          # iid | noniid
    local alpha_flag="$1"; shift     # "" | 0.1
    local tag="$1"; shift            # iid | noniid_a0.1
    local out_dir="${RESULTS_DIR}/${tag}"
    mkdir -p "${out_dir}"
    echo ""
    echo "=== [flower-ci] ${DATASET} / ${part}${alpha_flag:+ (alpha=$alpha_flag)} — all seeds ==="
    python -m experiments.e8_flower_tailscale \
        --mode sim \
        --dataset "${DATASET}" \
        --partition "${part}" ${alpha_flag:+--alpha "${alpha_flag}"} \
        --n-clients "${NCLIENTS}" \
        --rounds "${ROUNDS}" \
        --all-seeds \
        --results-dir "${out_dir}"

    echo "--- [flower-ci] aggregate ${tag} (mean ± CI95) ---"
    python scripts/aggregate_ci.py \
        --glob "${out_dir}/seeds/e8_flower_summary_seed*.csv" \
        --group config --metric test_acc_final \
        | tee "${out_dir}/e8_flower_ci_${tag}.txt"
}

run_cfg iid    ""    "iid"
run_cfg noniid "0.1" "noniid_a0.1"

echo ""
echo "[flower-ci] done $(date '+%F %T')"
echo "[flower-ci] CI summaries:"
ls -1 "${RESULTS_DIR}"/*/e8_flower_ci_*.txt 2>/dev/null || true
