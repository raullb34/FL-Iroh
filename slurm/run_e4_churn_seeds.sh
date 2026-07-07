#!/usr/bin/env bash
# =============================================================================
# FL-Iroh — paper-E4 churn resilience with MULTI-SEED CI + churn x non-IID
#
# Addresses two reviewer findings on the E4 table:
#   (a) single-seed results carry no uncertainty estimate -> replicate over the
#       5 master seeds in seeds.yaml (same registry as E2/E6);
#   (b) IID-only churn is nearly tautological -> add a Dirichlet alpha=0.1
#       configuration to measure the churn x non-IID interaction.
#
# Workload is CPU-only (Crop Recommendation + AgriMLP, mock transport).
# Cluster reorg (July 2026): use --partition=cpu (max 2 days); do NOT request
# GPUs for this job.
#
# Submit:   sbatch slurm/run_e4_churn_seeds.sh
# Monitor:  squeue --me ; tail -f results/logs/e4_seeds_<jobid>.out
# Results:  results/e5/seeds/seed<S>/  (per-seed CSVs, tag e5_churn_XXpct_seed<S>[_dirichlet0.1])
# =============================================================================
#SBATCH --job-name=fl_e4_churn_seeds
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=1-16:00:00
#SBATCH --output=results/logs/e4_seeds_%j.out
#SBATCH --error=results/logs/e4_seeds_%j.err

set -uo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "${REPO_ROOT}"
mkdir -p results/logs results/e5

VENV="${REPO_ROOT}/.venv"
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

NCPU="${SLURM_CPUS_PER_TASK:-16}"
export OMP_NUM_THREADS="${NCPU}" MKL_NUM_THREADS="${NCPU}"
export OPENBLAS_NUM_THREADS="${NCPU}" NUMEXPR_NUM_THREADS="${NCPU}"
export FL_TORCH_THREADS="${NCPU}"
export FL_MOCK_IROH=1   # in-process transport (same as published E4 runs)

ROUNDS="${ROUNDS:-100}"
NCLIENTS="${NCLIENTS:-10}"
DATASET="${DATASET:-crop}"

echo "=== paper-E4 multi-seed churn: dataset=${DATASET} R=${ROUNDS} K=${NCLIENTS} ==="

# (a) IID churn, all 5 master seeds (matches the published E4 configuration)
python -m experiments.e5_churn \
    --rounds "${ROUNDS}" --n-clients "${NCLIENTS}" --dataset "${DATASET}" \
    --churn-rates "0.0,0.1,0.3,0.5" --churn-mode select \
    --partition iid --all-seeds \
    --results-dir results/e5

# (b) churn x non-IID interaction (Dirichlet alpha=0.1), all 5 master seeds
python -m experiments.e5_churn \
    --rounds "${ROUNDS}" --n-clients "${NCLIENTS}" --dataset "${DATASET}" \
    --churn-rates "0.0,0.1,0.3,0.5" --churn-mode select \
    --partition dirichlet --alpha 0.1 --all-seeds \
    --results-dir results/e5

echo "=== done: $(date '+%F %T') ==="
