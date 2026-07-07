#!/usr/bin/env bash
# =============================================================================
# FL-Iroh — centralized upper-bound baselines (reviewer request)
#
# Trains each model on the FULL dataset with a single client (K=1, IID
# partition = the whole training set), which with E=1 local epoch per round
# and R rounds is equivalent to R epochs of plain centralized SGD using the
# exact same training loop, model, and seeds as the federated runs. This
# provides the "centralized upper bound" rows that make the federation gap
# interpretable in the paper (crop: literature reaches ~99% on this dataset).
#
#   * Crop Recommendation + AgriMLP  (paper E4 dataset)
#   * CIFAR-10 + SimpleCNN           (paper E2 dataset)
#
# NOTE: the air-quality centralized baseline (AirMLP/Prophet) requires a
# single-client mode in experiments/e7_air_quality_fl.py, which does not
# expose --n-clients yet; extend it before adding that row.
#
# CPU-only workload. Cluster reorg (July 2026): --partition=cpu (max 2 days);
# do NOT request GPUs for this job.
#
# Submit:   sbatch slurm/run_centralized_baselines.sh
# Results:  results/e2_central/  (tags e2_*_central, multi-seed)
# =============================================================================
#SBATCH --job-name=fl_central_base
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=results/logs/central_%j.out
#SBATCH --error=results/logs/central_%j.err

set -uo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "${REPO_ROOT}"
mkdir -p results/logs results/e2_central

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
export FL_MOCK_IROH=1

ROUNDS="${ROUNDS:-100}"

echo "=== centralized upper bounds (K=1, R=${ROUNDS}, multi-seed) ==="

# Crop Recommendation + AgriMLP
python -m experiments.e2_centralized_fl \
    --rounds "${ROUNDS}" --n-clients 1 --dataset crop \
    --partition iid --all-seeds \
    --results-dir results/e2_central

# CIFAR-10 + SimpleCNN
python -m experiments.e2_centralized_fl \
    --rounds "${ROUNDS}" --n-clients 1 --dataset cifar10 \
    --partition iid --all-seeds \
    --results-dir results/e2_central

echo "=== done: $(date '+%F %T') ==="
