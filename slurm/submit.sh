#!/usr/bin/env bash
# =============================================================================
# FL-Iroh — Submit the SLURM experiment array job
#
# Usage:
#   bash slurm/submit.sh                   # submit all tasks (0-10)
#   bash slurm/submit.sh --array=0         # re-run only task 0 (E1)
#   bash slurm/submit.sh --array=1-3       # re-run E2 variants only
#   bash slurm/submit.sh --array=4         # re-run E5 only
#   bash slurm/submit.sh --array=8         # re-run only E7 (air quality FL, 12 configs)
#   bash slurm/submit.sh --array=9         # re-run E8 baseline + energy estimation
#   bash slurm/submit.sh --array=10        # re-run multi-seed replication (CIs)
#
# Any extra flags are forwarded verbatim to sbatch, e.g.:
#   bash slurm/submit.sh --mail-type=END --mail-user=you@example.com
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ── Pre-flight checks ────────────────────────────────────────────────────────
if [[ ! -f "${REPO_ROOT}/.venv/bin/activate" ]]; then
    # Also accept conda env as fallback
    if ! (command -v conda &>/dev/null && conda env list 2>/dev/null | grep -q fl_iroh); then
        echo "[error] No virtual environment found."
        echo "        Run:  bash slurm/setup_env.sh"
        exit 1
    fi
fi

if [[ ! -d "${REPO_ROOT}/data/cifar-10-batches-py" ]]; then
    echo "[warn]  CIFAR-10 not found at data/cifar-10-batches-py/"
    echo "        Experiments will attempt to download on first run."
fi

# ── Ensure output directories exist before sbatch writes its logs ────────────
mkdir -p \
    "${REPO_ROOT}/results/logs" \
    "${REPO_ROOT}/results/metadata" \
    "${REPO_ROOT}/results/e1" \
    "${REPO_ROOT}/results/e2" \
    "${REPO_ROOT}/results/e3" \
    "${REPO_ROOT}/results/e5" \
    "${REPO_ROOT}/results/e6" \
    "${REPO_ROOT}/results/e7" \
    "${REPO_ROOT}/results/e8"

# ── Submit ────────────────────────────────────────────────────────────────────
cd "${REPO_ROOT}"
GIT_COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "[submit] git=${GIT_COMMIT}  $(date '+%F %T')"
echo "[submit] sbatch $* slurm/run_experiments.sh"

JOB_OUTPUT="$(sbatch "$@" slurm/run_experiments.sh)"
JOB_ID="$(echo "${JOB_OUTPUT}" | awk '{print $NF}')"

echo ""
echo "  ${JOB_OUTPUT}"
echo ""
echo "  Monitor:  squeue -j ${JOB_ID}"
echo "  Logs:     tail -f results/logs/slurm_${JOB_ID}_<task>.out"
echo "  Cancel:   scancel ${JOB_ID}"
echo "  Results:  python slurm/collect_results.py"
echo ""
