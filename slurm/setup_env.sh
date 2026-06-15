#!/usr/bin/env bash
# =============================================================================
# FL-Iroh — One-time HPC environment setup
#
# Usage:
#   bash slurm/setup_env.sh            # creates .venv with Python 3.11
#   bash slurm/setup_env.sh --conda    # creates conda env 'fl_iroh' instead
#
# After setup, submit experiments with:
#   bash slurm/submit.sh
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${REPO_ROOT}/.venv"
USE_CONDA=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --conda) USE_CONDA=1 ;;
        *) echo "[warn] Unknown option: $1" ;;
    esac
    shift
done

echo "=== FL-Iroh HPC environment setup ==="
echo "    REPO_ROOT = ${REPO_ROOT}"
cd "${REPO_ROOT}"

# ── Python environment ────────────────────────────────────────────────────────
if [[ -n "${USE_CONDA}" ]]; then
    if ! command -v conda &>/dev/null; then
        echo "[error] conda not found. Load the module first, e.g.:"
        echo "        module load anaconda3"
        exit 1
    fi
    # shellcheck source=/dev/null
    source "$(conda info --base)/etc/profile.d/conda.sh"
    if conda env list | grep -q '^fl_iroh '; then
        echo "[setup] Conda env 'fl_iroh' already exists — skipping creation"
    else
        conda create -n fl_iroh python=3.11 -y
    fi
    conda activate fl_iroh
    PYTHON="python"
else
    # Require Python 3.11+
    PYVER="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    PYMAJ="${PYVER%%.*}"
    PYMIN="${PYVER##*.}"
    if [[ "${PYMAJ}" -lt 3 ]] || { [[ "${PYMAJ}" -eq 3 ]] && [[ "${PYMIN}" -lt 9 ]]; }; then
        echo "[error] Python 3.9+ required (found ${PYVER})"
        echo "        Try: module load python/3.9  (or use --conda)"
        exit 1
    fi
    if [[ -d "${VENV}" ]]; then
        echo "[setup] .venv already exists — skipping creation"
    else
        echo "[setup] Creating virtualenv at ${VENV} ..."
        python3 -m venv "${VENV}"
    fi
    # shellcheck source=/dev/null
    source "${VENV}/bin/activate"
    PYTHON="python"
fi

# ── Upgrade build tools ───────────────────────────────────────────────────────
pip install --upgrade pip wheel setuptools

# ── PyTorch CPU-only wheels ───────────────────────────────────────────────────
# Uses the +cpu index to avoid pulling in CUDA libraries (~3 GB saved).
# Adjust the index URL for GPU nodes (remove --index-url to get the default).
echo "[setup] Installing PyTorch (CPU-only build) ..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# ── Project package + all dependencies ───────────────────────────────────────
echo "[setup] Installing fl-coap-iroh (with dev + baselines extras) ..."
pip install -e ".[dev,baselines]"

# ── CIFAR-10 dataset ──────────────────────────────────────────────────────────
echo "[setup] Downloading CIFAR-10 dataset ..."
"${PYTHON}" - <<'PYEOF'
import sys
from pathlib import Path

data_dir = Path("data")
data_dir.mkdir(exist_ok=True)

try:
    from fl_coap_iroh.data.partition import load_dataset
    load_dataset("cifar10", str(data_dir))
    print("[setup] CIFAR-10 ready.")
except Exception as exc:
    print(f"[warn]  CIFAR-10 download failed: {exc}")
    print("[warn]  Experiments will attempt to download on first run.")
    sys.exit(0)   # non-fatal
PYEOF

# ── Results directories ───────────────────────────────────────────────────────
mkdir -p \
    results/logs \
    results/metadata \
    results/e1 \
    results/e2 \
    results/e3 \
    results/e5 \
    results/e6 \
    results/e7 \
    results/e8
echo "[setup] Created results/ subdirectories."

# ── CmdStan (required for Prophet / E7 FedGAM) ───────────────────────────────
echo "[setup] Checking cmdstan for Prophet ..."
"${PYTHON}" - <<'PYEOF'
import sys
from pathlib import Path

cmdstan_home = Path.home() / ".cmdstan"
if cmdstan_home.exists() and any(cmdstan_home.iterdir()):
    print(f"[setup] cmdstan already installed at {cmdstan_home}")
    sys.exit(0)

print("[setup] Installing cmdstan (required for Prophet FedGAM) ...")
print("[setup] This may take 5-10 minutes on first run ...")
try:
    import cmdstanpy
    cmdstanpy.install_cmdstan()
    print("[setup] cmdstan installed OK")
except Exception as exc:
    print(f"[warn]  cmdstan install failed: {exc}")
    print("[warn]  Prophet will fall back to majority-class predictor in E7.")
    sys.exit(0)   # non-fatal — AirMLP experiments still work without cmdstan
PYEOF

# ── Smoke test ────────────────────────────────────────────────────────────────
echo ""
echo "[setup] Smoke test: importing fl_coap_iroh ..."
"${PYTHON}" -c "import fl_coap_iroh; print('[setup] Import OK')"

echo ""
echo "=== Setup complete ==="
echo ""
echo "  Environment:   ${VENV:-conda:fl_iroh}"
echo "  Submit jobs:   bash slurm/submit.sh"
echo "  Single task:   bash slurm/submit.sh --array=0"
echo "  Submit E7:     bash slurm/submit.sh --array=8"
echo "  Collect data:  python slurm/collect_results.py"
echo ""
echo "  FL_MOCK_IROH=1 sbatch slurm/run_experiments.sh"
echo "    → forces in-process mock transport (use if iroh UDP is blocked)"
