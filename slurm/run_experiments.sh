#!/usr/bin/env bash
# =============================================================================
# FL-Iroh — SLURM array job
#
# Runs experiments E1-E6 as a SLURM array of tasks.
# Submit with:  bash slurm/submit.sh
#
# Task mapping:
#   0 — E1  communication microbenchmark        (~1-2 h)
#   1 — E2  FL convergence, IID                 (~6-10 h)
#   2 — E2  FL convergence, Dirichlet α=0.1     (~6-10 h)
#   3 — E2  FL convergence, Dirichlet α=0.5+1.0 (~12-18 h)
#   4 — E5  churn resilience  (0/10/30/50 %)    (~20-28 h)
#   5 — E3  NAT traversal mock + E6 CoAP overhead (~1-2 h)
#   6 — E2  crop IID + Dirichlet α=0.1/0.5/1.0  (~12-18 h)
#   7 — E5  churn resilience, crop dataset       (~20-28 h)
#   8 — E7  air quality FL: 12 configs (AirMLP/AirLSTM+FedAvg, Prophet+FedGAM/FedAvg) (~2-4 h)
#   9 — E8  Flower+Tailscale baseline (sim) + software energy estimation (~1-2 h)
#  10 — Multi-seed replication: E2 (IID+α=0.1) and E7 over all seeds (F2/CIs) (~24-36 h)
#
# NOTE — paper ↔ code mapping (the paper renumbers experiments; code is stable):
#   paper E4 (churn)        ↔ experiments.e5_churn        ↔ results/e5/
#   paper E5 (CoAP)         ↔ experiments.e6_coap_overhead ↔ results/e6/
#   paper E6 (air quality)  ↔ experiments.e7_air_quality_fl ↔ results/e7/
# =============================================================================
#SBATCH --job-name=fl_iroh_exp
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=36:00:00
#SBATCH --array=0-10
#SBATCH --output=results/logs/slurm_%A_%a.out
#SBATCH --error=results/logs/slurm_%A_%a.err

set -uo pipefail

# SLURM_SUBMIT_DIR is set by SLURM to the directory where sbatch was called.
# Fall back to BASH_SOURCE-relative path for local/interactive runs.
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
RESULTS_ROOT="${REPO_ROOT}/results"
VENV="${REPO_ROOT}/.venv"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
ARRAY_JOB_ID="${SLURM_ARRAY_JOB_ID:-local}"
JOB_ID="${ARRAY_JOB_ID}_${TASK_ID}"

mkdir -p "${RESULTS_ROOT}/logs" "${RESULTS_ROOT}/metadata"

# ── Activate environment ─────────────────────────────────────────────────────
if [[ -f "${VENV}/bin/activate" ]]; then
    # shellcheck source=/dev/null
    source "${VENV}/bin/activate"
elif command -v conda &>/dev/null && conda env list 2>/dev/null | grep -q fl_iroh; then
    # shellcheck source=/dev/null
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate fl_iroh
else
    echo "[error] No .venv found and no conda env 'fl_iroh'."
    echo "        Run:  bash slurm/setup_env.sh"
    exit 1
fi

# ── Thread counts ────────────────────────────────────────────────────────────
NCPU="${SLURM_CPUS_PER_TASK:-16}"
export OMP_NUM_THREADS="${NCPU}"
export MKL_NUM_THREADS="${NCPU}"
export OPENBLAS_NUM_THREADS="${NCPU}"
export NUMEXPR_NUM_THREADS="${NCPU}"
# Torch inter-op and intra-op threads
export FL_TORCH_THREADS="${NCPU}"

# If iroh cannot bind UDP sockets on this HPC node, set FL_MOCK_IROH=1 in your
# sbatch environment.  Mock mode runs an in-process asyncio transport with
# realistic simulated delays so E2/E5 still produce comparable convergence data.
# FL_MOCK_IROH is intentionally NOT set here — iroh is used by default.

cd "${REPO_ROOT}"

# ── Metadata helpers ─────────────────────────────────────────────────────────
GIT_COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
START_TS="$(date +%s)"
META_FILE="${RESULTS_ROOT}/metadata/job_${JOB_ID}.json"

write_metadata() {
    local status="${1:-UNKNOWN}"
    local end_ts
    end_ts="$(date +%s)"
    cat > "${META_FILE}" <<ENDJSON
{
  "job_id":      "${JOB_ID}",
  "array_id":    "${ARRAY_JOB_ID}",
  "task_id":     ${TASK_ID},
  "git_commit":  "${GIT_COMMIT}",
  "status":      "${status}",
  "start_epoch": ${START_TS},
  "end_epoch":   ${end_ts},
  "elapsed_sec": $(( end_ts - START_TS )),
  "ncpu":        ${NCPU},
  "hostname":    "$(hostname -s)"
}
ENDJSON
    echo "[meta] $(date '+%H:%M:%S')  status=${status}"
}

trap 'write_metadata "FAILED"'    ERR
trap 'write_metadata "PREEMPTED"; exit 143' TERM

# ── Experiment runner ─────────────────────────────────────────────────────────
# Usage: run_exp LABEL RESULTS_SUBDIR PYTHON_MODULE [ARGS...]
#
# - Creates <subdir>/<label>.running sentinel while running
# - Tees stdout+stderr to <subdir>/<label>.log
# - Writes <label>.ok or <label>.failed on exit (does NOT abort the job)
run_exp() {
    local label="$1";  shift
    local subdir="$1"; shift
    local module="$1"; shift

    local out_dir="${RESULTS_ROOT}/${subdir}"
    mkdir -p "${out_dir}"

    local t0
    t0="$(date +%s)"
    local sep="========================================================================"
    printf '\n%s\n START  %-36s  %s\n%s\n' \
        "${sep}" "${label}" "$(date '+%H:%M:%S')" "${sep}"

    echo "${label} started at $(date -Iseconds)" > "${out_dir}/${label}.running"

    if python -m "${module}" --results-dir "${out_dir}" "$@" \
            2>&1 | tee "${out_dir}/${label}.log"; then
        local elapsed=$(( $(date +%s) - t0 ))
        echo "elapsed=${elapsed}s  status=OK" > "${out_dir}/${label}.ok"
        rm -f "${out_dir}/${label}.running"
        printf ' END    %-36s  elapsed=%ds  [OK]\n' "${label}" "${elapsed}"
    else
        local rc=$?
        local elapsed=$(( $(date +%s) - t0 ))
        echo "elapsed=${elapsed}s  rc=${rc}  status=FAILED" > "${out_dir}/${label}.failed"
        rm -f "${out_dir}/${label}.running"
        printf ' END    %-36s  elapsed=%ds  [FAILED rc=%d]\n' "${label}" "${elapsed}" "${rc}"
        echo "[warn]  ${label} failed — continuing remaining experiments in this task"
    fi
}

# ── Header ───────────────────────────────────────────────────────────────────
SEP="════════════════════════════════════════════════════════════════════════"
printf '%s\n' "${SEP}"
printf '  FL-Iroh  task=%-2s  job=%s  git=%s\n' "${TASK_ID}" "${JOB_ID}" "${GIT_COMMIT}"
printf '  host=%-20s  CPUs=%s  %s\n' "$(hostname -s)" "${NCPU}" "$(date '+%F %T')"
printf '%s\n\n' "${SEP}"

write_metadata "RUNNING"

# ── Dispatch ─────────────────────────────────────────────────────────────────
case "${TASK_ID}" in

    0)  # E1 — Communication microbenchmark
        # Measures Iroh/QUIC throughput vs HTTP/2 baseline at several payload sizes.
        run_exp "e1_microbenchmark" "e1" \
            "experiments.e1_microbenchmark" \
            --n-iter 30
        ;;

    1)  # E2 — FL convergence, IID data partition
        export FL_MOCK_IROH=1   # mock transport: asyncio.Queue bypasses iroh/Tokio threading
        run_exp "e2_iid" "e2" \
            "experiments.e2_centralized_fl" \
            --rounds 100 --n-clients 10 --dataset cifar10 \
            --partition iid
        ;;

    2)  # E2 — FL convergence, non-IID Dirichlet α=0.1 (high heterogeneity)
        export FL_MOCK_IROH=1
        run_exp "e2_noniid_01" "e2" \
            "experiments.e2_centralized_fl" \
            --rounds 100 --n-clients 10 --dataset cifar10 \
            --partition dirichlet --alpha 0.1
        ;;

    3)  # E2 — FL convergence, non-IID Dirichlet α=0.5 then α=1.0
        export FL_MOCK_IROH=1
        run_exp "e2_noniid_05" "e2" \
            "experiments.e2_centralized_fl" \
            --rounds 100 --n-clients 10 --dataset cifar10 \
            --partition dirichlet --alpha 0.5

        run_exp "e2_noniid_10" "e2" \
            "experiments.e2_centralized_fl" \
            --rounds 100 --n-clients 10 --dataset cifar10 \
            --partition dirichlet --alpha 1.0
        ;;

    4)  # E5 — Churn resilience (0 / 10 / 30 / 50 % per-round churn)
        export FL_MOCK_IROH=1
        run_exp "e5_churn" "e5" \
            "experiments.e5_churn" \
            --rounds 100 --n-clients 10 --dataset cifar10 \
            --churn-rates "0.0,0.1,0.3,0.5"
        ;;

    5)  # E3 (mock) — NAT traversal connection setup for 5 scenarios
        # NOTE: Real NAT scenarios (requiring Docker + netem overlays) must be
        # run separately via:  docker compose -f docker-compose.nat1.yml up
        # The --mock flag runs two in-process iroh nodes (LAN-equivalent).
        for SCENARIO in net_lan net_nat1 net_nat2 net_cgnat net_fw443; do
            run_exp "e3_${SCENARIO}" "e3" \
                "experiments.e3_nat_traversal" \
                --mock --n-iter 30 --scenario "${SCENARIO}"
        done

        # E6 — CoAP discovery overhead
        run_exp "e6_coap_overhead" "e6" \
            "experiments.e6_coap_overhead" \
            --n-iter 30
        ;;

    6)  # E2 — FL convergence with Crop Recommendation dataset (all partitions)
        export FL_MOCK_IROH=1
        run_exp "e2_crop_iid" "e2" \
            "experiments.e2_centralized_fl" \
            --rounds 100 --n-clients 10 --dataset crop \
            --partition iid

        run_exp "e2_crop_noniid_01" "e2" \
            "experiments.e2_centralized_fl" \
            --rounds 100 --n-clients 10 --dataset crop \
            --partition dirichlet --alpha 0.1

        run_exp "e2_crop_noniid_05" "e2" \
            "experiments.e2_centralized_fl" \
            --rounds 100 --n-clients 10 --dataset crop \
            --partition dirichlet --alpha 0.5

        run_exp "e2_crop_noniid_10" "e2" \
            "experiments.e2_centralized_fl" \
            --rounds 100 --n-clients 10 --dataset crop \
            --partition dirichlet --alpha 1.0
        ;;

    7)  # E5 — Churn resilience with Crop Recommendation dataset
        export FL_MOCK_IROH=1
        run_exp "e5_crop_churn" "e5" \
            "experiments.e5_churn" \
            --rounds 100 --n-clients 10 --dataset crop \
            --churn-rates "0.0,0.1,0.3,0.5"
        ;;

    8)  # E7 — Air quality FL: 7-day-ahead ICA forecasting (outdoor, CyL stations)
        export FL_MOCK_IROH=1
        # Neural models (AirMLP tabular + AirLSTM sequential) + FedAvg
        run_exp "e7_neural" "e7" \
            "experiments.e7_air_quality_fl" \
            --configs airmlp_geographic airmlp_iid airlstm_geographic airlstm_iid \
            --rounds 50

        # Prophet GAM: FedGAM vs FedAvg ablation (same model, both partitions,
        # full-history and 1-year variants) — 8 configs.  Real Stan MAP requires
        # cmdstan on this node; otherwise ProphetWrapper falls back to majority-class.
        run_exp "e7_prophet" "e7" \
            "experiments.e7_air_quality_fl" \
            --configs prophet_fedgam_geographic prophet_fedgam_iid \
                      prophet_fedavg_geographic prophet_fedavg_iid \
                      prophet_fedgam_1yr_geographic prophet_fedgam_1yr_iid \
                      prophet_fedavg_1yr_geographic prophet_fedavg_1yr_iid \
            --rounds 10
        ;;

    9)  # E8 — Flower (flwr) over Tailscale baseline + software energy estimation
        # F3: transport-isolated comparison against FL-Iroh using the SAME model
        # and training loop.  --mode sim runs the in-process Flower simulation
        # (rounds, final acc, payload bytes/round, wall time/round).  The 8-axis
        # operational comparison table is always written to results/e8/.
        export FL_MOCK_IROH=1
        run_exp "e8_flower_iid" "e8" \
            "experiments.e8_flower_tailscale" \
            --mode sim --dataset crop --partition iid \
            --n-clients 10 --rounds 50

        run_exp "e8_flower_noniid" "e8" \
            "experiments.e8_flower_tailscale" \
            --mode sim --dataset crop --partition noniid --alpha 0.1 \
            --n-clients 10 --rounds 50

        # F6: software energy estimation. Post-process per-round CPU times into
        # energy (mWh) under the RPi4B edge model (TDP=7W, 4 cores). Override
        # --tdp/--cores for x86 framing if desired. Skips silently if the source
        # CSV or its cpu-time column is absent.
        for SPEC in "e2/e2_iid_fl_metrics.csv:e2/e2_iid_energy.csv" \
                    "e7/e7_airmlp_iid_fl_metrics.csv:e7/e7_airmlp_iid_energy.csv"; do
            SRC="${RESULTS_ROOT}/${SPEC%%:*}"
            DST="${RESULTS_ROOT}/${SPEC##*:}"
            if [[ -f "${SRC}" ]] && head -1 "${SRC}" | grep -q "cpu"; then
                python -m scripts.energy_estimate \
                    --csv "${SRC}" --cpu-col cpu_time_s --rounds-col round \
                    --tdp 7 --cores 4 --out "${DST}" \
                    2>&1 | tee "${RESULTS_ROOT}/e8/energy_$(basename "${DST}" .csv).log" \
                    || echo "[warn] energy estimation failed for ${SRC}"
            else
                echo "[skip] energy: ${SRC} missing or has no cpu-time column"
            fi
        done
        ;;

    10) # Multi-seed replication for confidence intervals (F2)
        # Runs the canonical heterogeneity configs over every seed in
        # seeds.yaml 'replicate_seeds'. aggregate_ci.py then produces ±CI tables.
        export FL_MOCK_IROH=1
        run_exp "e2_iid_seeds" "e2" \
            "experiments.e2_centralized_fl" \
            --rounds 100 --n-clients 10 --dataset cifar10 \
            --partition iid --all-seeds

        run_exp "e2_noniid_01_seeds" "e2" \
            "experiments.e2_centralized_fl" \
            --rounds 100 --n-clients 10 --dataset cifar10 \
            --partition dirichlet --alpha 0.1 --all-seeds

        run_exp "e7_seeds" "e7" \
            "experiments.e7_air_quality_fl" \
            --configs airmlp_geographic airmlp_iid \
                      prophet_fedgam_geographic prophet_fedavg_geographic \
            --rounds 50 --all-seeds
        ;;

    *)  echo "ERROR: unexpected SLURM_ARRAY_TASK_ID=${TASK_ID}"
        exit 1
        ;;
esac

# ── Footer ───────────────────────────────────────────────────────────────────
trap - ERR
write_metadata "OK"
echo ""
printf '%s\n' "${SEP}"
printf '  Task %s finished  %s\n' "${TASK_ID}" "$(date '+%F %T')"
printf '%s\n' "${SEP}"
