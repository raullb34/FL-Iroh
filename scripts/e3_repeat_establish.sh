#!/usr/bin/env bash
# =============================================================================
# E3 — repeated COLD-START connection establishment (client side, RPi/Linux)
#
# Motivation (reviewer finding): the published E3 table reports per-TRANSFER
# statistics over a single established session per scenario (~4-12 distinct
# QUIC connections in total), which cannot estimate a population-level
# hole-punching success probability. This script runs REPS fully independent
# cold-start establishments per scenario: each repetition spawns a FRESH
# python process (fresh Iroh endpoint, fresh NAT mapping discovery) with
# --n-iter 1, so every run is one independent connection-establishment event.
#
# Per-connection metrics recorded (from e3_nat_traversal per-iter CSV):
#   conn_type (direct/relay), active_addr, establishment outcome + timing.
#
# Prerequisites:
#   * PC side: keep the server running via  scripts/_e3_server.sh <scenario>
#   * copy/paste the server endpoint JSON as with scripts/_e3_client.sh
#
# Usage (from repo root on the RPi):
#   scripts/e3_repeat_establish.sh net_cgnat 30 @results/e3/server_endpoint.json
#   scripts/e3_repeat_establish.sh net_nat2  30 '<server_endpoint JSON>'
#
# Outputs:
#   results/e3/establish/<scenario>/run_NNN/  - one per-iter CSV per cold start
#   results/e3/establish/<scenario>/summary.csv - outcome + wall time per run
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

SCEN="${1:-net_nat1}"
REPS="${2:-30}"
EP="${3:-@results/e3/server_endpoint.json}"
PAUSE="${4:-20}"   # seconds between runs, lets NAT mappings expire

export FL_CONN_DEBUG_ADDRS=1
unset FL_MOCK_IROH || true

OUT_ROOT="results/e3/establish/${SCEN}"
mkdir -p "${OUT_ROOT}"
SUMMARY="${OUT_ROOT}/summary.csv"
[[ -f "${SUMMARY}" ]] || echo "run,outcome,wall_s,results_dir" > "${SUMMARY}"

echo "== E3 repeated cold-start establishment: scenario=${SCEN} reps=${REPS} pause=${PAUSE}s =="

for i in $(seq 1 "${REPS}"); do
    RUN_DIR="${OUT_ROOT}/run_$(printf '%03d' "${i}")"
    mkdir -p "${RUN_DIR}"
    echo "--- run ${i}/${REPS} ($(date '+%H:%M:%S')) ---"
    t0=$(date +%s)
    if ./.venv/bin/python -u -m experiments.e3_nat_traversal \
            --role client --n-iter 1 --scenario "${SCEN}" \
            --server-endpoint "${EP}" \
            --results-dir "${RUN_DIR}" \
            > "${RUN_DIR}/client.log" 2>&1; then
        outcome="ok"
    else
        outcome="failed"
    fi
    wall=$(( $(date +%s) - t0 ))
    echo "${i},${outcome},${wall},${RUN_DIR}" >> "${SUMMARY}"
    echo "    ${outcome} (${wall}s)"
    # fresh NAT mapping for the next cold start
    [[ "${i}" -lt "${REPS}" ]] && sleep "${PAUSE}"
done

echo ""
echo "== summary (${SUMMARY}) =="
ok=$(grep -c ',ok,' "${SUMMARY}" || true)
fail=$(grep -c ',failed,' "${SUMMARY}" || true)
echo "   ok=${ok}  failed=${fail}"
echo "   per-connection conn_type/active_addr: ${OUT_ROOT}/run_*/e3_nat_${SCEN}_client.csv"
