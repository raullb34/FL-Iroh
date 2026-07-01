#!/usr/bin/env bash
# E3 real-hardware CLIENT launcher (Raspberry Pi / Linux).
#
# The RPi runs the CLIENT role (it has a working torch) and connects to the PC
# server endpoint, sending N small tensors while recording conn_type +
# active_addr per attempt with the fixed _active_wire_addr classifier.
#
# Usage (from repo root on the RPi):
#   scripts/_e3_client.sh net_nat1 30 '<server_endpoint JSON pasted from PC>'
#   scripts/_e3_client.sh net_nat1 30 @results/e3/server_endpoint.json   # if you scp'd it
#
# Outputs (results/e3/):
#   e3_nat_<scenario>_client.csv   - per-iter conn_type + active_addr
set -euo pipefail
cd "$(dirname "$0")/.." || exit 1

SCEN="${1:-net_nat1}"
NITER="${2:-30}"
EP="${3:-@results/e3/server_endpoint.json}"

export FL_CONN_DEBUG_ADDRS=1
unset FL_MOCK_IROH || true

echo "== E3 CLIENT (RPi) scenario=$SCEN n_iter=$NITER =="

exec ./.venv/bin/python -u -m experiments.e3_nat_traversal \
    --role client --n-iter "$NITER" --scenario "$SCEN" \
    --server-endpoint "$EP"
