#!/usr/bin/env bash
# E3 real server launcher (WSL). Runs the iroh server, prints endpoint JSON,
# and waits for client connections from the RPi.
cd "$(dirname "$0")/.." || exit 1
unset FL_MOCK_IROH
export FL_CONN_DEBUG_ADDRS=1
SCEN="${1:-net_cgnat}"
NITER="${2:-30}"
exec ./.venv/bin/python -u -m experiments.e3_nat_traversal --role server --n-iter "$NITER" --scenario "$SCEN"
