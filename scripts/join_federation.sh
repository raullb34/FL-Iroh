#!/usr/bin/env bash
# =============================================================================
# Join an FL-Iroh federation from any Linux/macOS machine or Raspberry Pi,
# behind any NAT/CGNAT — no port forwarding, VPN or public IP needed.
#
#   1) first time:   scripts/join_federation.sh --keygen <client-id>
#                    → prints this node's NodeId; send it to the coordinator,
#                      who adds it to the server allow-list
#   2) then:         scripts/join_federation.sh <server_endpoint.json> <client-id> <province-idx>
#
# The server endpoint JSON (NodeId + relay URL) is produced by
# scripts/run_wan_server.sh and can be shared by e-mail/chat: it contains no
# IP that needs to be reachable.  Task: air-quality ICA forecasting, one
# province per client (geographic partition), AirMLP model.
#
# Env overrides: PYTHON (default python3), FL_ROUNDS (default 30),
#                FL_RESOURCE_MONITOR (default 1: CPU/RSS/energy per round)
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON:-python3}"
VENV=".venv-edge"

setup() {
    if [[ ! -x "$VENV/bin/python" ]]; then
        echo "== creating $VENV (first run only) =="
        "$PY" -m venv "$VENV"
        "$VENV/bin/pip" install -q --upgrade pip
        # Always the CPU-only torch wheel.  On aarch64 the default PyPI wheel is a
        # CUDA/NVPL build for ARMv8.2+ servers and dies with "Illegal instruction"
        # in BatchNorm backward on Raspberry Pi (Cortex-A53/A72, ARMv8.0).
        "$VENV/bin/pip" install -q torch --index-url https://download.pytorch.org/whl/cpu
        "$VENV/bin/pip" install -q -e ".[edge]"
    fi
}

if [[ "${1:-}" == "--keygen" ]]; then
    CID="${2:?usage: $0 --keygen <client-id>}"
    setup
    mkdir -p keys
    echo "NodeId for ${CID} (send this to the coordinator):"
    "$VENV/bin/fl-keygen" "keys/${CID}.key"
    exit 0
fi

EP="${1:?usage: $0 <server_endpoint.json> <client-id> <province-idx>}"
CID="${2:?client-id}"
IDX="${3:?province index (0-8)}"
setup
[[ -f "keys/${CID}.key" ]] || { echo "run '$0 --keygen ${CID}' first"; exit 1; }

OUT="results/wan/${CID}"
mkdir -p "$OUT"
[[ -f "$OUT/nat_${CID}.json" ]] || "$VENV/bin/python" scripts/nat_classify.py --label "$CID" --out "$OUT" || true

export FL_RESOURCE_MONITOR="${FL_RESOURCE_MONITOR:-1}"
export FL_CLIENT_IDLE_TIMEOUT_S="${FL_CLIENT_IDLE_TIMEOUT_S:-900}"
unset FL_MOCK_IROH || true
exec "$VENV/bin/fl-node" \
    --node-id "$CID" --secret-key-file "keys/${CID}.key" \
    --server-endpoint-file "$EP" \
    --dataset air_quality --partition geographic --partition-idx "$IDX" \
    --rounds "${FL_ROUNDS:-30}" --scenario wan --results-dir "$OUT" \
    2>&1 | tee "$OUT/client.log"
