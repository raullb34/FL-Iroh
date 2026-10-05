#!/usr/bin/env bash
# =============================================================================
# Coordinator side of a real multi-device FL-Iroh federation over the WAN.
#
#   scripts/run_wan_server.sh <allowlist.txt> <min-clients> [rounds]
#
# allowlist.txt: one client NodeId per line (from `join_federation.sh --keygen`).
# Writes results/wan/server/server_endpoint.json — share it with the clients.
# The server may itself sit behind NAT/CGNAT: clients register over Iroh
# (ALPN fl-ctrl/1) by NodeId, not over CoAP by IP.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

ALLOW="${1:?usage: $0 <allowlist.txt> <min-clients> [rounds]}"
MIN="${2:?min clients}"
ROUNDS="${3:-30}"
OUT="results/wan/server"
mkdir -p "$OUT" keys
PYBIN="${PYBIN:-.venv/bin}"

unset FL_MOCK_IROH || true
export FL_CONN_CLASSIFY_SETTLE_S="${FL_CONN_CLASSIFY_SETTLE_S:-3}"
exec "$PYBIN/fl-server" \
    --node-id server --secret-key-file keys/server.key --allowlist-file "$ALLOW" \
    --dataset air_quality --partition geographic --min-clients "$MIN" \
    --rounds "$ROUNDS" --wait-clients-s 1800 --scenario wan --results-dir "$OUT" \
    2>&1 | tee "$OUT/server.log"
