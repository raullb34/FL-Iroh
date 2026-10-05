#!/usr/bin/env bash
# PC side of the WAN federation run (server + local clients pc1, pc2), kept in
# ONE foreground process tree so that the hosting wsl.exe keeps the WSL VM
# alive until the federation ends (WSL shuts the VM down when no wsl.exe
# session remains, killing detached processes).
#   usage: scripts/run_wan_pc_side.sh <out-dir> <min-clients> <rounds>
set -uo pipefail
cd "$(dirname "$0")/.."
OUT="${1:-results/wan}"; MIN="${2:-2}"; ROUNDS="${3:-30}"
mkdir -p "$OUT/server"
unset FL_MOCK_IROH
export FL_CONN_CLASSIFY_SETTLE_S=3 FL_CONNECT_TIMEOUT_S=20 FL_CONNECT_ATTEMPTS=2 FL_SEND_RETRIES=1

.venv/bin/fl-server --node-id server --secret-key-file keys/server.key \
    --allowlist-file keys/allow_fl.txt --dataset air_quality --partition geographic \
    --min-clients "$MIN" --rounds "$ROUNDS" --wait-clients-s 1800 --coap-port 5990 \
    --scenario wan --results-dir "$OUT/server" > "$OUT/server/server.log" 2>&1 &
SERVER=$!
until [[ -f "$OUT/server/server_endpoint.json" ]]; do sleep 2; done

for c in 1 2; do
    FL_RESOURCE_MONITOR=1 FL_CLIENT_IDLE_TIMEOUT_S=900 .venv/bin/fl-node --node-id "pc$c" \
        --secret-key-file "keys/pc$c.key" --server-endpoint-file "$OUT/server/server_endpoint.json" \
        --dataset air_quality --partition geographic --partition-idx "$c" --rounds "$ROUNDS" \
        --coap-port "599$c" --scenario wan --results-dir "$OUT/pc$c" > "$OUT/pc$c.log" 2>&1 &
done
wait "$SERVER"
wait
echo "PC side finished" >> "$OUT/server/server.log"
