#!/usr/bin/env bash
# PC side of the live Flower-over-Tailscale run (server + local clients 1, 2),
# kept in one process tree so the hosting wsl.exe keeps the WSL VM alive.
# The Raspberry Pi client connects to <PC Tailscale IP>:8080 (forwarded to WSL).
#   usage: scripts/run_h2h_flower_pc_side.sh <out-dir> <rounds>
set -uo pipefail
cd "$(dirname "$0")/.."
OUT="${1:-results/h2h/flower}"; ROUNDS="${2:-30}"
mkdir -p "$OUT"
COMMON=(--dataset air_quality --partition geographic --n-clients 3 --rounds "$ROUNDS" --results-dir "$OUT")

.venv/bin/python -m experiments.e8_flower_tailscale --mode server --server-address 0.0.0.0:8080 \
    "${COMMON[@]}" > "$OUT/server.log" 2>&1 &
SERVER=$!
sleep 15
for c in 1 2; do
    .venv/bin/python -m experiments.e8_flower_tailscale --mode client --server-address 127.0.0.1:8080 \
        --client-id "$c" "${COMMON[@]}" > "$OUT/pc$c.log" 2>&1 &
done
wait "$SERVER"
wait
echo "PC side finished" >> "$OUT/server.log"
