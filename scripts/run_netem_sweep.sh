#!/usr/bin/env bash
# E11 — degraded-network sweep (R1.8).
#
# Creates an isolated network namespace (no route to the host network, so no
# relay is reachable and every transfer uses the direct QUIC path), applies a
# tc-netem qdisc on its loopback for each condition, and runs
# experiments/e11_degraded_network.py inside it as an unprivileged user.
#
# Must run as root (e.g. `wsl -u root`, or sudo on a Linux host):
#   sudo scripts/run_netem_sweep.sh [rounds] [user]
# Results: results/e11/<condition>/ and results/e11/e11_summary.csv
set -euo pipefail

ROUNDS="${1:-20}"
RUN_AS="${2:-$(getent passwd 1000 | cut -d: -f1)}"
NS=fl-e11
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PY="$REPO/.venv/bin/python"

# name | netem parameters (one-way; RTT = 2 x delay, loss applies per direction)
# Note: netem jitter reorders packets unless a rate is set (packets are then
# serialised in order); reordering is not typical of real links and QUIC treats
# it as loss, so jitter conditions add a (non-binding) 1 Gbit/s rate.
CONDITIONS=(
  "baseline|"
  "delay_25|delay 25ms"
  "delay_50|delay 50ms"
  "delay_100|delay 100ms"
  "delay_250|delay 250ms"
  "jitter_50_10|delay 50ms 10ms rate 1gbit"
  "jitter_100_30|delay 100ms 30ms rate 1gbit"
  "jitter_50_10_reorder|delay 50ms 10ms distribution normal"
  "loss_1|loss 1%"
  "loss_5|loss 5%"
  "loss_10|loss 10%"
  "loss_20|loss 20%"
  "rate_10mbit|delay 10ms rate 10mbit"
  "rate_2mbit|delay 10ms rate 2mbit"
  "rate_512kbit|delay 10ms rate 512kbit"
  "rural_lte|delay 30ms 10ms loss 2% rate 5mbit"
  "harsh_rural|delay 75ms 25ms loss 10% rate 1mbit"
)

cleanup() { ip netns del "$NS" 2>/dev/null || true; }
trap cleanup EXIT
cleanup
ip netns add "$NS"
ip netns exec "$NS" ip link set lo up
# A non-loopback address for Iroh to advertise; traffic to it is routed via lo.
ip netns exec "$NS" ip link add dummy0 type dummy
ip netns exec "$NS" ip addr add 10.77.0.1/24 dev dummy0
ip netns exec "$NS" ip link set dummy0 up

cd "$REPO"
for entry in "${CONDITIONS[@]}"; do
  name="${entry%%|*}"; spec="${entry#*|}"
  ip netns exec "$NS" tc qdisc del dev lo root 2>/dev/null || true
  if [[ -n "$spec" ]]; then
    # shellcheck disable=SC2086
    ip netns exec "$NS" tc qdisc add dev lo root netem limit 10000 $spec
  fi
  echo "=== $name: ${spec:-none} ==="
  ip netns exec "$NS" tc qdisc show dev lo
  mkdir -p "results/e11/$name"; chown -R "$RUN_AS" results/e11
  ip netns exec "$NS" runuser -u "$RUN_AS" -- "$PY" -m experiments.e11_degraded_network \
      --condition "$name" --netem "${spec:-none}" --rounds "$ROUNDS" \
      > "results/e11/$name/run.log" 2>&1 || echo "  (condition $name exited non-zero)"
  tail -1 "results/e11/$name/run.log"
done

runuser -u "$RUN_AS" -- "$PY" - <<'PY'
import csv, json, pathlib
rows = [json.loads(p.read_text()) for p in sorted(pathlib.Path("results/e11").glob("*/e11_summary.json"))]
if rows:
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open("results/e11/e11_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print(f"wrote results/e11/e11_summary.csv ({len(rows)} conditions)")
PY
