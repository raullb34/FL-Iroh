#!/usr/bin/env bash
# scripts/apply_netem.sh — apply Linux tc-netem rules from a scenario YAML
#
# Usage (inside container or on host with iproute2):
#   apply_netem.sh <scenario.yaml> [interface]
#
# Requires: iproute2 (tc), python3, python3-yaml (for YAML parsing)
# Must run as root (or with CAP_NET_ADMIN).
#
# Safety: script is idempotent — it clears existing qdisc before adding.

set -euo pipefail

SCENARIO_FILE="${1:-}"
IFACE="${2:-eth0}"

if [[ -z "$SCENARIO_FILE" || ! -f "$SCENARIO_FILE" ]]; then
    echo "Usage: $0 <scenario.yaml> [interface=eth0]" >&2
    exit 1
fi

# Parse YAML with Python (minimal dep)
parse_field() {
    python3 -c "
import sys, yaml
doc = yaml.safe_load(open('$SCENARIO_FILE'))
key = '$1'
nested = key.split('.')
val = doc
for k in nested:
    val = val.get(k, {}) if isinstance(val, dict) else None
    if val is None:
        break
print(val or '', end='')
"
}

NAT=$(parse_field nat)
DELAY=$(parse_field netem.delay)
JITTER=$(parse_field netem.jitter)
CORR=$(parse_field netem.correlation)
LOSS=$(parse_field netem.loss)
BLOCK_UDP=$(parse_field firewall.block_udp)
ALLOW_TCP_443=$(parse_field firewall.allow_tcp_443)

echo "=== Applying scenario: $(parse_field name) on $IFACE ==="

# Clear any existing root qdisc
tc qdisc del dev "$IFACE" root 2>/dev/null || true

NETEM_CMD=""
if [[ -n "$DELAY" ]]; then
    NETEM_CMD="delay $DELAY"
    [[ -n "$JITTER" ]] && NETEM_CMD="$NETEM_CMD $JITTER"
    [[ -n "$CORR"   ]] && NETEM_CMD="$NETEM_CMD ${CORR}%"
fi
if [[ -n "$LOSS" ]]; then
    NETEM_CMD="$NETEM_CMD loss $LOSS"
fi

if [[ -n "$NETEM_CMD" ]]; then
    echo "  tc netem: $NETEM_CMD"
    tc qdisc add dev "$IFACE" root netem $NETEM_CMD
else
    echo "  No netem rules (LAN baseline)"
fi

# Firewall rules — strictly for fw443 scenario
if [[ "$BLOCK_UDP" == "True" ]]; then
    echo "  Blocking all UDP outbound (except port 443)"
    iptables -I OUTPUT -p udp ! --dport 443 -j DROP 2>/dev/null || true
fi

echo "=== Done ==="
