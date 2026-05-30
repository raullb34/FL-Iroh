#!/usr/bin/env python3
"""
scripts/churn_simulator.py — Simulate client churn during FL experiments.

Reads a scenario YAML file for churn parameters, then randomly pauses/unpauses
Docker containers on each FL round boundary, as signalled by a shared file or
a simple UDP heartbeat from the server.

Usage::
    python scripts/churn_simulator.py \\
        --scenario scenarios/net_churn30.yaml \\
        --clients fl-client-1,fl-client-2,fl-client-3,fl-client-4,fl-client-5 \\
        --rounds 50 \\
        --round-duration-sec 60 \\
        --seed 456

Signal file protocol:
    The FL server writes  ``results/round_<N>.signal``  at the start of each
    aggregation step.  This script watches for new files and triggers churn.
    (Alternative: use --poll-interval for a time-based approximation.)
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import random
import sys
import time
from pathlib import Path

import yaml

log = logging.getLogger("churn_simulator")
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)


def load_scenario(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


async def docker_pause(container: str) -> None:
    proc = await asyncio.create_subprocess_exec(
        "docker", "pause", container,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        log.warning("pause %s failed: %s", container, stderr.decode().strip())
    else:
        log.info("⏸  Paused %s", container)


async def docker_unpause(container: str) -> None:
    proc = await asyncio.create_subprocess_exec(
        "docker", "unpause", container,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        log.warning("unpause %s failed: %s", container, stderr.decode().strip())
    else:
        log.info("▶️  Unpaused %s", container)


async def run_churn(
    clients         : list[str],
    churn_rate      : float,
    resume_after    : int,
    rounds          : int,
    round_duration  : float,
    signal_dir      : str,
    rng             : random.Random,
) -> None:
    paused_until: dict[str, int] = {}   # container → round when it should resume

    for round_num in range(1, rounds + 1):
        # Wait for round signal file or use time-based polling
        sig = Path(signal_dir) / f"round_{round_num}.signal"
        deadline = time.monotonic() + round_duration
        while not sig.exists() and time.monotonic() < deadline:
            await asyncio.sleep(1.0)

        log.info("=== Round %d ===", round_num)

        # Resume containers that have waited long enough
        for name, resume_at in list(paused_until.items()):
            if round_num >= resume_at:
                await docker_unpause(name)
                del paused_until[name]

        # Determine churned set for this round
        available = [c for c in clients if c not in paused_until]
        n_churn = max(0, round(len(available) * churn_rate))
        churned = rng.sample(available, min(n_churn, len(available)))

        for name in churned:
            await docker_pause(name)
            paused_until[name] = round_num + resume_after

    # Resume all at the end
    for name in list(paused_until):
        await docker_unpause(name)


def main() -> None:
    parser = argparse.ArgumentParser(description="FL churn simulator")
    parser.add_argument("--scenario",            default="scenarios/net_churn30.yaml")
    parser.add_argument("--clients",             required=True,
                        help="Comma-separated Docker container names")
    parser.add_argument("--rounds",              type=int, default=50)
    parser.add_argument("--round-duration-sec",  type=float, default=60.0,
                        help="Fallback round duration if signal files are absent")
    parser.add_argument("--signal-dir",          default="results",
                        help="Directory watched for round_N.signal files")
    parser.add_argument("--seed",                type=int, default=456)
    args = parser.parse_args()

    scenario = load_scenario(args.scenario)
    churn_cfg = scenario.get("churn", {})
    churn_rate   = float(churn_cfg.get("rate", 0.30))
    resume_after = int(churn_cfg.get("resume_after", 1))

    client_list = [c.strip() for c in args.clients.split(",") if c.strip()]
    rng = random.Random(args.seed)

    log.info(
        "Starting churn: %d clients, %.0f%% per round, resumes after %d round(s)",
        len(client_list), churn_rate * 100, resume_after,
    )

    asyncio.run(run_churn(
        clients        = client_list,
        churn_rate     = churn_rate,
        resume_after   = resume_after,
        rounds         = args.rounds,
        round_duration = args.round_duration_sec,
        signal_dir     = args.signal_dir,
        rng            = rng,
    ))


if __name__ == "__main__":
    main()
