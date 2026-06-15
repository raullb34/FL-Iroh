"""
F6 — Software energy estimation for FL rounds (no hardware instrumentation).

Estimates the electrical energy consumed by computation using the standard
"TDP share x CPU-time" software model (as used by CodeCarbon / Green-Algorithms):

    power_per_core  = TDP / n_physical_cores          [W]
    energy_J        = power_per_core * cpu_core_seconds * PUE
    energy_mWh      = energy_J / 3.6

where ``cpu_core_seconds`` is the *process* CPU time (user+system), i.e. how many
core-seconds the workload actually burned. This attributes the per-core share of
the package TDP to the work done, and is the accepted software-only proxy when
RAPL / external power meters are unavailable.

This module exposes:

  * ``estimate_energy(cpu_core_seconds, tdp_watts, n_cores, pue) -> dict``
        pure analytic estimate (no dependencies).
  * ``EnergyMeter``  — context manager (psutil) that measures a code block's
        process CPU time and converts it to an energy estimate.
  * a CLI to (a) wrap a command, (b) post-process a results CSV, or
        (c) compute from explicit --cpu-seconds.

Defaults target a Raspberry Pi 4B class node (TDP ~= 7 W over 4 cores) to match
the solar-powered edge framing; override --tdp / --cores for x86 HPC nodes.

Examples
--------
    # explicit number
    python -m scripts.energy_estimate --cpu-seconds 42.0 --tdp 7 --cores 4

    # wrap a command and measure it
    python -m scripts.energy_estimate --tdp 7 --cores 4 \\
        --wrap "python -m experiments.e2_centralized_fl --rounds 16"

    # post-process a metrics CSV that has a cpu_time column, per round
    python -m scripts.energy_estimate --csv results/e2/e2_convergence.csv \\
        --cpu-col cpu_time_s --rounds-col round --tdp 7 --cores 4 \\
        --out results/e2/e2_energy.csv
"""
from __future__ import annotations

import argparse
import csv
import logging
import time
from pathlib import Path
from typing import Optional

log = logging.getLogger("energy_estimate")
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)

# Defaults: Raspberry Pi 4B (Cortex-A72 x4) ~= 7 W package under load.
DEFAULT_TDP_W = 7.0
DEFAULT_CORES = 4
# Power Usage Effectiveness / supply overhead (1.0 = bare SoC, no overhead).
DEFAULT_PUE = 1.0

JOULES_PER_MWH = 3.6  # 1 mWh = 3.6 J


def estimate_energy(
    cpu_core_seconds: float,
    tdp_watts: float = DEFAULT_TDP_W,
    n_cores: int = DEFAULT_CORES,
    pue: float = DEFAULT_PUE,
) -> dict:
    """Analytic software energy estimate. Pure function, no dependencies."""
    n_cores = max(1, int(n_cores))
    power_per_core = tdp_watts / n_cores
    energy_j = power_per_core * max(0.0, cpu_core_seconds) * pue
    return {
        "cpu_core_seconds": round(cpu_core_seconds, 4),
        "tdp_watts": tdp_watts,
        "n_cores": n_cores,
        "power_per_core_w": round(power_per_core, 4),
        "pue": pue,
        "energy_j": round(energy_j, 4),
        "energy_mwh": round(energy_j / JOULES_PER_MWH, 4),
    }


class EnergyMeter:
    """Context manager measuring a code block's process CPU time -> energy.

    Usage::
        with EnergyMeter(tdp_watts=7, n_cores=4) as m:
            ... do FL round ...
        print(m.result["energy_mwh"])
    """

    def __init__(
        self,
        tdp_watts: float = DEFAULT_TDP_W,
        n_cores: int = DEFAULT_CORES,
        pue: float = DEFAULT_PUE,
        include_children: bool = True,
    ):
        self.tdp_watts = tdp_watts
        self.n_cores = n_cores
        self.pue = pue
        self.include_children = include_children
        self.result: dict = {}
        self._proc = None
        self._t0 = 0.0
        self._cpu0 = 0.0

    def _cpu_seconds(self) -> float:
        t = self._proc.cpu_times()
        total = t.user + t.system
        if self.include_children:
            total += getattr(t, "children_user", 0.0) + getattr(t, "children_system", 0.0)
        return total

    def __enter__(self) -> "EnergyMeter":
        try:
            import psutil
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("EnergyMeter requires psutil (pip install psutil)") from e
        self._proc = psutil.Process()
        self._cpu0 = self._cpu_seconds()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        cpu = self._cpu_seconds() - self._cpu0
        wall = time.perf_counter() - self._t0
        self.result = estimate_energy(cpu, self.tdp_watts, self.n_cores, self.pue)
        self.result["wall_seconds"] = round(wall, 4)


def _wrap_command(cmd: str, tdp: float, cores: int, pue: float) -> dict:
    """Run a subprocess and estimate its energy from measured CPU time."""
    try:
        import psutil
    except ImportError as e:
        raise RuntimeError("--wrap requires psutil (pip install psutil)") from e
    import shlex
    import subprocess

    args = cmd if isinstance(cmd, list) else shlex.split(cmd, posix=False)
    t0 = time.perf_counter()
    proc = subprocess.Popen(args)
    p = psutil.Process(proc.pid)

    # Poll CPU time while the process (and its children) run, keeping the last
    # good reading — robust even when the process exits before a final read.
    cpu = 0.0

    def _tree_cpu() -> float:
        ct = p.cpu_times()
        total = ct.user + ct.system
        try:
            for ch in p.children(recursive=True):
                cct = ch.cpu_times()
                total += cct.user + cct.system
        except psutil.Error:
            pass
        return total

    while proc.poll() is None:
        try:
            cpu = _tree_cpu()
        except psutil.Error:
            break
        time.sleep(0.1)
    try:
        cpu = max(cpu, _tree_cpu())
    except psutil.Error:
        pass
    proc.wait()
    if cpu <= 0.0:
        cpu = time.perf_counter() - t0
        log.warning("Could not read CPU times; using wall time as proxy")
    res = estimate_energy(cpu, tdp, cores, pue)
    res["wall_seconds"] = round(time.perf_counter() - t0, 4)
    res["returncode"] = proc.returncode
    return res


def _post_process_csv(
    csv_path: Path, cpu_col: str, rounds_col: Optional[str],
    tdp: float, cores: int, pue: float, out_path: Optional[Path],
) -> list[dict]:
    rows_out: list[dict] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                cpu = float(row[cpu_col])
            except (KeyError, ValueError):
                continue
            est = estimate_energy(cpu, tdp, cores, pue)
            merged = dict(row)
            merged["energy_j"] = est["energy_j"]
            merged["energy_mwh"] = est["energy_mwh"]
            rows_out.append(merged)

    if not rows_out:
        log.warning("No rows with numeric '%s' found in %s", cpu_col, csv_path)
        return rows_out

    total_mwh = sum(r["energy_mwh"] for r in rows_out)
    n = len(rows_out)
    log.info("%d rows: total %.3f mWh, mean %.4f mWh/row", n, total_mwh, total_mwh / n)
    if rounds_col and rounds_col in rows_out[0]:
        log.info("(treated as per-round energy over column '%s')", rounds_col)

    if out_path:
        with out_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
            w.writeheader()
            w.writerows(rows_out)
        log.info("Wrote %s", out_path)
    return rows_out


def main() -> None:
    p = argparse.ArgumentParser(description="F6 — software energy estimation for FL rounds")
    p.add_argument("--tdp", type=float, default=DEFAULT_TDP_W, help="package TDP in watts")
    p.add_argument("--cores", type=int, default=DEFAULT_CORES, help="physical core count")
    p.add_argument("--pue", type=float, default=DEFAULT_PUE, help="supply/overhead factor")

    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--cpu-seconds", type=float, help="explicit process CPU core-seconds")
    g.add_argument("--wrap", type=str, help="run this command and measure its CPU time")
    g.add_argument("--csv", type=str, help="post-process a results CSV with a CPU-time column")

    p.add_argument("--cpu-col", default="cpu_time_s", help="CPU-time column for --csv")
    p.add_argument("--rounds-col", default=None, help="optional round-index column for --csv")
    p.add_argument("--out", default=None, help="output CSV path for --csv mode")
    args = p.parse_args()

    if args.cpu_seconds is not None:
        res = estimate_energy(args.cpu_seconds, args.tdp, args.cores, args.pue)
        log.info("Energy: %.4f mWh  (%.3f J, %.3f core-s @ %.1f W / %d cores)",
                 res["energy_mwh"], res["energy_j"], res["cpu_core_seconds"],
                 res["tdp_watts"], res["n_cores"])
    elif args.wrap:
        res = _wrap_command(args.wrap, args.tdp, args.cores, args.pue)
        log.info("Command rc=%s  wall=%.1fs  CPU=%.1f core-s  -> %.4f mWh (%.2f J)",
                 res.get("returncode"), res["wall_seconds"], res["cpu_core_seconds"],
                 res["energy_mwh"], res["energy_j"])
    else:
        _post_process_csv(
            Path(args.csv), args.cpu_col, args.rounds_col,
            args.tdp, args.cores, args.pue,
            Path(args.out) if args.out else None,
        )


if __name__ == "__main__":
    main()
