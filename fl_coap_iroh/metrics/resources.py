"""
Resource monitoring for constrained FL clients: CPU, memory and power/energy.

A background thread samples, every ``interval_s``:
  * CPU utilisation and cumulative CPU time of the monitored processes
    (this process by default, optionally extra processes by name, e.g.
    ``tailscaled`` or a Flower client, for like-for-like comparisons);
  * resident memory (RSS) of the monitored processes;
  * system-wide CPU utilisation;
  * instantaneous power, when a hardware source is available.

Power sources, tried in order (``FL_POWER_SOURCE`` forces one):
  ``hwmon``  Linux hwmon ``power*_input`` (µW), e.g. an INA219/INA226 shunt
             monitor enabled with ``dtoverlay=i2c-sensor,ina219``.
  ``pmic``   Raspberry Pi 5 on-board PMIC, via ``vcgencmd pmic_read_adc``
             (sum of V×I over all rails; SoC-side power, excludes USB loads).
  ``none``   no hardware power reading; energy is then *estimated* with a
             linear utilisation model  P = P_idle + (P_max − P_idle)·u,
             with defaults for a Raspberry Pi 4B (``FL_POWER_IDLE_W``,
             ``FL_POWER_MAX_W`` override them).  Rows are flagged
             ``energy_method=model`` so estimates are never mistaken for
             measurements.

Every sample carries a wall-clock timestamp, so traces from an external USB
power meter can be aligned offline with ``scripts/join_power_log.py``.

Usage::

    mon = ResourceMonitor(extra_process_names=["tailscaled"])
    mon.start()
    with mon.phase("train", round_num=3):
        ...
    mon.stop()
    mon.export_csv("results/e10/rpi_client")
"""
from __future__ import annotations

import csv
import glob
import logging
import os
import re
import shutil
import subprocess
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

log = logging.getLogger(__name__)

try:
    import psutil  # type: ignore[import]
except ImportError:  # pragma: no cover - optional dependency
    psutil = None

# Raspberry Pi 4B board power (5 V input), typical published figures.
DEFAULT_IDLE_W = 2.7
DEFAULT_MAX_W  = 6.4


# ---------------------------------------------------------------------------
# Power sources
# ---------------------------------------------------------------------------

class _HwmonPower:
    name = "hwmon"

    def __init__(self) -> None:
        self._files = sorted(glob.glob("/sys/class/hwmon/hwmon*/power*_input"))
        if not self._files:
            raise RuntimeError("no hwmon power inputs")
        # Some virtualised hosts (e.g. WSL2) expose power inputs that read 0.
        if not (self.read_w() or 0) > 0:
            raise RuntimeError("hwmon power inputs read zero")

    def read_w(self) -> Optional[float]:
        total = 0.0
        for f in self._files:
            try:
                total += int(Path(f).read_text().strip()) / 1e6
            except (OSError, ValueError):
                return None
        return total


_PMIC_RE = re.compile(r"^\s*(\S+)_([AV])\s+(?:current|volt)\(\d+\)=([\d.]+)[AV]", re.M)


class _PmicPower:
    """Raspberry Pi 5 PMIC rails (``vcgencmd pmic_read_adc``)."""
    name = "pmic"

    def __init__(self) -> None:
        if shutil.which("vcgencmd") is None or self.read_w() is None:
            raise RuntimeError("vcgencmd pmic_read_adc unavailable")

    def read_w(self) -> Optional[float]:
        try:
            out = subprocess.run(
                ["vcgencmd", "pmic_read_adc"], capture_output=True, text=True, timeout=2,
            ).stdout
        except (OSError, subprocess.SubprocessError):
            return None
        amps: dict[str, float] = {}
        volts: dict[str, float] = {}
        for rail, kind, value in _PMIC_RE.findall(out):
            (amps if kind == "A" else volts)[rail] = float(value)
        rails = amps.keys() & volts.keys()
        if not rails:
            return None
        return sum(amps[r] * volts[r] for r in rails)


def _open_power_source() -> Optional[object]:
    forced = os.environ.get("FL_POWER_SOURCE", "").lower()
    candidates = {"hwmon": _HwmonPower, "pmic": _PmicPower}
    order = [forced] if forced in candidates else (["hwmon", "pmic"] if not forced else [])
    for name in order:
        try:
            src = candidates[name]()
            log.info("Power source: %s", name)
            return src
        except Exception as exc:  # noqa: BLE001
            log.debug("power source %s unavailable: %s", name, exc)
    return None


# ---------------------------------------------------------------------------
# Monitor
# ---------------------------------------------------------------------------

@dataclass
class _PhaseAcc:
    name       : str
    round_num  : int
    t0_wall    : float
    t0_mono    : float
    cpu0_s     : float
    samples    : list[dict] = field(default_factory=list)


class ResourceMonitor:
    def __init__(
        self,
        interval_s: float = 0.2,
        extra_process_names: Optional[list[str]] = None,
        include_self: bool = True,
        n_cores: Optional[int] = None,
    ) -> None:
        if psutil is None:
            raise RuntimeError("psutil is required for ResourceMonitor (pip install psutil)")
        self.interval_s = interval_s
        self._names     = [n.lower() for n in (extra_process_names or [])]
        self._include_self = include_self
        self._n_cores   = n_cores or psutil.cpu_count(logical=True) or 1
        self._power     = _open_power_source()
        self._idle_w    = float(os.environ.get("FL_POWER_IDLE_W", DEFAULT_IDLE_W))
        self._max_w     = float(os.environ.get("FL_POWER_MAX_W", DEFAULT_MAX_W))
        self._procs     : dict[int, "psutil.Process"] = {}
        self._samples   : list[dict] = []
        self._phases    : list[dict] = []
        self._active    : list[_PhaseAcc] = []
        self._lock      = threading.Lock()
        self._stop      = threading.Event()
        self._thread    : Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    @property
    def power_method(self) -> str:
        return self._power.name if self._power is not None else "model"

    def _refresh_procs(self) -> None:
        wanted: dict[int, "psutil.Process"] = {}
        if self._include_self:
            wanted[os.getpid()] = self._procs.get(os.getpid()) or psutil.Process(os.getpid())
        if self._names:
            for p in psutil.process_iter(["name"]):
                if (p.info.get("name") or "").lower() in self._names:
                    wanted[p.pid] = self._procs.get(p.pid) or p
        for pid, p in wanted.items():
            if pid not in self._procs:
                p.cpu_percent(None)  # prime the per-process counter
        self._procs = wanted

    def _cpu_seconds(self) -> float:
        total = 0.0
        for p in list(self._procs.values()):
            try:
                t = p.cpu_times()
                total += t.user + t.system
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return total

    def _sample(self) -> dict:
        cpu_pct = rss = 0.0
        for p in list(self._procs.values()):
            try:
                cpu_pct += p.cpu_percent(None)
                rss     += p.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        sys_cpu = psutil.cpu_percent(None)
        power_w = self._power.read_w() if self._power is not None else None
        if power_w is None:
            # Linear utilisation model on whole-system CPU utilisation.
            power_w = self._idle_w + (self._max_w - self._idle_w) * sys_cpu / 100.0
        return {
            "t_wall"      : time.time(),
            "t_mono"      : time.monotonic(),
            "proc_cpu_pct": round(cpu_pct, 2),          # may exceed 100 on multi-core
            "sys_cpu_pct" : round(sys_cpu, 2),
            "rss_mb"      : round(rss / 2**20, 2),
            "power_w"     : round(power_w, 4),
            "n_procs"     : len(self._procs),
        }

    def _run(self) -> None:
        refresh_every = max(1, int(2.0 / self.interval_s))
        i = 0
        while not self._stop.is_set():
            if i % refresh_every == 0:
                self._refresh_procs()
            s = self._sample()
            with self._lock:
                self._samples.append(s)
                for acc in self._active:
                    acc.samples.append(s)
            i += 1
            self._stop.wait(self.interval_s)

    # ------------------------------------------------------------------
    def start(self) -> "ResourceMonitor":
        self._refresh_procs()
        psutil.cpu_percent(None)
        self._thread = threading.Thread(target=self._run, name="resource-monitor", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    @contextmanager
    def phase(self, name: str, round_num: int = 0) -> Iterator[None]:
        """Attribute CPU time, peak RSS and energy to a named phase."""
        acc = _PhaseAcc(name, round_num, time.time(), time.monotonic(), self._cpu_seconds())
        with self._lock:
            self._active.append(acc)
        try:
            yield
        finally:
            end_sample = self._sample()   # phases shorter than one interval still get a sample
            with self._lock:
                self._active.remove(acc)
                samples = list(acc.samples) + [end_sample]
            wall_s = time.monotonic() - acc.t0_mono
            cpu_s  = self._cpu_seconds() - acc.cpu0_s
            powers = [s["power_w"] for s in samples]
            mean_w = sum(powers) / len(powers) if powers else None
            self._phases.append({
                "phase"          : name,
                "round"          : round_num,
                "t_start_wall"   : round(acc.t0_wall, 3),
                "wall_s"         : round(wall_s, 4),
                "cpu_s"          : round(cpu_s, 4),
                "cpu_util_pct"   : round(100 * cpu_s / wall_s / self._n_cores, 2) if wall_s > 0 else None,
                "rss_peak_mb"    : max((s["rss_mb"] for s in samples), default=None),
                "power_mean_w"   : round(mean_w, 4) if mean_w is not None else None,
                "energy_j"       : round(mean_w * wall_s, 4) if mean_w is not None else None,
                "energy_method"  : self.power_method,
                "n_samples"      : len(samples),
            })

    # ------------------------------------------------------------------
    @property
    def phases(self) -> list[dict]:
        return list(self._phases)

    def export_csv(self, prefix: str | os.PathLike) -> dict[str, Path]:
        prefix = Path(prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        written: dict[str, Path] = {}
        for suffix, rows in (("resource_samples", self._samples), ("resource_phases", self._phases)):
            if not rows:
                continue
            path = prefix.with_name(f"{prefix.name}_{suffix}.csv")
            with path.open("w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
            written[suffix] = path
        return written


def maybe_monitor() -> Optional[ResourceMonitor]:
    """Return a started monitor when ``FL_RESOURCE_MONITOR=1`` and psutil is available."""
    if os.environ.get("FL_RESOURCE_MONITOR", "0") != "1":
        return None
    if psutil is None:
        log.warning("FL_RESOURCE_MONITOR=1 but psutil is not installed — monitoring disabled")
        return None
    names = [n for n in os.environ.get("FL_MONITOR_PROCS", "").split(",") if n]
    return ResourceMonitor(extra_process_names=names).start()
