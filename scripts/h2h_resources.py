"""Head-to-head resource comparison on the Raspberry Pi: FL-Iroh vs Flower+Tailscale.

Both runs monitored the FL client process plus the tailscaled daemon
(FL_MONITOR_PROCS=tailscaled). Statistics are computed over the federation
window (first to last per-round phase) from the 0.2 s samples:
CPU-seconds of the monitored processes per round, peak and median RSS, mean
system CPU, and energy per round from the Raspberry Pi 3B utilisation model
(1.4 W idle, 3.7 W full load).
"""
import csv
import statistics as st
import sys
from pathlib import Path

IDLE, MAX = 1.4, 3.7
base = Path(sys.argv[1] if len(sys.argv) > 1 else "results/h2h/rpi/results")
runs = {
    "Flower + Tailscale": base / "h2h/flower/e8_flower_client0",
    "FL-Iroh": base / "wan/rpi01/B_wan_rpi01",
}
for name, prefix in runs.items():
    ph = list(csv.DictReader(open(f"{prefix}_resource_phases.csv")))
    sm = list(csv.DictReader(open(f"{prefix}_resource_samples.csv")))
    t0 = min(float(p["t_start_wall"]) for p in ph)
    t1 = max(float(p["t_start_wall"]) + float(p["wall_s"]) for p in ph)
    w = [s for s in sm if t0 <= float(s["t_wall"]) <= t1]
    rounds = len({p["round"] for p in ph})
    dt = [float(b["t_wall"]) - float(a["t_wall"]) for a, b in zip(w, w[1:])]
    cpu_s = sum(float(s["proc_cpu_pct"]) / 100 * d for s, d in zip(w[1:], dt))
    energy = sum((IDLE + (MAX - IDLE) * float(s["sys_cpu_pct"]) / 100) * d for s, d in zip(w[1:], dt))
    rss = [float(s["rss_mb"]) for s in w]
    train = [float(p["wall_s"]) for p in ph if p["phase"] in ("train", "fit")]
    print(f"{name:20s} rounds={rounds:2d} window={t1 - t0:6.1f}s  CPU-s/round={cpu_s / rounds:5.2f}  "
          f"sysCPU={st.mean(float(s['sys_cpu_pct']) for s in w):5.1f}%  RSS med/peak={st.median(rss):5.0f}/{max(rss):5.0f} MB  "
          f"E/round={energy / rounds:5.2f} J  procs={max(int(s['n_procs']) for s in w)}  "
          f"train/fit phase med={st.median(train):.2f}s")
