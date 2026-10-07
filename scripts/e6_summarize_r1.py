"""
R1: imbalance-aware summary of E6 (paper) / e7 (code) air-quality results.

* Trivial baselines (majority class fitted on train, 7-day persistence) with
  accuracy, balanced accuracy, macro-F1 and per-class recall.
* Every configuration found under <results-dir>/seeds/seed*/e7_<config>_report.json:
  mean and 95 % Student-t CI over seeds of the same metrics, plus the summed
  confusion matrix.

Usage:  python scripts/e6_summarize_r1.py results/e7_local [results/e7_hpc ...]
Writes  <first results-dir>/e6_r1_summary.csv and prints a LaTeX-ready table.
"""
from __future__ import annotations

import csv
import json
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fl_coap_iroh.metrics.classification import classification_report  # noqa: E402

DATA = ROOT / "data" / "air-quailty" / "datasets" / "air_quality_fl_classification.csv"
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}
METRICS = ["accuracy", "balanced_accuracy", "macro_f1"]
CLASSES = ["Bajo", "Medio", "Alto"]


def baselines() -> dict[str, dict]:
    rows = [(r["provincia"], datetime.strptime(r["fecha"], "%Y-%m-%d"), int(r["label_ica"]), r["split"])
            for r in csv.DictReader(DATA.open(encoding="utf-8"))]
    by_key = {(p, d): y for p, d, y, _ in rows}
    train = [y for _, _, y, s in rows if s == "train"]
    test = [(p, d, y) for p, d, y, s in rows if s == "test"]
    maj = max(set(train), key=train.count)
    out = {"Majority class (train)": classification_report([y for _, _, y in test], [maj] * len(test))}
    yt, yp = [], []
    for p, d, y in test:
        prev = by_key.get((p, d - timedelta(days=7)))
        if prev is not None:
            yt.append(y)
            yp.append(prev)
    out["Persistence (t-7)"] = classification_report(yt, yp)
    return out


def ci(vals: list[float]) -> tuple[float, float]:
    m = statistics.mean(vals)
    if len(vals) < 2:
        return m, float("nan")
    return m, T95.get(len(vals) - 1, 1.96) * statistics.stdev(vals) / len(vals) ** 0.5


def flat(rep: dict) -> dict:
    row = {k: rep[k] for k in METRICS}
    for c in CLASSES:
        row[f"recall_{c}"] = rep["per_class"][c]["recall"]
    return row


def main() -> None:
    dirs = [Path(a) for a in sys.argv[1:]] or [ROOT / "results" / "e7_local"]
    reports: dict[str, list[dict]] = defaultdict(list)
    for d in dirs:
        for f in sorted(d.glob("seeds/seed*/e7_*_report.json")):
            cfg = f.name[len("e7_"):-len("_report.json")]
            reports[cfg].append(json.loads(f.read_text()))
    rows = []
    for name, rep in baselines().items():
        rows.append({"config": name, "n": 1, **{k: f"{100 * v:.2f}" for k, v in flat(rep).items()}})
    for cfg, reps in sorted(reports.items()):
        flats = [flat(r) for r in reps]
        row = {"config": cfg, "n": len(reps)}
        for k in flats[0]:
            m, h = ci([x[k] for x in flats])
            row[k] = f"{100 * m:.2f}" + (f" ± {100 * h:.2f}" if h == h else "")
        cm = [[sum(r["confusion_matrix"][i][j] for r in reps) for j in range(3)] for i in range(3)]
        row["confusion_sum"] = json.dumps(cm)
        rows.append(row)
    out = dirs[0] / "e6_r1_summary.csv"
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(f"{r['config']:<28} n={r['n']}  " + "  ".join(
            f"{k}={r.get(k, '')}" for k in METRICS + [f"recall_{c}" for c in CLASSES]))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
