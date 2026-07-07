"""
E6 (paper) trivial baselines for 7-day-ahead ICA forecasting.

Computes, from data/air-quailty/datasets/air_quality_fl_classification.csv:

  * test-year class distribution (terciles are balanced in-sample only),
  * majority-class baseline fitted on the TRAIN split, evaluated on test,
  * 7-day persistence baseline: predict for each test row the label observed
    7 days earlier at the same station (works regardless of whether label_ica
    is stored at t or t+7, since it measures P(class(x) == class(x+7))).

These are the "Majority class (train)" and "Persistence (class at t-7)" rows
of the E6 table in the paper. Stdlib only; no dependencies.

Usage:  python scripts/e6_trivial_baselines.py [path/to/csv]
"""
from __future__ import annotations

import csv
import sys
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

DEFAULT_CSV = (
    Path(__file__).resolve().parents[1]
    / "data" / "air-quailty" / "datasets" / "air_quality_fl_classification.csv"
)


def main() -> None:
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CSV
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append((
                r["provincia"],
                datetime.strptime(r["fecha"], "%Y-%m-%d"),
                int(r["label_ica"]),
                r["split"],
            ))

    label_by_key = {(p, d): y for p, d, y, _ in rows}
    train = [r for r in rows if r[3] == "train"]
    test = [r for r in rows if r[3] == "test"]
    print(f"total={len(rows)}  train={len(train)}  test={len(test)}")

    dist_train = Counter(y for _, _, y, _ in train)
    dist_test = Counter(y for _, _, y, _ in test)
    print("train class dist:",
          {k: f"{100 * v / len(train):.1f}%" for k, v in sorted(dist_train.items())})
    print("test  class dist:",
          {k: f"{100 * v / len(test):.1f}%" for k, v in sorted(dist_test.items())})

    maj_train = dist_train.most_common(1)[0][0]
    maj_acc = sum(1 for _, _, y, _ in test if y == maj_train) / len(test)
    print(f"majority-class baseline (train class={maj_train}): {100 * maj_acc:.2f}% on test")

    hits = n = 0
    for p, d, y, _ in test:
        prev = label_by_key.get((p, d - timedelta(days=7)))
        if prev is None:
            continue
        n += 1
        hits += (prev == y)
    print(f"7-day persistence baseline: {100 * hits / n:.2f}% "
          f"(n={n} of {len(test)} test rows with t-7 available)")


if __name__ == "__main__":
    main()
