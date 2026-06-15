#!/usr/bin/env python3
"""
aggregate_ci.py — aggregate replicated experiment summaries into mean ± CI95
with pairwise significance tests (F2).

Reads several per-seed summary CSVs (one row per config per seed), groups rows
by a key column, and for a chosen numeric metric reports:

    n, mean, std, sem, 95% CI (Student-t)

Optionally runs pairwise comparisons between two groups, reporting:
    * mean difference,
    * Welch's two-sample t-test p-value (unequal variances),
    * a percentile bootstrap 95% CI of the mean difference.

scipy is used when available for the exact t/CDF; otherwise a small built-in
fallback keeps the script dependency-light on minimal HPC environments.

Examples
--------
    # E7: FedGAM vs FedAvg clean comparison, limited-history regime
    python scripts/aggregate_ci.py \
        --glob 'results/e7/seeds/e7_summary_seed*.csv' \
        --group config --metric test_acc_final \
        --compare prophet_fedgam_1yr_geographic prophet_fedavg_1yr_geographic \
        --compare airlstm_geographic airmlp_geographic
"""
from __future__ import annotations

import argparse
import csv
import glob
import math
from collections import defaultdict


# --------------------------------------------------------------------------- #
# Statistics helpers (scipy when present, pure-python fallback otherwise)
# --------------------------------------------------------------------------- #
try:
    from scipy import stats as _scipy_stats  # type: ignore
except Exception:  # pragma: no cover - scipy optional
    _scipy_stats = None


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def _std(xs: list[float]) -> float:
    """Sample standard deviation (ddof=1)."""
    n = len(xs)
    if n < 2:
        return 0.0
    mu = _mean(xs)
    return math.sqrt(sum((x - mu) ** 2 for x in xs) / (n - 1))


def _t_critical(df: int, conf: float = 0.95) -> float:
    """Two-sided t critical value. Uses scipy if available, else a table."""
    if _scipy_stats is not None:
        return float(_scipy_stats.t.ppf(1 - (1 - conf) / 2, df))
    # Fallback table for 95% two-sided, df 1..30 then ~normal
    table95 = {
        1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
        7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
        13: 2.160, 14: 2.145, 15: 2.131, 20: 2.086, 25: 2.060, 30: 2.042,
    }
    if conf != 0.95:
        return 1.96  # crude
    if df in table95:
        return table95[df]
    keys = sorted(table95)
    if df < keys[0]:
        return table95[keys[0]]
    if df > 30:
        return 1.96
    # nearest lower key
    lo = max(k for k in keys if k <= df)
    return table95[lo]


def ci95(xs: list[float]) -> tuple[float, float, float, float]:
    """Return (mean, std, ci_low, ci_high) with a Student-t 95% CI."""
    n = len(xs)
    mu = _mean(xs)
    sd = _std(xs)
    if n < 2:
        return mu, sd, mu, mu
    sem = sd / math.sqrt(n)
    tc = _t_critical(n - 1, 0.95)
    return mu, sd, mu - tc * sem, mu + tc * sem


def welch_t_test(a: list[float], b: list[float]) -> tuple[float, float]:
    """Welch's t-test; return (t_stat, p_value two-sided)."""
    if _scipy_stats is not None:
        t, p = _scipy_stats.ttest_ind(a, b, equal_var=False)
        return float(t), float(p)
    # Manual Welch with normal-approx p-value fallback
    na, nb = len(a), len(b)
    va, vb = _std(a) ** 2, _std(b) ** 2
    se = math.sqrt(va / na + vb / nb) or 1e-12
    t = (_mean(a) - _mean(b)) / se
    # Normal approximation to the two-sided p-value
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))
    return t, p


def bootstrap_diff_ci(
    a: list[float], b: list[float], n_boot: int = 10_000, seed: int = 12345
) -> tuple[float, float]:
    """Percentile bootstrap 95% CI of the mean difference (a - b)."""
    import random

    rng = random.Random(seed)
    diffs = []
    for _ in range(n_boot):
        ra = [a[rng.randrange(len(a))] for _ in a]
        rb = [b[rng.randrange(len(b))] for _ in b]
        diffs.append(_mean(ra) - _mean(rb))
    diffs.sort()
    lo = diffs[int(0.025 * n_boot)]
    hi = diffs[int(0.975 * n_boot)]
    return lo, hi


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate replicated runs into CI95 + significance tests")
    ap.add_argument("--glob", required=True, help="Glob for per-seed summary CSVs")
    ap.add_argument("--group", default="config", help="Column to group by (default: config)")
    ap.add_argument("--metric", default="test_acc_final", help="Numeric metric column to aggregate")
    ap.add_argument(
        "--compare", nargs=2, action="append", metavar=("A", "B"), default=[],
        help="Two group values to compare (repeatable).",
    )
    ap.add_argument("--out", default=None, help="Optional CSV path to write aggregated stats")
    args = ap.parse_args()

    files = sorted(glob.glob(args.glob))
    if not files:
        raise SystemExit(f"No files matched glob: {args.glob}")
    print(f"Reading {len(files)} summary files:")
    for f in files:
        print(f"  - {f}")

    # group value -> list of metric values
    groups: dict[str, list[float]] = defaultdict(list)
    for path in files:
        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                gkey = row.get(args.group)
                raw = row.get(args.metric)
                if gkey is None or raw is None or raw == "":
                    continue
                try:
                    groups[gkey].append(float(raw))
                except ValueError:
                    continue

    if not groups:
        raise SystemExit(
            f"No data found for group='{args.group}' metric='{args.metric}'. "
            "Check column names."
        )

    # Per-group stats
    print(f"\n=== {args.metric} by {args.group} (mean ± CI95, Student-t) ===")
    hdr = f"{args.group:<34}  {'n':>2}  {'mean':>8}  {'std':>7}  {'ci95_low':>9}  {'ci95_high':>9}"
    print(hdr)
    print("-" * len(hdr))
    stat_rows = []
    for g in sorted(groups):
        xs = groups[g]
        mu, sd, lo, hi = ci95(xs)
        print(f"{g:<34}  {len(xs):>2}  {mu:>8.4f}  {sd:>7.4f}  {lo:>9.4f}  {hi:>9.4f}")
        stat_rows.append({
            "group": g, "n": len(xs), "mean": mu, "std": sd,
            "ci95_low": lo, "ci95_high": hi,
        })

    # Pairwise comparisons
    if args.compare:
        print("\n=== Pairwise comparisons (Welch t-test + bootstrap CI of diff) ===")
        for a_key, b_key in args.compare:
            if a_key not in groups or b_key not in groups:
                print(f"  [skip] '{a_key}' or '{b_key}' not found in data")
                continue
            a, b = groups[a_key], groups[b_key]
            diff = _mean(a) - _mean(b)
            t, p = welch_t_test(a, b)
            blo, bhi = bootstrap_diff_ci(a, b)
            sig = "significant" if p < 0.05 else "NOT significant"
            print(
                f"\n  {a_key}  vs  {b_key}\n"
                f"    mean diff (A-B) : {diff:+.4f}  ({100*diff:+.2f} pp)\n"
                f"    Welch t / p     : t={t:+.3f}, p={p:.4f}  -> {sig} at α=0.05\n"
                f"    bootstrap 95% CI: [{blo:+.4f}, {bhi:+.4f}]"
                + ("  (excludes 0)" if (blo > 0 or bhi < 0) else "  (includes 0)")
            )

    # Optional CSV out
    if args.out and stat_rows:
        with open(args.out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(stat_rows[0].keys()))
            w.writeheader()
            w.writerows(stat_rows)
        print(f"\nAggregated stats written to {args.out}")


if __name__ == "__main__":
    main()
