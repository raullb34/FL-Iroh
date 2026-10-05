"""Imbalance-aware classification metrics for the air-quality task (R1.12)."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Sequence

CLASS_NAMES = ["Bajo", "Medio", "Alto"]


def classification_report(
    y_true: Sequence[int], y_pred: Sequence[int], n_classes: int = 3,
) -> dict:
    """Accuracy, balanced accuracy, macro/weighted F1, per-class P/R/F1, confusion matrix."""
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
        precision_recall_fscore_support,
    )
    labels = list(range(n_classes))
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)
    names = CLASS_NAMES if n_classes == len(CLASS_NAMES) else [str(c) for c in labels]
    return {
        "n"                : len(y_true),
        "accuracy"         : float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1"         : float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1"      : float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "per_class"        : {
            names[c]: {"precision": float(p[c]), "recall": float(r[c]),
                       "f1": float(f[c]), "support": int(s[c])}
            for c in labels
        },
        "confusion_matrix" : confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def flat(report: dict) -> dict:
    """Flatten a report into scalar columns for summary CSVs."""
    row = {k: report[k] for k in ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1")}
    for name, m in report["per_class"].items():
        row[f"precision_{name}"] = m["precision"]
        row[f"recall_{name}"] = m["recall"]
        row[f"f1_{name}"] = m["f1"]
    return row


def save_report(report: dict, y_true: Sequence[int], y_pred: Sequence[int], prefix: Path) -> None:
    """Write ``<prefix>_report.json`` and ``<prefix>_predictions.csv``."""
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_name(prefix.name + "_report.json").write_text(json.dumps(report, indent=2))
    with prefix.with_name(prefix.name + "_predictions.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["y_true", "y_pred"])
        w.writerows(zip(y_true, y_pred))
