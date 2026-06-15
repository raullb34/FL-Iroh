"""
Dataset loading and FL partitioning (IID and Dirichlet non-IID).

Supported datasets:
  cifar10     — 50K train / 10K test, 10 classes, 32×32 RGB
  mnist       — 60K train / 10K test, 10 classes, 28×28 greyscale
  fmnist      — same shape as MNIST (Fashion-MNIST)
  crop        — Crop Recommendation (2200 samples, 7 features, 22 classes)
               Place CSV at data/Crop_recommendation.csv
               Download: kaggle datasets download -d atharvaingle/crop-recommendation-dataset
  air_quality — CyL daily air-quality (2011-2019, 10 provinces, 3 ICA classes)
               Requires: data/air-quailty/datasets/air_quality_fl_classification.csv
               Generate: python data/air-quailty/notebooks/preprocess_e7.py

Partitioning strategies:
  iid        — random shuffle, equal split
  dirichlet  — Dirichlet(α) label distribution; smaller α = more non-IID
  geographic — one Subset per province (natural non-IID; air_quality only)

Typical FL benchmark settings:
  α = 0.5  — moderate heterogeneity (default)
  α = 0.1  — severe heterogeneity

Seeds are taken from seeds.yaml or passed explicitly; always reproducible.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, Subset, TensorDataset
from torchvision import datasets, transforms

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

def _load_with_retry(loader_fn, max_attempts: int = 3, delay_sec: float = 3.0):
    """Call *loader_fn()* up to *max_attempts* times, sleeping between retries.

    This prevents race conditions when multiple containers simultaneously
    attempt to access/verify a pre-mounted dataset directory.
    """
    last_exc: Exception = RuntimeError("unreachable")
    for attempt in range(max_attempts):
        try:
            return loader_fn()
        except Exception as exc:
            last_exc = exc
            if attempt < max_attempts - 1:
                log.warning(
                    "Dataset load failed (attempt %d/%d): %s — retrying in %.0fs",
                    attempt + 1, max_attempts, exc, delay_sec,
                )
                time.sleep(delay_sec)
    raise last_exc


def load_cifar10(data_dir: str = "./data") -> tuple[Dataset, Dataset]:
    """Return (train_dataset, test_dataset) with standard CIFAR-10 transforms."""
    _mean = (0.4914, 0.4822, 0.4465)
    _std  = (0.2023, 0.1994, 0.2010)
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(_mean, _std),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(_mean, _std),
    ])

    def _load():
        train = datasets.CIFAR10(data_dir, train=True,  download=True, transform=train_tf)
        test  = datasets.CIFAR10(data_dir, train=False, download=True, transform=test_tf)
        return train, test

    return _load_with_retry(_load)


def load_mnist(data_dir: str = "./data") -> tuple[Dataset, Dataset]:
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    def _load():
        train = datasets.MNIST(data_dir, train=True,  download=True, transform=tf)
        test  = datasets.MNIST(data_dir, train=False, download=True, transform=tf)
        return train, test

    return _load_with_retry(_load)


def load_fmnist(data_dir: str = "./data") -> tuple[Dataset, Dataset]:
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ])

    def _load():
        train = datasets.FashionMNIST(data_dir, train=True,  download=True, transform=tf)
        test  = datasets.FashionMNIST(data_dir, train=False, download=True, transform=tf)
        return train, test

    return _load_with_retry(_load)


def load_crop(data_dir: str = "./data") -> tuple[Dataset, Dataset]:
    """
    Crop Recommendation Dataset — tabular, 7 features, 22 crop classes.

    Expected file: <data_dir>/Crop_recommendation.csv
    Download: kaggle datasets download -d atharvaingle/crop-recommendation-dataset

    Features: N, P, K, temperature, humidity, ph, rainfall (all float32)
    Labels:   22 crop types encoded as integers 0-21

    Split: 80% train / 20% test, stratified by class, seed=42.
    """
    import csv

    csv_path = Path(data_dir) / "Crop_recommendation.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Crop Recommendation CSV not found at {csv_path}.\n"
            "Download it with:\n"
            "  kaggle datasets download -d atharvaingle/crop-recommendation-dataset\n"
            "  unzip crop-recommendation-dataset.zip -d data/"
        )

    feature_cols = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    label_col = "label"

    rows: list[list[float]] = []
    raw_labels: list[str] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append([float(row[c]) for c in feature_cols])
            raw_labels.append(row[label_col].strip())

    # Encode string labels to integers (sorted for reproducibility)
    classes = sorted(set(raw_labels))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    int_labels = [class_to_idx[l] for l in raw_labels]

    X = np.array(rows, dtype=np.float32)
    y = np.array(int_labels, dtype=np.int64)

    # Z-score normalise features globally (acceptable since this is a
    # centralised preprocessing step before FL partitioning)
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

    # Stratified 80/20 split
    rng = np.random.default_rng(42)
    train_idx: list[int] = []
    test_idx: list[int] = []
    for c in range(len(classes)):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        split = max(1, int(len(idx) * 0.8))
        train_idx.extend(idx[:split].tolist())
        test_idx.extend(idx[split:].tolist())

    X_train = torch.tensor(X[train_idx])
    y_train = torch.tensor(y[train_idx])
    X_test  = torch.tensor(X[test_idx])
    y_test  = torch.tensor(y[test_idx])

    train_ds = TensorDataset(X_train, y_train)
    test_ds  = TensorDataset(X_test,  y_test)

    log.info(
        "Crop dataset loaded: %d train / %d test  |  %d classes  |  %d features",
        len(train_ds), len(test_ds), len(classes), len(feature_cols),
    )
    return train_ds, test_ds


def load_air_quality(data_dir: str = "./data") -> tuple[Dataset, Dataset]:
    """
    CyL Air Quality Dataset — tabular, 6 features, 3 ICA classes.

    Expected file: <data_dir>/air-quailty/datasets/air_quality_fl_classification.csv
    Generate with: python data/air-quailty/notebooks/preprocess_e7.py

    Features (z-score normalised): NO2, O3, PM_particle, CO, velmedia, prec
    Labels:  0=Bueno, 1=Regular, 2=Malo  (ICA thresholds on NO2)
    Clients: 10 provinces of Castilla y León  (natural geographic non-IID)
    Train:   2011–2018  |  Test: 2019
    """
    import csv

    csv_path = Path(data_dir) / "air-quailty" / "datasets" / "air_quality_fl_classification.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Air quality classification CSV not found at {csv_path}.\n"
            "Generate it with:\n"
            "  python data/air-quailty/notebooks/preprocess_e7.py"
        )

    feature_cols = ["NO2", "O3", "PM_particle", "CO", "velmedia", "prec"]
    label_col    = "label_ica"
    split_col    = "split"
    prov_col     = "provincia"

    train_feats: list[list[float]] = []
    train_labels: list[int] = []
    train_provs: list[str]  = []
    test_feats: list[list[float]] = []
    test_labels: list[int] = []
    test_provs: list[str]  = []

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                feats = [float(row[c]) for c in feature_cols]
            except (ValueError, KeyError):
                continue
            label = int(row[label_col])
            prov  = row[prov_col].strip()
            if row[split_col] == "train":
                train_feats.append(feats)
                train_labels.append(label)
                train_provs.append(prov)
            else:
                test_feats.append(feats)
                test_labels.append(label)
                test_provs.append(prov)

    X_train = torch.tensor(train_feats, dtype=torch.float32)
    y_train = torch.tensor(train_labels, dtype=torch.int64)
    X_test  = torch.tensor(test_feats,  dtype=torch.float32)
    y_test  = torch.tensor(test_labels, dtype=torch.int64)

    train_ds = TensorDataset(X_train, y_train)
    test_ds  = TensorDataset(X_test,  y_test)

    # Attach province metadata for geographic partition
    train_ds.provinces = train_provs  # type: ignore[attr-defined]
    test_ds.provinces  = test_provs   # type: ignore[attr-defined]

    log.info(
        "Air quality dataset loaded: %d train / %d test | 6 features | 3 classes",
        len(train_ds), len(test_ds),
    )
    return train_ds, test_ds


def load_air_quality_sequences(
    data_dir: str = "./data",
    window  : int = 7,
) -> tuple[Dataset, Dataset]:
    """
    CyL Air Quality Dataset as sliding sequences for the AirLSTM model.

    Same source CSV as :func:`load_air_quality`, but instead of one tabular row
    per day it emits windows of ``window`` consecutive (per-province) days::

        X[i] = features of days [t-window+1 .. t]   shape (window, 6)
        y[i] = label_ica at the window's last row   (already the ICA class of
               NO2 at t+7 thanks to the preprocessing shift — no leakage)

    Windows are built per province after sorting by date, so they never cross
    province boundaries. Consecutive *rows* are used (calendar gaps from missing
    days are tolerated), which is standard for this kind of LSTM baseline.

    Returns:
        (train_ds, test_ds) where X has shape (N, window, 6) and each dataset
        carries a ``.provinces`` list aligned with the window's last day, so the
        existing geographic partitioner works unchanged.
    """
    import csv
    from collections import defaultdict

    csv_path = Path(data_dir) / "air-quailty" / "datasets" / "air_quality_fl_classification.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Air quality classification CSV not found at {csv_path}.\n"
            "Generate it with:\n"
            "  python data/air-quailty/notebooks/preprocess_e7.py"
        )

    feature_cols = ["NO2", "O3", "PM_particle", "CO", "velmedia", "prec"]
    label_col    = "label_ica"
    split_col    = "split"
    prov_col     = "provincia"
    date_col     = "fecha"

    # Group rows by (province, split), preserving (date, feats, label) tuples
    grouped: dict[tuple[str, str], list[tuple[str, list[float], int]]] = defaultdict(list)
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                feats = [float(row[c]) for c in feature_cols]
                label = int(row[label_col])
            except (ValueError, KeyError):
                continue
            prov  = row[prov_col].strip()
            split = row[split_col]
            date  = row.get(date_col, "")
            grouped[(prov, split)].append((date, feats, label))

    def _build(split: str) -> tuple[list, list, list]:
        X: list[list[list[float]]] = []
        y: list[int] = []
        provs: list[str] = []
        prov_names = sorted({p for (p, s) in grouped if s == split})
        for prov in prov_names:
            rows = grouped[(prov, split)]
            rows.sort(key=lambda r: r[0])  # sort by date string YYYY-MM-DD
            for t in range(window - 1, len(rows)):
                win = rows[t - window + 1 : t + 1]
                X.append([r[1] for r in win])
                y.append(rows[t][2])
                provs.append(prov)
        return X, y, provs

    Xtr, ytr, ptr = _build("train")
    Xte, yte, pte = _build("test")

    X_train = torch.tensor(Xtr, dtype=torch.float32)
    y_train = torch.tensor(ytr, dtype=torch.int64)
    X_test  = torch.tensor(Xte, dtype=torch.float32)
    y_test  = torch.tensor(yte, dtype=torch.int64)

    train_ds = TensorDataset(X_train, y_train)
    test_ds  = TensorDataset(X_test,  y_test)
    train_ds.provinces = ptr  # type: ignore[attr-defined]
    test_ds.provinces  = pte  # type: ignore[attr-defined]

    log.info(
        "Air quality SEQUENCES loaded: %d train / %d test | window=%d | 6 features | 3 classes",
        len(train_ds), len(test_ds), window,
    )
    return train_ds, test_ds


def load_dataset(name: str, data_dir: str = "./data") -> tuple[Dataset, Dataset]:
    loaders = {
        "cifar10"    : load_cifar10,
        "mnist"      : load_mnist,
        "fmnist"     : load_fmnist,
        "crop"       : load_crop,
        "air_quality": load_air_quality,
    }
    if name not in loaders:
        raise ValueError(f"Unknown dataset '{name}'. Choose from {list(loaders)}")
    return loaders[name](data_dir)


# ---------------------------------------------------------------------------
# Partitioning helpers
# ---------------------------------------------------------------------------

def _get_labels(dataset: Dataset) -> np.ndarray:
    """Extract integer labels from any torchvision dataset."""
    if hasattr(dataset, "targets"):
        return np.array(dataset.targets, dtype=np.int64)
    # Fallback: iterate (slow, only for custom datasets)
    return np.array([int(dataset[i][1]) for i in range(len(dataset))], dtype=np.int64)


def partition_iid(
    dataset: Dataset,
    n_clients: int,
    seed: int = 42,
) -> list[Subset]:
    """
    Split *dataset* uniformly at random into *n_clients* equal (or near-equal) subsets.

    Each client receives samples from all classes in roughly equal proportion.
    """
    rng     = np.random.default_rng(seed)
    indices = rng.permutation(len(dataset)).tolist()
    splits  = [indices[i::n_clients] for i in range(n_clients)]
    subsets = [Subset(dataset, split) for split in splits]
    log.info(
        "IID partition: %d clients, ~%d samples each",
        n_clients, len(dataset) // n_clients,
    )
    return subsets


def partition_dirichlet(
    dataset   : Dataset,
    n_clients : int,
    alpha     : float  = 0.5,
    seed      : int    = 42,
    min_samples: int   = 10,
) -> list[Subset]:
    """
    Non-IID Dirichlet partition (standard FL benchmark).

    For each class c, sample a probability vector p ~ Dir(α · 1_{n_clients})
    and allocate fraction p[i] of class c to client i.

    Args:
        alpha: Concentration parameter.
               α → ∞  : approaches IID
               α = 0.5: moderate non-IID  (FL benchmark default)
               α = 0.1: severe non-IID
        min_samples: Minimum samples per client (redistributed if needed).
    """
    rng      = np.random.default_rng(seed)
    labels   = _get_labels(dataset)
    n_classes = int(labels.max()) + 1

    # Group indices by class
    class_idx: dict[int, np.ndarray] = {
        c: np.where(labels == c)[0] for c in range(n_classes)
    }
    client_idx: list[list[int]] = [[] for _ in range(n_clients)]

    for c in range(n_classes):
        idx = class_idx[c].copy()
        rng.shuffle(idx)
        proportions = rng.dirichlet(np.ones(n_clients) * alpha)
        proportions /= proportions.sum()
        splits = np.round(proportions * len(idx)).astype(int)
        # Fix rounding so splits sum exactly to len(idx)
        splits[-1] = len(idx) - splits[:-1].sum()
        off = 0
        for i, n in enumerate(splits):
            n = max(0, int(n))
            client_idx[i].extend(idx[off : off + n].tolist())
            off += n

    # Ensure every client has at least min_samples
    all_pool = [i for lst in client_idx for i in lst]
    rng.shuffle(all_pool)
    pool_ptr = 0
    for i, lst in enumerate(client_idx):
        while len(lst) < min_samples and pool_ptr < len(all_pool):
            lst.append(all_pool[pool_ptr])
            pool_ptr += 1

    subsets = [Subset(dataset, sorted(idx)) for idx in client_idx]
    sizes   = [len(s) for s in subsets]
    log.info(
        "Dirichlet(α=%.2f) partition: %d clients  min=%d  max=%d  mean=%.0f",
        alpha, n_clients, min(sizes), max(sizes), float(np.mean(sizes)),
    )
    return subsets


def partition_geographic(dataset: Dataset) -> list[Subset]:
    """
    Geographic partition for the air_quality dataset.

    Returns one Subset per province (10 clients for Castilla y León).
    Province membership is read from the `provinces` attribute attached by
    load_air_quality().  Provinces are sorted alphabetically so client index
    is deterministic without requiring a seed.

    Args:
        dataset: TensorDataset produced by load_air_quality(), with
                 a .provinces list attribute.

    Returns:
        List of Subset objects, one per province, sorted alphabetically.
    """
    if not hasattr(dataset, "provinces"):
        raise ValueError(
            "partition_geographic requires a dataset with a .provinces attribute. "
            "Use load_air_quality() to obtain one."
        )
    provinces: list[str] = dataset.provinces  # type: ignore[attr-defined]
    unique_provs = sorted(set(provinces))

    prov_to_idx: dict[str, list[int]] = {p: [] for p in unique_provs}
    for i, p in enumerate(provinces):
        prov_to_idx[p].append(i)

    subsets = [Subset(dataset, prov_to_idx[p]) for p in unique_provs]
    sizes   = [len(s) for s in subsets]
    log.info(
        "Geographic partition: %d provinces  min=%d  max=%d  mean=%.0f",
        len(unique_provs), min(sizes), max(sizes), float(sum(sizes) / len(sizes)),
    )
    return subsets


def partition_dataset(
    dataset   : Dataset,
    n_clients : int,
    strategy  : str   = "iid",
    alpha     : float = 0.5,
    seed      : int   = 42,
) -> list[Subset]:
    """Dispatch to the correct partitioning strategy."""
    if strategy == "iid":
        return partition_iid(dataset, n_clients, seed=seed)
    if strategy == "dirichlet":
        return partition_dirichlet(dataset, n_clients, alpha=alpha, seed=seed)
    if strategy == "geographic":
        return partition_geographic(dataset)
    raise ValueError(
        f"Unknown partition strategy '{strategy}'. Use 'iid', 'dirichlet', or 'geographic'."
    )


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def class_distribution(subset: Subset) -> dict[int, int]:
    """Count samples per class in *subset*."""
    ds      = subset.dataset
    labels  = _get_labels(ds)
    dist: dict[int, int] = {}
    for idx in subset.indices:
        label = int(labels[idx])
        dist[label] = dist.get(label, 0) + 1
    return dict(sorted(dist.items()))


def describe_partitions(subsets: list[Subset]) -> list[dict]:
    """Return a list of per-client distribution summaries (for logging/CSV)."""
    rows = []
    for i, s in enumerate(subsets):
        dist = class_distribution(s)
        rows.append({
            "client_id"    : i,
            "n_samples"    : len(s),
            "n_classes"    : len(dist),
            "distribution" : dist,
        })
    return rows
