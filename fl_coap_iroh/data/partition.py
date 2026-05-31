"""
Dataset loading and FL partitioning (IID and Dirichlet non-IID).

Supported datasets:
  cifar10  — 50K train / 10K test, 10 classes, 32×32 RGB
  mnist    — 60K train / 10K test, 10 classes, 28×28 greyscale
  fmnist   — same shape as MNIST (Fashion-MNIST)

Partitioning strategies:
  iid        — random shuffle, equal split
  dirichlet  — Dirichlet(α) label distribution; smaller α = more non-IID

Typical FL benchmark settings:
  α = 0.5  — moderate heterogeneity (default)
  α = 0.1  — severe heterogeneity

Seeds are taken from seeds.yaml or passed explicitly; always reproducible.
"""
from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, Subset
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


def load_dataset(name: str, data_dir: str = "./data") -> tuple[Dataset, Dataset]:
    loaders = {"cifar10": load_cifar10, "mnist": load_mnist, "fmnist": load_fmnist}
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
    raise ValueError(f"Unknown partition strategy '{strategy}'. Use 'iid' or 'dirichlet'.")


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
