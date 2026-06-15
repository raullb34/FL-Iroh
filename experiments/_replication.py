"""
experiments/_replication.py — multi-seed replication helpers (F2).

Experiments report point estimates from a single seed by default. For the
statistical claims in the paper (e.g. the FedGAM-vs-FedAvg gap, or geographic
vs IID) we run several *replicates*, each driven by a distinct master seed, and
report mean ± std with a t-based 95% confidence interval plus a significance
test (see scripts/aggregate_ci.py).

This module centralises two concerns:

  * ``load_replicate_seeds`` — read the ``replicate_seeds`` list from seeds.yaml.
  * ``derive_seeds``          — turn one master seed into a full per-replicate
                                seed dict, re-deriving every stochastic seed
                                (data_partition, model_init, experiment_e*)
                                deterministically so a replicate is fully
                                reproducible from its master alone.

Determinism is preserved: the same master seed always yields the same derived
dict, and master ``replicate_seeds[0]`` reproduces the canonical run because the
derivation is the identity offset 0 for that role-ordering only when the master
equals the original base seed. In practice we simply re-key everything off the
master so results are independent and comparable across replicates.
"""
from __future__ import annotations

from typing import Iterable

import yaml

# Fixed, role-specific offsets so different stochastic components of the same
# replicate do not share a seed (which would correlate, e.g., the data shuffle
# with the weight init). Offsets are arbitrary but stable.
_ROLE_OFFSETS: dict[str, int] = {
    "data_partition": 0,
    "model_init":     10_000,
    "churn_simulator": 20_000,
    "experiment_e1":  30_001,
    "experiment_e2":  30_002,
    "experiment_e3":  30_003,
    "experiment_e5":  30_005,
    "experiment_e6":  30_006,
    "experiment_e7":  30_007,
}


def load_replicate_seeds(seeds_file: str = "seeds.yaml") -> list[int]:
    """Return the ``replicate_seeds`` list from seeds.yaml (empty if absent)."""
    try:
        with open(seeds_file, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception:
        return []
    seeds = data.get("replicate_seeds", [])
    return [int(s) for s in seeds] if isinstance(seeds, Iterable) else []


def derive_seeds(base_seeds: dict, master: int) -> dict:
    """
    Build a per-replicate seed dict from a single ``master`` seed.

    Every stochastic role is re-derived as ``master + role_offset`` so that:
      * each replicate is independent (different master → different seeds),
      * each role is decorrelated (distinct offsets),
      * the result is fully reproducible from ``master`` alone.

    Non-seed keys in ``base_seeds`` are preserved unchanged.
    """
    derived = dict(base_seeds)
    for role, offset in _ROLE_OFFSETS.items():
        derived[role] = int(master) + offset
    derived["_master_seed"] = int(master)
    return derived
