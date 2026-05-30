"""
FedAvg aggregation (McMahan et al., 2017).

The aggregation is intentionally kept in a pure function so it can be:
  - called from FLServer (centralised topology)
  - called from a gateway (hierarchical topology)
  - called from a peer node (decentralised topology)

No FL algorithm contribution is claimed.  FedProx (Li et al., 2020) is
implemented as a client-side regularisation term in fl/client.py.
"""
from __future__ import annotations

import torch


def fedavg_aggregate(
    updates: list[tuple[dict[str, torch.Tensor], float]],
) -> dict[str, torch.Tensor]:
    """
    Weighted average of model state dicts (FedAvg).

    Args:
        updates: List of (state_dict, weight) pairs.
                 Weight is typically the number of local training samples.
                 Equal weights produce vanilla FedAvg.

    Returns:
        Aggregated state dict (same keys/shapes as inputs).

    Raises:
        ValueError: if *updates* is empty or all weights are zero.
    """
    if not updates:
        raise ValueError("fedavg_aggregate: no updates provided")

    total_weight = sum(w for _, w in updates)
    if total_weight <= 0.0:
        raise ValueError("fedavg_aggregate: total weight must be > 0")

    # Initialise accumulator with zeros matching the first update
    ref_state = updates[0][0]
    aggregated: dict[str, torch.Tensor] = {
        k: torch.zeros_like(v, dtype=torch.float32)
        for k, v in ref_state.items()
    }

    for state_dict, weight in updates:
        frac = weight / total_weight
        for key, param in state_dict.items():
            aggregated[key].add_(param.float() * frac)

    # Cast back to original dtypes to keep integer buffers (e.g. num_batches_tracked)
    result: dict[str, torch.Tensor] = {}
    for key, param in aggregated.items():
        orig_dtype = ref_state[key].dtype
        result[key] = param.to(orig_dtype)

    return result
