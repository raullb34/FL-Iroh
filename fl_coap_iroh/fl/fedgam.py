"""
fedgam.py — FedGAM aggregator for ProphetWrapper federated learning.

FedGAM averages only the Fourier seasonality coefficients that encode
SHARED periodic patterns (annual/weekly cycles common to all Castilla y León
provinces) and the wind-speed regressor coefficient.

Trend parameters (k, m, delta[]) encode LOCAL demographic/industrial patterns
and are deliberately NOT averaged — they are taken from the first update
(highest-weight client) and remain province-specific after each round.

Function signature is identical to fedavg_aggregate() so FLServer accepts it
transparently via the aggregator_fn= constructor parameter.

Keys treated as federatable seasonality (averaged with weighted mean):
  sin_year, cos_year — annual Fourier coefficients
  sin_week, cos_week — weekly Fourier coefficients
  beta_velmedia      — wind-speed regressor coefficient

Any key NOT in the above list is passed through from the first (heaviest-weight)
update unchanged.
"""
from __future__ import annotations

from collections import OrderedDict

import torch
from torch import Tensor

# Keys whose tensors are federated (weighted average across clients)
_FEDERATED_KEYS = frozenset({
    "sin_year",
    "cos_year",
    "sin_week",
    "cos_week",
    "beta_velmedia",
})


def fedgam_aggregate(
    updates: list[tuple[dict[str, Tensor], float]],
) -> dict[str, Tensor]:
    """
    Aggregate ProphetWrapper state dicts using FedGAM.

    Only Fourier seasonality coefficients (sin_year, cos_year, sin_week,
    cos_week) and the wind-speed regressor (beta_velmedia) are averaged.
    All other keys (trend: k, m, delta, changepoints, …) are passed through
    from the update with the highest sample weight.

    Args:
        updates: List of (state_dict, weight) pairs. Weight is the number of
                 local training samples. Zero-weight updates are ignored.

    Returns:
        Aggregated state dict compatible with ProphetWrapper.load_state_dict().
    """
    if not updates:
        raise ValueError("fedgam_aggregate received an empty updates list")

    # Filter out zero-weight updates (avoids division-by-zero)
    valid = [(sd, w) for sd, w in updates if w > 0]
    if not valid:
        valid = updates  # fall back to equal weighting if all weights are 0

    total_weight: float = sum(w for _, w in valid)

    # --- Find the "anchor" update (highest weight) for passthrough keys ---
    anchor_sd, _ = max(valid, key=lambda x: x[1])

    # --- Initialise aggregated dict from anchor ---
    aggregated: dict[str, Tensor] = OrderedDict()
    for key, tensor in anchor_sd.items():
        aggregated[key] = tensor.clone().float()

    # --- Weighted-average only the federatable seasonality keys ---
    # First pass: zero out federatable keys
    for key in _FEDERATED_KEYS:
        if key in aggregated:
            aggregated[key].zero_()

    # Second pass: accumulate weighted contributions
    for state_dict, weight in valid:
        frac = weight / total_weight
        for key in _FEDERATED_KEYS:
            if key in state_dict and key in aggregated:
                contribution = state_dict[key].to(aggregated[key].dtype)
                aggregated[key].add_(contribution * frac)

    # Cast back to original dtypes (federatable params are float32 in ProphetWrapper)
    for key in aggregated:
        orig_dtype = anchor_sd[key].dtype
        aggregated[key] = aggregated[key].to(orig_dtype)

    return aggregated
