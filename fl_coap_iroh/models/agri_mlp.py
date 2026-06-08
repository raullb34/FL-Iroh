"""
AgriMLP — three-hidden-layer MLP for tabular agricultural datasets.

Designed for the Crop Recommendation Dataset:
  input_dim  = 7  (N, P, K, temperature, humidity, ph, rainfall)
  num_classes = 22 crop types

~15K parameters — intentionally small for edge/IoT deployment (RPi, Arduino-class).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class AgriMLP(nn.Module):
    """
    MLP for tabular agricultural sensor data.

    Layers::
        Linear(input_dim→h1) → BatchNorm → ReLU → Dropout(p)
        Linear(h1→h2)        → BatchNorm → ReLU → Dropout(p)
        Linear(h2→h3)        → ReLU
        Linear(h3→num_classes)

    Default: 7→64→64→32→22 ≈ 15K params.
    """

    def __init__(
        self,
        input_dim  : int   = 7,
        hidden1    : int   = 64,
        hidden2    : int   = 64,
        hidden3    : int   = 32,
        num_classes: int   = 22,
        dropout    : float = 0.2,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden2, hidden3),
            nn.ReLU(inplace=True),
            nn.Linear(hidden3, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def serialised_size_bytes(self) -> int:
        return sum(p.numel() * 4 for p in self.parameters())
