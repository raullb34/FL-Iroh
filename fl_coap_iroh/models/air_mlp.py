"""
AirMLP — lightweight MLP for tabular air-quality classification.

Task: 3-class ICA (Índice de Calidad del Aire) prediction from daily
      atmospheric measurements at provincial IoT stations in Castilla y León.

Input features (6, all z-score normalised):
  NO2        (μg/m³)  — nitrogen dioxide
  O3         (μg/m³)  — tropospheric ozone
  PM_particle(μg/m³)  — PM2.5 where available, else PM10
  CO         (mg/m³)  — carbon monoxide
  velmedia   (m/s)    — mean wind speed (AEMET)
  prec       (mm)     — daily precipitation (AEMET)

Output classes (label_ica):
  0 — Bueno    (NO2 < 40  μg/m³)
  1 — Regular  (40 ≤ NO2 < 100)
  2 — Malo     (NO2 ≥ 100 μg/m³)

Architecture: 6→32→32→16→3  ≈ 3.7K parameters
Designed as the air-quality analogue of AgriMLP for E7 federated experiment.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class AirMLP(nn.Module):
    """
    MLP for air-quality IoT tabular data.

    Layers::
        Linear(input_dim→h1) → BatchNorm → ReLU → Dropout(p)
        Linear(h1→h2)        → BatchNorm → ReLU → Dropout(p)
        Linear(h2→h3)        → ReLU
        Linear(h3→num_classes)

    Default: 6→32→32→16→3 ≈ 3.7K params.
    """

    def __init__(
        self,
        input_dim  : int   = 6,
        hidden1    : int   = 32,
        hidden2    : int   = 32,
        hidden3    : int   = 16,
        num_classes: int   = 3,
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

    def size_bytes(self) -> int:
        return sum(p.numel() * 4 for p in self.parameters())
