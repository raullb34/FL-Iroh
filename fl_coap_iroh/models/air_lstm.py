"""
AirLSTM — lightweight recurrent model for sequential air-quality forecasting.

Companion to :class:`AirMLP`. Whereas AirMLP is a memoryless baseline that sees
a single day's measurements, AirLSTM consumes a short window of the preceding
``L`` days and is therefore able to exploit the temporal trajectory that the MLP
cannot. It serves as the realistic accuracy ceiling for the E6 air-quality task,
while remaining small enough for Cortex-A class edge devices.

Task: 3-class ICA (Índice de Calidad del Aire) forecasting 7 days ahead (t+7).

Input  : tensor of shape (batch, L, 6) — L consecutive days of the 6 features
         (NO2, O3, PM_particle, CO, velmedia, prec), all z-score normalised.
Label  : ICA class of NO2 at day t+7 (the window ends at day t; the 7-day shift
         is applied during preprocessing, so there is no same-day leakage).
Output : logits of shape (batch, 3).

Architecture: LSTM(6→48, 1 layer) → last hidden → Linear(48→24) → ReLU → Linear(24→3)
Parameters  : ≈ 12K  (≈ 48 KB at float32)

The point of including AirLSTM is twofold:
  1. Provide a stronger, sequence-aware ML baseline than the MLP.
  2. Demonstrate that FL-Iroh's algorithm-agnostic transport carries recurrent
     model tensors over the same Iroh/QUIC channel without protocol changes.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class AirLSTM(nn.Module):
    """
    Single-layer LSTM classifier for windowed air-quality data.

    Args:
        input_dim:   Number of features per timestep (default 6).
        hidden_dim:  LSTM hidden size (default 48).
        num_layers:  Number of stacked LSTM layers (default 1).
        head_dim:    Hidden size of the classification head (default 24).
        num_classes: Number of output classes (default 3).
        dropout:     Dropout applied to the pooled representation (default 0.2).
    """

    def __init__(
        self,
        input_dim  : int   = 6,
        hidden_dim : int   = 48,
        num_layers : int   = 1,
        head_dim   : int   = 24,
        num_classes: int   = 3,
        dropout    : float = 0.2,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size  = input_dim,
            hidden_size = hidden_dim,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, head_dim),
            nn.ReLU(inplace=True),
            nn.Linear(head_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Accept (N, L, F); if a 2D tensor (N, F) is passed, treat as L=1.
        if x.dim() == 2:
            x = x.unsqueeze(1)
        out, _ = self.lstm(x)          # (N, L, H)
        last = out[:, -1, :]           # (N, H) — representation at final timestep
        return self.head(last)         # (N, num_classes)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def size_bytes(self) -> int:
        return sum(p.numel() * 4 for p in self.parameters())
