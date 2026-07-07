"""
SimpleCNN — lightweight convolutional network for CIFAR-10 / FL experiments.

Architecture: 3 conv blocks → global-average pooling → linear head.
94,762 trainable parameters, runs on CPU in < 1 s/batch on a Pi 4.
Serialised state dict: ~0.38 MB (float32), adequate for transfer benchmarks.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """
    Small CNN (~95 K params) for CIFAR-10 (32×32 RGB).

    Block structure::
        Conv(c_in → 32, 3×3, pad=1) → BN → ReLU
        Conv(32   → 64, 3×3, pad=1) → BN → ReLU → MaxPool(2) → Drop(0.25)
        Conv(64   → 128,3×3, pad=1) → BN → ReLU → MaxPool(2) → Drop(0.25)
        GlobalAvgPool → Linear(128 → num_classes)

    Input  : (B, in_channels, H, W)  — default CIFAR-10: (B, 3, 32, 32)
    Output : (B, num_classes)
    """

    def __init__(self, num_classes: int = 10, in_channels: int = 3) -> None:
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),          # → 16×16
            nn.Dropout2d(0.25),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),          # → 8×8
            nn.Dropout2d(0.25),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # → 1×1
            nn.Flatten(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.head(x)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def serialised_size_bytes(self) -> int:
        """Approximate on-disk size of state_dict (float32)."""
        return sum(p.numel() * 4 for p in self.parameters())
