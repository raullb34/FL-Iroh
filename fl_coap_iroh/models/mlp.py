"""
SimpleMLP — two-hidden-layer MLP for MNIST / Fashion-MNIST.

~100 K parameters. Intentionally small so the focus of experiments stays
on the communication layer rather than computation time.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SimpleMLP(nn.Module):
    """
    MLP for flat-pixel datasets (MNIST / Fashion-MNIST).

    Layers::
        Flatten → Linear(input_dim→h1) → ReLU → Dropout(0.3)
                → Linear(h1→h2)        → ReLU
                → Linear(h2→num_classes)

    Default: input_dim=784 (28×28), h1=256, h2=128 → ~100 K params.
    """

    def __init__(
        self,
        input_dim  : int = 784,
        hidden1    : int = 256,
        hidden2    : int = 128,
        num_classes: int = 10,
        dropout    : float = 0.3,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return self.net(x)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def serialised_size_bytes(self) -> int:
        return sum(p.numel() * 4 for p in self.parameters())
