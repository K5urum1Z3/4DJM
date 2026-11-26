from __future__ import annotations

import torch
from torch import nn


class Stage1VerificationHead(nn.Module):
    """Classifier that consumes only the M occurrence scores."""

    def __init__(self, num_units: int, hidden_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(int(num_units)),
            nn.Linear(int(num_units), int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), 1),
        )

    def forward(self, occurrence_scores: torch.Tensor) -> torch.Tensor:
        if occurrence_scores.ndim != 2:
            raise ValueError("occurrence_scores must have shape [B,M]")
        return self.network(occurrence_scores).squeeze(-1)
