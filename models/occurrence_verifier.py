from __future__ import annotations

import torch
from torch import nn


class OccurrenceVerification(nn.Module):
    """Differentiable, time-permutation-invariant evidence occurrence pooling."""

    def __init__(self, temperature: float) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("occurrence temperature must be positive")
        self.temperature = float(temperature)

    def forward(self, evidence_scores: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if evidence_scores.ndim != 3:
            raise ValueError("evidence_scores must have shape [B,T,M]")
        weights = torch.softmax(evidence_scores.float() / self.temperature, dim=1)
        occurrence_scores = (weights * evidence_scores.float()).sum(dim=1)
        return occurrence_scores, weights
