from __future__ import annotations

import math

import torch
from torch import nn


class EvidenceScorerV2(nn.Module):
    """Visual-only evidence scorer: its sole input is cross-attended r[t,m]."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.network = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), 1),
        )

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        if r.ndim != 4 or r.shape[-1] != self.input_dim:
            raise ValueError(f"Expected r [B,T,M,{self.input_dim}], got {tuple(r.shape)}")
        return self.network(r).squeeze(-1)


class SemanticTimeEvidenceFieldV2(nn.Module):
    """Same semantic-to-video cross-attention as v1, followed by an r-only scorer."""

    def __init__(
        self,
        feature_dim: int,
        attention_dim: int = 256,
        num_heads: int = 8,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if attention_dim % num_heads:
            raise ValueError("attention_dim must be divisible by num_heads")
        self.feature_dim = int(feature_dim)
        self.attention_dim = int(attention_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.attention_dim // self.num_heads
        self.query_proj = nn.Linear(self.feature_dim, self.attention_dim, bias=False)
        self.key_proj = nn.Linear(self.feature_dim, self.attention_dim, bias=False)
        self.value_proj = nn.Linear(self.feature_dim, self.attention_dim, bias=False)
        self.evidence_scorer = EvidenceScorerV2(
            input_dim=self.attention_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

    def forward(
        self,
        visual_tokens: torch.Tensor,
        semantic_units: torch.Tensor,
        return_spatial_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
        if visual_tokens.ndim != 4:
            raise ValueError("visual_tokens must have shape [B,T,P,D]")
        if semantic_units.ndim != 3:
            raise ValueError("semantic_units must have shape [B,M,D]")
        batch, time, spatial, _ = visual_tokens.shape
        if semantic_units.shape[0] != batch or visual_tokens.shape[-1] != self.feature_dim:
            raise ValueError("Visual/semantic dimensions do not match the evidence field")

        # Preserve v1 cross-attention exactly; s affects evidence only through r.
        visual_tokens = visual_tokens.to(dtype=self.key_proj.weight.dtype)
        semantic_units = semantic_units.to(dtype=self.query_proj.weight.dtype)
        units = semantic_units.shape[1]
        q = self.query_proj(semantic_units).view(batch, units, self.num_heads, self.head_dim)
        k = self.key_proj(visual_tokens).view(batch, time, spatial, self.num_heads, self.head_dim)
        v = self.value_proj(visual_tokens).view(batch, time, spatial, self.num_heads, self.head_dim)
        score = torch.einsum("bmhd,btphd->bthmp", q, k) / math.sqrt(self.head_dim)
        spatial_weights = torch.softmax(score.float(), dim=-1).to(score.dtype)
        attended = torch.einsum("bthmp,btphd->bthmd", spatial_weights, v)
        r = attended.permute(0, 1, 3, 2, 4).reshape(
            batch, time, units, self.attention_dim
        )

        # Score each attended visual representation.
        evidence_logits = self.evidence_scorer(r)
        evidence_scores = torch.sigmoid(evidence_logits)
        return evidence_logits, evidence_scores, (spatial_weights if return_spatial_attention else None), r

    def load_v1_cross_attention(self, v1_state: dict[str, torch.Tensor]) -> None:
        for name in ("query_proj", "key_proj", "value_proj"):
            prefix = f"evidence_field.{name}."
            selected = {
                key[len(prefix):]: value
                for key, value in v1_state.items()
                if key.startswith(prefix)
            }
            if not selected:
                raise ValueError(f"v1 checkpoint is missing {prefix}")
            getattr(self, name).load_state_dict(selected, strict=True)
