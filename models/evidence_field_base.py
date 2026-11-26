from __future__ import annotations

import math

import torch
from torch import nn


class SemanticTimeEvidenceField(nn.Module):
    """Per-time, per-semantic-unit visual evidence over spatial tokens."""

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
        self.semantic_proj = nn.Linear(self.feature_dim, self.attention_dim, bias=False)
        self.evidence_scorer = nn.Sequential(
            nn.LayerNorm(self.attention_dim * 2),
            nn.Linear(self.attention_dim * 2, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), 1),
        )

    def forward(
        self,
        visual_tokens: torch.Tensor,
        semantic_units: torch.Tensor,
        return_spatial_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if visual_tokens.ndim != 4:
            raise ValueError("visual_tokens must have shape [B,T,P,D]")
        if semantic_units.ndim != 3:
            raise ValueError("semantic_units must have shape [B,M,D]")
        batch, time, spatial, dim = visual_tokens.shape
        if semantic_units.shape[0] != batch or visual_tokens.shape[-1] != self.feature_dim:
            raise ValueError("Visual/semantic dimensions do not match the evidence field")

        # Evidence attention is intentionally fp32 even when cached Qwen
        # features are BF16; gradients still flow through these differentiable casts.
        visual_tokens = visual_tokens.to(dtype=self.key_proj.weight.dtype)
        semantic_units = semantic_units.to(dtype=self.query_proj.weight.dtype)

        units = semantic_units.shape[1]
        q = self.query_proj(semantic_units).view(batch, units, self.num_heads, self.head_dim)
        k = self.key_proj(visual_tokens).view(batch, time, spatial, self.num_heads, self.head_dim)
        v = self.value_proj(visual_tokens).view(batch, time, spatial, self.num_heads, self.head_dim)

        # Scores are normalized over P only: each (t,m) independently queries Z[t].
        score = torch.einsum("bmhd,btphd->bthmp", q, k) / math.sqrt(self.head_dim)
        spatial_weights = torch.softmax(score.float(), dim=-1).to(score.dtype)
        attended = torch.einsum("bthmp,btphd->bthmd", spatial_weights, v)
        attended = attended.permute(0, 1, 3, 2, 4).reshape(
            batch, time, units, self.attention_dim
        )
        sem = self.semantic_proj(semantic_units)[:, None, :, :].expand(-1, time, -1, -1)
        evidence_input = torch.cat((attended, sem), dim=-1)
        evidence_logits = self.evidence_scorer(evidence_input).squeeze(-1)
        evidence_scores = torch.sigmoid(evidence_logits)
        return evidence_logits, evidence_scores, spatial_weights if return_spatial_attention else None
