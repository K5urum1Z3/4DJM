from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class TemporalSemanticPatternGenerator(nn.Module):
    """Generate one assertion-conditioned relative-time × latent-unit pattern.

    F_A[j,m] is the response for latent assertion-conditioned unit m at relative
    temporal position j. The latent units are not assigned human-readable names.
    """

    def __init__(
        self,
        language_dim: int,
        pattern_dim: int = 256,
        pattern_length: int = 4,
        num_heads: int = 8,
        tau_pattern: float = 1.0,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if pattern_dim <= 0 or pattern_dim % num_heads:
            raise ValueError("pattern_dim must be positive and divisible by num_heads")
        if pattern_length <= 0 or tau_pattern <= 0:
            raise ValueError("pattern_length and tau_pattern must be positive")
        self.language_dim = int(language_dim)
        self.pattern_dim = int(pattern_dim)
        self.pattern_length = int(pattern_length)
        self.tau_pattern = float(tau_pattern)
        self.eps = float(eps)

        self.semantic_projection = nn.Linear(self.language_dim, self.pattern_dim)
        self.pattern_query = nn.Parameter(torch.empty(1, 1, self.pattern_dim))
        nn.init.normal_(self.pattern_query, mean=0.0, std=self.pattern_dim ** -0.5)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.pattern_dim,
            num_heads=int(num_heads),
            batch_first=True,
        )
        self.relative_time_embeddings = nn.Parameter(
            torch.empty(self.pattern_length, self.pattern_dim)
        )
        nn.init.normal_(self.relative_time_embeddings, mean=0.0, std=self.pattern_dim ** -0.5)
        self.pattern_mlp = nn.Sequential(
            nn.LayerNorm(self.pattern_dim),
            nn.Linear(self.pattern_dim, self.pattern_dim),
            nn.GELU(),
            nn.Linear(self.pattern_dim, self.pattern_dim),
        )
        self.pattern_output_projection = nn.Linear(self.pattern_dim, self.pattern_dim, bias=False)
        self.semantic_output_projection = nn.Linear(self.language_dim, self.pattern_dim, bias=False)

    def forward(
        self, semantic_units: torch.Tensor, *, return_debug: bool = False
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        if semantic_units.ndim != 3 or semantic_units.shape[-1] != self.language_dim:
            raise ValueError(
                f"Expected semantic_units [B,M,{self.language_dim}], got {tuple(semantic_units.shape)}"
            )
        batch = semantic_units.shape[0]
        # The projection and cross-attention are part of the assertion-conditioned path.
        semantic_projected = self.semantic_projection(semantic_units)
        query = self.pattern_query.expand(batch, -1, -1)
        pooled, attention_weights = self.cross_attention(
            query, semantic_projected, semantic_projected,
            need_weights=return_debug,
            average_attn_weights=False,
        )
        pooled = pooled[:, 0, :]
        generator_input = pooled[:, None, :] + self.relative_time_embeddings[None, :, :]
        temporal_codes = self.pattern_mlp(generator_input)
        pattern_features = self.pattern_output_projection(temporal_codes)
        semantic_features = self.semantic_output_projection(semantic_units)
        pattern_features = F.normalize(pattern_features.float(), p=2, dim=-1, eps=self.eps)
        semantic_features = F.normalize(semantic_features.float(), p=2, dim=-1, eps=self.eps)
        similarities = torch.einsum("bkd,bmd->bkm", pattern_features, semantic_features)
        pattern = torch.sigmoid(similarities / self.tau_pattern)
        if not return_debug:
            return pattern, None
        return pattern, {
            "semantic_projected": semantic_projected,
            "pattern_query_context": pooled,
            "pattern_temporal_codes": temporal_codes,
            "pattern_features": pattern_features,
            "semantic_pattern_features": semantic_features,
            "pattern_similarity": similarities,
            "pattern_attention": attention_weights,
        }
