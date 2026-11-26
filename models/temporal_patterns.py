from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from models.temporal_semantic_pattern import TemporalSemanticPatternGenerator


class MultiPatternGenerator(TemporalSemanticPatternGenerator):
    """Fresh independent queries; shared projection/time embeddings/MLP.

    Output axes are [batch, pattern, relative time, semantic unit].
    All single-pattern parameters keep their Stage-2 names for recovery tests.
    """

    def __init__(self, language_dim, pattern_dim, pattern_length, num_heads,
                 tau_pattern, num_patterns, eps=1e-8):
        if int(num_patterns) <= 0:
            raise ValueError("num_patterns must be positive")
        super().__init__(language_dim, pattern_dim, pattern_length, num_heads, tau_pattern, eps)
        self.num_patterns = int(num_patterns)
        if self.num_patterns != 1:
            self.pattern_query = nn.Parameter(torch.empty(1, self.num_patterns, self.pattern_dim))
            nn.init.normal_(self.pattern_query, std=self.pattern_dim ** -0.5)

    def forward(self, semantic_units, *, return_debug=False):
        if semantic_units.ndim != 3 or semantic_units.shape[-1] != self.language_dim:
            raise ValueError("Invalid semantic_units dimensions")
        projected = self.semantic_projection(semantic_units)
        query = self.pattern_query.expand(semantic_units.shape[0], -1, -1)
        latent, attention = self.cross_attention(
            query, projected, projected, need_weights=return_debug, average_attn_weights=False)
        codes = self.pattern_mlp(latent[:, :, None, :] + self.relative_time_embeddings[None, None])
        pf = F.normalize(self.pattern_output_projection(codes).float(), dim=-1, eps=self.eps)
        sf = F.normalize(self.semantic_output_projection(semantic_units).float(), dim=-1, eps=self.eps)
        similarities = torch.einsum("bpkd,bmd->bpkm", pf, sf)
        pattern = torch.sigmoid(similarities / self.tau_pattern)
        debug = {"semantic_projected": projected, "pattern_query_context": latent,
                 "pattern_temporal_codes": codes, "pattern_features": pf,
                 "semantic_pattern_features": sf, "pattern_similarity": similarities,
                 "pattern_attention": attention} if return_debug else None
        return pattern, debug
