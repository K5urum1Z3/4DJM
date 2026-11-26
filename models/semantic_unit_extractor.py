from __future__ import annotations

import math

import torch
from torch import nn


class SemanticUnitExtractor(nn.Module):
    """Learn M assertion-conditioned latent queries over assertion tokens."""

    def __init__(
        self,
        language_dim: int,
        num_units: int,
        attention_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if attention_dim % num_heads:
            raise ValueError("attention_dim must be divisible by num_heads")
        self.language_dim = int(language_dim)
        self.num_units = int(num_units)
        self.attention_dim = int(attention_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.attention_dim // self.num_heads

        self.semantic_queries = nn.Parameter(torch.empty(self.num_units, self.language_dim))
        nn.init.normal_(self.semantic_queries, mean=0.0, std=self.language_dim ** -0.5)
        self.query_proj = nn.Linear(self.language_dim, self.attention_dim, bias=False)
        self.key_proj = nn.Linear(self.language_dim, self.attention_dim, bias=False)
        self.value_proj = nn.Linear(self.language_dim, self.attention_dim, bias=False)
        self.output_proj = nn.Linear(self.attention_dim, self.language_dim, bias=False)
        self.output_norm = nn.LayerNorm(self.language_dim)
        self.dropout = nn.Dropout(float(dropout))

    def forward(
        self,
        assertion_hidden: torch.Tensor,
        assertion_mask: torch.Tensor,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if assertion_hidden.ndim != 3:
            raise ValueError("assertion_hidden must have shape [B,L,D]")
        batch, length, dim = assertion_hidden.shape
        if dim != self.language_dim or assertion_mask.shape != (batch, length):
            raise ValueError("Assertion hidden/mask dimensions do not match the extractor")
        if not assertion_mask.bool().any(dim=1).all():
            raise ValueError("Each sample must contain at least one assertion token")

        # Keep the new evidence modules in fp32 for stable attention, while
        # retaining an autograd cast edge to the BF16 Qwen hidden states.
        assertion_hidden = assertion_hidden.to(dtype=self.key_proj.weight.dtype)
        queries = self.semantic_queries.unsqueeze(0).expand(batch, -1, -1)
        q = self.query_proj(queries).view(batch, self.num_units, self.num_heads, self.head_dim)
        k = self.key_proj(assertion_hidden).view(batch, length, self.num_heads, self.head_dim)
        v = self.value_proj(assertion_hidden).view(batch, length, self.num_heads, self.head_dim)
        scores = torch.einsum("bmhd,blhd->bhml", q, k) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~assertion_mask[:, None, None, :].bool(), torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores.float(), dim=-1).to(scores.dtype)
        context = torch.einsum("bhml,blhd->bmhd", weights, v).reshape(
            batch, self.num_units, self.attention_dim
        )
        semantic_units = self.output_norm(queries + self.dropout(self.output_proj(context)))
        return semantic_units, weights if return_attention else None
