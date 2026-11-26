from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class MultiScaleCenteredMatcher(nn.Module):
    """Joint softmax over all VALID (scale, pattern, window) candidates.

    No padding, candidate priors, diversity losses, or hidden score correction.
    """

    def __init__(self, pattern_length, num_scales, temporal_pool_type,
                 temporal_pool_kernel, temporal_pool_stride, tau_cfg, eps):
        super().__init__()
        if temporal_pool_type != "avg":
            raise ValueError("Only temporal average pooling is allowed")
        if min(pattern_length, num_scales, temporal_pool_kernel, temporal_pool_stride) <= 0:
            raise ValueError("Lengths/scales/pooling parameters must be positive")
        if min(tau_cfg, eps) <= 0:
            raise ValueError("tau_cfg and eps must be positive")
        self.pattern_length, self.num_scales = int(pattern_length), int(num_scales)
        self.kernel, self.stride = int(temporal_pool_kernel), int(temporal_pool_stride)
        self.tau_cfg, self.eps = float(tau_cfg), float(eps)

    def pyramid(self, evidence):
        if evidence.ndim != 3 or evidence.shape[1] < self.pattern_length:
            raise ValueError("Evidence must be [B,T,M] with T >= pattern_length")
        scales = [evidence.float()]
        for _ in range(1, self.num_scales):
            next_length = (scales[-1].shape[1] - self.kernel) // self.stride + 1
            if next_length < self.pattern_length:
                break  # Never generate invalid scales or fake windows.
            scales.append(F.avg_pool1d(scales[-1].transpose(1, 2), self.kernel,
                                      self.stride).transpose(1, 2))
        return scales

    def forward(self, evidence_scores, pattern, *, return_windows=False,
                allowed_patterns=None, allowed_scales=None):
        if pattern.ndim != 4:
            raise ValueError("Pattern must be [B,P,K,M]")
        b, p, k, m = pattern.shape
        if k != self.pattern_length or evidence_scores.shape[0] != b or evidence_scores.shape[2] != m:
            raise ValueError("Pattern/evidence shape mismatch")
        pattern = pattern.float()
        mean = pattern.mean(dim=2, keepdim=True)
        centered = pattern - mean
        pnorm = torch.linalg.vector_norm(centered.flatten(2), dim=-1)
        scales = self.pyramid(evidence_scores)
        windows, means, relatives, norms, scores, mappings = [], [], [], [], [], []
        for r, evidence in enumerate(scales):
            win = torch.stack([evidence[:, s:s+k] for s in range(evidence.shape[1]-k+1)], dim=1)
            wm = win.mean(dim=2, keepdim=True)
            relative = win - wm
            wnorm = torch.linalg.vector_norm(relative.flatten(2), dim=-1)
            numerator = torch.einsum("bpkm,bnkm->bpn", centered, relative)
            c = numerator / (pnorm[:, :, None] * wnorm[:, None, :] + self.eps)
            scores.append(c)
            windows.append(win); means.append(wm); relatives.append(relative); norms.append(wnorm)
            mappings.extend((r, pi, s) for pi in range(p) for s in range(win.shape[1]))
        candidates = torch.cat([c.flatten(1) for c in scores], dim=1)
        valid = torch.tensor(mappings, dtype=torch.long, device=candidates.device)
        keep = torch.ones(len(mappings), dtype=torch.bool, device=candidates.device)
        if allowed_patterns is not None:
            keep &= torch.isin(valid[:, 1], torch.tensor(allowed_patterns, device=valid.device))
        if allowed_scales is not None:
            keep &= torch.isin(valid[:, 0], torch.tensor(allowed_scales, device=valid.device))
        if not bool(keep.any()):
            raise ValueError("No valid candidates after functional ablation")
        beta = torch.softmax(candidates.masked_fill(~keep[None], -torch.inf) / self.tau_cfg, dim=1)
        cfg = (beta * candidates).sum(dim=1)
        return {"pattern_mean": mean, "pattern_centered": centered,
                "pattern_centered_norm": pnorm, "evidence_by_scale": scales,
                "windows": windows, "window_means": means, "windows_centered": relatives,
                "window_centered_norm": norms, "scores_by_scale": scores,
                "candidate_mapping": valid, "cosine_scores": candidates,
                "beta_cfg": beta, "configuration_score": cfg}
