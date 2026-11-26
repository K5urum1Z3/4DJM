from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from models.temporal_patterns import MultiPatternGenerator
from models.temporal_matching import MultiScaleCenteredMatcher
from utils.reason_loss import reason_cross_entropy


class Stage3VerificationHead(nn.Module):
    """Binary head over [occurrence scores ; one configuration score]."""

    def __init__(self, num_units: int, hidden_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(int(num_units) + 1),
            nn.Linear(int(num_units) + 1, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), 1),
        )

    def forward(self, verification_representation: torch.Tensor) -> torch.Tensor:
        if verification_representation.ndim != 2:
            raise ValueError("verification_representation must have shape [B,M+1]")
        return self.network(verification_representation).squeeze(-1)


class FourDSAVStage3(nn.Module):
    """Stage-3 verifier with the published parameter names and forward equations."""

    def __init__(self, stage1, config):
        nn.Module.__init__(self)
        if not config["freeze_stage1_v2"] or config["load_stage2_trained_weights"]:
            raise ValueError("Stage-3 requires frozen Stage-1 and no Stage-2 weight loading")
        self.stage1 = stage1
        self.stage1_frozen = True
        self.stage1.requires_grad_(False)
        self.stage1.eval()
        self.num_units = int(stage1.num_units)
        self.answer_loss_weight = float(config["loss"]["answer_loss_weight"])
        self.reason_loss_weight = float(config["loss"]["reason_loss_weight"])
        cfg = config["stage3"]
        self.pattern_generator = MultiPatternGenerator(
            language_dim=int(stage1.baseline.multimodal_base.config.text_config.hidden_size),
            pattern_dim=int(cfg["pattern_dim"]), pattern_length=int(cfg["pattern_length"]),
            num_heads=int(cfg["pattern_num_heads"]), tau_pattern=float(cfg["tau_pattern"]),
            num_patterns=int(cfg["num_patterns"]))
        self.configuration_matcher = MultiScaleCenteredMatcher(
            pattern_length=int(cfg["pattern_length"]), num_scales=int(cfg["num_scales"]),
            temporal_pool_type=cfg["temporal_pool_type"],
            temporal_pool_kernel=int(cfg["temporal_pool_kernel"]),
            temporal_pool_stride=int(cfg["temporal_pool_stride"]),
            tau_cfg=float(cfg["tau_cfg"]), eps=float(cfg["matcher_eps"]))
        self.verification_head = Stage3VerificationHead(
            self.num_units, int(cfg["head_hidden_dim"]), float(cfg["dropout"]))

    @property
    def baseline(self) -> nn.Module:
        return self.stage1.baseline

    def train(self, mode: bool = True) -> "FourDSAVStage3":
        super().train(mode)
        self.stage1.train(False if self.stage1_frozen else mode)
        return self

    def forward_with_cached_video(
        self,
        inputs: dict[str, torch.Tensor],
        layouts: list[Any],
        video_features: list[torch.Tensor],
        assertions: list[str],
        labels: torch.Tensor | None = None,
        compute_reason_loss: bool = True,
        return_debug: bool = False,
    ) -> dict[str, Any]:
        hidden = self.stage1._language_hidden(inputs, video_features)
        spans = self.stage1._assertion_spans(inputs["input_ids"], layouts, assertions)
        assertion_sequences = [hidden[index, start:end] for index, (start, end) in enumerate(spans)]
        lengths = torch.tensor([sequence.shape[0] for sequence in assertion_sequences], device=hidden.device)
        assertion_hidden = pad_sequence(assertion_sequences, batch_first=True)
        positions = torch.arange(assertion_hidden.shape[1], device=hidden.device)
        assertion_mask = positions.unsqueeze(0) < lengths.unsqueeze(1)

        visual_tokens = self.stage1.extract_visual_grid(video_features, inputs["video_grid_thw"])
        semantic_units, semantic_attention = self.stage1.semantic_unit_extractor(
            assertion_hidden, assertion_mask, return_attention=return_debug
        )
        evidence_logits, evidence_scores, spatial_attention, cross_attended = self.stage1.evidence_field(
            visual_tokens, semantic_units, return_spatial_attention=return_debug
        )
        occurrence_scores, occurrence_weights = self.stage1.occurrence_verifier(evidence_scores)
        pattern, pattern_debug = self.pattern_generator(semantic_units, return_debug=return_debug)
        matching = self.configuration_matcher(evidence_scores, pattern, return_windows=return_debug)
        configuration_score = matching["configuration_score"]
        representation = torch.cat([occurrence_scores.float(), configuration_score[:, None]], dim=-1)
        logits = self.verification_head(representation)

        binary_loss = None
        if labels is not None:
            binary_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits.float(), labels.float())
        reason_loss = None
        reason_tokens = 0
        if compute_reason_loss:
            reason_loss, reason_tokens = reason_cross_entropy(
                hidden,
                inputs["input_ids"],
                self.baseline.lm_head,
                layouts,
                token_chunk_size=self.baseline.reason_token_chunk,
            )
        loss = binary_loss * self.answer_loss_weight if binary_loss is not None else None
        if reason_loss is not None:
            loss = reason_loss * self.reason_loss_weight if loss is None else loss + reason_loss * self.reason_loss_weight

        result: dict[str, Any] = {
            "logits": logits,
            "loss": loss,
            "binary_loss": binary_loss,
            "reason_loss": reason_loss,
            "reason_token_count": reason_tokens,
            "semantic_units": semantic_units,
            "evidence_logits": evidence_logits,
            "evidence_scores": evidence_scores,
            "occurrence_scores": occurrence_scores,
            "occurrence_weights": occurrence_weights,
            "pattern": pattern,
            "pattern_mean": matching["pattern_mean"],
            "pattern_centered": matching["pattern_centered"],
            "pattern_centered_norm": matching["pattern_centered_norm"],
            "windows": matching["windows"],
            "window_means": matching["window_means"],
            "windows_centered": matching["windows_centered"],
            "window_centered_norm": matching["window_centered_norm"],
            "cosine_scores": matching["cosine_scores"],
            "beta_cfg": matching["beta_cfg"],
            "configuration_score": configuration_score,
            "verification_representation": representation,
            "assertion_spans": spans,
        }
        if return_debug:
            result.update({
                "visual_tokens": visual_tokens,
                "assertion_hidden": assertion_hidden,
                "assertion_mask": assertion_mask,
                "semantic_attention": semantic_attention,
                "spatial_attention": spatial_attention,
                "cross_attended_evidence": cross_attended,
                **{f"pattern_{key}": value for key, value in (pattern_debug or {}).items()},
            })
        return result

    def stage3_named_parameters(self) -> list[tuple[str, nn.Parameter]]:
        return [
            (f"pattern_generator.{name}", parameter)
            for name, parameter in self.pattern_generator.named_parameters()
            if parameter.requires_grad
        ] + [
            (f"verification_head.{name}", parameter)
            for name, parameter in self.verification_head.named_parameters()
            if parameter.requires_grad
        ]

    def stage3_state_dict(self) -> dict[str, torch.Tensor]:
        state: dict[str, torch.Tensor] = {}
        for prefix, module in (
            ("pattern_generator", self.pattern_generator),
            ("verification_head", self.verification_head),
        ):
            state.update({f"{prefix}.{name}": value for name, value in module.state_dict().items()})
        return state

    def load_stage3_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        modules = {
            "pattern_generator": self.pattern_generator,
            "verification_head": self.verification_head,
        }
        for prefix, module in modules.items():
            selected = {
                key[len(prefix) + 1:]: value for key, value in state.items()
                if key.startswith(prefix + ".")
            }
            if not selected:
                raise ValueError(f"Stage-3 checkpoint has no {prefix} parameters")
            module.load_state_dict(selected, strict=True)
        expected = tuple(f"{name}." for name in modules)
        unexpected = [key for key in state if not key.startswith(expected)]
        if unexpected:
            raise ValueError(f"Unexpected Stage-3 checkpoint keys: {unexpected[:8]}")
