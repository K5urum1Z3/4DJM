from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from models.evidence_base import FourDSAVStage1
from models.occurrence_verifier import OccurrenceVerification
from models.evidence_field import SemanticTimeEvidenceFieldV2
from models.semantic_unit_extractor import SemanticUnitExtractor
from models.occurrence_head import Stage1VerificationHead
from utils.reason_loss import reason_cross_entropy


class FourDSAVStage1V2(FourDSAVStage1):
    """Stage-1 v2; inherits the unchanged v1 backbone/selection/occurrence helpers."""

    def __init__(self, baseline: nn.Module, tokenizer: Any, config: dict[str, Any]) -> None:
        nn.Module.__init__(self)
        self.baseline = baseline
        self.tokenizer = tokenizer
        stage = config["stage1"]
        feature_dim = int(config["classifier"]["input_dim"])
        language_dim = int(baseline.multimodal_base.config.text_config.hidden_size)
        if feature_dim != language_dim:
            raise ValueError(
                f"Runtime feature dimensions differ ({feature_dim} vs {language_dim}); "
                "this configuration expects no additional projection"
            )
        self.num_units = int(stage["num_semantic_units"])
        self.answer_loss_weight = float(config["loss"].get("answer_loss_weight", 1.0))
        self.reason_loss_weight = float(config["loss"]["reason_loss_weight"])
        self.visual_merge_size = int(baseline.multimodal_base.visual.spatial_merge_size)
        self.semantic_unit_extractor = SemanticUnitExtractor(
            language_dim=language_dim,
            num_units=self.num_units,
            attention_dim=int(stage["attention_dim"]),
            num_heads=int(stage["num_attention_heads"]),
            dropout=float(stage["dropout"]),
        )
        self.evidence_field = SemanticTimeEvidenceFieldV2(
            feature_dim=feature_dim,
            attention_dim=int(stage["attention_dim"]),
            num_heads=int(stage["num_attention_heads"]),
            hidden_dim=int(stage["evidence_hidden_dim"]),
            dropout=float(stage["dropout"]),
        )
        self.occurrence_verifier = OccurrenceVerification(
            temperature=float(stage["occurrence_temperature"])
        )
        self.verification_head = Stage1VerificationHead(
            num_units=self.num_units,
            hidden_dim=int(stage["head_hidden_dim"]),
            dropout=float(stage["dropout"]),
        )
        self.baseline.head.requires_grad_(False)

    def initialize_shared_from_v1(self, v1_state: dict[str, torch.Tensor]) -> dict[str, Any]:
        """Initialize shared modules and retain the new evidence scorer."""
        modules = {
            "semantic_unit_extractor": self.semantic_unit_extractor,
            "verification_head": self.verification_head,
        }
        copied = {}
        for prefix, module in modules.items():
            selected = {
                key[len(prefix) + 1:]: value
                for key, value in v1_state.items()
                if key.startswith(prefix + ".")
            }
            if not selected:
                raise ValueError(f"v1 checkpoint has no {prefix} state")
            module.load_state_dict(selected, strict=True)
            copied[prefix] = sum(value.numel() for value in selected.values())
        self.evidence_field.load_v1_cross_attention(v1_state)
        copied["evidence_cross_attention"] = sum(
            parameter.numel()
            for module in (
                self.evidence_field.query_proj,
                self.evidence_field.key_proj,
                self.evidence_field.value_proj,
            )
            for parameter in module.parameters()
        )
        return {
            "copied_parameter_counts": copied,
            "v1_evidence_scorer_loaded": False,
            "v2_evidence_scorer_initialization": "fresh_default_initialization",
        }

    def load_stage1_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        modules = {
            "semantic_unit_extractor": self.semantic_unit_extractor,
            "evidence_field": self.evidence_field,
            "verification_head": self.verification_head,
        }
        for prefix, module in modules.items():
            selected = {
                key[len(prefix) + 1:]: value
                for key, value in state.items()
                if key.startswith(prefix + ".")
            }
            if not selected:
                raise ValueError(f"Stage-1 v2 checkpoint has no {prefix} parameters")
            module.load_state_dict(selected, strict=True)
        expected = tuple(f"{name}." for name in modules)
        unexpected = [key for key in state if not key.startswith(expected)]
        if unexpected:
            raise ValueError(f"Unexpected Stage-1 v2 checkpoint keys: {unexpected[:8]}")

    def forward_with_cached_video(
        self,
        inputs: dict[str, torch.Tensor],
        layouts: list[Any],
        video_features: list[torch.Tensor],
        assertions: list[str],
        labels: torch.Tensor | None = None,
        compute_reason_loss: bool = True,
        return_debug: bool = True,
    ) -> dict[str, Any]:
        hidden = self._language_hidden(inputs, video_features)
        spans = self._assertion_spans(inputs["input_ids"], layouts, assertions)
        sequences = [hidden[index, start:end] for index, (start, end) in enumerate(spans)]
        lengths = torch.tensor([item.shape[0] for item in sequences], device=hidden.device)
        assertion_hidden = pad_sequence(sequences, batch_first=True)
        positions = torch.arange(assertion_hidden.shape[1], device=hidden.device)
        assertion_mask = positions.unsqueeze(0) < lengths.unsqueeze(1)

        z = self.extract_visual_grid(video_features, inputs["video_grid_thw"])
        semantic_units, semantic_attention = self.semantic_unit_extractor(
            assertion_hidden, assertion_mask, return_attention=return_debug
        )
        evidence_logits, evidence_scores, spatial_attention, r = self.evidence_field(
            z, semantic_units, return_spatial_attention=return_debug
        )
        logits, occurrence_scores, occurrence_weights = self.predict_from_evidence(evidence_scores)

        binary_loss = None
        if labels is not None:
            binary_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits.float(), labels.float()
            )
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
        total_loss = binary_loss * self.answer_loss_weight if binary_loss is not None else None
        if reason_loss is not None:
            total_loss = (
                reason_loss * self.reason_loss_weight
                if total_loss is None
                else total_loss + reason_loss * self.reason_loss_weight
            )

        result: dict[str, Any] = {
            "logits": logits,
            "loss": total_loss,
            "binary_loss": binary_loss,
            "reason_loss": reason_loss,
            "reason_token_count": reason_tokens,
            "semantic_units": semantic_units,
            "evidence_logits": evidence_logits,
            "evidence_scores": evidence_scores,
            "occurrence_scores": occurrence_scores,
            "occurrence_weights": occurrence_weights,
            "assertion_spans": spans,
        }
        if return_debug:
            result.update({
                "visual_tokens": z,
                "assertion_hidden": assertion_hidden,
                "assertion_mask": assertion_mask,
                "semantic_attention": semantic_attention,
                "spatial_attention": spatial_attention,
                "r": r,
            })
        return result

    def forward(
        self,
        inputs: dict[str, torch.Tensor],
        layouts: list[Any],
        video_features: list[torch.Tensor],
        assertions: list[str],
        labels: torch.Tensor | None = None,
        compute_reason_loss: bool = True,
        return_debug: bool = False,
    ) -> dict[str, Any]:
        return self.forward_with_cached_video(
            inputs,
            layouts,
            video_features,
            assertions,
            labels=labels,
            compute_reason_loss=compute_reason_loss,
            return_debug=return_debug,
        )
