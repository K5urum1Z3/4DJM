from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from models.occurrence_verifier import OccurrenceVerification
from models.evidence_field_base import SemanticTimeEvidenceField
from models.semantic_unit_extractor import SemanticUnitExtractor
from models.occurrence_head import Stage1VerificationHead
from utils.reason_loss import reason_cross_entropy


class FourDSAVStage1(nn.Module):
    """Semantic evidence verifier with a Qwen3.5 language adapter."""

    def __init__(
        self,
        baseline: nn.Module,
        tokenizer: Any,
        config: dict[str, Any],
    ) -> None:
        super().__init__()
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
        self.evidence_field = SemanticTimeEvidenceField(
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

    @property
    def multimodal_base(self) -> nn.Module:
        return self.baseline.multimodal_base

    def stage1_named_parameters(self) -> list[tuple[str, nn.Parameter]]:
        modules = (
            ("semantic_unit_extractor", self.semantic_unit_extractor),
            ("evidence_field", self.evidence_field),
            ("verification_head", self.verification_head),
        )
        return [
            (f"{prefix}.{name}", parameter)
            for prefix, module in modules
            for name, parameter in module.named_parameters()
            if parameter.requires_grad
        ]

    def trainable_named_parameters(self) -> list[tuple[str, nn.Parameter]]:
        return [
            (f"baseline.backbone.{name}", parameter)
            for name, parameter in self.baseline.lora_named_parameters()
        ] + self.stage1_named_parameters()

    def parameter_count_report(self) -> dict[str, int | float]:
        lora_count = sum(p.numel() for _, p in self.baseline.lora_named_parameters())
        stage_count = sum(p.numel() for _, p in self.stage1_named_parameters())
        total = sum(p.numel() for p in self.parameters())
        trainable = lora_count + stage_count
        return {
            "total_parameters": total,
            "lora_parameters": lora_count,
            "stage1_parameters": stage_count,
            "trainable_parameters": trainable,
            "trainable_fraction": trainable / total if total else 0.0,
        }

    def train(self, mode: bool = True) -> "FourDSAVStage1":
        super().train(mode)
        self.baseline.train(mode)
        if mode:
            self.baseline.head.eval()
            self.baseline.lm_head.eval()
        return self

    def extract_visual_grid(self, features: list[torch.Tensor], grid: torch.Tensor) -> torch.Tensor:
        if grid.ndim == 3 and grid.shape[1] == 1:
            grid = grid[:, 0]
        if grid.ndim != 2 or grid.shape[-1] != 3 or len(features) != grid.shape[0]:
            raise ValueError("Expected one [T,H,W] video grid row per cached feature tensor")
        sequences = []
        for item, row in zip(features, grid):
            t, h, w = (int(value) for value in row.tolist())
            if h % self.visual_merge_size or w % self.visual_merge_size:
                raise ValueError(f"Video grid {t,h,w} is not divisible by merge size {self.visual_merge_size}")
            gh, gw = h // self.visual_merge_size, w // self.visual_merge_size
            expected = t * gh * gw
            if item.ndim != 2 or item.shape[0] != expected:
                raise ValueError(f"Visual token count mismatch: got {tuple(item.shape)}, expected {expected}xD")
            sequences.append(item.reshape(t, gh * gw, item.shape[-1]))
        shapes = {tuple(item.shape) for item in sequences}
        if len(shapes) != 1:
            raise ValueError(f"Variable temporal/spatial feature shapes in a batch are unsupported: {shapes}")
        return torch.stack(sequences, dim=0)

    def _assertion_spans(
        self,
        input_ids: torch.Tensor,
        layouts: list[Any],
        assertions: list[str],
    ) -> list[tuple[int, int]]:
        if len(layouts) != len(assertions) or input_ids.shape[0] != len(assertions):
            raise ValueError("Need one assertion and token layout per batch row")
        special = set(int(value) for value in self.tokenizer.all_special_ids)
        spans = []
        for row, (layout, assertion) in enumerate(zip(layouts, assertions)):
            ids = [int(value) for value in input_ids[row, :layout.user_end_index].detach().cpu().tolist()]
            candidates = [assertion, " " + assertion, "\n" + assertion]
            matches: list[tuple[int, int]] = []
            for text in candidates:
                pattern = self.tokenizer.encode(text, add_special_tokens=False)
                if not pattern:
                    continue
                stop = len(ids) - len(pattern)
                for start in range(max(0, stop + 1)):
                    end = start + len(pattern)
                    if end == layout.classification_token_index + 1 and ids[start:end] == pattern:
                        if not any(token in special for token in ids[start:end]):
                            matches.append((start, end))
            if not matches:
                raise RuntimeError(
                    f"Could not recover complete normal-text assertion span for batch row {row}"
                )
            # Prefer the shortest exact match, so preceding whitespace is not treated as assertion content.
            spans.append(min(matches, key=lambda span: span[1] - span[0]))
        return spans

    def _language_hidden(
        self,
        inputs: dict[str, torch.Tensor],
        video_features: list[torch.Tensor],
    ) -> torch.Tensor:
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        base = self.multimodal_base
        embeds = base.get_input_embeddings()(input_ids)
        flat_video = torch.cat(
            [item.to(device=embeds.device, dtype=embeds.dtype) for item in video_features], dim=0
        )
        _, video_mask = base.get_placeholder_mask(input_ids, embeds, video_features=flat_video)
        embeds = embeds.masked_scatter(video_mask, flat_video)
        position_ids = base.compute_3d_position_ids(
            input_ids=input_ids,
            inputs_embeds=embeds,
            image_grid_thw=None,
            video_grid_thw=inputs["video_grid_thw"],
            attention_mask=attention_mask,
            past_key_values=None,
            mm_token_type_ids=inputs["mm_token_type_ids"],
        )
        outputs = self.baseline.backbone(
            inputs_embeds=embeds,
            position_ids=position_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        return outputs.last_hidden_state

    def predict_from_evidence(
        self,
        evidence_scores: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        occurrence_scores, occurrence_weights = self.occurrence_verifier(evidence_scores)
        logits = self.verification_head(occurrence_scores)
        return logits, occurrence_scores, occurrence_weights

    def stage1_state_dict(self) -> dict[str, torch.Tensor]:
        state: dict[str, torch.Tensor] = {}
        for prefix, module in (
            ("semantic_unit_extractor", self.semantic_unit_extractor),
            ("evidence_field", self.evidence_field),
            ("verification_head", self.verification_head),
        ):
            state.update({f"{prefix}.{name}": value for name, value in module.state_dict().items()})
        return state

    def load_stage1_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        modules = {
            "semantic_unit_extractor": self.semantic_unit_extractor,
            "evidence_field": self.evidence_field,
            "verification_head": self.verification_head,
        }
        for prefix, module in modules.items():
            selected = {
                key[len(prefix) + 1 :]: value
                for key, value in state.items()
                if key.startswith(prefix + ".")
            }
            if not selected:
                raise ValueError(f"Stage-1 checkpoint has no {prefix} parameters")
            module.load_state_dict(selected, strict=True)
        expected_prefixes = tuple(name + "." for name in modules)
        unexpected = [key for key in state if not key.startswith(expected_prefixes)]
        if unexpected:
            raise ValueError(f"Unexpected Stage-1 checkpoint keys: {unexpected[:8]}")

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
        evidence_logits, evidence_scores, spatial_attention = self.evidence_field(
            z, semantic_units, return_spatial_attention=return_debug
        )
        logits, occurrence_scores, occurrence_weights = self.predict_from_evidence(evidence_scores)

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
        total_loss = binary_loss * self.answer_loss_weight if binary_loss is not None else None
        if reason_loss is not None:
            total_loss = reason_loss * self.reason_loss_weight if total_loss is None else total_loss + reason_loss * self.reason_loss_weight

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
        """Standard module entry point required by DistributedDataParallel.

        The single-GPU code may continue using ``forward_with_cached_video``;
        DDP must call the module itself so its reducer can register the forward
        and backward hooks around the nested Qwen/Stage-1 parameters.
        """
        return self.forward_with_cached_video(
            inputs,
            layouts,
            video_features,
            assertions,
            labels=labels,
            compute_reason_loss=compute_reason_loss,
            return_debug=return_debug,
        )
