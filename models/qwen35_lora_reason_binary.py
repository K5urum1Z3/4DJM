from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from torch import nn
from transformers import AutoModelForImageTextToText, AutoProcessor

from utils.reason_loss import ReasonTokenLayout, reason_cross_entropy
from utils.config import resolve_dtype


LANGUAGE_LORA_RE = re.compile(
    r"\.language_model\.layers\.(\d+)\."
    r"(?:(?:self_attn\.(?:q_proj|v_proj))|(?:linear_attn\.in_proj_qkv))\."
    r"lora_[AB](?:\.|$)"
)


def load_qwen35_backbone(
    model_path: str,
    device: str,
    torch_dtype: torch.dtype,
    attention_backend: str,
) -> nn.Module:
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        local_files_only=True,
        dtype=torch_dtype,
        attn_implementation=attention_backend,
        low_cpu_mem_usage=True,
        device_map={"": device},
    )
    model.requires_grad_(False)
    model.eval()
    return model


class _LanguageAdapterHead(nn.Module):
    """Language adapter and binary classification head."""

    def __init__(
        self,
        peft_backbone: PeftModel,
        tokenizer: Any,
        hidden_size: int,
        head_hidden_dim: int = 512,
        dropout: float = 0.1,
        use_cache: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = peft_backbone
        self.tokenizer = tokenizer
        self.use_cache = bool(use_cache)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, head_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_dim, 1),
        )

        config = self.backbone.base_model.model.config
        special_ids = set(tokenizer.all_special_ids)
        for token_name in ("vision_start_token_id", "vision_end_token_id", "image_token_id", "video_token_id"):
            value = getattr(config, token_name, None)
            if value is not None:
                special_ids.add(int(value))
        self.register_buffer(
            "special_token_ids",
            torch.tensor(sorted(special_ids), dtype=torch.long),
            persistent=False,
        )
        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.im_end_token_id = int(im_end_id) if im_end_id is not None else -1

    @property
    def multimodal_base(self) -> nn.Module:
        return self.backbone.base_model.model

    def train(self, mode: bool = True) -> "_LanguageAdapterHead":
        """Keep frozen vision eval-only while LM checkpointing/LoRA is trainable."""
        super().train(mode)
        if mode:
            self.multimodal_base.visual.eval()
            self.multimodal_base.language_model.train(True)
            self.head.train(True)
            for name, module in self.backbone.named_modules():
                if "lora_" in name:
                    module.train(True)
        else:
            self.backbone.eval()
            self.head.eval()
        return self

    def classify_features(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(features).squeeze(-1)

    def head_state_dict(self) -> dict[str, torch.Tensor]:
        return self.head.state_dict()

    def load_head_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.head.load_state_dict(state_dict)

    def lora_named_parameters(self) -> list[tuple[str, nn.Parameter]]:
        return [
            (name, parameter)
            for name, parameter in self.backbone.named_parameters()
            if parameter.requires_grad and "lora_" in name
        ]

    def head_named_parameters(self) -> list[tuple[str, nn.Parameter]]:
        return [(f"head.{name}", parameter) for name, parameter in self.head.named_parameters() if parameter.requires_grad]

    def trainable_named_parameters(self) -> list[tuple[str, nn.Parameter]]:
        return [(f"backbone.{name}", parameter) for name, parameter in self.lora_named_parameters()] + self.head_named_parameters()


class Qwen35LoraReasonBinaryClassifier(_LanguageAdapterHead):
    """Qwen3.5 reason classifier with a frozen-vision feature-cache path."""

    def __init__(
        self,
        *args: Any,
        lm_head: torch.nn.Module,
        reason_token_chunk: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.lm_head = lm_head
        self.reason_token_chunk = (
            int(reason_token_chunk) if reason_token_chunk is not None else None
        )
        self.lm_head.requires_grad_(False)
        self.lm_head.eval()

    def train(self, mode: bool = True) -> "Qwen35LoraReasonBinaryClassifier":
        super().train(mode)
        self.lm_head.eval()
        if hasattr(self, "_generation_model") and self._generation_model is not None:
            self._generation_model.eval()
        return self

    @property
    def generation_model(self) -> Any | None:
        """The outer conditional-generation wrapper, when retained for eval."""
        return getattr(self, "_generation_model", None)

    def forward(
        self,
        inputs: dict[str, torch.Tensor],
        layouts: list[ReasonTokenLayout],
        compute_reason_loss: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, Any]]:
        outputs = self.backbone(**inputs, use_cache=self.use_cache)
        hidden = outputs.last_hidden_state
        selected = torch.tensor(
            [layout.classification_token_index for layout in layouts],
            dtype=torch.long,
            device=hidden.device,
        )
        batch_indices = torch.arange(hidden.shape[0], device=hidden.device)
        features = hidden[batch_indices, selected].float()
        answer_logits = self.classify_features(features)
        reason_loss = None
        reason_token_count = 0
        if compute_reason_loss:
            reason_loss, reason_token_count = reason_cross_entropy(
                hidden,
                inputs["input_ids"],
                self.lm_head,
                layouts,
                token_chunk_size=self.reason_token_chunk,
            )
        return answer_logits, reason_loss, {
            "features": features,
            "selected_token_indices": selected,
            "reason_token_count": reason_token_count,
        }

    @torch.no_grad()
    def encode_video_features(
        self,
        pixel_values_videos: torch.Tensor,
        video_grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        self.multimodal_base.visual.eval()
        outputs = self.multimodal_base.get_video_features(
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            return_dict=True,
        )
        features = outputs.pooler_output
        if len(features) != 1:
            raise ValueError(
                "encode_video_features expects exactly one scene at a time"
            )
        return features[0].detach()

    def forward_with_cached_video(
        self,
        inputs: dict[str, torch.Tensor],
        layouts: list[ReasonTokenLayout],
        video_features: list[torch.Tensor],
        compute_reason_loss: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, Any]]:
        """Run the trainable language path while reusing frozen vision outputs."""
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        base = self.multimodal_base
        inputs_embeds = base.get_input_embeddings()(input_ids)
        flat_video = torch.cat(
            [item.to(inputs_embeds.device, inputs_embeds.dtype) for item in video_features],
            dim=0,
        )
        _, video_mask = base.get_placeholder_mask(
            input_ids,
            inputs_embeds=inputs_embeds,
            video_features=flat_video,
        )
        inputs_embeds = inputs_embeds.masked_scatter(video_mask, flat_video)
        position_ids = base.compute_3d_position_ids(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            image_grid_thw=None,
            video_grid_thw=inputs["video_grid_thw"],
            attention_mask=attention_mask,
            past_key_values=None,
            mm_token_type_ids=inputs["mm_token_type_ids"],
        )
        outputs = self.backbone(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attention_mask=attention_mask,
            use_cache=self.use_cache,
        )
        hidden = outputs.last_hidden_state
        selected = torch.tensor(
            [layout.classification_token_index for layout in layouts],
            dtype=torch.long,
            device=hidden.device,
        )
        batch_indices = torch.arange(hidden.shape[0], device=hidden.device)
        features = hidden[batch_indices, selected].float()
        answer_logits = self.classify_features(features)
        reason_loss = None
        reason_token_count = 0
        if compute_reason_loss:
            reason_loss, reason_token_count = reason_cross_entropy(
                hidden,
                input_ids,
                self.lm_head,
                layouts,
                token_chunk_size=self.reason_token_chunk,
            )
        return answer_logits, reason_loss, {
            "features": features,
            "selected_token_indices": selected,
            "reason_token_count": reason_token_count,
            "cached_video": True,
        }


def _language_targets(multimodal_model: torch.nn.Module) -> list[str]:
    language_model = getattr(multimodal_model, "language_model", None)
    layers = getattr(language_model, "layers", None)
    if layers is None:
        raise RuntimeError("Qwen3.5 language layers were not found")
    targets: list[str] = []
    for index, layer in enumerate(layers):
        if hasattr(layer, "linear_attn"):
            projection = getattr(layer.linear_attn, "in_proj_qkv", None)
            if projection is None:
                raise RuntimeError(f"Layer {index} has no linear_attn.in_proj_qkv")
            targets.append(f"language_model.layers.{index}.linear_attn.in_proj_qkv")
        elif hasattr(layer, "self_attn"):
            for projection_name in ("q_proj", "v_proj"):
                if not hasattr(layer.self_attn, projection_name):
                    raise RuntimeError(
                        f"Layer {index} has no self_attn.{projection_name}"
                    )
                targets.append(
                    f"language_model.layers.{index}.self_attn.{projection_name}"
                )
        else:
            raise RuntimeError(f"Unsupported Qwen3.5 attention layer {index}")
    return targets


def build_qwen35_lora_config(
    config: dict[str, Any], multimodal_model: torch.nn.Module
) -> LoraConfig:
    lora = config["lora"]
    return LoraConfig(
        task_type=None,
        target_modules=_language_targets(multimodal_model),
        r=int(lora["r"]),
        lora_alpha=int(lora["alpha"]),
        lora_dropout=float(lora["dropout"]),
        bias=str(lora.get("bias", "none")),
        inference_mode=False,
    )


def load_qwen35_lora_reason_runtime(
    config: dict[str, Any],
    device: str,
    adapter_path: str | Path | None = None,
    trainable: bool = True,
    keep_generation_model: bool = False,
) -> tuple[Any, Qwen35LoraReasonBinaryClassifier]:
    model_path = config["paths"]["model"]
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    dtype = resolve_dtype(config["model"]["torch_dtype"])
    outer = load_qwen35_backbone(
        model_path=model_path,
        device=device,
        torch_dtype=dtype,
        attention_backend=config["model"]["attention_backend"],
    )
    base_multimodal = outer.model
    lm_head = outer.lm_head
    lm_head.requires_grad_(False)
    lm_head.eval()

    if adapter_path is None:
        peft_backbone = get_peft_model(
            base_multimodal,
            build_qwen35_lora_config(config, base_multimodal),
        )
    else:
        peft_backbone = PeftModel.from_pretrained(
            base_multimodal, str(adapter_path), is_trainable=trainable
        )

    generation_model = None
    if keep_generation_model:
        outer.model = peft_backbone
        generation_model = outer
    else:
        del outer

    if bool(config["model"].get("gradient_checkpointing", True)):
        base_multimodal.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        base_multimodal.enable_input_require_grads()

    classifier = Qwen35LoraReasonBinaryClassifier(
        peft_backbone=peft_backbone,
        tokenizer=processor.tokenizer,
        hidden_size=int(base_multimodal.config.text_config.hidden_size),
        head_hidden_dim=int(config["classifier"]["hidden_dim"]),
        dropout=float(config["classifier"]["dropout"]),
        use_cache=bool(config["model"].get("use_cache", False)),
        lm_head=lm_head,
        reason_token_chunk=config.get("performance", {}).get("reason_token_chunk"),
    )
    if generation_model is not None:
        object.__setattr__(classifier, "_generation_model", generation_model)
        generation_model.eval()
    classifier.head.to(device)
    classifier.train(trainable)
    if not trainable:
        classifier.eval()
    return processor, classifier


def audit_qwen35_trainables(
    classifier: Qwen35LoraReasonBinaryClassifier,
) -> dict[str, Any]:
    lora = classifier.lora_named_parameters()
    head = classifier.head_named_parameters()
    invalid_lora = [name for name, _ in lora if LANGUAGE_LORA_RE.search(name) is None]
    vision_trainables = [
        name
        for name, parameter in classifier.backbone.named_parameters()
        if parameter.requires_grad and ".visual." in f".{name}"
    ]
    base_trainables = [
        name
        for name, parameter in classifier.backbone.named_parameters()
        if parameter.requires_grad and "lora_" not in name
    ]
    lm_head_trainables = [
        name
        for name, parameter in classifier.lm_head.named_parameters()
        if parameter.requires_grad
    ]
    return {
        "trainable_names": [f"backbone.{name}" for name, _ in lora]
        + [name for name, _ in head],
        "lora_parameter_count": sum(parameter.numel() for _, parameter in lora),
        "head_parameter_count": sum(parameter.numel() for _, parameter in head),
        "trainable_parameter_count": sum(
            parameter.numel() for _, parameter in lora + head
        ),
        "lora_tensor_count": len(lora),
        "lora_target_module_count": len(lora) // 2,
        "invalid_lora_names": invalid_lora,
        "vision_trainables": vision_trainables,
        "base_language_trainables": base_trainables,
        "lm_head_trainables": lm_head_trainables,
        "total_backbone_parameters": sum(
            parameter.numel() for parameter in classifier.multimodal_base.parameters()
        )
        + sum(parameter.numel() for parameter in classifier.lm_head.parameters()),
    }


def assert_qwen35_trainable_integrity(
    classifier: Qwen35LoraReasonBinaryClassifier,
) -> dict[str, Any]:
    report = audit_qwen35_trainables(classifier)
    failures = {
        key: report[key]
        for key in (
            "invalid_lora_names",
            "vision_trainables",
            "base_language_trainables",
            "lm_head_trainables",
        )
        if report[key]
    }
    if failures:
        raise AssertionError(f"Qwen3.5 trainable-integrity failure: {failures}")
    if report["lora_target_module_count"] != 40:
        raise AssertionError(
            "Expected 40 Qwen3.5 language LoRA targets, got "
            f"{report['lora_target_module_count']}"
        )
    if report["lora_parameter_count"] != 6_946_816:
        raise AssertionError(
            "Unexpected Qwen3.5 LoRA parameter count: "
            f"{report['lora_parameter_count']}"
        )
    return report
