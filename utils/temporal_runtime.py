from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from transformers import get_cosine_schedule_with_warmup

from models.qwen35_lora_reason_binary import LANGUAGE_LORA_RE
from utils.config import load_yaml

VARIANT = "stage3_frozen_stage1_custom_trainable_language_lora"


def load_stage3_lora_config(path):
    path = Path(path).resolve()
    patch = json.loads(path.read_text()) if path.suffix == ".json" else load_yaml(path)
    if "inherits" in patch:
        raise ValueError("Use the fully resolved Stage-3 configuration")
    config = patch
    training, loss, inputs = config["training"], config["loss"], config["input"]
    if config.get("variant") != VARIANT:
        raise ValueError("Not a Stage-3 language-LoRA variant config")
    if (training["freeze_stage1"] or not training["freeze_stage1_custom"]
            or not training["train_qwen_language_lora"]):
        raise ValueError("Require fixed custom Stage-1 and trainable language LoRA")
    if not config["freeze_stage1_v2"] or config["load_stage2_trained_weights"]:
        raise ValueError("Stage-1 custom warm-start and fresh Stage-3 are required")
    if config["stage3_lora_execution"]["use_cached_stage1_outputs"]:
        raise ValueError("Cached Stage-1 outputs are invalid with trainable LoRA")
    if not inputs["use_reason_as_assistant_target"] or any(inputs[k] for k in (
            "use_reason_as_classifier_input", "use_answer_as_input", "use_capability_as_input")):
        raise ValueError("Reason must be a causal assistant target only")
    if (not math.isfinite(float(loss["reason_loss_weight"]))
            or float(loss["reason_loss_weight"]) <= 0):
        raise ValueError("A positive finite reason weight is required")
    if float(loss["answer_loss_weight"]) <= 0:
        raise ValueError("A positive answer weight is required")
    if training["batch_size"] != 1 or training["gradient_accumulation_steps"] < 1:
        raise ValueError("Use batch size 1 and positive accumulation")
    if training["epochs"] < 1 or any(training[k] < 1 for k in (
            "validate_every_steps", "save_every_steps")):
        raise ValueError("Epoch count and intervals must be positive")
    if training["effective_batch_size"] != training["gradient_accumulation_steps"]:
        raise ValueError("Effective batch size mismatch")
    for key in ("lora_learning_rate", "stage3_learning_rate", "eps", "max_grad_norm"):
        value = float(training[key])
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"Invalid training.{key}")
    if training["optimizer"] != "adamw" or training["scheduler"] != "cosine":
        raise ValueError("Use AdamW with cosine scheduling")
    if training["best_metric"] != "validation_macro_f1":
        raise ValueError("Checkpoint selection must use validation macro-F1")
    return config


def frozen_custom_digest(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.stage1.stage1_state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((tuple(tensor.shape), tensor.dtype)).encode())
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def trainability_report(model, *, trainable=True):
    counts = {"lora": 0, "stage3": 0, "frozen": 0}
    allowed = set()
    for name, parameter in model.named_parameters():
        prefix = "stage1.baseline.backbone."
        is_lora = name.startswith(prefix) and bool(LANGUAGE_LORA_RE.search(name[len(prefix):]))
        is_stage3 = name.startswith(("pattern_generator.", "verification_head."))
        expected = trainable and (is_lora or is_stage3)
        if parameter.requires_grad != expected:
            raise RuntimeError(f"Trainability contract failed: {name}")
        kind = ("lora" if is_lora else "stage3") if expected else "frozen"
        counts[kind] += parameter.numel()
        if expected:
            allowed.add(id(parameter))
        elif parameter.grad is not None:
            raise RuntimeError(f"Frozen parameter has a gradient: {name}")
    if trainable and (not counts["lora"] or not counts["stage3"]):
        raise RuntimeError("Both language LoRA and Stage-3 must be trainable")
    for module in (model.stage1.semantic_unit_extractor, model.stage1.evidence_field,
                   model.stage1.verification_head, model.baseline.head,
                   model.baseline.lm_head, model.baseline.multimodal_base.visual):
        if module.training:
            raise RuntimeError("Fixed custom/vision/head module must remain eval-only")
    if trainable and not model.baseline.multimodal_base.language_model.training:
        raise RuntimeError("Language training/checkpointing must be enabled")
    return counts, allowed


def build_stage3_scheduler(
    optimizer: Optimizer, config: dict[str, Any], total_steps: int
) -> LRScheduler:
    training = config["training"]
    if str(training.get("scheduler", "cosine")).lower() != "cosine":
        raise ValueError("Stage-3 requires cosine scheduling")
    warmup = max(1, round(int(total_steps) * float(training.get("warmup_ratio", 0.03))))
    return get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup,
        num_training_steps=max(warmup + 1, int(total_steps)),
    )
