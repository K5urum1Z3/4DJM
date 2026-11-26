from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from transformers import get_cosine_schedule_with_warmup

from models.qwen35_lora_reason_binary import Qwen35LoraReasonBinaryClassifier


def build_lora_optimizer(
    classifier: Qwen35LoraReasonBinaryClassifier,
    config: dict[str, Any],
) -> torch.optim.AdamW:
    lora_params = [parameter for _, parameter in classifier.lora_named_parameters()]
    head_params = [parameter for _, parameter in classifier.head_named_parameters()]
    if not lora_params or not head_params:
        raise RuntimeError("Both LoRA and Binary Head must have trainable parameters")
    training = config["training"]
    optimizer = torch.optim.AdamW(
        [
            {
                "params": lora_params,
                "lr": float(training["lora_learning_rate"]),
                "weight_decay": float(training.get("lora_weight_decay", 0.0)),
            },
            {
                "params": head_params,
                "lr": float(training["head_learning_rate"]),
                "weight_decay": float(training.get("head_weight_decay", 0.01)),
            },
        ],
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    return optimizer


def optimizer_group_report(optimizer: Optimizer) -> list[dict[str, Any]]:
    return [
        {
            "parameter_count": sum(parameter.numel() for parameter in group["params"]),
            "learning_rate": float(group["lr"]),
            "weight_decay": float(group.get("weight_decay", 0.0)),
        }
        for group in optimizer.param_groups
    ]


def build_lora_scheduler(
    optimizer: Optimizer,
    config: dict[str, Any],
    total_optimizer_steps: int,
) -> LRScheduler:
    warmup_ratio = float(config["training"].get("warmup_ratio", 0.03))
    warmup_steps = max(1, int(round(total_optimizer_steps * warmup_ratio)))
    return get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max(warmup_steps + 1, int(total_optimizer_steps)),
    )


def save_lora_checkpoint(
    output_dir: str | Path,
    classifier: Qwen35LoraReasonBinaryClassifier,
    optimizer: Optimizer,
    scheduler: LRScheduler | None,
    metadata: dict[str, Any],
) -> dict[str, str]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    adapter_dir = target / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    classifier.backbone.save_pretrained(adapter_dir)
    head_path = target / "head.pt"
    torch.save({"head_state_dict": classifier.head_state_dict(), "metadata": metadata}, head_path)
    state_path = target / "trainer_state.pt"
    torch.save(
        {
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "metadata": metadata,
        },
        state_path,
    )
    return {"adapter": str(adapter_dir), "head": str(head_path), "trainer_state": str(state_path)}


def load_lora_head(path: str | Path, classifier: Qwen35LoraReasonBinaryClassifier) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    classifier.load_head_state_dict(payload["head_state_dict"])
    return payload


def load_lora_trainer_state(
    path: str | Path,
    optimizer: Optimizer,
    scheduler: LRScheduler | None,
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    if scheduler is not None and payload.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(payload["scheduler_state_dict"])
    return payload

