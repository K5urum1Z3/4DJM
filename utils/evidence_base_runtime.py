from __future__ import annotations

from pathlib import Path
import json
from typing import Any

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from transformers import get_cosine_schedule_with_warmup

from models.evidence_base import FourDSAVStage1
from models.qwen35_lora_reason_binary import load_qwen35_lora_reason_runtime


def _load_weights(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=True)


def load_qwen35_stage1_runtime(
    config: dict[str, Any],
    device: str,
    *,
    trainable: bool = True,
    checkpoint_dir: str | Path | None = None,
    keep_generation_model: bool = False,
) -> tuple[Any, FourDSAVStage1]:
    """Load the Qwen3.5 backbone, language adapter, and evidence modules."""
    stage = config["stage1"]
    init_mode = str(stage["init_mode"])
    checkpoint = Path(checkpoint_dir).resolve() if checkpoint_dir else None
    if checkpoint is None and init_mode == "baseline_checkpoint":
        checkpoint = Path(config["paths"]["baseline_checkpoint"]).resolve()
    if checkpoint is not None:
        adapter = checkpoint / "adapter"
        if not adapter.is_dir():
            raise FileNotFoundError(f"Stage-1 initialization adapter not found: {adapter}")
    else:
        adapter = None
        if init_mode != "formal_fresh":
            raise ValueError(f"Unsupported Stage-1 init_mode: {init_mode}")

    processor, baseline = load_qwen35_lora_reason_runtime(
        config,
        device,
        adapter_path=adapter,
        trainable=trainable,
        keep_generation_model=keep_generation_model,
    )
    if checkpoint is not None:
        baseline_head = checkpoint / "baseline_head.pt"
        if not baseline_head.is_file():
            baseline_head = checkpoint / "head.pt"
        if baseline_head.is_file():
            payload = _load_weights(baseline_head)
            baseline.load_head_state_dict(payload["head_state_dict"])

    model = FourDSAVStage1(baseline, processor.tokenizer, config).to(device)
    if checkpoint is not None and (checkpoint / "stage1_modules.pt").is_file():
        payload = _load_weights(checkpoint / "stage1_modules.pt")
        model.load_stage1_state_dict(payload["stage1_state_dict"])
    model.train(trainable)
    return processor, model


def build_stage1_optimizer(model: FourDSAVStage1, config: dict[str, Any]) -> torch.optim.AdamW:
    lora = [p for _, p in model.baseline.lora_named_parameters() if p.requires_grad]
    stage1 = [p for _, p in model.stage1_named_parameters() if p.requires_grad]
    if not lora or not stage1:
        raise RuntimeError("Stage-1 optimizer requires both trainable LoRA and Stage-1 parameters")
    training = config["training"]
    return torch.optim.AdamW(
        [
            {
                "params": lora,
                "lr": float(training["lora_learning_rate"]),
                "weight_decay": float(training.get("lora_weight_decay", 0.0)),
            },
            {
                "params": stage1,
                "lr": float(training["stage1_learning_rate"]),
                "weight_decay": float(training.get("stage1_weight_decay", 0.01)),
            },
        ],
        betas=tuple(float(x) for x in training.get("betas", [0.9, 0.999])),
        eps=float(training.get("eps", 1.0e-8)),
    )


def build_stage1_scheduler(
    optimizer: Optimizer, config: dict[str, Any], total_steps: int
) -> LRScheduler:
    if str(config["training"].get("scheduler", "cosine")).lower() != "cosine":
        raise ValueError("Stage-1 currently supports the baseline cosine scheduler only")
    warmup = max(1, round(int(total_steps) * float(config["training"].get("warmup_ratio", 0.03))))
    return get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup,
        num_training_steps=max(warmup + 1, int(total_steps)),
    )


def stage1_optimizer_group_report(optimizer: Optimizer) -> list[dict[str, Any]]:
    return [
        {
            "parameter_count": sum(p.numel() for p in group["params"]),
            "learning_rate": float(group["lr"]),
            "weight_decay": float(group.get("weight_decay", 0.0)),
        }
        for group in optimizer.param_groups
    ]


def save_stage1_checkpoint(
    output_dir: str | Path,
    model: FourDSAVStage1,
    optimizer: Optimizer | None,
    scheduler: LRScheduler | None,
    metadata: dict[str, Any],
) -> dict[str, str]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    adapter_dir = target / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.baseline.backbone.save_pretrained(adapter_dir)
    stage_path = target / "stage1_modules.pt"
    torch.save({"stage1_state_dict": model.stage1_state_dict(), "metadata": metadata}, stage_path)
    (target / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    resolved_config = metadata.get("resolved_config")
    if isinstance(resolved_config, dict):
        (target / "config.json").write_text(
            json.dumps(resolved_config, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    baseline_head_path = target / "baseline_head.pt"
    torch.save({"head_state_dict": model.baseline.head_state_dict()}, baseline_head_path)
    state_path = target / "trainer_state.pt"
    torch.save(
        {
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "metadata": metadata,
        },
        state_path,
    )
    return {
        "adapter": str(adapter_dir),
        "stage1_modules": str(stage_path),
        "baseline_head": str(baseline_head_path),
        "trainer_state": str(state_path),
    }


def load_stage1_state(model: FourDSAVStage1, path: str | Path) -> dict[str, Any]:
    payload = _load_weights(Path(path))
    model.load_stage1_state_dict(payload["stage1_state_dict"])
    return payload


def load_stage1_trainer_state(
    path: str | Path,
    optimizer: Optimizer,
    scheduler: LRScheduler | None,
) -> dict[str, Any]:
    payload = _load_weights(Path(path))
    if payload.get("optimizer_state_dict") is None:
        raise ValueError("Checkpoint does not contain optimizer state")
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    if scheduler is not None:
        state = payload.get("scheduler_state_dict")
        if state is None:
            raise ValueError("Checkpoint does not contain scheduler state")
        scheduler.load_state_dict(state)
    return payload
