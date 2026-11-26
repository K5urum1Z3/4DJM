"""Load, optimize and save the complete verification model."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import torch

from models.model import FourDSAVStage3LoRA
from utils.config import dump_json
from utils.evidence_runtime import load_qwen35_stage1_v2_runtime
from utils.temporal_runtime import (
    frozen_custom_digest, load_stage3_lora_config, trainability_report as original_trainability,
)

PROJECT = Path(__file__).resolve().parents[1]


def load_lora_config(path):
    path = Path(path).resolve()
    config = load_stage3_lora_config(path)
    config["stage"] = 3
    config["lora_execution"] = copy.deepcopy(config["stage3_lora_execution"])
    stage = config["stage"]
    if config["variant"] != "stage3_frozen_stage1_custom_trainable_language_lora":
        raise ValueError("Require the published Stage-3 language-LoRA variant")
    training, inputs, loss = config["training"], config["input"], config["loss"]
    if training["freeze_stage1"] or not training["freeze_stage1_custom"] or not training["train_qwen_language_lora"]:
        raise ValueError("Require fixed custom modules and trainable language LoRA")
    if config["lora_execution"]["use_cached_stage1_outputs"]:
        raise ValueError("Live LoRA cannot reuse cached upstream semantic/evidence outputs")
    if not inputs["use_reason_as_assistant_target"] or any(inputs[k] for k in (
            "use_reason_as_classifier_input", "use_answer_as_input", "use_capability_as_input")):
        raise ValueError("Reason may only be an assistant generation target")
    if training["batch_size"] != 1 or training["effective_batch_size"] != training["gradient_accumulation_steps"]:
        raise ValueError("Batch1 accumulation protocol mismatch")
    for value in (training["lora_learning_rate"], training[f"stage{stage}_learning_rate"],
                  training["eps"], training["max_grad_norm"], loss["answer_loss_weight"], loss["reason_loss_weight"]):
        if not math.isfinite(float(value)) or float(value) <= 0:
            raise ValueError("Rates, loss weights and numerical scales must be positive finite values")
    if any(int(training[k]) < 1 for k in ("epochs", "gradient_accumulation_steps", "validate_every_steps", "save_every_steps")):
        raise ValueError("Counts must be positive")
    if training["optimizer"] != "adamw" or training["scheduler"] != "cosine" or training["best_metric"] != "validation_macro_f1":
        raise ValueError("Require AdamW/cosine and validation-only selection")
    return config


def check_protection(config):
    manifest = config.get("upstream_protection_manifest")
    if not manifest:
        return {"protected_files": 0, "unchanged": True}
    hashes = json.loads(Path(manifest).read_text())
    for relative, expected in hashes.items():
        digest = hashlib.sha256()
        with (PROJECT / relative).open("rb") as stream:
            for chunk in iter(lambda: stream.read(8*1024*1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise RuntimeError(f"Original protected file changed: {relative}")
    return {"protected_files": len(hashes), "unchanged": True}


def trainability_report(model, *, trainable=True):
    return original_trainability(model, trainable=trainable)


def load_lora_runtime(config, device, *, trainable, checkpoint_dir=None):
    stage = config["stage"]
    source = Path(config["paths"]["stage1_v2_checkpoint"]).resolve()
    checkpoint = Path(checkpoint_dir).resolve() if checkpoint_dir else None
    if checkpoint:
        if not (checkpoint / "COMPLETE.json").is_file():
            raise ValueError("Checkpoint save is incomplete")
        metadata = json.loads((checkpoint / "metadata.json").read_text())
        if metadata["variant"] != config["variant"] or metadata["resolved_config"] != config:
            raise ValueError("Checkpoint/config variant mismatch")
        if metadata["stage1_checkpoint"] != str(source):
            raise ValueError("Stage-1 identity mismatch")
    processor, stage1, upstream = load_qwen35_stage1_v2_runtime(
        config, device, trainable=trainable, checkpoint_dir=checkpoint or source)
    model = FourDSAVStage3LoRA(stage1, config).to(device)
    if checkpoint:
        state = torch.load(checkpoint / f"stage{stage}_modules.pt", map_location="cpu", weights_only=True)
        getattr(model, f"load_stage{stage}_state_dict")(state[f"stage{stage}_state_dict"])
        if frozen_custom_digest(model) != metadata["frozen_custom_sha256"]:
            raise ValueError("Fixed Stage-1 custom state changed")
    if not trainable:
        model.requires_grad_(False)
    model.train(trainable)
    counts, _ = trainability_report(model, trainable=trainable)
    return processor, model, {"variant": config["variant"], "upstream": upstream,
        "stage1_custom_frozen": True, "qwen_base_frozen": True,
        "language_lora_trainable": trainable, "parameter_counts": counts,
        "added_module_initialization": "restore" if checkpoint else "fresh",
        "trained_stage2_or_stage3_loaded": bool(checkpoint), "optimizer_state_restored": False}


def build_optimizer(model, config):
    _, allowed = trainability_report(model)
    stage, training = config["stage"], config["training"]
    optimizer = torch.optim.AdamW([
        {"params": [p for _, p in model.baseline.lora_named_parameters()],
         "name": "language_lora", "lr": float(training["lora_learning_rate"]),
         "weight_decay": float(training["lora_weight_decay"])},
        {"params": [p for _, p in model.stage3_named_parameters()],
         "name": f"stage{stage}", "lr": float(training[f"stage{stage}_learning_rate"]),
         "weight_decay": float(training[f"stage{stage}_weight_decay"])}],
        betas=tuple(training["betas"]), eps=float(training["eps"]))
    actual = [id(p) for group in optimizer.param_groups for p in group["params"]]
    if set(actual) != allowed or len(actual) != len(allowed):
        raise RuntimeError("Optimizer contains missing, duplicate or unauthorized parameters")
    return optimizer


def save_checkpoint(path, model, optimizer, scheduler, metadata):
    target = Path(path).resolve()
    config = metadata["resolved_config"]
    root = Path(config["paths"]["output_dir"]).resolve()
    if not target.is_relative_to(root) or target == root:
        raise ValueError("Checkpoint must stay inside its variant output tree")
    if target.exists():
        raise FileExistsError(f"Never overwrite: {target}")
    if frozen_custom_digest(model) != metadata["frozen_custom_sha256"]:
        raise RuntimeError("Stage-1 custom weights changed")
    target.mkdir(parents=True)
    stage = config["stage"]
    model.baseline.backbone.save_pretrained(target / "adapter", safe_serialization=True)
    torch.save({"stage1_state_dict": model.stage1.stage1_state_dict()}, target / "stage1_modules.pt")
    torch.save({"head_state_dict": model.baseline.head_state_dict()}, target / "baseline_head.pt")
    torch.save({f"stage{stage}_state_dict": model.stage3_state_dict()}, target / f"stage{stage}_modules.pt")
    torch.save({"optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(),
        "metadata": metadata, "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state() if any(p.is_cuda for p in model.parameters()) else None},
        target / "trainer_state.pt")
    dump_json(config, target / "config.json")
    dump_json(metadata, target / "metadata.json")
    dump_json({"complete": True, "global_step": metadata["global_step"]}, target / "COMPLETE.json")

