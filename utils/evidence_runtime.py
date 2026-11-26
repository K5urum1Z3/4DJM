from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from models.evidence_model import FourDSAVStage1V2
from models.qwen35_lora_reason_binary import load_qwen35_lora_reason_runtime
from utils.evidence_base_runtime import load_qwen35_stage1_runtime


def _load_tensor_payload(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=True)


def _load_baseline_for_adapter(
    config: dict[str, Any], device: str, checkpoint: Path, trainable: bool
):
    adapter = checkpoint / "adapter"
    if not adapter.is_dir():
        raise FileNotFoundError(f"Qwen3.5 adapter not found: {adapter}")
    processor, baseline = load_qwen35_lora_reason_runtime(
        config, device, adapter_path=adapter, trainable=trainable
    )
    head_path = checkpoint / "baseline_head.pt"
    if not head_path.is_file():
        head_path = checkpoint / "head.pt"
    if head_path.is_file():
        payload = _load_tensor_payload(head_path)
        key = "head_state_dict" if "head_state_dict" in payload else "state_dict"
        baseline.load_head_state_dict(payload[key])
    return processor, baseline


def load_qwen35_stage1_v2_runtime(
    config: dict[str, Any],
    device: str,
    *,
    trainable: bool = True,
    init_mode: str | None = None,
    checkpoint_dir: str | Path | None = None,
) -> tuple[Any, FourDSAVStage1V2, dict[str, Any]]:
    """Load an isolated v2 model from a v2 checkpoint or an explicit init mode."""
    init_mode = init_mode or str(config["stage1"]["init_mode"])
    init_report: dict[str, Any] = {"init_mode": init_mode}

    if checkpoint_dir is not None:
        checkpoint = Path(checkpoint_dir).resolve()
        processor, baseline = _load_baseline_for_adapter(config, device, checkpoint, trainable)
        model = FourDSAVStage1V2(baseline, processor.tokenizer, config).to(device)
        payload = _load_tensor_payload(checkpoint / "stage1_modules.pt")
        model.load_stage1_state_dict(payload["stage1_state_dict"])
        init_report["checkpoint"] = str(checkpoint)
    elif init_mode == "development_warmstart":
        v1_checkpoint = Path(config["paths"]["stage1_v1_checkpoint"]).resolve()
        processor, v1_model = load_qwen35_stage1_runtime(
            config, device, trainable=trainable, checkpoint_dir=v1_checkpoint
        )
        model = FourDSAVStage1V2(v1_model.baseline, processor.tokenizer, config).to(device)
        init_report.update(model.initialize_shared_from_v1(v1_model.stage1_state_dict()))
        init_report["warmstart_checkpoint"] = str(v1_checkpoint)
        del v1_model
    elif init_mode == "formal_fresh":
        baseline_checkpoint = Path(config["paths"]["baseline_checkpoint"]).resolve()
        processor, baseline = _load_baseline_for_adapter(
            config, device, baseline_checkpoint, trainable
        )
        model = FourDSAVStage1V2(baseline, processor.tokenizer, config).to(device)
        init_report.update({
            "baseline_checkpoint": str(baseline_checkpoint),
            "stage1_initialization": "fresh_random_stage1_v2",
        })
    else:
        raise ValueError(f"Unsupported Stage-1 v2 init_mode: {init_mode}")

    if not trainable:
        for _, parameter in model.trainable_named_parameters():
            parameter.requires_grad_(False)
    model.train(trainable)
    return processor, model, init_report
