from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from datasets import FourDSAVDataset, SceneMapper
from train_temporal import evaluate
from utils.config import dump_json, set_seed
from utils.qwen35_reason_runtime import Qwen35VisualFeatureCache
from utils.model_runtime import (
    check_protection, frozen_custom_digest, load_lora_config, load_lora_runtime,
)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Evaluate the trained assertion verification model")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--memory-budget-gib", type=float)
    args = parser.parse_args()
    if args.split == "test" and not args.allow_test:
        raise ValueError("Explicit --allow-test required; test is not a selection split")
    checkpoint, output = Path(args.checkpoint_dir).resolve(), Path(args.output_dir).resolve()
    if not (checkpoint / "COMPLETE.json").is_file():
        raise ValueError("Checkpoint is missing its completed-save marker")
    if output.exists():
        raise FileExistsError("Use a new evaluation output directory")
    config = load_lora_config(checkpoint / "config.json")
    check_protection(config)
    set_seed(config["seed"])
    torch.set_num_threads(4)
    if args.memory_budget_gib is not None:
        total_memory = torch.cuda.get_device_properties(0).total_memory
        fraction = args.memory_budget_gib * 1024**3 / total_memory
        if not 0 < fraction < 1:
            raise ValueError("Invalid shared GPU memory budget")
        torch.cuda.set_per_process_memory_fraction(fraction, 0)
    processor, model, report = load_lora_runtime(
        config, "cuda:0", trainable=False, checkpoint_dir=checkpoint)
    before = frozen_custom_digest(model)
    dataset = FourDSAVDataset(config["paths"][f"{args.split}_annotations"],
                             SceneMapper(config["paths"]["mapping"]))
    _, errors = dataset.validate_mappings()
    if errors:
        raise ValueError(errors)
    cache = Qwen35VisualFeatureCache(processor, model.baseline, config, "cuda:0")
    metrics, rows = evaluate(model, dataset, cache, processor, config, "cuda:0",
                            progress=lambda event: print(json.dumps(event), flush=True))
    if before != frozen_custom_digest(model):
        raise RuntimeError("Evaluation changed frozen custom weights")
    output.mkdir(parents=True)
    dump_json({"split": args.split, "checkpoint": str(checkpoint), "initialization": report,
               "metrics": metrics, "test_used_for_selection": False}, output / "metrics.json")
    dump_json(rows, output / "predictions.json")
    dump_json({"status": "completed", "split": args.split,
               "checkpoint": str(checkpoint)}, output / "status.json")
    print(json.dumps(metrics["overall"], indent=2))


if __name__ == "__main__":
    main()
