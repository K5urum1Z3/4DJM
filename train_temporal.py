from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from datasets import FourDSAVDataset, SceneMapper
from utils.config import dump_json, set_seed
from utils.metrics import binary_metrics, grouped_metrics
from utils.qwen35_reason_runtime import (
    Qwen35VisualFeatureCache, collate_cached_reason_samples, prepare_cached_reason_sample,
)
from utils.model_runtime import (
    PROJECT, build_optimizer, check_protection, frozen_custom_digest,
    load_lora_config, load_lora_runtime, save_checkpoint, trainability_report,
)
from utils.temporal_runtime import build_stage3_scheduler


def make_batch(model, dataset, index, cache, processor, config, device, *, with_reason):
    sample = dataset[index]
    reason = str(dataset.rows[index].get("reason", "")).strip() if with_reason else None
    if with_reason and not reason:
        raise ValueError(f"Missing reason target: {sample['sample_id']}")
    prepared = prepare_cached_reason_sample(sample, reason, cache, processor, config)
    inputs, layouts, features, _ = collate_cached_reason_samples(
        [prepared], processor.tokenizer.pad_token_id, device,
        bool(config["performance"].get("pin_memory", True)))
    labels = torch.tensor([sample["label"]], device=device, dtype=torch.float32)
    output = model.forward_with_cached_video(inputs, layouts, features, [sample["assertion"]],
        labels=labels, compute_reason_loss=with_reason, return_debug=False)
    if not torch.isfinite(output["logits"]).all() or not torch.isfinite(output["loss"]):
        raise RuntimeError(f"Non-finite forward: {sample['sample_id']}")
    if with_reason and output["reason_token_count"] <= 0:
        raise RuntimeError("Reason supervision has no target tokens")
    return output, sample, layouts


@torch.no_grad()
def evaluate(model, dataset, cache, processor, config, device, *, progress=None):
    model.eval()
    rows, losses = [], []
    for index in range(len(dataset)):
        output, sample, _ = make_batch(model, dataset, index, cache, processor, config,
                                       device, with_reason=False)
        logit = float(output["logits"][0])
        rows.append({"sample_id": sample["sample_id"], "source": sample["source"],
            "capability": sample["capability"], "label": int(sample["label"]),
            "logit": logit, "probability": float(torch.sigmoid(output["logits"][0]))})
        losses.append(float(output["binary_loss"]))
        if progress and ((index + 1) % 50 == 0 or index + 1 == len(dataset)):
            progress({"type": "evaluation_progress", "processed": index + 1, "count": len(dataset)})
    threshold = float(config["training"]["threshold"])
    metrics = {"overall": binary_metrics([r["label"] for r in rows],
        [r["probability"] for r in rows], threshold),
        "binary_loss_mean": sum(losses) / len(losses),
        "by_source": grouped_metrics(rows, "source", threshold),
        "by_capability": grouped_metrics(rows, "capability", threshold),
        "reason_used_as_input": False}
    return metrics, rows


def gradient_summary(values):
    tensors = [v.detach().float() for v in values if v is not None]
    return {"present": len(tensors), "missing": len(values) - len(tensors),
        "finite": all(bool(torch.isfinite(v).all()) for v in tensors),
        "norm": math.sqrt(sum(float(v.square().sum()) for v in tensors))}


def runtime_preflight(model, dataset, cache, processor, config, device):
    """One real sample, gradient checks only; no optimizer step/save."""
    before = frozen_custom_digest(model)
    model.train(True)
    trainability_report(model)
    lora = [p for _, p in model.baseline.lora_named_parameters()]
    added = [p for _, p in model.stage3_named_parameters()]
    output, sample, layouts = make_batch(model, dataset, 0, cache, processor, config,
                                         device, with_reason=True)
    classification = gradient_summary(torch.autograd.grad(
        output["binary_loss"], lora, retain_graph=True, allow_unused=True))
    reason = gradient_summary(torch.autograd.grad(
        output["reason_loss"], lora, retain_graph=True, allow_unused=True))
    reason_to_stage3 = torch.autograd.grad(
        output["reason_loss"], added, retain_graph=True, allow_unused=True)
    total = gradient_summary(torch.autograd.grad(output["loss"], lora + added, allow_unused=True))
    if any(v is not None for v in reason_to_stage3):
        raise RuntimeError("Reason loss unexpectedly bypassed into added modules")
    for name, report in (("classification_to_lora", classification), ("reason_to_lora", reason),
                         ("total_to_trainable", total)):
        if not report["finite"] or report["missing"] or report["norm"] <= 0:
            raise RuntimeError(f"Preflight gradient failure: {name}: {report}")
    del output
    model.eval()
    with torch.no_grad():
        with_target, _, _ = make_batch(model, dataset, 0, cache, processor, config,
                                        device, with_reason=True)
        without_target, _, _ = make_batch(model, dataset, 0, cache, processor, config,
                                           device, with_reason=False)
        # Check causal isolation of the assertion logits.
        delta = float((with_target["logits"] - without_target["logits"]).abs().max())
        tolerance = config["lora_execution"]
        if not torch.allclose(with_target["logits"], without_target["logits"],
                atol=float(tolerance["causal_logit_atol"]), rtol=float(tolerance["causal_logit_rtol"])):
            raise RuntimeError(f"Causal reason-target leakage check failed: {delta}")
    if frozen_custom_digest(model) != before:
        raise RuntimeError("Preflight changed frozen custom weights")
    if not all(layout.user_end_index < layout.reason_start_index for layout in layouts):
        raise RuntimeError("Assistant reason must follow the user boundary")
    model.train(True)
    trainability_report(model)
    return {"passed": True, "sample_id": sample["sample_id"],
        "classification_to_lora": classification, "reason_to_lora": reason,
        "total_to_trainable": total, "reason_to_added_modules_is_none": True,
        "causal_logit_max_delta": delta, "frozen_custom_sha256": before,
        "optimizer_steps": 0}


def main():
    parser = argparse.ArgumentParser(description="Train temporal patterns and the verification head with language LoRA")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir")
    parser.add_argument("--memory-budget-gib", type=float)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-config-only", action="store_true")
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--formal-training", action="store_true")
    args = parser.parse_args()
    config = load_lora_config(args.config)
    if args.check_config_only:
        print(json.dumps(config, ensure_ascii=False, indent=2))
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for a real-model preflight/training")
    root = Path(config["paths"]["output_dir"]).resolve()
    run = Path(args.run_dir).resolve() if args.run_dir else root / time.strftime("run_%Y%m%d_%H%M%S")
    if not run.is_relative_to(root) or run == root or run.exists():
        raise ValueError("Use a NEW run directory inside the independent output root")
    check_protection(config)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(4)
    set_seed(config["seed"])
    device = "cuda:0"
    if args.memory_budget_gib is not None:
        total_memory = torch.cuda.get_device_properties(0).total_memory
        fraction = args.memory_budget_gib * 1024**3 / total_memory
        if not 0 < fraction < 1:
            raise ValueError("Invalid shared GPU memory budget")
        torch.cuda.set_per_process_memory_fraction(fraction, 0)
    mapper = SceneMapper(config["paths"]["mapping"])
    train = FourDSAVDataset(config["paths"]["train_annotations"], mapper)
    validation = FourDSAVDataset(config["paths"]["validation_annotations"], mapper)
    for name, dataset in (("train", train), ("validation", validation)):
        _, errors = dataset.validate_mappings()
        if errors:
            raise RuntimeError(f"{name} media mapping errors: {errors}")
    if any(not str(row.get("reason", "")).strip() for row in train.rows):
        raise ValueError("All training rows must provide a nonempty reason target")
    run.mkdir(parents=True)
    dump_json(config, run / "config.json")
    log_file = (run / "training_metrics.jsonl").open("x", encoding="utf-8", buffering=1)
    def log(event):
        line = json.dumps({"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **event})
        print(line, flush=True)
        log_file.write(line + "\n")
    dump_json({"status": "preflight", "formal_training_requested": args.formal_training}, run / "status.json")
    try:
        processor, model, initialization = load_lora_runtime(config, device, trainable=True)
        cache = Qwen35VisualFeatureCache(processor, model.baseline, config, device)
        torch.cuda.reset_peak_memory_stats()
        preflight = runtime_preflight(model, train, cache, processor, config, device)
        preflight["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 1024**3
        preflight["peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 1024**3
        preflight["memory_budget_gib"] = args.memory_budget_gib
        dump_json(preflight, run / "runtime_preflight.json")
        log({"type": "preflight_pass", **preflight})
        if args.preflight_only:
            dump_json({"status": "preflight_passed", "optimizer_steps": 0}, run / "status.json")
            return
        if not config["lora_execution"]["lazy_video_cache"]:
            log({"type": "visual_cache_ready", **cache.preload((train, validation), progress_callback=log)})
        # Reset the training RNG after preflight.
        set_seed(config["seed"])
        before = frozen_custom_digest(model)
        training = config["training"]
        accumulation = int(training["gradient_accumulation_steps"])
        epochs = int(training["epochs"])
        total_steps = epochs * math.ceil(len(train) / accumulation)
        optimizer = build_optimizer(model, config)
        scheduler = build_stage3_scheduler(optimizer, config, total_steps)
        parameters = [p for group in optimizer.param_groups for p in group["params"]]
        shuffle = torch.Generator().manual_seed(config["seed"])
        step, best, best_path = 0, -1., None
        log({"type": "run_start", "variant": config["variant"], "optimizer_steps": total_steps,
            "initialization": initialization, "loss": config["loss"], "test_split_loaded": False})
        def checkpoint_and_validate(epoch, *, validate):
            nonlocal best, best_path
            metrics = None
            if validate:
                metrics, rows = evaluate(model, validation, cache, processor, config, device, progress=log)
                dump_json({"global_step": step, "metrics": metrics, "predictions": rows},
                          run / f"validation/step_{step:07d}.json")
                log({"type": "validation", "global_step": step, "metrics": metrics})
            if frozen_custom_digest(model) != before:
                raise RuntimeError("Stage-1 custom weights changed during training")
            check_protection(config)
            model.train(True)
            trainability_report(model)
            path = run / f"checkpoints/step_{step:07d}"
            metadata = {"variant": config["variant"], "global_step": step, "epoch": epoch,
                "total_optimizer_steps": total_steps, "resolved_config": config,
                "stage1_checkpoint": str(Path(config["paths"]["stage1_v2_checkpoint"]).resolve()),
                "frozen_custom_sha256": before, "validation": metrics,
                "initialization": initialization, "test_used_for_selection": False}
            save_checkpoint(path, model, optimizer, scheduler, metadata)
            if metrics and metrics["overall"]["macro_f1"] > best:
                best = metrics["overall"]["macro_f1"]
                best_path = str(path)
                # Record the selected checkpoint path.
                dump_json({"checkpoint": best_path, "global_step": step,
                    "validation_macro_f1": best}, run / "best_checkpoint.json")
            dump_json({"checkpoint": str(path), "global_step": step}, run / "last_checkpoint.json")
        checkpoint_and_validate(0, validate=True)
        dump_json({"status": "training", "total_optimizer_steps": total_steps}, run / "status.json")
        for epoch in range(1, epochs + 1):
            model.train(True)
            loader = DataLoader(range(len(train)), batch_size=1, shuffle=True,
                                generator=shuffle, collate_fn=list, num_workers=0)
            optimizer.zero_grad(set_to_none=True)
            sums = {"loss": 0., "binary_loss": 0., "reason_loss": 0.}
            group_count, reason_tokens = 0, 0
            started = time.perf_counter()
            for micro, indices in enumerate(loader):
                output, _, _ = make_batch(model, train, indices[0], cache, processor,
                                          config, device, with_reason=True)
                size = min(accumulation, len(loader) - (micro // accumulation) * accumulation)
                (output["loss"] / size).backward()
                for key in sums:
                    sums[key] += float(output[key].detach())
                reason_tokens += output["reason_token_count"]
                group_count += 1
                del output  # Do not retain the previous language graph on the next forward.
                if (micro + 1) % accumulation and micro + 1 != len(loader):
                    continue
                trainability_report(model)
                if any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters):
                    raise RuntimeError("Missing or non-finite trainable gradients")
                norm = torch.nn.utils.clip_grad_norm_(parameters, float(training["max_grad_norm"]))
                if not torch.isfinite(norm):
                    raise RuntimeError("Non-finite gradient norm")
                used_lr = {g["name"]: g["lr"] for g in optimizer.param_groups}
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                log({"type": "optimizer_step", "epoch": epoch, "global_step": step,
                    "micro_batches": group_count, "reason_token_count": reason_tokens,
                    **{k: v / group_count for k, v in sums.items()}, "learning_rates": used_lr,
                    "gradient_norm": float(norm), "step_seconds": time.perf_counter() - started})
                end_epoch = micro + 1 == len(loader)
                validate = step % int(training["validate_every_steps"]) == 0 or end_epoch
                save = step % int(training["save_every_steps"]) == 0 or validate
                if save:
                    checkpoint_and_validate(epoch, validate=validate)
                sums = {key: 0. for key in sums}
                group_count, reason_tokens = 0, 0
                started = time.perf_counter()
        if step != total_steps:
            raise RuntimeError("Training budget mismatch")
        dump_json({"status": "completed", "global_step": step, "best_checkpoint": best_path,
                   "best_validation_macro_f1": best, "test_split_loaded": False}, run / "status.json")
        log({"type": "run_complete", "global_step": step, "best_checkpoint": best_path})
    except BaseException as error:
        dump_json({"status": "failed_or_interrupted", "error": repr(error)}, run / "status.json")
        raise
    finally:
        log_file.close()


if __name__ == "__main__":
    main()
