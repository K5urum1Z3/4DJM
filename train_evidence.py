from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from datasets import FourDSAVDataset, SceneMapper
from utils.evidence_evaluation import evaluate_stage1
from utils.config import dump_json, load_yaml, set_seed
from utils.qwen35_reason_runtime import (
    Qwen35VisualFeatureCache,
    collate_cached_reason_samples,
    prepare_cached_reason_sample,
)
from utils.evidence_base_runtime import (
    build_stage1_optimizer,
    build_stage1_scheduler,
    load_stage1_trainer_state,
    save_stage1_checkpoint,
    stage1_optimizer_group_report,
)
from utils.evidence_runtime import load_qwen35_stage1_v2_runtime
from utils.evidence_diagnostics import (
    counterfactual_summary,
    evaluate_v2_diagnostics,
    render_heatmap_comparisons,
    restore_signed_v1_cache,
    stratified_indices,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the isolated Qwen3.5 4DSAV Stage-1 v2 verifier")
    parser.add_argument("--config", default="configs/evidence.yaml")
    parser.add_argument("--resume-dir")
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-optimizer-steps", type=int)
    parser.add_argument("--max-validation-samples", type=int)
    parser.add_argument("--run-dir")
    parser.add_argument("--init-mode", choices=("development_warmstart", "formal_fresh"))
    parser.add_argument("--tiny-overfit", action="store_true")
    parser.add_argument("--lazy-video-cache", action="store_true")
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--diagnostic-steps", default="", help="Comma-separated optimizer steps, e.g. 100,250,500")
    parser.add_argument("--diagnostic-count", type=int, default=200)
    parser.add_argument("--formal-training", action="store_true")
    return parser.parse_args()


def select_tiny_indices(train_set: FourDSAVDataset, seed: int, count: int = 16) -> list[int]:
    """Deterministic label/capability-balanced subset; reads train annotations only."""
    rng = random.Random(seed)
    groups: dict[tuple[str, int], list[int]] = {}
    for index in range(len(train_set)):
        sample = train_set[index]
        key = (str(sample["capability"]), int(sample["label"]))
        groups.setdefault(key, []).append(index)
    for values in groups.values():
        rng.shuffle(values)
    selected: list[int] = []
    keys = sorted(groups)
    while len(selected) < min(count, len(train_set)):
        progressed = False
        for key in keys:
            if groups[key] and len(selected) < count:
                selected.append(groups[key].pop())
                progressed = True
        if not progressed:
            break
    return selected


def evaluate_fixed_train_subset(model, train_set, indices, cache, processor, config, device) -> dict:
    model.eval()
    losses, binary_losses, reason_losses, labels, predictions = [], [], [], [], []
    with torch.no_grad():
        for index in indices:
            sample = train_set[index]
            prepared = prepare_cached_reason_sample(
                sample, str(train_set.rows[index]["reason"]).strip(), cache, processor, config
            )
            inputs, layouts, features, _ = collate_cached_reason_samples(
                [prepared], processor.tokenizer.pad_token_id, device, False
            )
            label = torch.tensor([sample["label"]], dtype=torch.float32, device=device)
            output = model.forward_with_cached_video(
                inputs, layouts, features, [sample["assertion"]], labels=label,
                compute_reason_loss=True, return_debug=False,
            )
            prob = torch.sigmoid(output["logits"].float())
            labels.append(int(sample["label"]))
            predictions.append(int(prob[0].item() >= float(config["training"]["threshold"])))
            losses.append(float(output["loss"].float().cpu()))
            binary_losses.append(float(output["binary_loss"].float().cpu()))
            reason_losses.append(float(output["reason_loss"].float().cpu()))
            del output, inputs, features, label
    model.train(True)
    correct = sum(a == b for a, b in zip(labels, predictions))
    return {
        "sample_count": len(indices),
        "true_count": sum(labels),
        "false_count": len(labels) - sum(labels),
        "total_loss_mean": sum(losses) / max(1, len(losses)),
        "binary_loss_mean": sum(binary_losses) / max(1, len(binary_losses)),
        "reason_loss_mean": sum(reason_losses) / max(1, len(reason_losses)),
        "accuracy": correct / max(1, len(labels)),
        "correct": correct,
        "predictions": predictions,
        "labels": labels,
    }


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    formal = args.formal_training and args.max_train_samples is None and args.max_optimizer_steps is None
    if args.max_train_samples is None and args.max_optimizer_steps is None and not args.formal_training:
        raise SystemExit("Refusing an unbounded run without --formal-training")
    if formal:
        init_mode = args.init_mode or config["stage1"].get("init_mode")
        if init_mode != "formal_fresh":
            raise SystemExit("Formal training requires stage1.init_mode=formal_fresh")
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")
    set_seed(int(config["seed"]))
    device = "cuda:0"
    mapper = SceneMapper(config["paths"]["mapping"])
    train_set = FourDSAVDataset(config["paths"]["train_annotations"], mapper)
    val_set = FourDSAVDataset(config["paths"]["validation_annotations"], mapper)
    for split, dataset in (("train", train_set), ("validation", val_set)):
        _, errors = dataset.validate_mappings()
        if errors:
            raise RuntimeError(f"{split} mapping audit failed: {errors}")
    if args.tiny_overfit:
        train_indices = select_tiny_indices(
            train_set, int(config["seed"]), int(config["stage1"].get("tiny_sample_count", 16))
        )
    else:
        limit = len(train_set) if args.max_train_samples is None else min(args.max_train_samples, len(train_set))
        train_indices = list(range(limit))
    count = len(train_indices)
    if args.max_train_samples is not None and not args.tiny_overfit:
        train_indices = train_indices[:args.max_train_samples]
        count = len(train_indices)
    training = config["training"]
    batch_size = int(training["batch_size"])
    accumulation = int(training["gradient_accumulation_steps"])
    epochs = int(training["epochs"])
    steps_per_epoch = math.ceil(count / (batch_size * accumulation))
    if args.max_optimizer_steps is not None:
        if args.max_optimizer_steps < 1:
            raise ValueError("--max-optimizer-steps must be positive")
        total_steps = int(args.max_optimizer_steps)
        if args.tiny_overfit:
            epochs = max(1, math.ceil(total_steps / max(1, steps_per_epoch)))
    else:
        total_steps = steps_per_epoch * epochs

    processor, model, init_report = load_qwen35_stage1_v2_runtime(
        config,
        device,
        trainable=True,
        init_mode=args.init_mode,
        checkpoint_dir=args.resume_dir,
    )
    optimizer = build_stage1_optimizer(model, config)
    scheduler = build_stage1_scheduler(optimizer, config, total_steps)
    start_epoch = 0
    global_step = 0
    best_macro_f1 = -1.0
    if args.resume_dir:
        state = load_stage1_trainer_state(Path(args.resume_dir) / "trainer_state.pt", optimizer, scheduler)
        metadata = state.get("metadata", {})
        start_epoch = int(metadata.get("epoch", 0))
        global_step = int(metadata.get("global_step", 0))
        best_macro_f1 = float(metadata.get("best_macro_f1", -1.0))

    run_dir = Path(args.run_dir or config["paths"]["output_dir"]).resolve()
    checkpoints = run_dir / "checkpoints"
    logs = run_dir / "logs"
    checkpoints.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    config_copy = run_dir / "config_snapshot.yaml"
    config_copy.write_text(Path(args.config).read_text(encoding="utf-8"), encoding="utf-8")
    metrics_path = run_dir / "training_metrics.jsonl"
    metrics_file = metrics_path.open("a", encoding="utf-8")

    def log(event: dict) -> None:
        event.setdefault("timestamp", time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        line = json.dumps(event, ensure_ascii=False)
        metrics_file.write(line + "\n")
        metrics_file.flush()
        print(line, flush=True)

    log({
        "type": "run_start",
        "formal": formal,
        "config": str(Path(args.config).resolve()),
        "train_samples": count,
        "validation_samples": len(val_set),
        "batch_size": batch_size,
        "gradient_accumulation_steps": accumulation,
        "effective_batch_size": batch_size * accumulation,
        "epochs": epochs,
        "num_frames": int(config["video"]["num_frames"]),
        "init_mode": init_report["init_mode"],
        "init_report": init_report,
        "tiny_sample_ids": [str(train_set.rows[index]["sample_id"]) for index in train_indices] if args.tiny_overfit else None,
        "max_optimizer_steps": args.max_optimizer_steps,
        "optimizer_groups": stage1_optimizer_group_report(optimizer),
        "model_counts": model.parameter_count_report(),
        "run_dir": str(run_dir),
    })

    cache = Qwen35VisualFeatureCache(processor, model.baseline, config, device)
    diagnostic_steps = {int(value) for value in args.diagnostic_steps.split(",") if value.strip()}
    diagnostic_results: dict[int, dict] = {}
    diagnostic_indices: list[int] = []
    if args.lazy_video_cache:
        log({"type": "visual_cache_ready", "lazy": True, **cache.stats()})
    else:
        cache_stats = cache.preload((train_set, val_set))
        log({"type": "visual_cache_ready", "lazy": False, **cache_stats})
    tiny_initial = None
    if args.tiny_overfit:
        tiny_initial = evaluate_fixed_train_subset(
            model, train_set, train_indices, cache, processor, config, device
        )
        log({"type": "tiny_initial", **tiny_initial})
    if diagnostic_steps:
        if init_report["init_mode"] != "development_warmstart":
            raise RuntimeError("Short-training diagnostics require the development_warmstart initialization")
        cache_report = restore_signed_v1_cache(
            cache,
            Path(config["paths"]["project_root"]) / "outputs/stage1_postaudit/validation_visual_feature_cache.pt",
            config["paths"]["stage1_v1_checkpoint"],
            config,
        )
        diagnostic_indices = stratified_indices(val_set, args.diagnostic_count, int(config["seed"]))
        log({"type": "diagnostic_set_fixed", "sample_count": len(diagnostic_indices),
             "sample_ids": [str(val_set.rows[index]["sample_id"]) for index in diagnostic_indices],
             "signed_v1_cache": cache_report})
        initial_diag = evaluate_v2_diagnostics(
            model, val_set, diagnostic_indices, cache, processor, config, device,
            step=0, output_path=run_dir / "temporal_variance_step0000.json",
            retain_counterfactuals=True,
        )
        diagnostic_results[0] = initial_diag
        model.train(True)
    pin = bool(config.get("performance", {}).get("pin_memory", True))
    loader_generator = torch.Generator().manual_seed(int(config["seed"]))
    for epoch in range(start_epoch, epochs):
        model.train(True)
        loader = DataLoader(
            train_indices, batch_size=batch_size, shuffle=True,
            generator=loader_generator, num_workers=0, collate_fn=list,
        )
        optimizer.zero_grad(set_to_none=True)
        epoch_started = time.perf_counter()
        batch_started = None
        accumulated_samples = 0
        accumulated_micro_batches = 0
        accumulated_binary_loss = 0.0
        accumulated_reason_loss = 0.0
        accumulated_total_loss = 0.0
        loss_sum = 0.0
        seen = 0
        for micro_index, indices in enumerate(loader, start=1):
            if batch_started is None:
                batch_started = time.perf_counter()
            samples = [train_set[index] for index in indices]
            prepared = [
                prepare_cached_reason_sample(
                    sample,
                    str(train_set.rows[index]["reason"]).strip(),
                    cache,
                    processor,
                    config,
                )
                for index, sample in zip(indices, samples)
            ]
            inputs, layouts, features, _ = collate_cached_reason_samples(
                prepared, processor.tokenizer.pad_token_id, device, pin
            )
            labels = torch.tensor([sample["label"] for sample in samples], dtype=torch.float32, device=device)
            output = model.forward_with_cached_video(
                inputs,
                layouts,
                features,
                [sample["assertion"] for sample in samples],
                labels=labels,
                compute_reason_loss=True,
                return_debug=False,
            )
            loss = output["loss"]
            if loss is None or not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite Stage-1 loss at epoch={epoch + 1}, micro={micro_index}")
            (loss / accumulation).backward()
            loss_sum += float(loss.detach()) * len(samples)
            seen += len(samples)
            accumulated_samples += len(samples)
            accumulated_micro_batches += 1
            accumulated_binary_loss += float(output["binary_loss"].detach()) * len(samples)
            accumulated_reason_loss += float(output["reason_loss"].detach()) * len(samples)
            accumulated_total_loss += float(loss.detach()) * len(samples)
            final_micro = micro_index == len(loader)
            if micro_index % accumulation == 0 or final_micro:
                gradient_groups: dict[str, list[torch.Tensor]] = {"lora": [], "stage1": []}
                missing_gradient_parameters = {"lora": 0, "stage1": 0}
                for name, parameter in model.trainable_named_parameters():
                    group_name = "lora" if name.startswith("baseline.backbone.") else "stage1"
                    if parameter.grad is None:
                        missing_gradient_parameters[group_name] += 1
                    else:
                        gradient_groups[group_name].append(parameter.grad.detach().float())
                gradient_report = {}
                for group_name, gradients in gradient_groups.items():
                    if not gradients:
                        raise RuntimeError(f"No gradients reached the {group_name} parameter group")
                    finite = all(bool(torch.isfinite(gradient).all()) for gradient in gradients)
                    if not finite:
                        raise RuntimeError(f"Non-finite gradient in the {group_name} parameter group")
                    squared_norm = sum(torch.sum(gradient.square()) for gradient in gradients)
                    gradient_report[group_name] = {
                        "grad_norm": float(torch.sqrt(squared_norm).cpu()),
                        "parameters_with_grad": len(gradients),
                        "parameters_without_grad": missing_gradient_parameters[group_name],
                        "finite": finite,
                    }
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [p for _, p in model.trainable_named_parameters()],
                    float(training["max_grad_norm"]),
                )
                if not torch.isfinite(grad_norm):
                    raise RuntimeError(f"Non-finite gradient norm at global_step={global_step + 1}")
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                seconds = time.perf_counter() - batch_started
                log({
                    "type": "optimizer_step",
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "micro_batches": accumulated_micro_batches,
                    "samples_in_optimizer_step": accumulated_samples,
                    "samples_seen_epoch": seen,
                    "loss_running_mean": loss_sum / max(1, seen),
                    "binary_loss": accumulated_binary_loss / max(1, accumulated_samples),
                    "reason_loss": accumulated_reason_loss / max(1, accumulated_samples),
                    "total_loss": accumulated_total_loss / max(1, accumulated_samples),
                    "lora_lr": float(optimizer.param_groups[0]["lr"]),
                    "stage1_lr": float(optimizer.param_groups[1]["lr"]),
                    "gradient_norm": float(grad_norm),
                    "gradient_groups": gradient_report,
                    "step_seconds": seconds,
                    "samples_per_second": accumulated_samples / max(seconds, 1e-9),
                    "torch_allocated_mib": torch.cuda.memory_allocated() / 2**20,
                    "torch_reserved_mib": torch.cuda.memory_reserved() / 2**20,
                    "torch_peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                    "torch_peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
                })
                if global_step in diagnostic_steps:
                    result = evaluate_v2_diagnostics(
                        model, val_set, diagnostic_indices, cache, processor, config, device,
                        step=global_step,
                        output_path=run_dir / f"temporal_variance_step{global_step:04d}.json",
                        retain_counterfactuals=True,
                    )
                    diagnostic_results[global_step] = result
                    dump_json({
                        "step": global_step,
                        "diagnostics": counterfactual_summary(model, result),
                    }, run_dir / f"counterfactuals_step{global_step:04d}.json")
                    log({"type": "temporal_diagnostics", "global_step": global_step,
                         "r_to_L_retention": result["variance_retention"]["r_to_L"],
                         "E_temporal_std": result["E_temporal_std"],
                         "occurrence_normalized_entropy": result["occurrence_normalized_entropy"]})
                    model.train(True)
                batch_started = None
                accumulated_samples = 0
                accumulated_micro_batches = 0
                accumulated_binary_loss = 0.0
                accumulated_reason_loss = 0.0
                accumulated_total_loss = 0.0
                if global_step % int(training.get("save_every_steps", 500)) == 0:
                    save_stage1_checkpoint(
                        checkpoints / f"step_{global_step:07d}", model, optimizer, scheduler,
                        {"epoch": epoch + 1, "global_step": global_step, "best_macro_f1": best_macro_f1,
                         "resolved_config": config},
                    )
                if not args.skip_validation and global_step % int(training.get("validate_every_steps", 500)) == 0:
                    val_metrics, _ = evaluate_stage1(
                        model, val_set, cache, processor, config, device, args.max_validation_samples
                    )
                    macro = float(val_metrics["overall"]["macro_f1"])
                    log({"type": "validation", "epoch": epoch + 1, "global_step": global_step, "metrics": val_metrics})
                    if macro > best_macro_f1:
                        best_macro_f1 = macro
                        save_stage1_checkpoint(
                            checkpoints / "best", model, optimizer, scheduler,
                            {"epoch": epoch + 1, "global_step": global_step,
                             "best_macro_f1": best_macro_f1, "validation": val_metrics,
                             "resolved_config": config},
                        )
                    model.train(True)
                if args.max_optimizer_steps is not None and global_step >= args.max_optimizer_steps:
                    break
        if not args.skip_validation:
            val_metrics, _ = evaluate_stage1(
                model, val_set, cache, processor, config, device, args.max_validation_samples
            )
            macro = float(val_metrics["overall"]["macro_f1"])
            log({
                "type": "epoch_end", "epoch": epoch + 1, "global_step": global_step,
                "elapsed_seconds": time.perf_counter() - epoch_started,
                "mean_loss": loss_sum / max(1, seen), "validation": val_metrics,
            })
            if macro > best_macro_f1:
                best_macro_f1 = macro
                save_stage1_checkpoint(
                    checkpoints / "best", model, optimizer, scheduler,
                    {"epoch": epoch + 1, "global_step": global_step,
                     "best_macro_f1": best_macro_f1, "validation": val_metrics,
                     "resolved_config": config},
                )
            save_stage1_checkpoint(
                checkpoints / "last", model, optimizer, scheduler,
                {"epoch": epoch + 1, "global_step": global_step,
                 "best_macro_f1": best_macro_f1, "validation": val_metrics,
                 "resolved_config": config},
            )
        if args.max_optimizer_steps is not None and global_step >= args.max_optimizer_steps:
            break
    save_stage1_checkpoint(
        checkpoints / "last", model, optimizer, scheduler,
        {"epoch": min(epochs, epoch + 1), "global_step": global_step,
         "best_macro_f1": best_macro_f1, "resolved_config": config,
         "validation_skipped": bool(args.skip_validation)},
    )
    if args.tiny_overfit:
        tiny_final = evaluate_fixed_train_subset(
            model, train_set, train_indices, cache, processor, config, device
        )
        tiny_summary = {
            "sample_ids": [str(train_set.rows[index]["sample_id"]) for index in train_indices],
            "capability_distribution": {
                capability: sum(str(train_set.rows[index]["capability"]) == capability for index in train_indices)
                for capability in sorted({str(train_set.rows[index]["capability"]) for index in train_indices})
            },
            "initial": tiny_initial,
            "final": tiny_final,
            "optimizer_steps": global_step,
            "loss_decreased": tiny_final["total_loss_mean"] < tiny_initial["total_loss_mean"],
            "accuracy_delta": tiny_final["accuracy"] - tiny_initial["accuracy"],
        }
        dump_json(tiny_summary, run_dir / "tiny_overfit_summary.json")
        log({"type": "tiny_final", "summary": tiny_summary})
    if diagnostic_results:
        temporal_summary = {
            "diagnostic_sample_count": len(diagnostic_indices),
            "fixed_sample_ids": [str(val_set.rows[index]["sample_id"]) for index in diagnostic_indices],
            "checkpoints": [
                {key: value for key, value in diagnostic_results[step].items() if not key.startswith("_")}
                for step in sorted(diagnostic_results)
            ],
        }
        dump_json(temporal_summary, run_dir / "temporal_variance.json")
        heatmap_by_step = {step: result["heatmap_e"] for step, result in diagnostic_results.items()}
        selected_heatmaps = heatmap_by_step[min(heatmap_by_step)]
        heatmap_ids = list(selected_heatmaps)
        labels_by_id = {sample_id: int(item["label"]) for sample_id, item in selected_heatmaps.items()}
        v1_debug_path = Path(config["paths"]["project_root"]) / "outputs/stage1_postaudit/stage1_predictions_debug.jsonl"
        wanted = set(heatmap_ids)
        v1_heatmaps = {}
        with v1_debug_path.open("r", encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                sample_id = str(row.get("sample_id"))
                if sample_id in wanted:
                    v1_heatmaps[sample_id] = {
                        "label": int(row["label"]),
                        "E": row["evidence_scores"],
                    }
        if len(v1_heatmaps) != len(heatmap_ids):
            raise RuntimeError("Could not recover all fixed v1 heatmap samples from the saved audit")
        step_e = heatmap_by_step
        heatmap_files = render_heatmap_comparisons(
            heatmap_ids, labels_by_id, v1_heatmaps, step_e, run_dir / "evidence_heatmaps"
        )
        dump_json({"sample_ids": heatmap_ids, "files": heatmap_files}, run_dir / "evidence_heatmaps/index.json")
        final_diag_step = max(diagnostic_results)
        dump_json({
            "step": final_diag_step,
            "diagnostics": counterfactual_summary(model, diagnostic_results[final_diag_step]),
        }, run_dir / "counterfactuals.json")
    log({
        "type": "run_end",
        "global_step": global_step,
        "best_validation_macro_f1": best_macro_f1,
        "torch_peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "torch_peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
    })
    metrics_file.close()


if __name__ == "__main__":
    main()
