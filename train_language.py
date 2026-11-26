from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from datasets import FourDSAVDataset, SceneMapper
from models.qwen35_lora_reason_binary import (
    assert_qwen35_trainable_integrity,
    load_qwen35_lora_reason_runtime,
)
from utils.config import dump_json, load_yaml, set_seed
from utils.lora_runtime import (
    build_lora_optimizer,
    build_lora_scheduler,
    load_lora_head,
    load_lora_trainer_state,
    optimizer_group_report,
    save_lora_checkpoint,
)
from utils.metrics import binary_metrics, grouped_metrics
from utils.qwen35_reason_runtime import (
    Qwen35VisualFeatureCache,
    collate_cached_reason_samples,
    prefetch_map,
    prepare_cached_reason_sample,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Qwen3.5 hybrid-attention LoRA with answer and reason losses"
    )
    parser.add_argument("--config", default="configs/language.yaml")
    parser.add_argument("--resume-dir", default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-validation-samples", type=int, default=None)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--formal-training", action="store_true")
    return parser.parse_args()


def _prepare_batch(
    indices: list[int],
    dataset: FourDSAVDataset,
    cache: Qwen35VisualFeatureCache,
    processor,
    config: dict,
    include_reason: bool,
):
    prepared = []
    samples = []
    for index in indices:
        sample = dataset[index]
        reason = str(dataset.rows[index]["reason"]).strip() if include_reason else None
        if include_reason and not reason:
            raise RuntimeError(f"Empty reason: {sample['sample_id']}")
        prepared.append(
            prepare_cached_reason_sample(sample, reason, cache, processor, config)
        )
        samples.append(sample)
    return prepared, samples


@torch.no_grad()
def evaluate(classifier, dataset, cache, processor, config, device, limit=None):
    classifier.eval()
    labels, probabilities, sources, capabilities = [], [], [], []
    count = len(dataset) if limit is None else min(limit, len(dataset))
    batch_size = int(config["training"].get("evaluation_batch_size", 4))
    pin_memory = bool(config.get("performance", {}).get("pin_memory", True))
    prefetch_depth = int(config.get("performance", {}).get("prefetch_batches", 0))
    loader = DataLoader(
        list(range(count)), batch_size=batch_size, shuffle=False,
        num_workers=0, collate_fn=list,
    )

    def prepare(indices):
        return _prepare_batch(
            indices, dataset, cache, processor, config, include_reason=False
        )

    for prepared, samples in prefetch_map(loader, prepare, prefetch_depth):
        inputs, layouts, video_features, _ = collate_cached_reason_samples(
            prepared, processor.tokenizer.pad_token_id, device, pin_memory
        )
        logits, reason_loss, _ = classifier.forward_with_cached_video(
            inputs, layouts, video_features, compute_reason_loss=False
        )
        if reason_loss is not None:
            raise AssertionError("Validation unexpectedly created reason loss")
        probabilities.extend(torch.sigmoid(logits).float().cpu().tolist())
        labels.extend(int(sample["label"]) for sample in samples)
        sources.extend(sample["source"] for sample in samples)
        capabilities.extend(sample["capability"] for sample in samples)

    threshold = float(config["training"]["threshold"])
    records = [
        {"label": label, "probability": probability, "source": source,
         "capability": capability}
        for label, probability, source, capability in zip(
            labels, probabilities, sources, capabilities
        )
    ]
    return {
        "overall": binary_metrics(labels, probabilities, threshold),
        "by_source": grouped_metrics(records, "source", threshold),
        "by_capability": grouped_metrics(records, "capability", threshold),
    }


def main() -> None:
    args = parse_args()
    if args.max_train_samples is None and not args.formal_training:
        raise SystemExit(
            "Refusing an unbounded run. Supply --max-train-samples or add "
            "--formal-training explicitly."
        )
    config = load_yaml(args.config)
    if not bool(config.get("performance", {}).get("cache_visual_features", False)):
        raise ValueError("The optimized Qwen3.5 profile requires visual feature caching")
    set_seed(int(config["seed"]))
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")
    device = "cuda:0"

    mapper = SceneMapper(config["paths"]["mapping"])
    train_set = FourDSAVDataset(config["paths"]["train_annotations"], mapper)
    validation_set = FourDSAVDataset(
        config["paths"]["validation_annotations"], mapper
    )
    for split_name, dataset in (("train", train_set), ("validation", validation_set)):
        _, errors = dataset.validate_mappings()
        if errors:
            raise RuntimeError(f"{split_name} mapping audit failed: {errors}")

    train_count = len(train_set) if args.max_train_samples is None else min(
        args.max_train_samples, len(train_set)
    )
    batch_size = int(config["training"]["batch_size"])
    accumulation = int(config["training"]["gradient_accumulation_steps"])
    epochs = int(config["training"]["epochs"])
    total_optimizer_steps = math.ceil(
        train_count / (batch_size * accumulation)
    ) * epochs

    adapter_path = None
    if args.resume_dir:
        adapter_path = Path(args.resume_dir) / "adapter"
        if not adapter_path.is_dir():
            raise FileNotFoundError(f"Missing resume adapter: {adapter_path}")
    processor, classifier = load_qwen35_lora_reason_runtime(
        config, device, adapter_path=adapter_path, trainable=True
    )
    trainable_report = assert_qwen35_trainable_integrity(classifier)
    optimizer = build_lora_optimizer(classifier, config)
    scheduler = build_lora_scheduler(optimizer, config, total_optimizer_steps)
    if args.resume_dir:
        load_lora_head(Path(args.resume_dir) / "head.pt", classifier)
        load_lora_trainer_state(
            Path(args.resume_dir) / "trainer_state.pt", optimizer, scheduler
        )

    run_dir = Path(args.run_dir or config["paths"]["output_dir"]).resolve()
    output_dir = run_dir / "checkpoints"
    logs_dir = run_dir / "logs"
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = logs_dir / "train_metrics.jsonl"
    if not args.resume_dir and metrics_path.exists() and any(
        '"type": "optimizer_step"' in line
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
    ):
        raise RuntimeError(f"Refusing to append a fresh run to {metrics_path}")
    metrics_handle = metrics_path.open("a", encoding="utf-8")

    def write_event(event: dict) -> None:
        metrics_handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        metrics_handle.flush()
        print(json.dumps(event, ensure_ascii=False), flush=True)

    answer_weight = float(config["loss"]["answer_loss_weight"])
    reason_weight = float(config["loss"]["reason_loss_weight"])
    pin_memory = bool(config.get("performance", {}).get("pin_memory", True))
    prefetch_depth = int(config.get("performance", {}).get("prefetch_batches", 0))
    visual_cache = Qwen35VisualFeatureCache(processor, classifier, config, device)
    cache_started = time.time()
    cache_stats = visual_cache.preload(
        (train_set, validation_set),
        progress_callback=lambda event: write_event({
            "type": "visual_cache_progress",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            **event,
        }),
    )
    write_event({
        "type": "visual_cache_ready",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        **cache_stats,
        "build_seconds": time.time() - cache_started,
    })

    training_start = time.time()
    write_event({
        "type": "run_start",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "train_samples": train_count,
        "validation_samples": len(validation_set),
        "batch_size": batch_size,
        "gradient_accumulation_steps": accumulation,
        "effective_batch_size": batch_size * accumulation,
        "epochs": epochs,
        "num_frames": int(config["video"]["num_frames"]),
        "answer_loss_weight": answer_weight,
        "reason_loss_weight": reason_weight,
        "cache_visual_features": True,
        "prefetch_batches": prefetch_depth,
        "reason_token_chunk": classifier.reason_token_chunk,
        "run_dir": str(run_dir),
    })

    generator = torch.Generator().manual_seed(int(config["seed"]))
    loader = DataLoader(
        list(range(train_count)), batch_size=batch_size, shuffle=True,
        generator=generator, num_workers=0, collate_fn=list,
    )
    start_epoch = global_step = 0
    best_macro_f1 = -1.0
    resume_optimizer_steps_in_epoch = 0
    if args.resume_dir:
        state = torch.load(
            Path(args.resume_dir) / "trainer_state.pt",
            map_location="cpu", weights_only=True,
        )
        metadata = state.get("metadata", {})
        start_epoch = int(metadata.get("epoch", 0))
        global_step = int(metadata.get("global_step", 0))
        best_macro_f1 = float(metadata.get("best_macro_f1", -1.0))
        steps_per_epoch = math.ceil(len(loader) / accumulation)
        resume_optimizer_steps_in_epoch = global_step - start_epoch * steps_per_epoch
        if resume_optimizer_steps_in_epoch < 0 or resume_optimizer_steps_in_epoch > steps_per_epoch:
            raise RuntimeError(
                "Checkpoint metadata is inconsistent with the current dataset: "
                f"epoch={start_epoch}, global_step={global_step}, "
                f"steps_per_epoch={steps_per_epoch}"
            )
        if resume_optimizer_steps_in_epoch == steps_per_epoch:
            start_epoch += 1
            resume_optimizer_steps_in_epoch = 0

    print(json.dumps({
        "trainable": trainable_report,
        "optimizer_groups": optimizer_group_report(optimizer),
        "visual_cache": cache_stats,
        "resume_dir": args.resume_dir,
        "resume_optimizer_steps_in_epoch": resume_optimizer_steps_in_epoch,
    }, indent=2), flush=True)
    optimizer.zero_grad(set_to_none=True)
    history = []
    try:
        for epoch in range(start_epoch, epochs):
            classifier.train(True)
            epoch_answer_sum = epoch_reason_sum = epoch_total_sum = 0.0
            epoch_reason_tokens = epoch_samples = 0
            started = time.time()
            accumulation_started = None
            step_answer_sum = step_reason_sum = step_total_sum = 0.0
            step_reason_tokens = step_samples = 0

            def prepare(indices):
                return _prepare_batch(
                    indices, train_set, visual_cache, processor, config,
                    include_reason=True,
                )

            skipped_micro_batches = (
                min(len(loader), resume_optimizer_steps_in_epoch * accumulation)
                if epoch == start_epoch else 0
            )
            if skipped_micro_batches:
                write_event({
                    "type": "resume_position",
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "epoch": epoch,
                    "global_step": global_step,
                    "skipped_micro_batches": skipped_micro_batches,
                    "skipped_samples": min(train_count, skipped_micro_batches * batch_size),
                })
            for batch_number, (prepared, samples) in enumerate(
                prefetch_map(loader, prepare, prefetch_depth), start=1
            ):
                if batch_number <= skipped_micro_batches:
                    continue
                if accumulation_started is None:
                    accumulation_started = time.perf_counter()
                    step_answer_sum = step_reason_sum = step_total_sum = 0.0
                    step_reason_tokens = step_samples = 0
                inputs, layouts, video_features, _ = collate_cached_reason_samples(
                    prepared, processor.tokenizer.pad_token_id, device, pin_memory
                )
                targets = torch.tensor(
                    [sample["label"] for sample in samples],
                    dtype=torch.float32, device=device,
                )
                logits, reason_loss, aux = classifier.forward_with_cached_video(
                    inputs, layouts, video_features, compute_reason_loss=True
                )
                if reason_loss is None or int(aux["reason_token_count"]) < 1:
                    raise RuntimeError("Missing reason supervision in Qwen3.5 batch")
                answer_loss = F.binary_cross_entropy_with_logits(logits, targets)
                total_loss = answer_weight * answer_loss + reason_weight * reason_loss
                if not torch.isfinite(total_loss):
                    ids = [sample["sample_id"] for sample in samples]
                    raise RuntimeError(
                        f"Non-finite loss for {ids}: answer={answer_loss.item()} "
                        f"reason={reason_loss.item()}"
                    )
                (total_loss / accumulation).backward()

                sample_count = len(samples)
                a, r, t = map(float, (
                    answer_loss.detach(), reason_loss.detach(), total_loss.detach()
                ))
                reason_tokens = int(aux["reason_token_count"])
                step_answer_sum += a * sample_count
                step_reason_sum += r * sample_count
                step_total_sum += t * sample_count
                step_reason_tokens += reason_tokens
                step_samples += sample_count
                epoch_answer_sum += a * sample_count
                epoch_reason_sum += r * sample_count
                epoch_total_sum += t * sample_count
                epoch_reason_tokens += reason_tokens
                epoch_samples += sample_count

                if batch_number % accumulation == 0 or batch_number == len(loader):
                    gradient_norm = torch.nn.utils.clip_grad_norm_(
                        [p for _, p in classifier.trainable_named_parameters()],
                        float(config["training"]["max_grad_norm"]),
                    )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    step_seconds = time.perf_counter() - accumulation_started
                    write_event({
                        "type": "optimizer_step",
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                        "epoch": epoch,
                        "global_step": global_step,
                        "answer_loss": step_answer_sum / step_samples,
                        "reason_loss": step_reason_sum / step_samples,
                        "total_loss": step_total_sum / step_samples,
                        "reason_token_count": step_reason_tokens,
                        "samples_in_step": step_samples,
                        "lora_learning_rate": float(optimizer.param_groups[0]["lr"]),
                        "head_learning_rate": float(optimizer.param_groups[1]["lr"]),
                        "gradient_norm": float(gradient_norm),
                        "optimizer_step_seconds": step_seconds,
                        "samples_per_second": step_samples / max(step_seconds, 1e-9),
                        "torch_memory_allocated_mib": torch.cuda.memory_allocated() / 2**20,
                        "torch_memory_reserved_mib": torch.cuda.memory_reserved() / 2**20,
                        "torch_peak_memory_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                        "torch_peak_memory_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
                    })
                    accumulation_started = None
                    if global_step % int(config["training"]["save_every_steps"]) == 0:
                        save_lora_checkpoint(
                            output_dir / f"checkpoint_step_{global_step:07d}",
                            classifier, optimizer, scheduler,
                            {"epoch": epoch, "global_step": global_step,
                             "best_macro_f1": best_macro_f1,
                             "config": str(Path(args.config).resolve()),
                             "visual_cache": cache_stats},
                        )
                    if global_step % int(config["training"]["validate_every_steps"]) == 0:
                        metrics = evaluate(
                            classifier, validation_set, visual_cache, processor,
                            config, device, args.max_validation_samples,
                        )
                        record = {"epoch": epoch, "global_step": global_step,
                                  "validation": metrics}
                        history.append(record)
                        write_event({"type": "validation", **record})
                        macro_f1 = float(metrics["overall"]["macro_f1"])
                        if macro_f1 > best_macro_f1:
                            best_macro_f1 = macro_f1
                            save_lora_checkpoint(
                                output_dir / "best", classifier, optimizer,
                                scheduler, {**record,
                                            "best_macro_f1": best_macro_f1},
                            )
                        classifier.train(True)

            record = {
                "epoch": epoch,
                "global_step": global_step,
                "mean_answer_loss": epoch_answer_sum / max(1, epoch_samples),
                "mean_reason_loss": epoch_reason_sum / max(1, epoch_samples),
                "mean_total_loss": epoch_total_sum / max(1, epoch_samples),
                "reason_token_count": epoch_reason_tokens,
                "sample_count": epoch_samples,
                "elapsed_seconds": time.time() - started,
            }
            history.append(record)
            write_event({"type": "epoch_end", **record})
            save_lora_checkpoint(
                output_dir / "last", classifier, optimizer, scheduler,
                {**record, "best_macro_f1": best_macro_f1},
            )
            dump_json(history, run_dir / "training_history.json")
        write_event({
            "type": "run_end",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "wall_clock_seconds": time.time() - training_start,
            "global_step": global_step,
            "best_macro_f1": best_macro_f1,
            "visual_cache": cache_stats,
            "torch_peak_memory_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
            "torch_peak_memory_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        })
    finally:
        metrics_handle.close()


if __name__ == "__main__":
    main()
