from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from datasets import FourDSAVDataset, SceneMapper
from models.evidence_base import FourDSAVStage1
from utils.config import dump_json, load_yaml, set_seed
from utils.metrics import binary_metrics, grouped_metrics
from utils.qwen35_reason_runtime import (
    Qwen35VisualFeatureCache,
    collate_cached_reason_samples,
    prepare_cached_reason_sample,
)


@torch.no_grad()
def evaluate_stage1(
    model: FourDSAVStage1,
    dataset: FourDSAVDataset,
    cache: Qwen35VisualFeatureCache,
    processor,
    config: dict,
    device: str,
    max_samples: int | None = None,
    batch_size: int | None = None,
    return_debug: bool = False,
) -> tuple[dict, list[dict]]:
    model.eval()
    count = len(dataset) if max_samples is None else min(len(dataset), max_samples)
    batch_size = batch_size or int(config["training"].get("evaluation_batch_size", 2))
    loader = DataLoader(list(range(count)), batch_size=batch_size, shuffle=False, collate_fn=list)
    records: list[dict] = []
    evidence_values: list[torch.Tensor] = []
    occurrence_values: list[torch.Tensor] = []
    semantic_cosines: list[float] = []
    for indices in loader:
        samples = [dataset[index] for index in indices]
        prepared = [
            prepare_cached_reason_sample(sample, None, cache, processor, config)
            for sample in samples
        ]
        inputs, layouts, features, media = collate_cached_reason_samples(
            prepared,
            processor.tokenizer.pad_token_id,
            device,
            bool(config.get("performance", {}).get("pin_memory", True)),
        )
        output = model.forward_with_cached_video(
            inputs,
            layouts,
            features,
            [sample["assertion"] for sample in samples],
            labels=None,
            compute_reason_loss=False,
            return_debug=return_debug,
        )
        if torch.is_grad_enabled() or output["loss"] is not None:
            raise AssertionError("Evaluation must be no-grad and answer-only")
        probs = torch.sigmoid(output["logits"].float()).cpu().tolist()
        logits = output["logits"].float().cpu().tolist()
        evidence_values.append(output["evidence_scores"].detach().float().cpu().flatten())
        occurrence_values.append(output["occurrence_scores"].detach().float().cpu().flatten())
        for units in output["semantic_units"].detach().float():
            normalized = torch.nn.functional.normalize(units, dim=-1)
            cosine = normalized @ normalized.T
            off_diagonal = cosine[~torch.eye(cosine.shape[0], dtype=torch.bool)]
            semantic_cosines.append(float(off_diagonal.mean()))
        for i, (sample, layout, logit, probability) in enumerate(zip(samples, layouts, logits, probs)):
            record = {
                "sample_id": sample["sample_id"],
                "source": sample["source"],
                "scene_id": sample["scene_id"],
                "capability": sample["capability"],
                "label": int(sample["label"]),
                "logit": float(logit),
                "probability": float(probability),
                "prediction": int(probability >= float(config["training"]["threshold"])),
                "assertion_span": list(output["assertion_spans"][i]),
                "selected_token": layout.classification_token_string,
                "sampled_frame_indices": media[i]["selected_indices"],
            }
            if return_debug:
                record["evidence_scores"] = output["evidence_scores"][i].float().cpu().tolist()
                record["occurrence_weights"] = output["occurrence_weights"][i].float().cpu().tolist()
                record["semantic_attention"] = output["semantic_attention"][i].float().cpu().tolist()
                record["spatial_attention"] = output["spatial_attention"][i].float().cpu().tolist()
            records.append(record)
    threshold = float(config["training"]["threshold"])
    metric_rows = [
        {**row, "probability": row["probability"]} for row in records
    ]
    metrics = {
        "overall": binary_metrics(
            [row["label"] for row in records], [row["probability"] for row in records], threshold
        ),
        "by_source": grouped_metrics(metric_rows, "source", threshold),
        "by_capability": grouped_metrics(metric_rows, "capability", threshold),
        "evidence_diagnostics": {
            "E_mean": float(torch.cat(evidence_values).mean()),
            "E_std": float(torch.cat(evidence_values).std(unbiased=False)),
            "E_min": float(torch.cat(evidence_values).min()),
            "E_max": float(torch.cat(evidence_values).max()),
            "occurrence_score_mean": float(torch.cat(occurrence_values).mean()),
            "occurrence_score_std": float(torch.cat(occurrence_values).std(unbiased=False)),
            "semantic_pairwise_cosine_mean": sum(semantic_cosines) / max(1, len(semantic_cosines)),
            "semantic_pairwise_cosine_max": max(semantic_cosines) if semantic_cosines else None,
        },
    }
    return metrics, records
