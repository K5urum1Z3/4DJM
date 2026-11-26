from __future__ import annotations

import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

from utils.metrics import binary_metrics
from utils.qwen35_reason_runtime import (
    CachedSceneVideo,
    Qwen35VisualFeatureCache,
    collate_cached_reason_samples,
    prepare_cached_reason_sample,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def restore_signed_v1_cache(
    cache: Qwen35VisualFeatureCache,
    cache_path: str | Path,
    v1_checkpoint: str | Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    cache_path, checkpoint = Path(cache_path), Path(v1_checkpoint)
    model_path = Path(config["paths"]["model"])
    signature = {
        "stage1_config_sha256": sha256(checkpoint / "config.json"),
        "validation_annotation_sha256": sha256(Path(config["paths"]["validation_annotations"])),
        "scene_mapping_sha256": sha256(Path(config["paths"]["mapping"])),
        "video_sampling": json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))["video"],
        "base_model": config["paths"]["model"],
        "base_model_config_sha256": sha256(model_path / "config.json"),
        "base_model_index_sha256": sha256(model_path / "model.safetensors.index.json"),
    }
    payload = torch.load(cache_path, map_location="cpu", weights_only=True)
    if payload.get("signature") != signature:
        raise RuntimeError("Signed v1 validation feature cache signature mismatch")
    for entry in payload["entries"]:
        key = tuple(entry["key"])
        cache._entries[key] = CachedSceneVideo(
            key=key,
            replacement_text=entry["replacement_text"],
            video_grid_thw=entry["video_grid_thw"].to(cache.device),
            features=entry["features"].to(cache.device),
            metadata=entry["metadata"],
        )
    del payload
    return cache.stats()


def stratified_indices(dataset: Any, count: int, seed: int = 20260926) -> list[int]:
    groups: dict[tuple[str, str, int], list[int]] = defaultdict(list)
    for index, row in enumerate(dataset.rows):
        label = int(bool(row.get("answer", row.get("label", False))))
        groups[(str(row["source"]), str(row["capability"]), label)].append(index)
    rng = random.Random(seed)
    for values in groups.values():
        rng.shuffle(values)
    result: list[int] = []
    keys = sorted(groups)
    while len(result) < min(count, len(dataset)):
        progress = False
        for key in keys:
            if groups[key] and len(result) < count:
                result.append(groups[key].pop())
                progress = True
        if not progress:
            break
    return result


def temporal_stats(value: torch.Tensor) -> dict[str, float]:
    x = value.detach().float().cpu().numpy().astype(np.float64, copy=False)
    if x.shape[0] < 2:
        return {"normalized_temporal_variance_fraction": 0.0, "mean_temporal_std": 0.0,
                "median_temporal_std": 0.0, "global_std": 0.0}
    time_std = np.std(x, axis=0)
    global_std = float(np.std(x))
    return {
        "normalized_temporal_variance_fraction": float(np.mean(np.square(time_std)) / (global_std**2 + 1e-18)),
        "mean_temporal_std": float(np.mean(time_std)),
        "median_temporal_std": float(np.median(time_std)),
        "global_std": global_std,
    }


def _summary(values: list[float]) -> dict[str, float]:
    x = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "median": float(np.median(x)),
        "q10": float(np.quantile(x, 0.10)),
        "q90": float(np.quantile(x, 0.90)),
    }


def _correlation(a: np.ndarray, b: np.ndarray) -> float | None:
    a, b = np.asarray(a, dtype=np.float64).reshape(-1), np.asarray(b, dtype=np.float64).reshape(-1)
    if a.size != b.size or a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def evaluate_v2_diagnostics(
    model: Any,
    dataset: Any,
    sample_indices: list[int],
    cache: Qwen35VisualFeatureCache,
    processor: Any,
    config: dict[str, Any],
    device: str,
    *,
    step: int,
    output_path: str | Path,
    retain_counterfactuals: bool = False,
) -> dict[str, Any]:
    """Fixed validation-only diagnostic pass using batch size one for v1 compatibility."""
    model.eval()
    per = {name: [] for name in ("Z", "r", "L", "E")}
    e_temporal_mean, e_temporal_median, entropy, semantic_cosine = [], [], [], []
    logits, probabilities, labels, rows = [], [], [], []
    retained: list[dict[str, Any]] = []
    heatmap_e: dict[str, list[list[float]]] = {}

    with torch.no_grad():
        for diagnostic_position, index in enumerate(sample_indices, start=1):
            if diagnostic_position == 1 or diagnostic_position % 25 == 0 or diagnostic_position == len(sample_indices):
                print(
                    f"stage1_v2_diagnostics step={step} "
                    f"samples={diagnostic_position}/{len(sample_indices)}",
                    flush=True,
                )
            sample = dataset[index]
            prepared = prepare_cached_reason_sample(sample, None, cache, processor, config)
            inputs, layouts, features, _ = collate_cached_reason_samples(
                [prepared], processor.tokenizer.pad_token_id, device, False
            )
            output = model.forward_with_cached_video(
                inputs, layouts, features, [sample["assertion"]],
                compute_reason_loss=False, return_debug=True,
            )
            z = output["visual_tokens"][0]
            r = output["r"][0]
            l = output["evidence_logits"][0]
            e = output["evidence_scores"][0]
            o = output["occurrence_scores"][0]
            w = output["occurrence_weights"][0]
            s = output["semantic_units"][0].float()
            for name, tensor in (("Z", z), ("r", r), ("L", l), ("E", e)):
                per[name].append(temporal_stats(tensor))
            e_temporal_mean.append(float(e.float().std(dim=0, unbiased=False).mean().cpu()))
            e_temporal_median.append(float(e.float().std(dim=0, unbiased=False).median().cpu()))
            w_float = w.float().clamp_min(1e-12)
            ent = -(w_float * w_float.log()).sum(dim=0) / math.log(max(2, w.shape[0]))
            entropy.extend(ent.cpu().tolist())
            s_norm = torch.nn.functional.normalize(s, dim=-1)
            sim = s_norm @ s_norm.T
            offdiag = sim[~torch.eye(sim.shape[0], dtype=torch.bool, device=sim.device)]
            if offdiag.numel():
                semantic_cosine.append(float(offdiag.mean().cpu()))
            logit = float(output["logits"][0].float().cpu())
            prob = float(torch.sigmoid(output["logits"][0].float()).cpu())
            label = int(sample["label"])
            logits.append(logit)
            probabilities.append(prob)
            labels.append(label)
            rows.append({
                "sample_id": str(sample["sample_id"]), "source": str(sample["source"]),
                "scene_id": str(sample["scene_id"]), "capability": str(sample["capability"]),
                "label": label, "logit": logit, "probability": prob,
                "prediction": int(prob >= float(config["training"]["threshold"])),
            })
            if retain_counterfactuals and len(retained) < 64:
                retained.append({
                    "sample_id": str(sample["sample_id"]),
                    "video_key": [str(sample["source"]), str(sample["scene_id"])],
                    "r": r.detach().float().cpu(),
                    "E": e.detach().float().cpu(),
                    "logit": logit,
                })
            # Keep a stable 10-positive / 10-negative probe set.
            if sum(1 for item in heatmap_e.values() if item["label"] == 1) < 10 and label == 1:
                heatmap_e[str(sample["sample_id"])] = {"label": label, "E": e.detach().float().cpu().tolist()}
            elif sum(1 for item in heatmap_e.values() if item["label"] == 0) < 10 and label == 0:
                heatmap_e[str(sample["sample_id"])] = {"label": label, "E": e.detach().float().cpu().tolist()}
            del output, z, r, l, e, o, w, s, inputs, features

    result = {
        "step": int(step),
        "sample_count": len(rows),
        "samples": rows,
        "classification_metrics": binary_metrics(labels, probabilities, float(config["training"]["threshold"])),
        "temporal_variation": {
            name: {metric: _summary([row[metric] for row in values]) for metric in values[0]}
            for name, values in per.items()
        },
        "variance_retention": {},
        "E_temporal_std": {
            "mean_per_sample": _summary(e_temporal_mean),
            "median_per_sample": _summary(e_temporal_median),
        },
        "occurrence_normalized_entropy": _summary(entropy),
        "semantic_unit_pairwise_cosine_mean": _summary(semantic_cosine),
        "shapes": {},
        "heatmap_e": heatmap_e,
        "_retained": retained,
    }
    for name, source, target in (("Z_to_r", "Z", "r"), ("r_to_L", "r", "L"), ("L_to_E", "L", "E")):
        source_values = [item["normalized_temporal_variance_fraction"] for item in per[source]]
        target_values = [item["normalized_temporal_variance_fraction"] for item in per[target]]
        sample_ratios = [b / max(a, 1e-18) for a, b in zip(source_values, target_values)]
        result["variance_retention"][name] = {
            "ratio_of_mean_normalized_variances": float(np.mean(target_values) / max(np.mean(source_values), 1e-18)),
            "per_sample_ratio": _summary(sample_ratios),
        }
    # Replace placeholder with actual observed tensor shapes from the first sample.
    if sample_indices:
        sample = dataset[sample_indices[0]]
        prepared = prepare_cached_reason_sample(sample, None, cache, processor, config)
        inputs, layouts, features, _ = collate_cached_reason_samples(
            [prepared], processor.tokenizer.pad_token_id, device, False
        )
        with torch.no_grad():
            out = model.forward_with_cached_video(
                inputs, layouts, features, [sample["assertion"]],
                compute_reason_loss=False, return_debug=True,
            )
        result["shapes"] = {
            "Z": list(out["visual_tokens"].shape),
            "H_A": list(out["assertion_hidden"].shape),
            "S_A": list(out["semantic_units"].shape),
            "r": list(out["r"].shape),
            "L_A": list(out["evidence_logits"].shape),
            "E_A": list(out["evidence_scores"].shape),
            "o": list(out["occurrence_scores"].shape),
            "logit": list(out["logits"].shape),
        }
        del out, inputs, features
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {key: value for key, value in result.items() if not key.startswith("_")}
    output_path.write_text(json.dumps(serializable, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    result["_retained"] = retained
    return result


@torch.no_grad()
def counterfactual_summary(model: Any, diagnostic_result: dict[str, Any]) -> dict[str, Any]:
    model.eval()
    entries = diagnostic_result.get("_retained", [])
    if not entries:
        raise ValueError("No retained r/E tensors for counterfactual analysis")
    field = model.evidence_field
    scorer = field.evidence_scorer
    normal_e, zero_e, normal_logit, zero_logit = [], [], [], []
    for item in entries:
        r = item["r"].unsqueeze(0).to(next(scorer.parameters()).device)
        e = item["E"].unsqueeze(0).to(r.device)
        e0 = torch.sigmoid(scorer(torch.zeros_like(r)))
        o0, _ = model.occurrence_verifier(e0)
        l0 = model.verification_head(o0)
        normal_e.append(e.cpu().numpy().reshape(-1))
        zero_e.append(e0.detach().cpu().numpy().reshape(-1))
        normal_logit.append(float(item["logit"]))
        zero_logit.append(float(l0[0].detach().cpu()))
    a, b = np.concatenate(normal_e), np.concatenate(zero_e)
    z_summary = {
        "sample_count": len(entries),
        "E_mae": float(np.mean(np.abs(a - b))),
        "E_pearson": _correlation(a, b),
        "mean_abs_logit_diff": float(np.mean(np.abs(np.asarray(normal_logit) - np.asarray(zero_logit)))),
        "prediction_flip_rate": float(np.mean((np.asarray(normal_logit) >= 0) != (np.asarray(zero_logit) >= 0))),
    }

    pairs = []
    for index, item in enumerate(entries):
        replacement = next((candidate for candidate in entries if candidate["video_key"] != item["video_key"]), None)
        if replacement is not None:
            pairs.append((item, replacement))
    cross_diffs, cross_logits = [], []
    shuffle_diffs, shuffle_logit_diffs, shuffle_consistency = [], [], []
    for original, replacement in pairs:
        device = next(scorer.parameters()).device
        r = original["r"].unsqueeze(0).to(device)
        r_other = replacement["r"].unsqueeze(0).to(device)
        e = original["E"].unsqueeze(0).to(device)
        e_other = torch.sigmoid(scorer(r_other))
        o_other, _ = model.occurrence_verifier(e_other)
        l_other = model.verification_head(o_other)
        cross_diffs.append(float((e - e_other).abs().mean().cpu()))
        cross_logits.append(abs(float(original["logit"]) - float(l_other[0].cpu())))
        e_shuffle = torch.sigmoid(scorer(r.flip(1)))
        o, _ = model.occurrence_verifier(e)
        o_shuffle, _ = model.occurrence_verifier(e_shuffle)
        l = model.verification_head(o)
        l_shuffle = model.verification_head(o_shuffle)
        shuffle_diffs.append(float((e - e_shuffle).abs().mean().cpu()))
        shuffle_logit_diffs.append(abs(float((l - l_shuffle).abs().max().cpu())))
        shuffle_consistency.append(float((e_shuffle - e.flip(1)).abs().max().cpu()))
    return {
        "r_zero": z_summary,
        "cross_video_r_replacement": {
            "pair_count": len(pairs),
            "mean_E_mae": float(np.mean(cross_diffs)) if cross_diffs else None,
            "mean_abs_logit_diff": float(np.mean(cross_logits)) if cross_logits else None,
        },
        "temporal_shuffle": {
            "pair_count": len(pairs),
            "mean_E_profile_mae": float(np.mean(shuffle_diffs)) if shuffle_diffs else None,
            "mean_abs_logit_diff": float(np.mean(shuffle_logit_diffs)) if shuffle_logit_diffs else None,
            "max_error_vs_exact_E_permutation": float(max(shuffle_consistency)) if shuffle_consistency else None,
        },
    }


def render_heatmap_comparisons(
    sample_ids: list[str],
    label_by_id: dict[str, int],
    v1_e: dict[str, Any],
    step_e: dict[int, dict[str, Any]],
    output_dir: str | Path,
) -> list[str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    colors = [(0, 48, 105), (0, 133, 119), (246, 196, 83), (211, 67, 65)]

    def color(value: float) -> tuple[int, int, int]:
        x = min(1.0, max(0.0, float(value))) * (len(colors) - 1)
        lo = int(x)
        hi = min(len(colors) - 1, lo + 1)
        alpha = x - lo
        return tuple(int(colors[lo][i] * (1 - alpha) + colors[hi][i] * alpha) for i in range(3))

    steps = sorted(step_e)
    panels = [("v1", v1_e)] + [(f"v2 step {step}", step_e[step]) for step in steps]
    cell, margin, title_h = 32, 12, 28
    panel_w = cell * 8 + margin * 2
    for sample_id in sample_ids:
        if sample_id not in v1_e or any(sample_id not in values for values in step_e.values()):
            continue
        label = int(label_by_id[sample_id])
        image = Image.new("RGB", (panel_w * len(panels), title_h + cell * 8 + margin * 2 + 20), "white")
        draw = ImageDraw.Draw(image)
        for panel_i, (title, source) in enumerate(panels):
            x0 = panel_i * panel_w + margin
            draw.text((panel_i * panel_w + margin, 6), title, fill="black")
            for t in range(8):
                for m in range(8):
                    px = x0 + m * cell
                    py = title_h + margin + t * cell
                    value = float(source[sample_id]["E"][t][m])
                    draw.rectangle((px, py, px + cell - 1, py + cell - 1), fill=color(value))
        draw.text((8, image.height - 18), f"{sample_id} | GT={label} | E scale [0,1]", fill="black")
        target = output_dir / f"{sample_id}_gt{label}.png"
        image.save(target)
        saved.append(str(target))
    return saved
