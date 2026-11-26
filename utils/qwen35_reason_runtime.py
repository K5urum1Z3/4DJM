from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import time
from typing import Any, Callable, Iterable, Iterator, TypeVar

import torch
from transformers.video_utils import VideoMetadata

from models.qwen35_lora_reason_binary import Qwen35LoraReasonBinaryClassifier
from utils.reason_loss import ReasonTokenLayout, build_reason_token_layout
from utils.video_utils import MultiViewMosaicSampler


def build_reason_sampler(config: dict[str, Any]) -> MultiViewMosaicSampler:
    video = config["video"]
    return MultiViewMosaicSampler(
        num_frames=int(video["num_frames"]),
        grid_rows=int(video["grid_rows"]),
        grid_cols=int(video["grid_cols"]),
        tile_size=int(video["tile_size"]),
        background_rgb=tuple(video.get("background_rgb", [0, 0, 0])),
    )


@dataclass(frozen=True)
class CachedSceneVideo:
    key: tuple[str, str]
    replacement_text: str
    video_grid_thw: torch.Tensor
    features: torch.Tensor
    metadata: dict[str, Any]


@dataclass(frozen=True)
class PreparedQwen35Sample:
    inputs: dict[str, torch.Tensor]
    layout: ReasonTokenLayout
    video_features: torch.Tensor
    metadata: dict[str, Any]


class Qwen35VisualFeatureCache:
    """GPU cache for deterministic, frozen Qwen3.5 scene video features."""

    def __init__(
        self,
        processor: Any,
        classifier: Qwen35LoraReasonBinaryClassifier,
        config: dict[str, Any],
        device: str,
    ) -> None:
        self.processor = processor
        self.classifier = classifier
        self.config = config
        self.device = device
        self.sampler = build_reason_sampler(config)
        self._entries: dict[tuple[str, str], CachedSceneVideo] = {}

    @staticmethod
    def scene_key(sample: dict[str, Any]) -> tuple[str, str]:
        return str(sample["source"]), str(sample["scene_id"])

    def _video_metadata(self, metadata: Any) -> VideoMetadata:
        return VideoMetadata(
            total_num_frames=metadata.common_frame_count,
            fps=metadata.fps,
            width=metadata.mosaic_width,
            height=metadata.mosaic_height,
            duration=metadata.common_frame_count / metadata.fps,
            frames_indices=list(metadata.selected_indices),
        )

    def build(self, sample: dict[str, Any]) -> CachedSceneVideo:
        key = self.scene_key(sample)
        existing = self._entries.get(key)
        if existing is not None:
            return existing

        frames, metadata = self.sampler.sample(sample["media"])
        processed, replacements = self.processor._process_videos(
            [frames],
            video_metadata=[self._video_metadata(metadata)],
            do_sample_frames=bool(
                self.config["video"].get("processor_do_sample_frames", False)
            ),
            do_resize=bool(self.config["video"].get("processor_do_resize", False)),
            return_tensors="pt",
            return_metadata=True,
        )
        if len(replacements) != 1:
            raise RuntimeError(f"Expected one video replacement for scene {key}")
        pixel_values = processed["pixel_values_videos"].to(
            self.device, non_blocking=True
        )
        grid = processed["video_grid_thw"].to(self.device, non_blocking=True)
        features = self.classifier.encode_video_features(pixel_values, grid)
        del pixel_values, processed
        entry = CachedSceneVideo(
            key=key,
            replacement_text=replacements[0],
            video_grid_thw=grid.detach(),
            features=features.detach(),
            metadata=metadata.as_dict(),
        )
        self._entries[key] = entry
        return entry

    def get(self, sample: dict[str, Any]) -> CachedSceneVideo:
        return self._entries.get(self.scene_key(sample)) or self.build(sample)

    def preload(
        self,
        datasets: Iterable[Any],
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        unique: dict[tuple[str, str], dict[str, Any]] = {}
        for dataset in datasets:
            for index in range(len(dataset)):
                sample = dataset[index]
                unique.setdefault(self.scene_key(sample), sample)
        total = len(unique)
        preload_started = time.monotonic()
        for position, key in enumerate(sorted(unique), start=1):
            scene_started = time.monotonic()
            self.build(unique[key])
            if progress_callback is not None:
                progress_callback({
                    "completed": position,
                    "total": total,
                    "scene_key": list(key),
                    "scene_seconds": round(time.monotonic() - scene_started, 3),
                    "elapsed_seconds": round(time.monotonic() - preload_started, 3),
                    **self.stats(),
                })
        return self.stats()

    def stats(self) -> dict[str, Any]:
        bytes_used = sum(
            item.features.numel() * item.features.element_size()
            + item.video_grid_thw.numel() * item.video_grid_thw.element_size()
            for item in self._entries.values()
        )
        return {
            "cached_video_scenes": len(self._entries),
            "cached_video_bytes": bytes_used,
            "cached_video_gib": bytes_used / 2**30,
            "device": self.device,
        }


def prepare_cached_reason_sample(
    sample: dict[str, Any],
    reason: str | None,
    cache: Qwen35VisualFeatureCache,
    processor: Any,
    config: dict[str, Any],
    add_generation_prompt: bool = False,
) -> PreparedQwen35Sample:
    if add_generation_prompt and reason is not None and reason.strip():
        raise ValueError("A generation prompt cannot be combined with a reason target")
    cached = cache.get(sample)
    prompt = config["prompt"]["template"].format(assertion=sample["assertion"])
    user_message = {
        "role": "user",
        "content": [
            {"type": "video", "video": None},
            {"type": "text", "text": prompt},
        ],
    }
    messages = [user_message]
    if reason is not None and reason.strip():
        messages.append({"role": "assistant", "content": reason})
    rendered_user = processor.apply_chat_template(
        [user_message], tokenize=False, add_generation_prompt=False
    )
    template_kwargs: dict[str, Any] = {}
    enable_thinking: bool | None = None
    if add_generation_prompt:
        enable_thinking = bool(config.get("input", {}).get("enable_thinking", False))
        template_kwargs["enable_thinking"] = enable_thinking
    rendered = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        **template_kwargs,
    )
    if any(
        value and value in rendered_user
        for value in (sample["sample_id"], sample["capability"])
    ):
        raise RuntimeError(f"Prompt leakage guard triggered for {sample['sample_id']}")
    expanded, _ = processor.get_text_with_replacements(
        [rendered], videos_replacements=[cached.replacement_text]
    )
    tokenized = processor.tokenizer(
        expanded,
        padding=False,
        return_tensors="pt",
    )
    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]
    mm_types = torch.tensor(
        processor.create_mm_token_type_ids(input_ids.tolist()), dtype=torch.long
    )
    inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "mm_token_type_ids": mm_types,
        "video_grid_thw": cached.video_grid_thw.detach().cpu(),
    }
    layout = build_reason_token_layout(
        input_ids,
        attention_mask,
        processor.tokenizer,
        expect_reason=reason is not None and bool(reason.strip()),
    )
    return PreparedQwen35Sample(
        inputs=inputs,
        layout=layout,
        video_features=cached.features,
        metadata={
            **cached.metadata,
            "rendered_user_prompt": rendered_user,
            "rendered_training_text": rendered,
            "add_generation_prompt": add_generation_prompt,
            "enable_thinking": enable_thinking,
        },
    )


def collate_cached_reason_samples(
    samples: list[PreparedQwen35Sample],
    pad_token_id: int,
    device: str,
    pin_memory: bool,
) -> tuple[
    dict[str, torch.Tensor], list[ReasonTokenLayout], list[torch.Tensor], list[dict[str, Any]]
]:
    if not samples:
        raise ValueError("Cannot collate an empty Qwen3.5 batch")
    max_length = max(int(item.inputs["input_ids"].shape[1]) for item in samples)
    batch_size = len(samples)
    input_ids = torch.full(
        (batch_size, max_length), int(pad_token_id), dtype=torch.long
    )
    attention_mask = torch.zeros((batch_size, max_length), dtype=torch.long)
    mm_types = torch.zeros((batch_size, max_length), dtype=torch.long)
    for row, item in enumerate(samples):
        length = int(item.inputs["input_ids"].shape[1])
        input_ids[row, :length] = item.inputs["input_ids"][0]
        attention_mask[row, :length] = item.inputs["attention_mask"][0]
        mm_types[row, :length] = item.inputs["mm_token_type_ids"][0]
    grid = torch.cat([item.inputs["video_grid_thw"] for item in samples], dim=0)
    inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "mm_token_type_ids": mm_types,
        "video_grid_thw": grid,
    }
    if pin_memory and torch.cuda.is_available():
        inputs = {key: value.pin_memory() for key, value in inputs.items()}
    inputs = {
        key: value.to(device, non_blocking=pin_memory) for key, value in inputs.items()
    }
    return (
        inputs,
        [item.layout for item in samples],
        [item.video_features for item in samples],
        [item.metadata for item in samples],
    )


T = TypeVar("T")


def prefetch_map(
    iterable: Iterable[T],
    function: Callable[[T], Any],
    depth: int,
) -> Iterator[Any]:
    """Prepare a small number of CPU batches while the GPU handles the current one."""
    if depth <= 0:
        for item in iterable:
            yield function(item)
        return
    iterator = iter(iterable)
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="qwen35-prefetch") as pool:
        pending: deque[Future[Any]] = deque()
        for _ in range(depth):
            try:
                pending.append(pool.submit(function, next(iterator)))
            except StopIteration:
                break
        while pending:
            future = pending.popleft()
            yield future.result()
            try:
                pending.append(pool.submit(function, next(iterator)))
            except StopIteration:
                pass
