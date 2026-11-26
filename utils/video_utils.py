from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

import cv2
import numpy as np
from PIL import Image
from datasets.fourdsav_dataset import MediaSpec


def natural_key(value: str | Path) -> list[Any]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", str(value))]


@dataclass(frozen=True)
class ViewMetadata:
    path: str
    frame_count: int
    fps: float
    width: int
    height: int
    duration: float


@dataclass(frozen=True)
class MosaicMetadata:
    source: str
    scene_id: str
    view_count: int
    common_frame_count: int
    fps: float
    selected_indices: tuple[int, ...]
    mosaic_width: int
    mosaic_height: int
    views: tuple[ViewMetadata, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "scene_id": self.scene_id,
            "view_count": self.view_count,
            "common_frame_count": self.common_frame_count,
            "fps": self.fps,
            "selected_indices": list(self.selected_indices),
            "mosaic_width": self.mosaic_width,
            "mosaic_height": self.mosaic_height,
            "views": [view.__dict__ for view in self.views],
        }


class MultiViewMosaicSampler:
    """Deterministic synchronized all-view sampler.

    Sampling depends only on media metadata and fixed configuration. It never
    receives assertion text, labels, reasons, evidence IDs, or sample IDs.
    """

    def __init__(
        self,
        num_frames: int = 4,
        grid_rows: int = 7,
        grid_cols: int = 7,
        tile_size: int = 96,
        background_rgb: tuple[int, int, int] = (0, 0, 0),
    ):
        if num_frames < 1 or grid_rows < 1 or grid_cols < 1 or tile_size < 8:
            raise ValueError("Invalid sampler dimensions")
        self.num_frames = num_frames
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.tile_size = tile_size
        self.background_rgb = tuple(int(value) for value in background_rgb)
        self._probe_cache: dict[tuple[str, ...], tuple[ViewMetadata, ...]] = {}
        self._sample_cache: dict[tuple[str, str], tuple[list[Image.Image], MosaicMetadata]] = {}

    @property
    def max_views(self) -> int:
        return self.grid_rows * self.grid_cols

    def clear_sample_cache(self) -> None:
        """Release decoded mosaic images while retaining cheap probe metadata."""
        self._sample_cache.clear()

    def _probe_image_sequence(self, path: Path, frame_glob: str, fps: float) -> ViewMetadata:
        files = sorted(path.glob(frame_glob), key=natural_key)
        if not files:
            raise RuntimeError(f"No image frames in {path} matching {frame_glob}")
        image = cv2.imread(str(files[0]), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Cannot decode first image frame: {files[0]}")
        height, width = image.shape[:2]
        count = len(files)
        return ViewMetadata(str(path), count, fps, width, height, count / fps)

    def _probe_video(self, path: Path) -> ViewMetadata:
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open video: {path}")
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        capture.release()
        if count <= 0 or fps <= 0 or width <= 0 or height <= 0:
            raise RuntimeError(
                f"Invalid video metadata for {path}: frames={count}, fps={fps}, size={width}x{height}"
            )
        return ViewMetadata(str(path), count, fps, width, height, count / fps)

    def probe(self, spec: MediaSpec) -> tuple[ViewMetadata, ...]:
        cache_key = tuple(str(path) for path in spec.views)
        if cache_key in self._probe_cache:
            return self._probe_cache[cache_key]
        if len(spec.views) > self.max_views:
            raise RuntimeError(
                f"{spec.source}/{spec.scene_id} has {len(spec.views)} views, exceeding fixed grid capacity {self.max_views}"
            )
        if spec.kind == "image_sequence":
            if spec.default_fps is None or spec.frame_glob is None:
                raise RuntimeError(f"Image sequence metadata is incomplete for {spec.scene_path}")
            metadata = tuple(
                self._probe_image_sequence(path, spec.frame_glob, spec.default_fps) for path in spec.views
            )
        else:
            metadata = tuple(self._probe_video(path) for path in spec.views)
        self._probe_cache[cache_key] = metadata
        return metadata

    def _uniform_indices(self, common_count: int) -> tuple[int, ...]:
        if common_count <= 0:
            raise RuntimeError("Cannot sample a zero-frame scene")
        if self.num_frames == 1:
            return (common_count // 2,)
        indices = np.linspace(0, common_count - 1, num=self.num_frames)
        return tuple(int(round(value)) for value in indices)

    def _letterbox(self, frame_bgr: np.ndarray) -> np.ndarray:
        if frame_bgr is None or frame_bgr.size == 0:
            raise RuntimeError("Decoder returned an empty frame")
        height, width = frame_bgr.shape[:2]
        scale = min(self.tile_size / width, self.tile_size / height)
        target_w = max(1, int(round(width * scale)))
        target_h = max(1, int(round(height * scale)))
        resized = cv2.resize(frame_bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)
        tile = np.empty((self.tile_size, self.tile_size, 3), dtype=np.uint8)
        tile[:, :] = self.background_rgb[::-1]
        x0 = (self.tile_size - target_w) // 2
        y0 = (self.tile_size - target_h) // 2
        tile[y0 : y0 + target_h, x0 : x0 + target_w] = resized
        return tile

    def _compose(self, decoded: list[np.ndarray]) -> Image.Image:
        canvas = np.empty(
            (self.grid_rows * self.tile_size, self.grid_cols * self.tile_size, 3), dtype=np.uint8
        )
        canvas[:, :] = self.background_rgb[::-1]
        for view_index, frame in enumerate(decoded):
            row, col = divmod(view_index, self.grid_cols)
            y0, x0 = row * self.tile_size, col * self.tile_size
            canvas[y0 : y0 + self.tile_size, x0 : x0 + self.tile_size] = self._letterbox(frame)
        return Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))

    def _sample_image_sequences(self, spec: MediaSpec, indices: tuple[int, ...]) -> list[Image.Image]:
        frame_lists = [sorted(path.glob(spec.frame_glob or "*.jpg"), key=natural_key) for path in spec.views]
        output: list[Image.Image] = []
        for frame_index in indices:
            decoded: list[np.ndarray] = []
            for files in frame_lists:
                frame = cv2.imread(str(files[frame_index]), cv2.IMREAD_COLOR)
                if frame is None:
                    raise RuntimeError(f"Cannot decode image frame {files[frame_index]}")
                decoded.append(frame)
            output.append(self._compose(decoded))
        return output

    def _sample_videos(self, spec: MediaSpec, indices: tuple[int, ...]) -> list[Image.Image]:
        frames_by_time: list[list[np.ndarray]] = [[] for _ in indices]
        target_slots = {frame_index: slot for slot, frame_index in enumerate(indices)}
        last_frame_index = max(indices)
        for path in spec.views:
            # Open one view at a time. Opening dozens of FFmpeg-backed
            # VideoCapture objects concurrently can create thousands of codec
            # threads for multi-view scenes. Each view is still decoded once
            # in frame order, and the selected synchronized indices are
            # accumulated before mosaics are composed, so the visual input is
            # unchanged.
            capture = cv2.VideoCapture(str(path))
            if not capture.isOpened():
                capture.release()
                raise RuntimeError(f"Cannot open video view {path}")
            try:
                for frame_index in range(last_frame_index + 1):
                    ok = capture.grab()
                    if not ok:
                        raise RuntimeError(f"Cannot decode frame {frame_index} from {path}")
                    time_slot = target_slots.get(frame_index)
                    if time_slot is not None:
                        ok, frame = capture.retrieve()
                        if not ok or frame is None:
                            raise RuntimeError(f"Cannot decode frame {frame_index} from {path}")
                        frames_by_time[time_slot].append(frame)
            finally:
                capture.release()
        return [self._compose(decoded) for decoded in frames_by_time]

    def sample(self, spec: MediaSpec) -> tuple[list[Image.Image], MosaicMetadata]:
        cache_key = (spec.source, spec.scene_id)
        if cache_key in self._sample_cache:
            return self._sample_cache[cache_key]
        views = self.probe(spec)
        common_count = min(view.frame_count for view in views)
        indices = self._uniform_indices(common_count)
        if spec.kind == "image_sequence":
            frames = self._sample_image_sequences(spec, indices)
        else:
            frames = self._sample_videos(spec, indices)
        fps_values = [view.fps for view in views if math.isfinite(view.fps) and view.fps > 0]
        fps = float(median(fps_values)) if fps_values else float(spec.default_fps or 30.0)
        metadata = MosaicMetadata(
            source=spec.source,
            scene_id=spec.scene_id,
            view_count=len(spec.views),
            common_frame_count=common_count,
            fps=fps,
            selected_indices=indices,
            mosaic_width=self.grid_cols * self.tile_size,
            mosaic_height=self.grid_rows * self.tile_size,
            views=views,
        )
        self._sample_cache[cache_key] = (frames, metadata)
        return frames, metadata
