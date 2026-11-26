from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from torch.utils.data import Dataset

from utils.config import load_json, load_yaml


class MappingError(RuntimeError):
    """Raised when an annotation scene cannot be mapped safely to real media."""


def _natural_key(value: str | Path) -> list[Any]:
    import re

    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", str(value))]


@dataclass(frozen=True)
class MediaSpec:
    source: str
    scene_id: str
    kind: str
    scene_path: Path
    views: tuple[Path, ...]
    frame_glob: str | None
    default_fps: float | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "scene_id": self.scene_id,
            "kind": self.kind,
            "scene_path": str(self.scene_path),
            "views": [str(path) for path in self.views],
            "frame_glob": self.frame_glob,
            "default_fps": self.default_fps,
        }


class SceneMapper:
    def __init__(self, mapping_path: str | Path):
        self.mapping_path = Path(mapping_path).resolve()
        self.config = load_yaml(self.mapping_path)
        self.project_root = Path(self.config["project_root"]).resolve()
        self.sources = self.config["sources"]
        self._source_lookup = {name.lower(): name for name in self.sources}
        self._cache: dict[tuple[str, str], MediaSpec] = {}

    def canonical_source(self, source: str) -> str:
        try:
            return self._source_lookup[source.lower()]
        except KeyError as exc:
            raise MappingError(f"Unknown source {source!r}") from exc

    def resolve(self, source: str, scene_id: str) -> MediaSpec:
        canonical = self.canonical_source(source)
        key = (canonical, scene_id)
        if key in self._cache:
            return self._cache[key]

        source_cfg = self.sources[canonical]
        scene_cfg = source_cfg.get("scenes", {}).get(scene_id)
        if scene_cfg is None:
            raise MappingError(f"No mapping for {canonical}/{scene_id}")
        if scene_cfg.get("status") == "blocked":
            raise MappingError(
                f"Blocked mapping for {canonical}/{scene_id}: {scene_cfg.get('reason', 'unspecified reason')}"
            )

        scene_path = (self.project_root / scene_cfg["path"]).resolve()
        if not scene_path.is_dir():
            raise MappingError(f"Mapped scene directory is missing: {scene_path}")
        view_pattern = source_cfg["view_glob"].format(path=str(scene_path))
        # pathlib cannot glob an arbitrary absolute pattern.
        import glob

        views = tuple(sorted((Path(item).resolve() for item in glob.glob(view_pattern)), key=_natural_key))
        views = tuple(path for path in views if path.is_dir() or path.is_file())
        if not views:
            raise MappingError(f"No views match {view_pattern}")

        kind = source_cfg["kind"]
        if kind not in {"video", "image_sequence"}:
            raise MappingError(f"Unsupported media kind {kind!r} for {canonical}")
        spec = MediaSpec(
            source=canonical,
            scene_id=scene_id,
            kind=kind,
            scene_path=scene_path,
            views=views,
            frame_glob=source_cfg.get("frame_glob"),
            default_fps=float(source_cfg["default_fps"]) if "default_fps" in source_cfg else None,
        )
        self._cache[key] = spec
        return spec

    def audit_pairs(self, pairs: Iterable[tuple[str, str]]) -> tuple[dict[tuple[str, str], MediaSpec], dict[tuple[str, str], str]]:
        resolved: dict[tuple[str, str], MediaSpec] = {}
        errors: dict[tuple[str, str], str] = {}
        for source, scene_id in sorted(set(pairs)):
            try:
                resolved[(source, scene_id)] = self.resolve(source, scene_id)
            except MappingError as exc:
                errors[(source, scene_id)] = str(exc)
        return resolved, errors


class FourDSAVDataset(Dataset):
    REQUIRED_FIELDS = {
        "source",
        "scene_id",
        "sample_id",
        "capability",
        "assertion",
        "answer",
        "reason",
        "evidence_time_ids",
    }

    def __init__(self, annotation_path: str | Path, mapper: SceneMapper):
        self.annotation_path = Path(annotation_path).resolve()
        self.mapper = mapper
        rows = load_json(self.annotation_path)
        if not isinstance(rows, list):
            raise ValueError(f"Annotation root must be a list: {self.annotation_path}")
        for index, row in enumerate(rows):
            missing = self.REQUIRED_FIELDS - set(row)
            if missing:
                raise ValueError(f"Row {index} in {self.annotation_path} misses {sorted(missing)}")
            if not isinstance(row["answer"], bool):
                raise ValueError(f"Row {index} has a non-boolean answer")
            if not isinstance(row["assertion"], str) or not row["assertion"].strip():
                raise ValueError(f"Row {index} has an empty assertion")
        self.rows: list[dict[str, Any]] = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        media = self.mapper.resolve(row["source"], row["scene_id"])
        return {
            "assertion": row["assertion"],
            "label": float(row["answer"]),
            "sample_id": row["sample_id"],
            "source": row["source"],
            "scene_id": row["scene_id"],
            "capability": row["capability"],
            "media": media,
        }

    def scene_pairs(self) -> set[tuple[str, str]]:
        return {(row["source"], row["scene_id"]) for row in self.rows}

    def validate_mappings(self) -> tuple[dict[tuple[str, str], MediaSpec], dict[tuple[str, str], str]]:
        return self.mapper.audit_pairs(self.scene_pairs())
