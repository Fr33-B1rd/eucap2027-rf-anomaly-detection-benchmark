"""Manifest, proposal, and atomic annotation storage for the IAD pilot."""

from __future__ import annotations

from collections import OrderedDict
import csv
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import numpy as np

from dataio.iad import IADBundle, IADDataError, hash_iad_sample, safe_load_iad_pickle
from iad_labeling.schema import (
    AnnotationValidationError,
    MASK_SHAPE,
    binary_mask,
    build_annotation,
    tight_bbox,
    validate_annotator,
    validate_sample_hash,
    validate_sample_id,
    validate_stored_annotation,
)


class ManifestValidationError(ValueError):
    """Raised when the pilot manifest violates its data contract."""


class ProposalValidationError(ValueError):
    """Raised when a proposal artifact is malformed."""


class StoragePathError(ValueError):
    """Raised when an identifier could escape a configured storage root."""


@dataclass(frozen=True)
class SampleRecord:
    sample_id: str
    sample_hash: str
    modulation: str
    split: str
    source_index: int
    label: int
    shape: tuple[int, ...]
    source_path: Path

    def public_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "hash": self.sample_hash,
            "modulation": self.modulation,
            "split": self.split,
            "source_index": self.source_index,
            "label": self.label,
            "shape": list(self.shape),
        }


def _is_within(path: Path, root: Path) -> bool:
    try:
        return os.path.commonpath((str(path), str(root))) == str(root)
    except ValueError:
        return False


def _safe_child(root: Path, *parts: str) -> Path:
    candidate = root.joinpath(*parts).resolve(strict=False)
    if not _is_within(candidate, root):
        raise StoragePathError("resolved path escapes its configured root")
    return candidate


def _parse_shape(value: Any) -> tuple[int, ...]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ManifestValidationError("shape must be a JSON integer array") from exc
    if not isinstance(value, (list, tuple)) or not value:
        raise ManifestValidationError("shape must be a non-empty integer array")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise ManifestValidationError("shape must contain integers")
    shape = tuple(value)
    if shape not in ((64, 64), (1, 64, 64)):
        raise ManifestValidationError(
            f"pilot sample shape must be [1, 64, 64] (or [64, 64]), got {list(shape)}"
        )
    return shape


def _integer(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise ManifestValidationError(f"{field} must be an integer")
    try:
        converted = int(value)
    except (TypeError, ValueError) as exc:
        raise ManifestValidationError(f"{field} must be an integer") from exc
    if isinstance(value, float) and value != converted:
        raise ManifestValidationError(f"{field} must be an integer")
    return converted


def _manifest_rows(path: Path) -> list[Mapping[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows: list[Mapping[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ManifestValidationError(
                        f"invalid JSON on manifest line {line_number}"
                    ) from exc
                if not isinstance(row, Mapping):
                    raise ManifestValidationError(
                        f"manifest line {line_number} must be a JSON object"
                    )
                rows.append(row)
        return rows
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            document = json.load(handle)
        if isinstance(document, Mapping):
            document = document.get("samples")
        if not isinstance(document, list) or not all(
            isinstance(row, Mapping) for row in document
        ):
            raise ManifestValidationError(
                "JSON manifest must be a list or an object with a 'samples' list"
            )
        return document
    raise ManifestValidationError("manifest extension must be .jsonl, .csv, or .json")


class _PickleCache:
    """Small LRU cache; IAD source pickles are large, so the bound is deliberate."""

    def __init__(self, max_items: int = 2) -> None:
        self.max_items = max_items
        self._items: OrderedDict[Path, IADBundle] = OrderedDict()

    def load(self, path: Path, *, modulation: str) -> IADBundle:
        cached = self._items.pop(path, None)
        if cached is not None:
            self._items[path] = cached
            return cached
        # The restricted loader allow-lists only the constructors required by
        # the IAD tensor bundle. A pickle path is never accepted over HTTP.
        try:
            bundle = safe_load_iad_pickle(path, modulation=modulation)
        except IADDataError as exc:
            raise ManifestValidationError(str(exc)) from exc
        self._items[path] = bundle
        while len(self._items) > self.max_items:
            self._items.popitem(last=False)
        return bundle


class PilotManifest:
    """Validated read-only view of a pilot manifest and its source tensors."""

    REQUIRED_FIELDS = {
        "sample_id",
        "hash",
        "modulation",
        "split",
        "source_index",
        "label",
        "shape",
        "source_path",
    }
    MODULATIONS = frozenset({"16QAM", "CHIRP", "GMSK", "QPSK"})

    def __init__(
        self,
        manifest_path: str | os.PathLike[str],
        *,
        source_root: str | os.PathLike[str] | None = None,
        pickle_cache_items: int = 2,
    ) -> None:
        self.path = Path(manifest_path).expanduser().resolve(strict=True)
        if not self.path.is_file():
            raise ManifestValidationError("manifest path must be a file")
        rows = _manifest_rows(self.path)
        if not rows:
            raise ManifestValidationError("pilot manifest is empty")

        raw_source_paths: list[Path] = []
        for row_number, row in enumerate(rows, start=1):
            missing = self.REQUIRED_FIELDS - set(row)
            if missing:
                raise ManifestValidationError(
                    f"manifest row {row_number} lacks fields: {sorted(missing)}"
                )
            raw_path = Path(str(row["source_path"])).expanduser()
            if not raw_path.is_absolute():
                raw_path = self.path.parent / raw_path
            raw_source_paths.append(raw_path.resolve(strict=True))

        if source_root is None:
            common_path = Path(os.path.commonpath([str(path) for path in raw_source_paths]))
            inferred_root = common_path if common_path.is_dir() else common_path.parent
            self.source_root = inferred_root.resolve(strict=True)
        else:
            self.source_root = Path(source_root).expanduser().resolve(strict=True)
        if not self.source_root.is_dir():
            raise ManifestValidationError("source_root must be a directory")

        records: list[SampleRecord] = []
        seen_ids: set[str] = set()
        for row_number, (row, source_path) in enumerate(
            zip(rows, raw_source_paths), start=1
        ):
            try:
                sample_id = validate_sample_id(row["sample_id"])
                sample_hash = validate_sample_hash(row["hash"], sample_id)
            except AnnotationValidationError as exc:
                raise ManifestValidationError(
                    f"invalid identity in manifest row {row_number}: {exc}"
                ) from exc
            if sample_id in seen_ids:
                raise ManifestValidationError(f"duplicate sample_id: {sample_id}")
            seen_ids.add(sample_id)
            if not _is_within(source_path, self.source_root):
                raise ManifestValidationError(
                    f"source_path escapes source_root in row {row_number}"
                )
            if not source_path.is_file() or source_path.suffix.lower() not in {
                ".pkl",
                ".pickle",
            }:
                raise ManifestValidationError(
                    f"source_path must be an existing pickle in row {row_number}"
                )
            modulation = str(row["modulation"])
            if modulation not in self.MODULATIONS:
                raise ManifestValidationError(
                    f"unsupported modulation in row {row_number}: {modulation}"
                )
            split = str(row["split"])
            if split not in {"train", "test"}:
                raise ManifestValidationError(
                    f"split must be train or test in row {row_number}"
                )
            source_index = _integer(row["source_index"], "source_index")
            if source_index < 0:
                raise ManifestValidationError("source_index must be non-negative")
            label = _integer(row["label"], "label")
            if label not in {0, 1}:
                raise ManifestValidationError("label must be 0 or 1")
            records.append(
                SampleRecord(
                    sample_id=sample_id,
                    sample_hash=sample_hash,
                    modulation=modulation,
                    split=split,
                    source_index=source_index,
                    label=label,
                    shape=_parse_shape(row["shape"]),
                    source_path=source_path,
                )
            )

        self.records = tuple(records)
        self._by_id = {record.sample_id: record for record in records}
        self._pickle_cache = _PickleCache(max_items=max(1, pickle_cache_items))
        self._global_contrast: tuple[float, float] | None = None

    def __len__(self) -> int:
        return len(self.records)

    def get(self, sample_id: str) -> SampleRecord:
        try:
            sample_id = validate_sample_id(sample_id)
        except AnnotationValidationError as exc:
            raise KeyError(str(exc)) from exc
        try:
            return self._by_id[sample_id]
        except KeyError as exc:
            raise KeyError(f"unknown sample_id: {sample_id}") from exc

    def load_array(self, sample_id: str) -> np.ndarray:
        record = self.get(sample_id)
        bundle = self._pickle_cache.load(
            record.source_path, modulation=record.modulation
        )
        tensor = bundle.train_data if record.split == "train" else bundle.test_data
        labels = bundle.train_labels if record.split == "train" else bundle.test_labels
        if record.source_index >= tensor.shape[0]:
            raise ManifestValidationError(
                f"source_index is out of range for {record.source_path.name}:{record.split}"
            )
        if int(labels[record.source_index]) != record.label:
            raise ManifestValidationError(
                "manifest label does not match the source tensor label"
            )
        source_sample = tensor[record.source_index]
        if tuple(int(size) for size in source_sample.shape) != record.shape:
            raise ManifestValidationError(
                "manifest shape does not match the source tensor shape"
            )
        actual_id, actual_hash = hash_iad_sample(source_sample)
        if actual_id != record.sample_id or actual_hash != record.sample_hash:
            raise ManifestValidationError(
                "manifest content hash does not match the source tensor sample"
            )
        sample = source_sample.detach().cpu().numpy()
        if sample.shape == (1, 64, 64):
            sample = sample[0]
        if sample.shape != MASK_SHAPE:
            raise ManifestValidationError(
                f"source sample has shape {list(sample.shape)}, expected [64, 64]"
            )
        if not np.issubdtype(sample.dtype, np.number):
            raise ManifestValidationError("source sample must be numeric")
        sample = sample.astype(np.float32, copy=False)
        if not bool(np.isfinite(sample).all()):
            raise ManifestValidationError("source sample contains NaN or infinite values")
        return sample

    def global_contrast(self) -> tuple[float, float]:
        if self._global_contrast is None:
            pixels: list[np.ndarray] = []
            for record in sorted(
                self.records, key=lambda item: str(item.source_path).casefold()
            ):
                sample = self.load_array(record.sample_id)
                pixels.append(sample.reshape(-1))
            all_pixels = np.concatenate(pixels)
            minimum, maximum = (
                float(value) for value in np.percentile(all_pixels, (1.0, 99.0))
            )
            if maximum <= minimum:
                minimum = float(all_pixels.min())
                maximum = float(all_pixels.max())
            if not np.isfinite((minimum, maximum)).all():
                raise ManifestValidationError("cannot determine finite global contrast")
            if maximum <= minimum:
                maximum = minimum + 1.0
            self._global_contrast = (minimum, maximum)
        return self._global_contrast


def render_uint8(
    sample: np.ndarray,
    *,
    mode: str,
    global_range: tuple[float, float],
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
) -> tuple[np.ndarray, tuple[float, float]]:
    """Contrast-map a 64x64 source tensor without resizing it."""
    array = np.asarray(sample, dtype=np.float32)
    if array.shape != MASK_SHAPE or not bool(np.isfinite(array).all()):
        raise ValueError("sample must be a finite 64x64 array")
    if mode == "global":
        lower, upper = (float(value) for value in global_range)
    elif mode == "percentile":
        lower, upper = (
            float(value)
            for value in np.percentile(array, (lower_percentile, upper_percentile))
        )
        if upper <= lower:
            lower, upper = (float(value) for value in global_range)
    else:
        raise ValueError("contrast mode must be 'global' or 'percentile'")
    if not np.isfinite((lower, upper)).all() or upper <= lower:
        raise ValueError("contrast range must contain two increasing finite values")
    mapped = np.clip((array - lower) / (upper - lower), 0.0, 1.0)
    return np.rint(mapped * 255.0).astype(np.uint8), (lower, upper)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


class ProposalStore:
    """Read independent per-sample proposal NPZ files.

    Only the binary proposal mask and bbox are returned to the browser. The
    score_map is shape-checked when present but is intentionally never exposed.
    """

    def __init__(self, root: str | os.PathLike[str] | None) -> None:
        self.root: Path | None
        if root is None:
            self.root = None
        else:
            self.root = Path(root).expanduser().resolve(strict=True)
            if not self.root.is_dir():
                raise ProposalValidationError("proposal root must be a directory")

    def _artifact_path(self, sample_id: str) -> Path | None:
        validate_sample_id(sample_id)
        if self.root is None:
            return None
        return _safe_child(self.root, f"{sample_id}.npz")

    def exists(self, sample_id: str) -> bool:
        path = self._artifact_path(sample_id)
        return path is not None and path.is_file()

    def _validate_sidecar(self, sample_id: str) -> Path | None:
        """Validate an optional sidecar and return only its audited path.

        Sidecar contents are deliberately discarded after identity validation.
        They may contain generator diagnostics that must never reach the UI.
        """
        if self.root is None:
            return None
        candidates = (
            _safe_child(self.root, f"{sample_id}.json"),
            _safe_child(self.root, f"{sample_id}.provenance.json"),
            _safe_child(self.root, f"{sample_id}.npz.json"),
        )
        for path in candidates:
            if not path.is_file():
                continue
            if path.stat().st_size > 1024 * 1024:
                raise ProposalValidationError("proposal provenance sidecar is too large")
            with path.open("r", encoding="utf-8") as handle:
                document = json.load(handle)
            if not isinstance(document, Mapping):
                raise ProposalValidationError(
                    "proposal provenance sidecar must contain a JSON object"
                )
            sidecar_id = document.get("sample_id")
            if sidecar_id is not None and sidecar_id != sample_id:
                raise ProposalValidationError("proposal sidecar sample_id mismatch")
            return path
        return None

    def load(self, sample_id: str) -> dict[str, Any] | None:
        path = self._artifact_path(sample_id)
        if path is None or not path.is_file():
            return None
        if path.stat().st_size > 128 * 1024 * 1024:
            raise ProposalValidationError("proposal NPZ is unexpectedly large")
        try:
            with np.load(path, allow_pickle=False) as artifact:
                if "proposal_mask" not in artifact:
                    raise ProposalValidationError("proposal NPZ lacks proposal_mask")
                raw_mask = np.asarray(artifact["proposal_mask"])
                if raw_mask.shape == (1, 64, 64):
                    raw_mask = raw_mask[0]
                try:
                    mask = binary_mask(raw_mask)
                except AnnotationValidationError as exc:
                    raise ProposalValidationError(
                        f"invalid proposal_mask: {exc}"
                    ) from exc

                if "score_map" in artifact:
                    score_map = np.asarray(artifact["score_map"])
                    if score_map.shape == (1, 64, 64):
                        score_map = score_map[0]
                    if score_map.shape != MASK_SHAPE:
                        raise ProposalValidationError(
                            "score_map must have shape [64, 64]"
                        )
                    if not np.issubdtype(score_map.dtype, np.number) or not bool(
                        np.isfinite(score_map).all()
                    ):
                        raise ProposalValidationError(
                            "score_map must contain finite numbers"
                        )

                bbox_key = (
                    "bbox_xyxy"
                    if "bbox_xyxy" in artifact
                    else "bbox"
                    if "bbox" in artifact
                    else None
                )
                if bbox_key is not None:
                    bbox_array = np.asarray(artifact[bbox_key]).reshape(-1)
                    if bbox_array.size == 0:
                        bbox = None
                    elif bbox_array.size != 4 or not np.issubdtype(
                        bbox_array.dtype, np.number
                    ):
                        raise ProposalValidationError(
                            "bbox must contain four numeric coordinates"
                        )
                    elif not bool(np.isfinite(bbox_array).all()):
                        raise ProposalValidationError("bbox contains non-finite values")
                    else:
                        bbox = [int(value) for value in bbox_array]
                        if not np.array_equal(bbox_array, bbox):
                            raise ProposalValidationError(
                                "bbox coordinates must be integers"
                            )
                    expected_bbox = tight_bbox(mask)
                    if bbox != expected_bbox:
                        raise ProposalValidationError(
                            "proposal bbox is not the tight bbox of proposal_mask"
                        )
                else:
                    bbox = tight_bbox(mask)

                if "metadata_json" in artifact:
                    metadata_array = np.asarray(artifact["metadata_json"])
                    if metadata_array.size != 1:
                        raise ProposalValidationError(
                            "metadata_json must be a scalar JSON string"
                        )
                    metadata_text = str(metadata_array.item())
                    if len(metadata_text.encode("utf-8")) > 1024 * 1024:
                        raise ProposalValidationError("embedded metadata is too large")
                    metadata_document = json.loads(metadata_text)
                    if not isinstance(metadata_document, Mapping):
                        raise ProposalValidationError(
                            "embedded metadata_json must contain a JSON object"
                        )
                    metadata_id = metadata_document.get("sample_id")
                    if metadata_id is not None and metadata_id != sample_id:
                        raise ProposalValidationError(
                            "embedded metadata sample_id mismatch"
                        )
        except (OSError, ValueError) as exc:
            if isinstance(exc, ProposalValidationError):
                raise
            raise ProposalValidationError(f"cannot read proposal NPZ: {exc}") from exc

        sidecar_path = self._validate_sidecar(sample_id)
        provenance: dict[str, Any] = {
            "artifact_file": path.name,
            "artifact_sha256": _sha256_file(path),
            "displayed_component": "proposal_mask",
            "heatmap_values_exposed_to_ui": False,
        }
        if sidecar_path is not None:
            provenance["sidecar_file"] = sidecar_path.name
            provenance["sidecar_sha256"] = _sha256_file(sidecar_path)
        return {
            "mask_64": mask.tolist(),
            "bbox_xyxy": bbox,
            "bbox_format": "xyxy_exclusive",
            "provenance": provenance,
        }

    def provenance(self, sample_id: str) -> Mapping[str, Any] | None:
        proposal = self.load(sample_id)
        return None if proposal is None else proposal["provenance"]


class AnnotationStore:
    """Per-annotator JSON storage with atomic replacement."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        root_path = Path(root).expanduser()
        root_path.mkdir(parents=True, exist_ok=True)
        self.root = root_path.resolve(strict=True)
        if not self.root.is_dir():
            raise StoragePathError("annotation root must be a directory")

    def path_for(self, sample_id: str, annotator: str) -> Path:
        validate_sample_id(sample_id)
        validate_annotator(annotator)
        return _safe_child(self.root, annotator, f"{sample_id}.json")

    def load(self, sample_id: str, annotator: str) -> dict[str, Any] | None:
        path = self.path_for(sample_id, annotator)
        if not path.is_file():
            return None
        if path.stat().st_size > 1024 * 1024:
            raise AnnotationValidationError("stored annotation is unexpectedly large")
        with path.open("r", encoding="utf-8") as handle:
            record = json.load(handle)
        canonical = validate_stored_annotation(record)
        if canonical["sample_id"] != sample_id or canonical["annotator"] != annotator:
            raise AnnotationValidationError("stored annotation path/identity mismatch")
        return canonical

    def save(
        self,
        *,
        sample_id: str,
        sample_hash: str,
        sample_label: int,
        annotator: str,
        payload: Mapping[str, Any],
        proposal_provenance: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        path = self.path_for(sample_id, annotator)
        existing = self.load(sample_id, annotator)
        record = build_annotation(
            payload,
            sample_id=sample_id,
            sample_hash=sample_hash,
            sample_label=sample_label,
            annotator=annotator,
            proposal_provenance=proposal_provenance,
            created_at=existing["created_at"] if existing else None,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        # Resolve again after directory creation to reject a pre-existing symlink.
        path = self.path_for(sample_id, annotator)

        temporary_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="\n",
                dir=path.parent,
                prefix=f".{sample_id}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary_name = handle.name
                json.dump(record, handle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, path)
            temporary_name = None
        finally:
            if temporary_name is not None:
                try:
                    Path(temporary_name).unlink()
                except FileNotFoundError:
                    pass
        return record

    def summaries(self, annotator: str) -> dict[str, dict[str, Any]]:
        validate_annotator(annotator)
        directory = _safe_child(self.root, annotator)
        if not directory.is_dir():
            return {}
        output: dict[str, dict[str, Any]] = {}
        for path in directory.glob("iad-sha256-*.json"):
            try:
                record = self.load(path.stem, annotator)
            except (AnnotationValidationError, json.JSONDecodeError, OSError):
                continue
            if record is not None:
                output[path.stem] = {
                    "status": record["status"],
                    "confidence": record["confidence"],
                    "updated_at": record["updated_at"],
                }
        return output
