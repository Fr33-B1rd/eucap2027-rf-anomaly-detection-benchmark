"""Stable sample manifests, duplicate audits, and pilot selection for IAD."""

from __future__ import annotations

import csv
import gc
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import torch

from dataio.iad.pickle_io import (
    IADDataError,
    discover_iad_sources,
    safe_load_iad_pickle,
)


MANIFEST_FIELDS = (
    "sample_id",
    "hash",
    "modulation",
    "split",
    "source_index",
    "label",
    "shape",
    "source_path",
)
HASH_RE = re.compile(r"^sha256:(?P<digest>[0-9a-f]{64})$")
SAMPLE_ID_PREFIX = "iad-sha256-"
HASH_DOMAIN = b"IAD_SAMPLE_V1\0"
SAMPLE_HASH_SCHEMA_VERSION = 1
DATASET_FINGERPRINT_DOMAIN = b"IAD_DATASET_V1\0"


@dataclass(frozen=True)
class IADManifestRecord:
    """One logical sample reference; no tensor data is copied into a manifest."""

    sample_id: str
    hash: str
    modulation: str
    split: str
    source_index: int
    label: int
    shape: tuple[int, ...]
    source_path: str

    def __post_init__(self) -> None:
        match = HASH_RE.fullmatch(self.hash)
        if match is None:
            raise IADDataError(
                f"invalid manifest hash {self.hash!r}; expected sha256:<64 hex>"
            )
        expected_id = f"{SAMPLE_ID_PREFIX}{match.group('digest')}"
        if self.sample_id != expected_id:
            raise IADDataError(
                f"sample_id/hash mismatch: expected {expected_id!r}, "
                f"got {self.sample_id!r}"
            )
        if not self.modulation:
            raise IADDataError("manifest modulation must not be empty")
        if self.split not in {"train", "test"}:
            raise IADDataError(
                f"manifest split must be 'train' or 'test', got {self.split!r}"
            )
        if self.source_index < 0:
            raise IADDataError("manifest source_index must be non-negative")
        if self.label not in {0, 1}:
            raise IADDataError("manifest label must be 0 or 1")
        if not self.shape or any(size <= 0 for size in self.shape):
            raise IADDataError(
                f"manifest shape must contain positive sizes, got {self.shape}"
            )
        if not self.source_path:
            raise IADDataError("manifest source_path must not be empty")

    def to_dict(self, *, csv_shape: bool = False) -> dict[str, Any]:
        shape: list[int] | str = list(self.shape)
        if csv_shape:
            shape = json.dumps(shape, separators=(",", ":"))
        return {
            "sample_id": self.sample_id,
            "hash": self.hash,
            "modulation": self.modulation,
            "split": self.split,
            "source_index": self.source_index,
            "label": self.label,
            "shape": shape,
            "source_path": self.source_path,
        }

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> "IADManifestRecord":
        missing = set(MANIFEST_FIELDS).difference(row)
        if missing:
            raise IADDataError(f"manifest row is missing fields: {sorted(missing)}")
        shape_value = row["shape"]
        if isinstance(shape_value, str):
            try:
                shape_value = json.loads(shape_value)
            except json.JSONDecodeError as exc:
                raise IADDataError(f"invalid JSON shape {row['shape']!r}") from exc
        if not isinstance(shape_value, (list, tuple)):
            raise IADDataError(f"manifest shape must be an array, got {shape_value!r}")
        try:
            shape = tuple(int(size) for size in shape_value)
            source_index = int(row["source_index"])
            label = int(row["label"])
        except (TypeError, ValueError) as exc:
            raise IADDataError(
                "manifest source_index, label, and shape must be integral"
            ) from exc
        return cls(
            sample_id=str(row["sample_id"]),
            hash=str(row["hash"]),
            modulation=str(row["modulation"]).strip().upper(),
            split=str(row["split"]),
            source_index=source_index,
            label=label,
            shape=shape,
            source_path=str(row["source_path"]),
        )


def _canonical_numpy_sample(
    sample: torch.Tensor | np.ndarray,
) -> tuple[np.ndarray, str]:
    if isinstance(sample, torch.Tensor):
        tensor = sample.detach()
        if tensor.device.type != "cpu":
            tensor = tensor.cpu()
        try:
            array = tensor.contiguous().numpy()
        except (TypeError, RuntimeError) as exc:
            raise IADDataError(
                f"sample dtype {tensor.dtype} cannot be hashed as NumPy bytes"
            ) from exc
    else:
        array = np.asarray(sample)

    if array.dtype.hasobject:
        raise IADDataError("object arrays cannot be content-hashed safely")
    if not (
        np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.bool_)
    ):
        raise IADDataError(f"unsupported sample dtype for hashing: {array.dtype}")

    array = np.ascontiguousarray(array)
    dtype = array.dtype
    if dtype.byteorder == ">" or (dtype.byteorder == "=" and sys.byteorder == "big"):
        little_dtype = dtype.newbyteorder("<")
        array = array.astype(little_dtype, copy=True)
        dtype = little_dtype
    else:
        dtype = dtype.newbyteorder("<")
    return array, dtype.str


def hash_iad_sample(
    sample: torch.Tensor | np.ndarray,
) -> tuple[str, str]:
    """Return a stable sample ID and SHA-256 content hash.

    The hash includes a schema domain, canonical little-endian dtype, shape,
    and logical C-order sample bytes.  It deliberately excludes file name,
    modulation, split, and index so byte-identical samples group together
    across source files.
    """

    array, dtype_string = _canonical_numpy_sample(sample)
    header = json.dumps(
        {"dtype": dtype_string, "shape": list(array.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    hasher = hashlib.sha256()
    hasher.update(HASH_DOMAIN)
    hasher.update(len(header).to_bytes(8, "big"))
    hasher.update(header)
    hasher.update(memoryview(array).cast("B"))
    digest = hasher.hexdigest()
    return f"{SAMPLE_ID_PREFIX}{digest}", f"sha256:{digest}"


def sha256_file(path: str | Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Return a streaming SHA-256 fingerprint without loading a file at once."""

    if chunk_size <= 0:
        raise IADDataError("chunk_size must be positive")
    source = Path(path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"source file does not exist: {source}")
    if not source.is_file():
        raise IADDataError(f"source path is not a file: {source}")
    hasher = hashlib.sha256()
    with source.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return f"sha256:{hasher.hexdigest()}"


def _manifest_json_line(record: IADManifestRecord) -> str:
    return json.dumps(
        record.to_dict(),
        ensure_ascii=False,
        separators=(",", ":"),
    )


def manifest_jsonl_sha256(records: Iterable[IADManifestRecord]) -> str:
    """Fingerprint the exact UTF-8 JSONL byte representation written below."""

    hasher = hashlib.sha256()
    for record in records:
        hasher.update(_manifest_json_line(record).encode("utf-8"))
        hasher.update(b"\n")
    return f"sha256:{hasher.hexdigest()}"


def build_iad_manifest(
    dataset_dir: str | Path,
    *,
    progress: Callable[[str], None] | None = None,
) -> list[IADManifestRecord]:
    """Build an in-memory manifest while loading one source file at a time."""

    records: list[IADManifestRecord] = []
    for source in discover_iad_sources(dataset_dir):
        if progress is not None:
            progress(f"loading {source.path.name}")
        bundle = safe_load_iad_pickle(
            source.path,
            modulation=source.modulation,
        )
        for split, data, labels in (
            ("train", bundle.train_data, bundle.train_labels),
            ("test", bundle.test_data, bundle.test_labels),
        ):
            if progress is not None:
                progress(
                    f"hashing {bundle.modulation}/{split} "
                    f"({int(data.shape[0])} samples)"
                )
            for source_index in range(int(data.shape[0])):
                sample = data[source_index]
                sample_id, content_hash = hash_iad_sample(sample)
                records.append(
                    IADManifestRecord(
                        sample_id=sample_id,
                        hash=content_hash,
                        modulation=bundle.modulation,
                        split=split,
                        source_index=source_index,
                        label=int(labels[source_index]),
                        shape=tuple(int(size) for size in sample.shape),
                        source_path=str(bundle.source_path),
                    )
                )
        del bundle
        gc.collect()
    return records


def _record_identity(record: IADManifestRecord) -> tuple[Any, ...]:
    return (
        record.modulation,
        0 if record.split == "train" else 1,
        record.source_path.casefold(),
        record.source_index,
    )


def _group_summary(
    content_hash: str,
    rows: Sequence[IADManifestRecord],
) -> dict[str, Any]:
    ordered = sorted(rows, key=_record_identity)
    paths = sorted({row.source_path for row in rows}, key=str.casefold)
    modulations = sorted({row.modulation for row in rows})
    splits = sorted({row.split for row in rows})
    labels = sorted({row.label for row in rows})
    return {
        "hash": content_hash,
        "sample_id": ordered[0].sample_id,
        "record_count": len(ordered),
        "cross_file": len(paths) > 1,
        "train_test_leakage": set(splits) == {"test", "train"},
        "modulations": modulations,
        "splits": splits,
        "labels": labels,
        "records": [
            {
                "modulation": row.modulation,
                "split": row.split,
                "source_index": row.source_index,
                "label": row.label,
                "source_path": row.source_path,
            }
            for row in ordered
        ],
    }


def _source_provenance(
    rows: Sequence[IADManifestRecord],
) -> tuple[list[dict[str, Any]], str]:
    records_by_path: dict[str, list[IADManifestRecord]] = defaultdict(list)
    for row in rows:
        records_by_path[row.source_path].append(row)

    entries: list[dict[str, Any]] = []
    for source_path, source_rows in records_by_path.items():
        source = Path(source_path).expanduser().resolve()
        modulations = sorted({row.modulation for row in source_rows})
        if len(modulations) != 1:
            raise IADDataError(
                f"source_path maps to multiple modulations: {source} -> {modulations}"
            )
        entries.append(
            {
                "modulation": modulations[0],
                "source_path": str(source),
                "file_name": source.name,
                "size_bytes": source.stat().st_size,
                "file_sha256": sha256_file(source),
                "record_count": len(source_rows),
            }
        )
    entries.sort(
        key=lambda entry: (
            entry["modulation"],
            entry["file_name"].casefold(),
            entry["source_path"].casefold(),
        )
    )

    # Exclude absolute paths so the dataset fingerprint remains stable after
    # moving an otherwise identical frozen dataset to another machine.
    dataset_hasher = hashlib.sha256()
    dataset_hasher.update(DATASET_FINGERPRINT_DOMAIN)
    for entry in entries:
        portable = {
            "file_name": entry["file_name"],
            "file_sha256": entry["file_sha256"],
            "modulation": entry["modulation"],
            "size_bytes": entry["size_bytes"],
        }
        encoded = json.dumps(
            portable,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        dataset_hasher.update(len(encoded).to_bytes(8, "big"))
        dataset_hasher.update(encoded)
    return entries, f"sha256:{dataset_hasher.hexdigest()}"


def audit_iad_manifest(
    records: Iterable[IADManifestRecord],
) -> dict[str, Any]:
    """Detect duplicate content groups and exact train-test leakage."""

    rows = list(records)
    by_hash: dict[str, list[IADManifestRecord]] = defaultdict(list)
    for row in rows:
        by_hash[row.hash].append(row)

    duplicate_groups = [
        _group_summary(content_hash, group)
        for content_hash, group in sorted(by_hash.items())
        if len(group) > 1
    ]
    cross_file_groups = [group for group in duplicate_groups if group["cross_file"]]
    leakage_groups = [
        group for group in duplicate_groups if group["train_test_leakage"]
    ]

    per_modulation = Counter(row.modulation for row in rows)
    per_split_label = Counter((row.split, row.label) for row in rows)
    source_files, dataset_fingerprint = _source_provenance(rows)
    return {
        "manifest_schema_version": 1,
        "sample_hash_schema": {
            "version": SAMPLE_HASH_SCHEMA_VERSION,
            "algorithm": "sha256",
            "domain_utf8": "IAD_SAMPLE_V1\\0",
            "canonicalization": (
                "little-endian dtype + shape header + logical C-order sample bytes"
            ),
        },
        "manifest_jsonl_sha256": manifest_jsonl_sha256(rows),
        "dataset_fingerprint": dataset_fingerprint,
        "source_files": source_files,
        "record_count": len(rows),
        "unique_hash_count": len(by_hash),
        "duplicate_hash_group_count": len(duplicate_groups),
        "duplicate_record_excess_count": sum(
            group["record_count"] - 1 for group in duplicate_groups
        ),
        "cross_file_duplicate_group_count": len(cross_file_groups),
        "train_test_leakage_group_count": len(leakage_groups),
        "per_modulation_record_count": dict(sorted(per_modulation.items())),
        "per_split_label_record_count": {
            f"{split}/label_{label}": count
            for (split, label), count in sorted(per_split_label.items())
        },
        "duplicate_groups": duplicate_groups,
        "train_test_leakage_groups": leakage_groups,
    }


def write_manifest_jsonl(
    records: Iterable[IADManifestRecord],
    path: str | Path,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(_manifest_json_line(record))
            handle.write("\n")
    return output.resolve()


def write_manifest_csv(
    records: Iterable[IADManifestRecord],
    path: str | Path,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow(record.to_dict(csv_shape=True))
    return output.resolve()


def write_manifest_audit(audit: Mapping[str, Any], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(audit, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return output.resolve()


def load_manifest(path: str | Path) -> list[IADManifestRecord]:
    """Load and validate a JSONL or CSV manifest."""

    source = Path(path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"manifest does not exist: {source}")
    if not source.is_file():
        raise IADDataError(f"manifest path is not a file: {source}")

    rows: list[IADManifestRecord] = []
    if source.suffix.lower() == ".jsonl":
        with source.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise IADDataError("row is not a JSON object")
                    rows.append(IADManifestRecord.from_mapping(value))
                except (json.JSONDecodeError, IADDataError) as exc:
                    raise IADDataError(
                        f"{source}:{line_number}: invalid manifest row: {exc}"
                    ) from exc
    elif source.suffix.lower() == ".csv":
        with source.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise IADDataError(f"CSV manifest has no header: {source}")
            missing = set(MANIFEST_FIELDS).difference(reader.fieldnames)
            if missing:
                raise IADDataError(
                    f"CSV manifest is missing columns: {sorted(missing)}"
                )
            for line_number, row in enumerate(reader, start=2):
                try:
                    rows.append(IADManifestRecord.from_mapping(row))
                except IADDataError as exc:
                    raise IADDataError(
                        f"{source}:{line_number}: invalid manifest row: {exc}"
                    ) from exc
    else:
        raise IADDataError(f"manifest must end in .jsonl or .csv, got {source.name!r}")
    if not rows:
        raise IADDataError(f"manifest contains no records: {source}")
    return rows


def _seeded_hash_order(seed: int, modulation: str, content_hash: str) -> str:
    payload = f"{seed}\0{modulation}\0{content_hash}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def select_anomaly_pilot(
    records: Iterable[IADManifestRecord],
    *,
    per_modulation: int = 50,
    seed: int = 2026,
) -> list[IADManifestRecord]:
    """Select deterministic, globally hash-unique anomalous test samples.

    Selection treats each content hash as a group.  A deterministic bipartite
    matching step prevents a hash duplicated across modulations from appearing
    twice while still satisfying each modulation's quota when possible.
    """

    if per_modulation <= 0:
        raise IADDataError("per_modulation must be a positive integer")

    rows = list(records)
    modulations = sorted({row.modulation for row in rows})
    if not modulations:
        raise IADDataError("cannot select a pilot from an empty manifest")

    representatives: dict[str, dict[str, IADManifestRecord]] = {
        modulation: {} for modulation in modulations
    }
    for row in sorted(rows, key=_record_identity):
        if row.split != "test" or row.label != 1:
            continue
        representatives[row.modulation].setdefault(row.hash, row)

    candidates: dict[str, list[str]] = {}
    for modulation in modulations:
        hashes = list(representatives[modulation])
        hashes.sort(
            key=lambda content_hash: (
                _seeded_hash_order(seed, modulation, content_hash),
                content_hash,
            )
        )
        candidates[modulation] = hashes
        if len(hashes) < per_modulation:
            raise IADDataError(
                f"modulation {modulation!r} has only {len(hashes)} unique "
                f"test anomalies; cannot select {per_modulation}"
            )

    slots = [
        (modulation, slot_index)
        for modulation in sorted(
            modulations,
            key=lambda name: (len(candidates[name]), name),
        )
        for slot_index in range(per_modulation)
    ]
    hash_to_slot: dict[str, tuple[str, int]] = {}
    slot_to_hash: dict[tuple[str, int], str] = {}

    def assign(
        slot: tuple[str, int],
        seen_hashes: set[str],
        seen_slots: set[tuple[str, int]],
    ) -> bool:
        if slot in seen_slots:
            return False
        seen_slots.add(slot)
        modulation, _ = slot
        for content_hash in candidates[modulation]:
            if content_hash in seen_hashes:
                continue
            seen_hashes.add(content_hash)
            previous_slot = hash_to_slot.get(content_hash)
            if previous_slot is None or assign(
                previous_slot,
                seen_hashes,
                seen_slots,
            ):
                hash_to_slot[content_hash] = slot
                slot_to_hash[slot] = content_hash
                return True
        return False

    for slot in slots:
        if not assign(slot, set(), set()):
            counts = {
                modulation: len(candidates[modulation]) for modulation in modulations
            }
            raise IADDataError(
                "not enough globally unique anomaly hashes to satisfy pilot "
                f"quota {per_modulation} per modulation; candidates={counts}"
            )

    selected: list[IADManifestRecord] = []
    for modulation in modulations:
        selected_hashes = {
            slot_to_hash[(modulation, slot_index)]
            for slot_index in range(per_modulation)
        }
        for content_hash in candidates[modulation]:
            if content_hash in selected_hashes:
                selected.append(representatives[modulation][content_hash])

    if len({row.hash for row in selected}) != len(selected):
        raise AssertionError("pilot matching produced duplicate hashes")
    expected = len(modulations) * per_modulation
    if len(selected) != expected:
        raise AssertionError(
            f"pilot matching produced {len(selected)} records, expected {expected}"
        )
    return selected
