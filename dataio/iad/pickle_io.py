"""Safe, read-only loading for the public IAD ``*_Train_Test.pkl`` files.

The upstream dataset is distributed as a Python pickle containing PyTorch
tensors and NumPy label arrays.  A normal ``pickle.load`` would permit an
untrusted file to import and execute arbitrary callables.  This module uses a
small allow-list and routes embedded PyTorch storage payloads through
``torch.load(..., weights_only=True)``.

Pickle is not a sandbox.  The allow-list materially narrows the attack surface,
but callers should still only use dataset files obtained from a trusted source.
"""

from __future__ import annotations

import collections
import importlib
import io
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
import torch


IAD_FILE_RE = re.compile(r"^(?P<modulation>.+)_Train_Test\.pkl$")
REQUIRED_KEYS = frozenset({"train_data", "train_label", "test_data", "test_label"})


class IADDataError(ValueError):
    """Base class for malformed or unsupported IAD input."""


class UnsafeIADPickleError(IADDataError):
    """Raised when a pickle requests a global outside the safe allow-list."""


@dataclass(frozen=True)
class IADSource:
    """A discovered, canonical IAD source file."""

    modulation: str
    path: Path


@dataclass
class IADBundle:
    """Validated tensors and labels from one IAD modulation file."""

    modulation: str
    source_path: Path
    train_data: torch.Tensor
    train_labels: np.ndarray
    test_data: torch.Tensor
    test_labels: np.ndarray


def _numpy_frombuffer(
    buffer: bytes | bytearray | memoryview,
    dtype: np.dtype[Any],
    shape: tuple[int, ...],
    order: str,
    axis_order: tuple[int, ...] | None = None,
) -> np.ndarray:
    """Compatibility replacement for NumPy 2.x's private ``_frombuffer``.

    IAD files rewritten under NumPy 2.x refer to ``numpy._core``.  The
    benchmark environment pins NumPy 1.26, so resolving the private symbol
    by module name is not portable.  This equivalent uses public NumPy APIs.
    """

    array = np.frombuffer(buffer, dtype=dtype).reshape(shape, order=order)
    if axis_order is not None:
        array = array.transpose(axis_order)
    return array


def _numpy_reconstruct() -> Any:
    """Return NumPy's legacy ndarray reconstruction helper."""

    for module_name in ("numpy._core.multiarray", "numpy.core.multiarray"):
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError:
            continue
        return getattr(module, "_reconstruct")
    raise RuntimeError("NumPy ndarray reconstruction helper is unavailable")


def _safe_load_torch_storage(payload: bytes) -> Any:
    """Load an embedded tensor storage without enabling arbitrary pickle code."""

    if not isinstance(payload, bytes):
        raise pickle.UnpicklingError("torch storage payload must be immutable bytes")
    try:
        return torch.load(
            io.BytesIO(payload),
            map_location="cpu",
            weights_only=True,
        )
    except Exception as exc:  # torch raises several version-specific types
        raise pickle.UnpicklingError(f"safe torch storage load failed: {exc}") from exc


def _allowed_globals() -> dict[tuple[str, str], Any]:
    reconstruct = _numpy_reconstruct()
    return {
        ("collections", "OrderedDict"): collections.OrderedDict,
        ("torch._utils", "_rebuild_tensor_v2"): torch._utils._rebuild_tensor_v2,
        ("torch.storage", "_load_from_bytes"): _safe_load_torch_storage,
        ("numpy", "dtype"): np.dtype,
        ("numpy", "ndarray"): np.ndarray,
        ("numpy._core.numeric", "_frombuffer"): _numpy_frombuffer,
        ("numpy.core.numeric", "_frombuffer"): _numpy_frombuffer,
        ("numpy._core.multiarray", "_reconstruct"): reconstruct,
        ("numpy.core.multiarray", "_reconstruct"): reconstruct,
    }


class _RestrictedIADUnpickler(pickle.Unpickler):
    """Unpickler that permits only the constructors needed by IAD."""

    def find_class(self, module: str, name: str) -> Any:
        allowed = _allowed_globals()
        key = (module, name)
        if key not in allowed:
            raise UnsafeIADPickleError(f"unsafe pickle global denied: {module}.{name}")
        return allowed[key]


def _restricted_load(handle: BinaryIO) -> Any:
    return _RestrictedIADUnpickler(handle).load()


def canonical_modulation_from_path(path: str | Path) -> str:
    """Extract a canonical modulation from an exact IAD file name."""

    name = Path(path).name
    match = IAD_FILE_RE.fullmatch(name)
    if match is None:
        raise IADDataError(
            f"expected file name matching '*_Train_Test.pkl', got {name!r}"
        )
    modulation = match.group("modulation").strip().upper()
    if not modulation:
        raise IADDataError(f"empty modulation in IAD file name: {name!r}")
    return modulation


def discover_iad_sources(dataset_dir: str | Path) -> list[IADSource]:
    """Discover only exact ``*_Train_Test.pkl`` files in ``dataset_dir``.

    The lookup is deliberately non-recursive.  Files ending in ``.pkl.bak``
    cannot match the regular expression and are explicitly ignored.
    """

    root = Path(dataset_dir).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"IAD dataset directory does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"IAD dataset path is not a directory: {root}")

    sources: list[IADSource] = []
    modulation_paths: dict[str, Path] = {}
    for path in root.iterdir():
        if not path.is_file():
            continue
        if path.name.endswith(".bak"):
            continue
        match = IAD_FILE_RE.fullmatch(path.name)
        if match is None:
            continue
        resolved = path.resolve()
        modulation = canonical_modulation_from_path(resolved)
        if modulation in modulation_paths:
            raise IADDataError(
                "multiple IAD files resolve to modulation "
                f"{modulation!r}: {modulation_paths[modulation]} and {resolved}"
            )
        modulation_paths[modulation] = resolved
        sources.append(IADSource(modulation=modulation, path=resolved))

    sources.sort(key=lambda item: (item.modulation, str(item.path).casefold()))
    if not sources:
        raise FileNotFoundError(
            f"no '*_Train_Test.pkl' files found directly in {root}; "
            "'.bak' files are intentionally excluded"
        )
    return sources


def _as_binary_labels(value: Any, *, name: str, path: Path) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu":
            value = value.detach().cpu()
        array = value.detach().numpy()
    else:
        array = np.asarray(value)

    if array.ndim != 1:
        raise IADDataError(
            f"{path.name}: {name} must be one-dimensional, got {array.shape}"
        )
    if not np.issubdtype(array.dtype, np.number):
        raise IADDataError(f"{path.name}: {name} must be numeric, got {array.dtype}")
    if not np.isfinite(array).all():
        raise IADDataError(f"{path.name}: {name} contains NaN or infinity")
    if not np.equal(array, np.rint(array)).all():
        raise IADDataError(f"{path.name}: {name} contains non-integral label values")

    labels = array.astype(np.int64, copy=False)
    unique = set(np.unique(labels).tolist())
    if not unique.issubset({0, 1}):
        raise IADDataError(
            f"{path.name}: {name} must be binary 0/1, got {sorted(unique)}"
        )
    return labels


def _validate_data_tensor(
    value: Any,
    *,
    name: str,
    path: Path,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise IADDataError(
            f"{path.name}: {name} must be a torch.Tensor, got {type(value).__name__}"
        )
    tensor = value.detach()
    if tensor.device.type != "cpu":
        tensor = tensor.cpu()
    if tensor.ndim != 4:
        raise IADDataError(
            f"{path.name}: {name} must have shape [N,C,H,W], got {tuple(tensor.shape)}"
        )
    if tensor.shape[0] <= 0 or any(size <= 0 for size in tensor.shape[1:]):
        raise IADDataError(
            f"{path.name}: {name} has an empty dimension: {tuple(tensor.shape)}"
        )
    if tensor.dtype == torch.bool or not (
        tensor.dtype.is_floating_point or tensor.dtype.is_complex
    ):
        raise IADDataError(
            f"{path.name}: {name} must contain floating-point spectrograms, "
            f"got {tensor.dtype}"
        )
    if not bool(torch.isfinite(tensor).all().item()):
        raise IADDataError(f"{path.name}: {name} contains NaN or infinity")
    return tensor


def safe_load_iad_pickle(
    path: str | Path,
    *,
    modulation: str | None = None,
) -> IADBundle:
    """Safely deserialize and validate one IAD pickle without modifying it."""

    source_path = Path(path).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"IAD pickle does not exist: {source_path}")
    if not source_path.is_file():
        raise IADDataError(f"IAD pickle path is not a file: {source_path}")
    canonical = canonical_modulation_from_path(source_path)
    if modulation is not None and canonical != modulation.strip().upper():
        raise IADDataError(
            f"modulation mismatch for {source_path.name}: "
            f"name implies {canonical!r}, caller supplied {modulation!r}"
        )

    try:
        with source_path.open("rb") as handle:
            obj = _restricted_load(handle)
    except UnsafeIADPickleError:
        raise
    except (OSError, EOFError, pickle.UnpicklingError, RuntimeError) as exc:
        raise IADDataError(
            f"failed to safely load IAD pickle {source_path}: {exc}"
        ) from exc

    if not isinstance(obj, dict):
        raise IADDataError(
            f"{source_path.name}: top-level pickle object must be a dict"
        )
    missing = REQUIRED_KEYS.difference(obj)
    if missing:
        raise IADDataError(
            f"{source_path.name}: missing required keys {sorted(missing)}"
        )

    train_data = _validate_data_tensor(
        obj["train_data"], name="train_data", path=source_path
    )
    test_data = _validate_data_tensor(
        obj["test_data"], name="test_data", path=source_path
    )
    train_labels = _as_binary_labels(
        obj["train_label"], name="train_label", path=source_path
    )
    test_labels = _as_binary_labels(
        obj["test_label"], name="test_label", path=source_path
    )

    if train_data.shape[0] != train_labels.shape[0]:
        raise IADDataError(
            f"{source_path.name}: train_data has {train_data.shape[0]} samples "
            f"but train_label has {train_labels.shape[0]}"
        )
    if test_data.shape[0] != test_labels.shape[0]:
        raise IADDataError(
            f"{source_path.name}: test_data has {test_data.shape[0]} samples "
            f"but test_label has {test_labels.shape[0]}"
        )
    if tuple(train_data.shape[1:]) != tuple(test_data.shape[1:]):
        raise IADDataError(
            f"{source_path.name}: train/test sample shapes differ: "
            f"{tuple(train_data.shape[1:])} vs {tuple(test_data.shape[1:])}"
        )

    train_unique = set(np.unique(train_labels).tolist())
    test_unique = set(np.unique(test_labels).tolist())
    if train_unique != {0}:
        raise IADDataError(
            f"{source_path.name}: expected normal-only train labels {{0}}, "
            f"got {sorted(train_unique)}. The original inverse-label pickle "
            "may have been supplied instead of the current converted file."
        )
    if test_unique != {0, 1}:
        raise IADDataError(
            f"{source_path.name}: expected both test labels 0 and 1, "
            f"got {sorted(test_unique)}"
        )

    return IADBundle(
        modulation=canonical,
        source_path=source_path,
        train_data=train_data,
        train_labels=train_labels,
        test_data=test_data,
        test_labels=test_labels,
    )
