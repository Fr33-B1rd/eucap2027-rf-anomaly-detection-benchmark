"""Classical pre-label proposals for IAD spectrograms.

The implementation is deliberately independent of learned anomaly detectors:

1. fit a pixelwise log-domain median/MAD background model on normal training
   images only;
2. convert a test image to a robust positive-residual score map;
3. apply a two-dimensional robust local threshold (a median/MAD CFAR
   equivalent) with a strong-response override;
4. connect and filter binary components without an opening operation that
   would erase thin diagonal chirp ridges.

The resulting proposal is intended for review or refinement.  It is not
presented as ground truth.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class PrelabelConfig:
    """Fully serializable parameters for :class:`RobustCFARPrelabeler`."""

    log_epsilon: float = 1e-6
    mad_consistency_scale: float = 1.4826
    mad_floor: float = 1e-3
    score_mode: str = "positive"
    score_clip: float = 50.0
    gaussian_sigma: float = 0.6
    global_score_threshold: float = 6.0
    local_window_size: int = 11
    local_threshold_scale: float = 3.5
    local_mad_floor: float = 0.5
    strong_score_threshold: float = 10.0
    closing_iterations: int = 1
    fill_holes: bool = True
    min_component_area: int = 6
    min_ridge_length: float = 10.0
    min_ridge_linearity: float = 0.85
    connectivity: int = 8
    bbox_padding: int = 0

    def __post_init__(self) -> None:
        if self.log_epsilon <= 0:
            raise ValueError("log_epsilon must be positive")
        if self.mad_consistency_scale <= 0:
            raise ValueError("mad_consistency_scale must be positive")
        if self.mad_floor <= 0:
            raise ValueError("mad_floor must be positive")
        if self.score_mode not in {"positive", "absolute"}:
            raise ValueError("score_mode must be 'positive' or 'absolute'")
        if self.score_clip <= 0:
            raise ValueError("score_clip must be positive")
        if self.gaussian_sigma < 0:
            raise ValueError("gaussian_sigma cannot be negative")
        if self.global_score_threshold < 0:
            raise ValueError("global_score_threshold cannot be negative")
        if self.local_window_size < 3 or self.local_window_size % 2 == 0:
            raise ValueError("local_window_size must be an odd integer >= 3")
        if self.local_threshold_scale < 0:
            raise ValueError("local_threshold_scale cannot be negative")
        if self.local_mad_floor <= 0:
            raise ValueError("local_mad_floor must be positive")
        if self.strong_score_threshold < self.global_score_threshold:
            raise ValueError(
                "strong_score_threshold must be >= global_score_threshold"
            )
        if self.closing_iterations < 0:
            raise ValueError("closing_iterations cannot be negative")
        if self.min_component_area < 1:
            raise ValueError("min_component_area must be at least 1")
        if self.min_ridge_length <= 0:
            raise ValueError("min_ridge_length must be positive")
        if not 0.0 <= self.min_ridge_linearity <= 1.0:
            raise ValueError("min_ridge_linearity must be in [0, 1]")
        if self.connectivity not in {4, 8}:
            raise ValueError("connectivity must be 4 or 8")
        if self.bbox_padding < 0:
            raise ValueError("bbox_padding cannot be negative")

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible parameters with no implicit values."""

        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "PrelabelConfig":
        """Construct a config while rejecting unknown parameter names."""

        known = {field.name for field in fields(cls)}
        unknown = sorted(set(values).difference(known))
        if unknown:
            raise ValueError(f"Unknown prelabel parameters: {unknown}")
        return cls(**dict(values))


@dataclass
class PrelabelResult:
    """One pre-label proposal and its auditable diagnostics."""

    score_map: np.ndarray
    binary_proposal: np.ndarray
    bbox: dict[str, int] | None
    diagnostics: dict[str, Any]

    def save_npz(
        self,
        path: str | Path,
        *,
        extra_arrays: Mapping[str, np.ndarray] | None = None,
        extra_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Save numeric artifacts without object-dtype pickle payloads."""

        bbox_array = (
            np.empty((0,), dtype=np.int32)
            if self.bbox is None
            else np.asarray(
                [
                    self.bbox["x0"],
                    self.bbox["y0"],
                    self.bbox["x1"],
                    self.bbox["y1"],
                ],
                dtype=np.int32,
            )
        )
        metadata = dict(extra_metadata or {})
        metadata["bbox"] = self.bbox
        metadata["diagnostics"] = self.diagnostics
        payload: dict[str, np.ndarray] = {
            "score_map": np.asarray(self.score_map, dtype=np.float32),
            "binary_proposal": np.asarray(self.binary_proposal, dtype=np.uint8),
            # Keep the original descriptive name and provide the short name
            # used by the annotation UI.  They deliberately contain the same
            # bytes so consumers cannot silently disagree about proposals.
            "proposal_mask": np.asarray(self.binary_proposal, dtype=np.uint8),
            "bbox_xyxy": bbox_array,
            "metadata_json": np.asarray(
                json.dumps(metadata, sort_keys=True, allow_nan=False)
            ),
        }
        if extra_arrays:
            payload.update(
                {name: np.asarray(value) for name, value in extra_arrays.items()}
            )
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_path, **payload)


def _as_normal_stack(images: np.ndarray) -> np.ndarray:
    array = np.asarray(images)
    if array.ndim == 4:
        if array.shape[1] != 1:
            raise ValueError(
                "Expected normal images with shape [N, 1, H, W] or [N, H, W]"
            )
        array = array[:, 0]
    if array.ndim == 2:
        array = array[None, ...]
    if array.ndim != 3:
        raise ValueError(
            "Expected normal images with shape [N, 1, H, W] or [N, H, W]"
        )
    if array.shape[0] < 1 or array.shape[1] < 2 or array.shape[2] < 2:
        raise ValueError("Normal stack must contain at least one non-trivial image")
    output = np.asarray(array, dtype=np.float32)
    if not np.isfinite(output).all():
        raise ValueError("Normal training images contain NaN or infinite values")
    return output


def _as_single_image(image: np.ndarray, expected_shape: tuple[int, int]) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 3:
        if array.shape[0] != 1:
            raise ValueError("Expected one image with shape [1, H, W] or [H, W]")
        array = array[0]
    if array.ndim != 2:
        raise ValueError("Expected one image with shape [1, H, W] or [H, W]")
    if tuple(array.shape) != expected_shape:
        raise ValueError(
            f"Image shape {tuple(array.shape)} does not match fitted shape "
            f"{expected_shape}"
        )
    output = np.asarray(array, dtype=np.float32)
    if not np.isfinite(output).all():
        raise ValueError("Input image contains NaN or infinite values")
    return output


def _component_geometry(
    rows: np.ndarray,
    cols: np.ndarray,
) -> tuple[float, float]:
    """Return PCA major-axis extent and linearity for a connected component."""

    area = int(rows.size)
    if area <= 1:
        return 1.0, 1.0
    row_values = np.asarray(rows, dtype=np.float64)
    col_values = np.asarray(cols, dtype=np.float64)
    centered_rows = row_values - float(row_values.mean())
    centered_cols = col_values - float(col_values.mean())
    denominator = max(area - 1, 1)
    # Elementwise reductions avoid routing this two-variable calculation
    # through BLAS/LAPACK, which can conflict with another loaded MKL runtime.
    variance_row = float(
        np.sum(centered_rows * centered_rows) / denominator
    )
    covariance_row_col = float(
        np.sum(centered_rows * centered_cols) / denominator
    )
    variance_col = float(
        np.sum(centered_cols * centered_cols) / denominator
    )
    discriminant = math.hypot(
        variance_row - variance_col,
        2.0 * covariance_row_col,
    )
    major_eigenvalue = max(
        0.0,
        0.5 * (variance_row + variance_col + discriminant),
    )
    minor_eigenvalue = max(
        0.0,
        0.5 * (variance_row + variance_col - discriminant),
    )
    # Analytic principal direction for a 2x2 symmetric covariance matrix.
    # This avoids invoking a threaded LAPACK runtime for a tiny problem.
    if abs(covariance_row_col) > 1e-15:
        direction_row = covariance_row_col
        direction_col = major_eigenvalue - variance_row
        direction_norm = math.hypot(
            direction_row,
            direction_col,
        )
        direction_row /= direction_norm
        direction_col /= direction_norm
    elif variance_row >= variance_col:
        direction_row, direction_col = 1.0, 0.0
    else:
        direction_row, direction_col = 0.0, 1.0
    projection = (
        centered_rows * direction_row
        + centered_cols * direction_col
    )
    major_extent = float(np.ptp(projection) + 1.0)
    total_variance = major_eigenvalue + minor_eigenvalue
    linearity = (
        float(major_eigenvalue / total_variance)
        if total_variance > 1e-12
        else 1.0
    )
    return major_extent, linearity


class RobustCFARPrelabeler:
    """Normal-only robust background model plus classical local detection."""

    def __init__(self, config: PrelabelConfig | None = None) -> None:
        self.config = config or PrelabelConfig()
        self.background_median_log: np.ndarray | None = None
        self.background_scale: np.ndarray | None = None
        self.image_shape: tuple[int, int] | None = None
        self.fit_diagnostics: dict[str, Any] | None = None

    @property
    def is_fitted(self) -> bool:
        return self.background_median_log is not None

    def _log_transform(self, images: np.ndarray) -> np.ndarray:
        nonnegative = np.maximum(images, 0.0)
        return np.log(nonnegative + self.config.log_epsilon)

    def fit(self, normal_train: np.ndarray) -> "RobustCFARPrelabeler":
        """Fit exclusively from a stack of normal training spectrograms."""

        stack = _as_normal_stack(normal_train)
        transformed = self._log_transform(stack)
        median_log = np.median(transformed, axis=0)
        raw_mad = np.median(
            np.abs(transformed - median_log[None, ...]),
            axis=0,
        )
        robust_scale = self.config.mad_consistency_scale * raw_mad
        effective_scale = np.maximum(robust_scale, self.config.mad_floor)

        self.background_median_log = median_log.astype(np.float32, copy=False)
        self.background_scale = effective_scale.astype(np.float32, copy=False)
        self.image_shape = tuple(int(value) for value in stack.shape[-2:])
        self.fit_diagnostics = {
            "normal_train_count": int(stack.shape[0]),
            "image_shape": list(self.image_shape),
            "negative_input_fraction": float((stack < 0).mean()),
            "background_log_min": float(median_log.min()),
            "background_log_median": float(np.median(median_log)),
            "background_log_max": float(median_log.max()),
            "raw_mad_median": float(np.median(raw_mad)),
            "effective_scale_min": float(effective_scale.min()),
            "effective_scale_median": float(np.median(effective_scale)),
            "effective_scale_max": float(effective_scale.max()),
            "mad_floor_fraction": float(
                (robust_scale < self.config.mad_floor).mean()
            ),
        }
        return self

    def _require_fitted(self) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
        if (
            self.background_median_log is None
            or self.background_scale is None
            or self.image_shape is None
        ):
            raise RuntimeError("Call fit(normal_train) before generating proposals")
        return (
            self.background_median_log,
            self.background_scale,
            self.image_shape,
        )

    def score(self, image: np.ndarray) -> np.ndarray:
        """Compute a continuous robust residual score map."""

        median_log, scale, image_shape = self._require_fitted()
        image_2d = _as_single_image(image, image_shape)
        residual = self._log_transform(image_2d) - median_log
        if self.config.score_mode == "positive":
            residual = np.maximum(residual, 0.0)
        else:
            residual = np.abs(residual)
        score = residual / scale
        np.clip(score, 0.0, self.config.score_clip, out=score)
        if self.config.gaussian_sigma > 0:
            score = ndimage.gaussian_filter(
                score,
                sigma=self.config.gaussian_sigma,
                mode="reflect",
            )
        return np.asarray(score, dtype=np.float32)

    def _local_threshold(
        self, score_map: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        window = self.config.local_window_size
        local_median = ndimage.median_filter(
            score_map,
            size=(window, window),
            mode="reflect",
        )
        local_abs_deviation = ndimage.median_filter(
            np.abs(score_map - local_median),
            size=(window, window),
            mode="reflect",
        )
        local_scale = np.maximum(
            self.config.mad_consistency_scale * local_abs_deviation,
            self.config.local_mad_floor,
        )
        threshold = (
            local_median + self.config.local_threshold_scale * local_scale
        )
        return (
            threshold.astype(np.float32, copy=False),
            local_median.astype(np.float32, copy=False),
            local_scale.astype(np.float32, copy=False),
        )

    def prepare_threshold_cache(
        self,
        image: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Cache score and local threshold for repeated global-threshold trials."""

        score_map = self.score(image)
        local_threshold, _, _ = self._local_threshold(score_map)
        return score_map, local_threshold

    def proposal_from_threshold_cache(
        self,
        score_map: np.ndarray,
        local_threshold: np.ndarray,
        *,
        global_score_threshold: float | None = None,
        strong_score_threshold: float | None = None,
    ) -> np.ndarray:
        """Apply candidate/component rules to an already prepared score cache."""

        _, _, image_shape = self._require_fitted()
        score = np.asarray(score_map, dtype=np.float32)
        local = np.asarray(local_threshold, dtype=np.float32)
        if score.shape != image_shape or local.shape != image_shape:
            raise ValueError(
                "Prepared score and local threshold must match fitted image "
                f"shape {image_shape}"
            )
        if not np.isfinite(score).all() or not np.isfinite(local).all():
            raise ValueError("Prepared threshold cache contains non-finite values")
        global_threshold = (
            self.config.global_score_threshold
            if global_score_threshold is None
            else float(global_score_threshold)
        )
        strong_threshold = (
            self.config.strong_score_threshold
            if strong_score_threshold is None
            else float(strong_score_threshold)
        )
        if global_threshold < 0:
            raise ValueError("global_score_threshold cannot be negative")
        if strong_threshold < global_threshold:
            raise ValueError(
                "strong_score_threshold must be >= global_score_threshold"
            )
        globally_high = score >= global_threshold
        locally_high = score >= local
        strongly_high = score >= strong_threshold
        candidate = globally_high & (locally_high | strongly_high)
        proposal, _, _ = self._filter_components(candidate, score)
        return proposal.astype(np.uint8, copy=False)

    def _connectivity_structure(self) -> np.ndarray:
        if self.config.connectivity == 8:
            return np.ones((3, 3), dtype=bool)
        return ndimage.generate_binary_structure(2, 1)

    def _filter_components(
        self,
        candidate: np.ndarray,
        score_map: np.ndarray,
    ) -> tuple[np.ndarray, list[dict[str, Any]], int]:
        structure = self._connectivity_structure()
        processed = np.asarray(candidate, dtype=bool)
        if self.config.closing_iterations > 0 and processed.any():
            processed = ndimage.binary_closing(
                processed,
                structure=structure,
                iterations=self.config.closing_iterations,
            )
        if self.config.fill_holes and processed.any():
            processed = ndimage.binary_fill_holes(processed)

        labeled, component_count = ndimage.label(processed, structure=structure)
        retained = np.zeros_like(processed, dtype=bool)
        component_rows: list[dict[str, Any]] = []
        for component_id in range(1, component_count + 1):
            rows, cols = np.where(labeled == component_id)
            area = int(rows.size)
            if area == 0:
                continue
            major_extent, linearity = _component_geometry(rows, cols)
            is_ridge = (
                major_extent >= self.config.min_ridge_length
                and linearity >= self.config.min_ridge_linearity
            )
            keep = area >= self.config.min_component_area or is_ridge
            component_mask = labeled == component_id
            component_scores = score_map[component_mask]
            component_rows.append(
                {
                    "component_id": int(component_id),
                    "area": area,
                    "x0": int(cols.min()),
                    "y0": int(rows.min()),
                    "x1": int(cols.max()) + 1,
                    "y1": int(rows.max()) + 1,
                    "major_extent": major_extent,
                    "linearity": linearity,
                    "ridge_rule": bool(is_ridge),
                    "retained": bool(keep),
                    "peak_score": float(component_scores.max()),
                    "mean_score": float(component_scores.mean()),
                }
            )
            if keep:
                retained |= component_mask
        return retained, component_rows, int(component_count)

    def _tight_bbox(self, proposal: np.ndarray) -> dict[str, int] | None:
        rows, cols = np.where(proposal)
        if rows.size == 0:
            return None
        height, width = proposal.shape
        padding = self.config.bbox_padding
        return {
            "x0": max(0, int(cols.min()) - padding),
            "y0": max(0, int(rows.min()) - padding),
            "x1": min(width, int(cols.max()) + 1 + padding),
            "y1": min(height, int(rows.max()) + 1 + padding),
        }

    def prelabel(self, image: np.ndarray) -> PrelabelResult:
        """Generate one score map, binary proposal, bbox, and diagnostics."""

        score_map = self.score(image)
        local_threshold, local_median, local_scale = self._local_threshold(
            score_map
        )
        globally_high = score_map >= self.config.global_score_threshold
        locally_high = score_map >= local_threshold
        strongly_high = score_map >= self.config.strong_score_threshold
        candidate = globally_high & (locally_high | strongly_high)
        proposal, component_rows, raw_component_count = self._filter_components(
            candidate,
            score_map,
        )
        bbox = self._tight_bbox(proposal)
        retained_components = sum(
            bool(component["retained"]) for component in component_rows
        )
        diagnostics: dict[str, Any] = {
            "max_score": float(score_map.max()),
            "mean_score": float(score_map.mean()),
            "global_candidate_pixels": int(globally_high.sum()),
            "local_candidate_pixels": int((globally_high & locally_high).sum()),
            "strong_override_pixels": int(
                (globally_high & strongly_high & ~locally_high).sum()
            ),
            "candidate_pixels_before_morphology": int(candidate.sum()),
            "proposal_pixels": int(proposal.sum()),
            "proposal_fraction": float(proposal.mean()),
            "raw_component_count": raw_component_count,
            "retained_component_count": int(retained_components),
            "local_median_min": float(local_median.min()),
            "local_median_median": float(np.median(local_median)),
            "local_median_max": float(local_median.max()),
            "local_scale_min": float(local_scale.min()),
            "local_scale_median": float(np.median(local_scale)),
            "local_scale_max": float(local_scale.max()),
            "local_threshold_min": float(local_threshold.min()),
            "local_threshold_median": float(np.median(local_threshold)),
            "local_threshold_max": float(local_threshold.max()),
            "bbox": bbox,
            "components": component_rows,
            "config": self.config.to_dict(),
        }
        return PrelabelResult(
            score_map=score_map,
            binary_proposal=proposal.astype(np.uint8, copy=False),
            bbox=bbox,
            diagnostics=diagnostics,
        )

    def save_calibration(
        self,
        path: str | Path,
        *,
        extra_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Persist the fitted normal-only calibration and all parameters."""

        median_log, scale, image_shape = self._require_fitted()
        metadata = {
            "image_shape": list(image_shape),
            "config": self.config.to_dict(),
            "fit_diagnostics": self.fit_diagnostics,
        }
        if extra_metadata:
            metadata["provenance"] = dict(extra_metadata)
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_path,
            background_median_log=np.asarray(median_log, dtype=np.float32),
            background_scale=np.asarray(scale, dtype=np.float32),
            metadata_json=np.asarray(
                json.dumps(metadata, sort_keys=True, allow_nan=False)
            ),
        )

    @classmethod
    def load_calibration(cls, path: str | Path) -> "RobustCFARPrelabeler":
        """Restore a calibration saved by :meth:`save_calibration`."""

        with np.load(path, allow_pickle=False) as payload:
            metadata = json.loads(str(payload["metadata_json"].item()))
            model = cls(PrelabelConfig.from_dict(metadata["config"]))
            model.background_median_log = np.asarray(
                payload["background_median_log"], dtype=np.float32
            )
            model.background_scale = np.asarray(
                payload["background_scale"], dtype=np.float32
            )
        model.image_shape = tuple(int(value) for value in metadata["image_shape"])
        model.fit_diagnostics = metadata.get("fit_diagnostics")
        if model.background_median_log.shape != model.image_shape:
            raise ValueError("Calibration median shape does not match metadata")
        if model.background_scale.shape != model.image_shape:
            raise ValueError("Calibration scale shape does not match metadata")
        return model
