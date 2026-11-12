# PatchCore model + feature extractor + 2D-array dataset wrapper.
#
# This module is dataset-neutral: dataset-specific loaders live in
# `datasets/<name>/bundle.py` and feed numpy arrays into `ArrayDataset2D`.
#
# Generic helpers (log, set_seed, signed_log1p_np, compute_finite_mean_std)
# live in `core.utils`.

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.random_projection import SparseRandomProjection
from torch.utils.data import DataLoader, Dataset
from torchvision.models import (
    ResNet50_Weights,
    ResNet18_Weights,
    Wide_ResNet50_2_Weights,
    resnet50,
    resnet18,
    wide_resnet50_2,
)

from core.utils import log, signed_log1p_np


class ArrayDataset2D(Dataset):
    """Generic 2D-array dataset (N,H,W) → (3,input_size,input_size) tensors.

    Supports global or per-image z-score normalization, optional signed-log1p
    transform, edge masking, and square resize. Grayscale input is replicated
    to 3 channels for ImageNet-pretrained CNN backbones.
    """

    def __init__(
        self,
        data_parts: list[np.ndarray],
        labels: np.ndarray,
        mean: float,
        std: float,
        input_size: int,
        clip_z: float,
        normalization: str,
        mask_top_rows: int = 0,
        mask_bottom_rows: int = 0,
        mask_left_cols: int = 0,
        mask_right_cols: int = 0,
    ) -> None:
        self.data_parts = data_parts
        self.labels = labels
        self.mean = mean
        self.std = std
        self.input_size = input_size
        self.clip_z = clip_z
        self.normalization = normalization
        self.mask_top_rows = max(int(mask_top_rows), 0)
        self.mask_bottom_rows = max(int(mask_bottom_rows), 0)
        self.mask_left_cols = max(int(mask_left_cols), 0)
        self.mask_right_cols = max(int(mask_right_cols), 0)
        self.lengths = [int(part.shape[0]) for part in data_parts]
        self.cum_lengths = np.cumsum(self.lengths)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def _resolve_part(self, index: int):
        part_idx = int(np.searchsorted(self.cum_lengths, index, side="right"))
        prev = 0 if part_idx == 0 else int(self.cum_lengths[part_idx - 1])
        local_idx = index - prev
        return self.data_parts[part_idx], local_idx

    def _get_raw(self, index: int) -> np.ndarray:
        part, local_idx = self._resolve_part(index)
        return np.asarray(part[local_idx], dtype=np.float32)

    def get_processed_image(self, index: int) -> np.ndarray:
        x = self._prepare_tensor(self._get_raw(index))
        return x[0].cpu().numpy()

    def _prepare_tensor(self, array_2d: np.ndarray) -> torch.Tensor:
        array_2d = np.nan_to_num(
            array_2d,
            nan=self.mean,
            posinf=self.mean,
            neginf=self.mean,
        )
        if (
            self.mask_top_rows
            or self.mask_bottom_rows
            or self.mask_left_cols
            or self.mask_right_cols
        ):
            array_2d = np.array(array_2d, copy=True)
            if self.mask_top_rows:
                array_2d[: self.mask_top_rows, :] = self.mean
            if self.mask_bottom_rows:
                array_2d[-self.mask_bottom_rows :, :] = self.mean
            if self.mask_left_cols:
                array_2d[:, : self.mask_left_cols] = self.mean
            if self.mask_right_cols:
                array_2d[:, -self.mask_right_cols :] = self.mean
        if self.normalization in {"log_zscore", "log_per_image_zscore"}:
            array_2d = signed_log1p_np(array_2d).astype(np.float32, copy=False)
        x = torch.from_numpy(array_2d).float().unsqueeze(0)
        if self.normalization in {"zscore", "log_zscore"}:
            x = (x - self.mean) / self.std
        elif self.normalization in {"per_image_zscore", "log_per_image_zscore"}:
            img_mean = x.mean()
            img_std = torch.clamp(x.std(unbiased=False), min=1e-6)
            x = (x - img_mean) / img_std
        else:
            raise ValueError(f"Unsupported normalization mode: {self.normalization}")
        if self.clip_z > 0:
            x = torch.clamp(x, -self.clip_z, self.clip_z)
        if self.input_size > 0 and tuple(x.shape[-2:]) != (self.input_size, self.input_size):
            x = F.interpolate(
                x.unsqueeze(0),
                size=(self.input_size, self.input_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        x = x.repeat(3, 1, 1)
        return x

    def __getitem__(self, index: int):
        raw = self._get_raw(index)
        x = self._prepare_tensor(raw)
        label = int(self.labels[index])
        return x, label, index


class ResNetFeatureExtractor(nn.Module):
    def __init__(self, backbone: str, layers: list[str], pretrained: bool = True) -> None:
        super().__init__()
        self.layers = layers
        if backbone == "resnet18":
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            base = resnet18(weights=weights)
        elif backbone == "resnet50":
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            base = resnet50(weights=weights)
        elif backbone == "wide_resnet50_2":
            weights = Wide_ResNet50_2_Weights.DEFAULT if pretrained else None
            base = wide_resnet50_2(weights=weights)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = {}
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        if "layer1" in self.layers:
            outputs["layer1"] = x
        x = self.layer2(x)
        if "layer2" in self.layers:
            outputs["layer2"] = x
        x = self.layer3(x)
        if "layer3" in self.layers:
            outputs["layer3"] = x
        x = self.layer4(x)
        if "layer4" in self.layers:
            outputs["layer4"] = x
        return outputs


class PatchCore:
    def __init__(
        self,
        backbone: str,
        layers: list[str],
        device: torch.device,
        sample_patches_per_image: int,
        candidate_pool_size: int,
        coreset_ratio: float,
        max_memory_bank_size: int,
        projection_dim: int,
        query_chunk_size: int,
        seed: int,
        pretrained: bool = True,
    ) -> None:
        self.backbone = backbone
        self.layers = layers
        self.device = device
        self.sample_patches_per_image = sample_patches_per_image
        self.candidate_pool_size = candidate_pool_size
        self.coreset_ratio = coreset_ratio
        self.max_memory_bank_size = max_memory_bank_size
        self.projection_dim = projection_dim
        self.query_chunk_size = query_chunk_size
        self.seed = seed
        self.feature_extractor = ResNetFeatureExtractor(
            backbone=backbone, layers=layers, pretrained=pretrained
        ).to(device)
        self.feature_extractor.eval()
        self.patch_grid_size = None
        self.embedding_dim = None
        self.total_train_patches = 0
        self.memory_bank = None
        self.memory_bank_norms = None

    def _compute_patch_embeddings(self, images: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(images)
        feature_maps = [features[layer] for layer in self.layers]
        reference_size = feature_maps[0].shape[-2:]

        aligned = []
        for feature_map in feature_maps:
            if feature_map.shape[-2:] != reference_size:
                feature_map = F.interpolate(
                    feature_map,
                    size=reference_size,
                    mode="bilinear",
                    align_corners=False,
                )
            aligned.append(feature_map)

        embedding_map = torch.cat(aligned, dim=1)
        self.patch_grid_size = tuple(int(v) for v in embedding_map.shape[-2:])
        patch_embeddings = embedding_map.permute(0, 2, 3, 1).reshape(
            embedding_map.shape[0], -1, embedding_map.shape[1]
        )
        self.embedding_dim = int(patch_embeddings.shape[-1])
        return patch_embeddings

    def _sample_candidate_patches(self, patch_embeddings: torch.Tensor) -> torch.Tensor:
        batch_size, num_patches, embedding_dim = patch_embeddings.shape
        sample_count = min(self.sample_patches_per_image, num_patches)
        sampled = []
        for batch_index in range(batch_size):
            perm = torch.randperm(num_patches, device=patch_embeddings.device)[:sample_count]
            sampled.append(patch_embeddings[batch_index, perm])
        return torch.stack(sampled, dim=0).reshape(-1, embedding_dim)

    def _kcenter_greedy(self, candidate_embeddings: torch.Tensor, num_select: int) -> torch.Tensor:
        candidate_embeddings = candidate_embeddings.to(self.device, non_blocking=True)
        num_candidates = candidate_embeddings.shape[0]
        first_index = int(torch.randint(0, num_candidates, (1,), device=self.device).item())
        selected = torch.empty(num_select, dtype=torch.long, device=self.device)
        selected[0] = first_index
        center = candidate_embeddings[first_index]
        min_distances = ((candidate_embeddings - center) ** 2).sum(dim=1)

        for i in range(1, num_select):
            next_index = torch.argmax(min_distances)
            selected[i] = next_index
            center = candidate_embeddings[next_index]
            distances = ((candidate_embeddings - center) ** 2).sum(dim=1)
            min_distances = torch.minimum(min_distances, distances)
            if i % 250 == 0 or i == num_select - 1:
                log(f"k-center progress: {i + 1}/{num_select}")
        return selected.cpu()

    def fit(self, train_loader: DataLoader) -> None:
        sampled_candidates = []
        total_batches = len(train_loader)
        with torch.no_grad():
            for batch_idx, (images, _, _) in enumerate(train_loader, start=1):
                images = images.to(self.device, non_blocking=True)
                patch_embeddings = self._compute_patch_embeddings(images)
                self.total_train_patches += int(
                    patch_embeddings.shape[0] * patch_embeddings.shape[1]
                )
                sampled = self._sample_candidate_patches(patch_embeddings)
                sampled_candidates.append(sampled.cpu().to(torch.float16))
                if batch_idx == 1 or batch_idx % 20 == 0 or batch_idx == total_batches:
                    log(
                        f"Collected candidate patches from batch {batch_idx}/{total_batches}"
                    )

        candidates = torch.cat(sampled_candidates, dim=0).float()
        if candidates.shape[0] > self.candidate_pool_size:
            perm = torch.randperm(candidates.shape[0])[: self.candidate_pool_size]
            candidates = candidates[perm]

        target_size = max(1, int(math.ceil(candidates.shape[0] * self.coreset_ratio)))
        target_size = min(target_size, self.max_memory_bank_size, candidates.shape[0])

        if self.projection_dim > 0 and candidates.shape[1] > self.projection_dim:
            projector = SparseRandomProjection(
                n_components=self.projection_dim, random_state=self.seed
            )
            projected = projector.fit_transform(candidates.numpy())
            projected = torch.from_numpy(np.asarray(projected, dtype=np.float32))
        else:
            projected = candidates

        log(
            f"Running k-center on {candidates.shape[0]} sampled candidates to select {target_size}"
        )
        selected_indices = self._kcenter_greedy(projected, target_size)
        self.memory_bank = candidates[selected_indices].to(self.device, non_blocking=True)
        self.memory_bank_norms = (self.memory_bank**2).sum(dim=1)
        log(
            f"Memory bank ready: {self.memory_bank.shape[0]} vectors from "
            f"{self.total_train_patches} estimated train patches"
        )

    def _nearest_neighbor_distances(self, queries: torch.Tensor) -> torch.Tensor:
        results = []
        memory_bank_t = self.memory_bank.t().contiguous()
        for start in range(0, queries.shape[0], self.query_chunk_size):
            end = min(start + self.query_chunk_size, queries.shape[0])
            query_chunk = queries[start:end].to(self.device, non_blocking=True)
            query_norms = (query_chunk**2).sum(dim=1, keepdim=True)
            distances = (
                query_norms
                + self.memory_bank_norms.unsqueeze(0)
                - 2.0 * query_chunk @ memory_bank_t
            )
            distances = torch.clamp(distances, min=0.0)
            results.append(torch.sqrt(distances.min(dim=1).values).cpu())
        return torch.cat(results, dim=0)

    def predict(self, test_loader: DataLoader):
        all_scores = []
        all_labels = []
        all_indices = []
        all_maps = []
        total_batches = len(test_loader)
        with torch.no_grad():
            for batch_idx, (images, labels, indices) in enumerate(test_loader, start=1):
                raw_h, raw_w = int(images.shape[-2]), int(images.shape[-1])
                images = images.to(self.device, non_blocking=True)
                patch_embeddings = self._compute_patch_embeddings(images)
                flat_embeddings = patch_embeddings.reshape(-1, self.embedding_dim)
                patch_scores = self._nearest_neighbor_distances(flat_embeddings)
                patch_scores = patch_scores.reshape(images.shape[0], *self.patch_grid_size)
                image_scores = patch_scores.amax(dim=(1, 2))
                anomaly_maps = F.interpolate(
                    patch_scores.unsqueeze(1),
                    size=(raw_h, raw_w),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)

                all_scores.append(image_scores.cpu())
                all_labels.append(labels.cpu())
                all_indices.append(indices.cpu())
                all_maps.append(anomaly_maps.cpu())
                if batch_idx == 1 or batch_idx % 20 == 0 or batch_idx == total_batches:
                    log(f"Scored batch {batch_idx}/{total_batches}")

        return (
            torch.cat(all_scores).numpy(),
            torch.cat(all_labels).numpy(),
            torch.cat(all_indices).numpy(),
            torch.cat(all_maps).numpy(),
        )

    def export_test_patch_artifacts(
        self,
        test_loader: DataLoader,
        output_path: str,
        selection: str = "label1",
        embedding_dtype: str = "float16",
    ) -> dict:
        if self.memory_bank is None:
            raise RuntimeError("Call fit() before export_test_patch_artifacts()")

        if selection not in {"all", "label1"}:
            raise ValueError(f"Unsupported export selection: {selection}")
        if embedding_dtype not in {"float16", "float32"}:
            raise ValueError(f"Unsupported embedding dtype: {embedding_dtype}")

        selected_indices = []
        selected_labels = []
        selected_scores = []
        selected_patch_scores = []
        selected_patch_embeddings = []

        total_batches = len(test_loader)
        with torch.no_grad():
            for batch_idx, (images, labels, indices) in enumerate(test_loader, start=1):
                images = images.to(self.device, non_blocking=True)
                labels_np = labels.cpu().numpy()
                indices_np = indices.cpu().numpy()

                patch_embeddings = self._compute_patch_embeddings(images)
                batch_size, _, embedding_dim = patch_embeddings.shape
                flat_embeddings = patch_embeddings.reshape(-1, embedding_dim)
                patch_scores = self._nearest_neighbor_distances(flat_embeddings)
                patch_scores = patch_scores.reshape(batch_size, *self.patch_grid_size)
                image_scores = patch_scores.amax(dim=(1, 2)).cpu().numpy()

                if selection == "all":
                    keep_mask = np.ones(batch_size, dtype=bool)
                else:
                    keep_mask = labels_np == 1

                keep_indices = np.where(keep_mask)[0]
                if keep_indices.size > 0:
                    embeddings_np = patch_embeddings[keep_indices].reshape(
                        keep_indices.size,
                        self.patch_grid_size[0],
                        self.patch_grid_size[1],
                        embedding_dim,
                    ).cpu().numpy()
                    patch_scores_np = patch_scores[keep_indices].cpu().numpy()
                    if embedding_dtype == "float16":
                        embeddings_np = embeddings_np.astype(np.float16, copy=False)
                        patch_scores_np = patch_scores_np.astype(np.float16, copy=False)
                    else:
                        embeddings_np = embeddings_np.astype(np.float32, copy=False)
                        patch_scores_np = patch_scores_np.astype(np.float32, copy=False)

                    selected_patch_embeddings.append(embeddings_np)
                    selected_patch_scores.append(patch_scores_np)
                    selected_labels.append(labels_np[keep_indices].astype(np.int64, copy=False))
                    selected_indices.append(indices_np[keep_indices].astype(np.int64, copy=False))
                    selected_scores.append(image_scores[keep_indices].astype(np.float32, copy=False))

                if batch_idx == 1 or batch_idx % 20 == 0 or batch_idx == total_batches:
                    log(f"Prepared patch artifact export for batch {batch_idx}/{total_batches}")

        if selected_patch_embeddings:
            patch_embeddings_out = np.concatenate(selected_patch_embeddings, axis=0)
            patch_scores_out = np.concatenate(selected_patch_scores, axis=0)
            labels_out = np.concatenate(selected_labels, axis=0)
            indices_out = np.concatenate(selected_indices, axis=0)
            image_scores_out = np.concatenate(selected_scores, axis=0)
        else:
            patch_embeddings_out = np.empty(
                (0, self.patch_grid_size[0], self.patch_grid_size[1], self.embedding_dim),
                dtype=np.float16 if embedding_dtype == "float16" else np.float32,
            )
            patch_scores_out = np.empty(
                (0, self.patch_grid_size[0], self.patch_grid_size[1]),
                dtype=np.float16 if embedding_dtype == "float16" else np.float32,
            )
            labels_out = np.empty((0,), dtype=np.int64)
            indices_out = np.empty((0,), dtype=np.int64)
            image_scores_out = np.empty((0,), dtype=np.float32)

        np.savez_compressed(
            output_path,
            sample_indices=indices_out,
            labels=labels_out,
            image_scores=image_scores_out,
            patch_scores=patch_scores_out,
            patch_embeddings=patch_embeddings_out,
            patch_grid_size=np.asarray(self.patch_grid_size, dtype=np.int64),
            embedding_dim=np.asarray([self.embedding_dim], dtype=np.int64),
        )

        return {
            "num_selected_samples": int(indices_out.shape[0]),
            "patch_grid_size": self.patch_grid_size,
            "embedding_dim": int(self.embedding_dim),
            "selection": selection,
            "output_path": output_path,
        }
