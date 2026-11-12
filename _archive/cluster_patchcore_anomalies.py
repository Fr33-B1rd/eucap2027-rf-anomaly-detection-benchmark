# Clusters IAD PatchCore anomaly samples using anomaly-region descriptors and
# exported patch-level semantic features.

import argparse
import json
import math
import os
import pickle
import sys
import time
from dataclasses import dataclass

import hdbscan
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import umap
from PIL import Image
from scipy import ndimage
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def install_numpy_pickle_compat() -> None:
    compat_modules = {
        "numpy._core": np.core,
        "numpy._core.numeric": np.core.numeric,
        "numpy._core.multiarray": np.core.multiarray,
        "numpy._core.umath": np.core.umath,
        "numpy._core._multiarray_umath": np.core._multiarray_umath,
    }
    for module_name, module_obj in compat_modules.items():
        sys.modules.setdefault(module_name, module_obj)


def normalize_map(anomaly_map: np.ndarray) -> np.ndarray:
    anomaly_map = anomaly_map.astype(np.float32, copy=False)
    min_value = float(anomaly_map.min())
    max_value = float(anomaly_map.max())
    if max_value <= min_value:
        return np.zeros_like(anomaly_map, dtype=np.float32)
    return (anomaly_map - min_value) / (max_value - min_value)


def resize_array(array: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    pil = Image.fromarray(array.astype(np.float32))
    resized = pil.resize((size[1], size[0]), resample=Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32)


@dataclass
class SampleRecord:
    dataset_name: str
    sample_index: int
    true_label: int
    score: float
    prediction: int
    raw_image: np.ndarray
    anomaly_map: np.ndarray
    patch_embeddings: np.ndarray
    patch_scores: np.ndarray


def load_dataset_pickle(path: str) -> dict:
    install_numpy_pickle_compat()
    with open(path, "rb") as f:
        return pickle.load(f)


def load_records(
    dataset_name: str,
    dataset_pkl: str,
    output_dir: str,
) -> list[SampleRecord]:
    data_obj = load_dataset_pickle(dataset_pkl)
    raw_test = np.asarray(data_obj["test_data"], dtype=np.float32)

    score_csv = os.path.join(output_dir, "test_scores.csv")
    score_df = pd.read_csv(score_csv)
    anomaly_npz = np.load(os.path.join(output_dir, "anomaly_outputs.npz"))
    patch_npz_path = os.path.join(output_dir, "test_patch_artifacts.npz")
    if not os.path.exists(patch_npz_path):
        raise FileNotFoundError(
            f"{dataset_name}: missing patch artifact export at {patch_npz_path}"
        )
    patch_npz = np.load(patch_npz_path)

    labels = anomaly_npz["labels"].astype(np.int64)
    scores = anomaly_npz["scores"].astype(np.float32)
    anomaly_maps = anomaly_npz["anomaly_maps"].astype(np.float32)
    patch_indices = patch_npz["sample_indices"].astype(np.int64)
    patch_labels = patch_npz["labels"].astype(np.int64)
    patch_embeddings = patch_npz["patch_embeddings"].astype(np.float32)
    patch_scores = patch_npz["patch_scores"].astype(np.float32)

    patch_lookup = {}
    for i, sample_index in enumerate(patch_indices):
        patch_lookup[int(sample_index)] = {
            "label": int(patch_labels[i]),
            "patch_embeddings": patch_embeddings[i],
            "patch_scores": patch_scores[i],
        }

    if raw_test.shape[0] != labels.shape[0]:
        raise ValueError(f"{dataset_name}: test data and output sizes do not match")

    records = []
    for i in range(raw_test.shape[0]):
        patch_entry = patch_lookup.get(i)
        records.append(
            SampleRecord(
                dataset_name=dataset_name,
                sample_index=i,
                true_label=int(labels[i]),
                score=float(scores[i]),
                prediction=int(score_df.iloc[i]["prediction"]),
                raw_image=raw_test[i, 0],
                anomaly_map=anomaly_maps[i],
                patch_embeddings=None if patch_entry is None else patch_entry["patch_embeddings"],
                patch_scores=None if patch_entry is None else patch_entry["patch_scores"],
            )
        )
    return records


def largest_connected_region(
    normalized_map: np.ndarray, threshold_percentile: float
) -> tuple[np.ndarray, dict]:
    threshold_value = float(np.percentile(normalized_map, threshold_percentile))
    binary_mask = normalized_map >= threshold_value

    if not np.any(binary_mask):
        flat_index = int(np.argmax(normalized_map))
        binary_mask = np.zeros_like(normalized_map, dtype=bool)
        binary_mask.flat[flat_index] = True

    labeled, num_components = ndimage.label(binary_mask)
    if num_components == 0:
        flat_index = int(np.argmax(normalized_map))
        binary_mask = np.zeros_like(normalized_map, dtype=bool)
        binary_mask.flat[flat_index] = True
        labeled, num_components = ndimage.label(binary_mask)

    component_sizes = ndimage.sum(binary_mask, labeled, index=np.arange(1, num_components + 1))
    largest_component = int(np.argmax(component_sizes)) + 1
    mask = labeled == largest_component

    y_coords, x_coords = np.where(mask)
    y_min, y_max = int(y_coords.min()), int(y_coords.max())
    x_min, x_max = int(x_coords.min()), int(x_coords.max())

    props = {
        "threshold_value": threshold_value,
        "num_components": int(num_components),
        "y_min": y_min,
        "y_max": y_max,
        "x_min": x_min,
        "x_max": x_max,
    }
    return mask, props


def extract_feature_vector(
    raw_image: np.ndarray,
    anomaly_map: np.ndarray,
    patch_embeddings: np.ndarray,
    patch_scores: np.ndarray,
    threshold_percentile: float,
    patch_size: int,
) -> tuple[np.ndarray, dict, np.ndarray]:
    normalized_map = normalize_map(anomaly_map)
    mask, props = largest_connected_region(
        normalized_map, threshold_percentile=threshold_percentile
    )

    y_min = props["y_min"]
    y_max = props["y_max"]
    x_min = props["x_min"]
    x_max = props["x_max"]

    map_crop = normalized_map[y_min : y_max + 1, x_min : x_max + 1]
    raw_crop = raw_image[y_min : y_max + 1, x_min : x_max + 1]
    mask_crop = mask[y_min : y_max + 1, x_min : x_max + 1]

    masked_map_values = normalized_map[mask]
    masked_raw_values = raw_image[mask]

    image_height, image_width = raw_image.shape
    bbox_height = y_max - y_min + 1
    bbox_width = x_max - x_min + 1
    area = int(mask.sum())
    area_ratio = area / float(image_height * image_width)

    center_y, center_x = ndimage.center_of_mass(mask.astype(np.float32))
    center_y = float(center_y / image_height)
    center_x = float(center_x / image_width)

    map_resized = resize_array(map_crop, (patch_size, patch_size))
    raw_resized = resize_array(raw_crop, (patch_size, patch_size))
    masked_crop = raw_crop * mask_crop.astype(np.float32)
    masked_resized = resize_array(masked_crop, (patch_size, patch_size))

    row_profile = mask.astype(np.float32).mean(axis=1)
    col_profile = mask.astype(np.float32).mean(axis=0)
    row_profile = resize_array(row_profile[np.newaxis, :], (1, patch_size)).reshape(-1)
    col_profile = resize_array(col_profile[np.newaxis, :], (1, patch_size)).reshape(-1)

    stats = np.array(
        [
            center_y,
            center_x,
            bbox_height / image_height,
            bbox_width / image_width,
            area_ratio,
            bbox_width / max(bbox_height, 1),
            float(masked_map_values.mean()),
            float(masked_map_values.std()),
            float(masked_map_values.max()),
            float(normalized_map.mean()),
            float(raw_crop.mean()),
            float(raw_crop.std()),
            float(masked_raw_values.mean()),
            float(masked_raw_values.std()),
            float(masked_raw_values.max()),
            float(masked_raw_values.min()),
            float(np.square(masked_raw_values).mean()),
            float(props["num_components"]),
        ],
        dtype=np.float32,
    )

    patch_grid_shape = tuple(int(v) for v in patch_scores.shape)
    patch_mask = resize_array(mask.astype(np.float32), patch_grid_shape)
    patch_mask = patch_mask >= 0.25
    if not np.any(patch_mask):
        flat_index = int(np.argmax(patch_scores))
        patch_mask = np.zeros_like(patch_scores, dtype=bool)
        patch_mask.flat[flat_index] = True

    patch_score_region = patch_scores[patch_mask].astype(np.float32)
    patch_embedding_region = patch_embeddings[patch_mask].astype(np.float32)

    patch_score_region = patch_score_region - patch_score_region.min()
    if float(patch_score_region.max()) > 0:
        patch_score_region = patch_score_region / patch_score_region.max()
    weights = patch_score_region + 1e-6
    semantic_embedding = np.average(patch_embedding_region, axis=0, weights=weights)
    semantic_norm = float(np.linalg.norm(semantic_embedding))
    if semantic_norm > 0:
        semantic_embedding = semantic_embedding / semantic_norm

    feature_vector = np.concatenate(
        [
            stats,
            map_resized.reshape(-1),
            raw_resized.reshape(-1),
            masked_resized.reshape(-1),
            row_profile,
            col_profile,
            semantic_embedding.astype(np.float32, copy=False),
        ]
    ).astype(np.float32)

    metadata = {
        "center_y": center_y,
        "center_x": center_x,
        "bbox_height": int(bbox_height),
        "bbox_width": int(bbox_width),
        "area": int(area),
        "area_ratio": area_ratio,
        "threshold_value": float(props["threshold_value"]),
        "num_components": int(props["num_components"]),
        "y_min": int(y_min),
        "y_max": int(y_max),
        "x_min": int(x_min),
        "x_max": int(x_max),
        "semantic_patch_count": int(patch_mask.sum()),
    }
    return feature_vector, metadata, mask


def save_projection_plot(
    projection: np.ndarray,
    cluster_labels: np.ndarray,
    dataset_names: list[str],
    output_path: str,
) -> None:
    unique_datasets = sorted(set(dataset_names))
    dataset_markers = {
        name: marker
        for name, marker in zip(unique_datasets, ["o", "s", "^", "D", "P", "X", "v", "*"])
    }

    non_noise_labels = sorted(label for label in np.unique(cluster_labels) if label != -1)
    palette = sns.color_palette("tab20", n_colors=max(len(non_noise_labels), 1))
    color_map = {-1: (0.7, 0.7, 0.7)}
    for index, label in enumerate(non_noise_labels):
        color_map[label] = palette[index % len(palette)]

    plt.figure(figsize=(10, 8))
    for dataset_name in unique_datasets:
        dataset_mask = np.array(dataset_names) == dataset_name
        for label in [-1] + non_noise_labels:
            mask = dataset_mask & (cluster_labels == label)
            if not np.any(mask):
                continue
            legend_label = f"{dataset_name} / noise" if label == -1 else f"{dataset_name} / c{label}"
            plt.scatter(
                projection[mask, 0],
                projection[mask, 1],
                s=28,
                alpha=0.85 if label != -1 else 0.45,
                c=[color_map[label]],
                marker=dataset_markers[dataset_name],
                label=legend_label,
                edgecolors="none",
            )

    plt.title("HDBSCAN Clustering Projection (UMAP)")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.legend(fontsize=8, ncol=2, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()


def save_cluster_montage(
    cluster_id: int,
    rows: list[dict],
    output_path: str,
    num_images: int = 10,
) -> None:
    selected = rows[: min(num_images, len(rows))]
    cols = 5
    grid_rows = int(math.ceil(len(selected) / cols))
    fig, axes = plt.subplots(grid_rows, cols, figsize=(4 * cols, 3.5 * grid_rows))
    axes = np.array(axes).reshape(-1)

    for ax in axes:
        ax.axis("off")

    for i, row in enumerate(selected):
        ax = axes[i]
        ax.imshow(row["raw_image"], cmap="gray")
        ax.imshow(row["anomaly_map_norm"], cmap="jet", alpha=0.45)
        ax.contour(row["largest_mask"].astype(np.float32), levels=[0.5], colors="white", linewidths=0.8)
        ax.set_title(
            f"{row['dataset_name']} idx={row['sample_index']}\nscore={row['score']:.2f}",
            fontsize=9,
        )
        ax.axis("off")

    fig.suptitle(f"Cluster {cluster_id} Representative Samples", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cluster PatchCore-detected anomalies across IAD datasets"
    )
    parser.add_argument(
        "--dataset-root",
        default="Datasets/IAD",
        help="Directory containing the original IAD pickle files",
    )
    parser.add_argument(
        "--outputs-root",
        default="./outputs",
        help="Directory containing PatchCore output folders",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where clustering results will be saved",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["16QAM", "CHIRP", "GMSK", "QPSK"],
        help="Dataset names to merge and cluster",
    )
    parser.add_argument(
        "--threshold-percentile",
        type=float,
        default=90.0,
        help="Percentile used to threshold the anomaly map before taking the largest region",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=8,
        help="Spatial size used to summarize cropped anomaly regions",
    )
    parser.add_argument("--min-cluster-size", type=int, default=30)
    parser.add_argument("--min-samples", type=int, default=10)
    parser.add_argument("--pca-dim", type=int, default=20)
    parser.add_argument("--umap-neighbors", type=int, default=25)
    parser.add_argument("--umap-min-dist", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    dataset_to_output = {
        "16QAM": "16QAM_patchcore_wide_resnet50_2",
        "CHIRP": "CHIRP_patchcore_wide_resnet50_2",
        "GMSK": "GMSK_patchcore_wide_resnet50_2",
        "QPSK": "qpsk_patchcore_wide_resnet50_2",
    }

    all_records = []
    for dataset_name in args.datasets:
        dataset_pkl = os.path.join(args.dataset_root, f"{dataset_name}_Train_Test.pkl")
        output_dir = os.path.join(args.outputs_root, dataset_to_output[dataset_name])
        log(f"Loading {dataset_name} from {dataset_pkl} and {output_dir}")
        all_records.extend(load_records(dataset_name, dataset_pkl, output_dir))

    anomaly_records = [record for record in all_records if record.true_label == 1]
    log(f"Selected {len(anomaly_records)} anomaly samples with label=1")
    missing_patch = [
        (record.dataset_name, record.sample_index)
        for record in anomaly_records
        if record.patch_embeddings is None or record.patch_scores is None
    ]
    if missing_patch:
        first_missing = missing_patch[0]
        raise ValueError(
            "Missing exported patch artifacts for anomaly sample "
            f"{first_missing[0]} idx={first_missing[1]}"
        )

    feature_rows = []
    metadata_rows = []
    montage_rows = []
    for idx, record in enumerate(anomaly_records, start=1):
        feature_vector, region_meta, largest_mask = extract_feature_vector(
            raw_image=record.raw_image,
            anomaly_map=record.anomaly_map,
            patch_embeddings=record.patch_embeddings,
            patch_scores=record.patch_scores,
            threshold_percentile=args.threshold_percentile,
            patch_size=args.patch_size,
        )
        normalized_map = normalize_map(record.anomaly_map)
        feature_rows.append(feature_vector)
        metadata_rows.append(
            {
                "dataset_name": record.dataset_name,
                "sample_index": record.sample_index,
                "true_label": record.true_label,
                "score": record.score,
                "prediction": record.prediction,
                **region_meta,
            }
        )
        montage_rows.append(
            {
                "dataset_name": record.dataset_name,
                "sample_index": record.sample_index,
                "score": record.score,
                "raw_image": record.raw_image,
                "anomaly_map_norm": normalized_map,
                "largest_mask": largest_mask,
            }
        )
        if idx == 1 or idx % 500 == 0 or idx == len(anomaly_records):
            log(f"Extracted clustering features for {idx}/{len(anomaly_records)} anomalies")

    features = np.stack(feature_rows).astype(np.float32)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    pca_dim = min(args.pca_dim, features_scaled.shape[0], features_scaled.shape[1])
    pca = PCA(n_components=pca_dim, random_state=args.seed)
    features_pca = pca.fit_transform(features_scaled)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=False,
    )
    cluster_labels = clusterer.fit_predict(features_pca)

    reducer = umap.UMAP(
        n_neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
        n_components=2,
        metric="euclidean",
        random_state=args.seed,
    )
    projection = reducer.fit_transform(features_pca)

    valid_mask = cluster_labels != -1
    unique_clusters = sorted(label for label in np.unique(cluster_labels) if label != -1)
    num_noise = int((cluster_labels == -1).sum())

    silhouette_value = None
    davies_bouldin_value = None
    if len(unique_clusters) >= 2 and np.sum(valid_mask) >= 2:
        silhouette_value = float(
            silhouette_score(features_pca[valid_mask], cluster_labels[valid_mask])
        )
        davies_bouldin_value = float(
            davies_bouldin_score(features_pca[valid_mask], cluster_labels[valid_mask])
        )

    output_table = []
    for row, montage_row, cluster_label, point in zip(
        metadata_rows, montage_rows, cluster_labels, projection
    ):
        row = dict(row)
        row["cluster_label"] = int(cluster_label)
        row["umap_x"] = float(point[0])
        row["umap_y"] = float(point[1])
        output_table.append(row)
        montage_row["cluster_label"] = int(cluster_label)
        montage_row["umap_x"] = float(point[0])
        montage_row["umap_y"] = float(point[1])

    result_df = pd.DataFrame(output_table)
    result_df.to_csv(os.path.join(args.output_dir, "cluster_assignments.csv"), index=False)

    save_projection_plot(
        projection=projection,
        cluster_labels=cluster_labels,
        dataset_names=result_df["dataset_name"].tolist(),
        output_path=os.path.join(args.output_dir, "umap_cluster_projection.png"),
    )

    summary_rows = []
    for cluster_id in unique_clusters:
        cluster_df = result_df[result_df["cluster_label"] == cluster_id].copy()
        cluster_features = features_pca[cluster_labels == cluster_id]
        centroid = cluster_features.mean(axis=0, keepdims=True)
        distances = np.linalg.norm(cluster_features - centroid, axis=1)
        cluster_df["centroid_distance"] = distances
        cluster_df = cluster_df.sort_values(
            by=["centroid_distance", "score"], ascending=[True, False]
        )

        dataset_counts = (
            cluster_df["dataset_name"].value_counts().sort_index().to_dict()
        )
        summary_rows.append(
            {
                "cluster_label": int(cluster_id),
                "count": int(cluster_df.shape[0]),
                "mean_score": float(cluster_df["score"].mean()),
                "std_score": float(cluster_df["score"].std(ddof=0)),
                "dataset_counts": dataset_counts,
            }
        )

        cluster_lookup = {
            (row["dataset_name"], row["sample_index"]): row for row in montage_rows
        }
        representative_rows = []
        for _, rep_row in cluster_df.head(10).iterrows():
            representative_rows.append(
                cluster_lookup[(rep_row["dataset_name"], int(rep_row["sample_index"]))]
            )

        save_cluster_montage(
            cluster_id=cluster_id,
            rows=representative_rows,
            output_path=os.path.join(args.output_dir, f"cluster_{cluster_id:02d}_montage.png"),
            num_images=10,
        )

    with open(os.path.join(args.output_dir, "cluster_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2)

    metrics = {
        "datasets": args.datasets,
        "num_total_records": len(all_records),
        "num_anomaly_records": len(anomaly_records),
        "feature_dim": int(features.shape[1]),
        "pca_dim": int(features_pca.shape[1]),
        "threshold_percentile": args.threshold_percentile,
        "patch_size": args.patch_size,
        "semantic_embedding_dim": int(anomaly_records[0].patch_embeddings.shape[-1]),
        "min_cluster_size": args.min_cluster_size,
        "min_samples": args.min_samples,
        "num_clusters": len(unique_clusters),
        "num_noise_points": num_noise,
        "clustered_points": int(valid_mask.sum()),
        "silhouette": silhouette_value,
        "davies_bouldin": davies_bouldin_value,
        "pca_explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
    }
    with open(os.path.join(args.output_dir, "clustering_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    log(
        f"Finished clustering: clusters={metrics['num_clusters']}, noise={metrics['num_noise_points']}, "
        f"silhouette={metrics['silhouette']}, davies_bouldin={metrics['davies_bouldin']}"
    )
    log(f"Artifacts saved to {args.output_dir}")


if __name__ == "__main__":
    main()
