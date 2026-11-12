import argparse
import json
import math
import os

import hdbscan
import numpy as np
import pandas as pd
import umap
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from cluster_patchcore_anomalies import (
    largest_connected_region,
    load_records,
    log,
    normalize_map,
    resize_array,
    save_cluster_montage,
    save_projection_plot,
)


def parse_int_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare mean, gram, and mean+gram feature modes for anomaly clustering"
    )
    parser.add_argument(
        "--dataset-root",
        default="Datasets/IAD",
        help="Directory containing original IAD pickle files",
    )
    parser.add_argument(
        "--outputs-root",
        default="./outputs",
        help="Directory containing PatchCore output folders",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where comparison results will be saved",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["16QAM", "CHIRP", "GMSK", "QPSK"],
        help="Datasets to merge for clustering",
    )
    parser.add_argument("--threshold-percentile", type=float, default=90.0)
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--gram-dim", type=int, default=32)
    parser.add_argument(
        "--feature-modes",
        nargs="+",
        default=["mean", "gram", "mean+gram"],
        choices=["mean", "gram", "mean+gram"],
        help="Feature construction modes to compare",
    )
    parser.add_argument(
        "--pca-dims",
        default="10,20,30,40,60,80",
        help="Comma-separated PCA dimensions to test",
    )
    parser.add_argument(
        "--min-cluster-sizes",
        default="10,20,30,50,80",
        help="Comma-separated HDBSCAN min_cluster_size values to test",
    )
    parser.add_argument(
        "--min-samples-list",
        default="3,5,10,15",
        help="Comma-separated HDBSCAN min_samples values to test",
    )
    parser.add_argument("--umap-neighbors", type=int, default=25)
    parser.add_argument("--umap-min-dist", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=10)
    return parser.parse_args()


def build_region_feature_cache(args):
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

    base_features = []
    mean_embeddings = []
    patch_regions = []
    patch_weights = []
    metadata_rows = []
    montage_rows = []

    for idx, record in enumerate(anomaly_records, start=1):
        normalized_map = normalize_map(record.anomaly_map)
        mask, props = largest_connected_region(
            normalized_map, threshold_percentile=args.threshold_percentile
        )

        y_min = props["y_min"]
        y_max = props["y_max"]
        x_min = props["x_min"]
        x_max = props["x_max"]

        map_crop = normalized_map[y_min : y_max + 1, x_min : x_max + 1]
        raw_crop = record.raw_image[y_min : y_max + 1, x_min : x_max + 1]
        mask_crop = mask[y_min : y_max + 1, x_min : x_max + 1]

        masked_map_values = normalized_map[mask]
        masked_raw_values = record.raw_image[mask]

        image_height, image_width = record.raw_image.shape
        bbox_height = y_max - y_min + 1
        bbox_width = x_max - x_min + 1
        area = int(mask.sum())
        area_ratio = area / float(image_height * image_width)

        center_y, center_x = np.argwhere(mask).mean(axis=0)
        center_y = float(center_y / image_height)
        center_x = float(center_x / image_width)

        map_resized = resize_array(map_crop, (args.patch_size, args.patch_size))
        raw_resized = resize_array(raw_crop, (args.patch_size, args.patch_size))
        masked_crop = raw_crop * mask_crop.astype(np.float32)
        masked_resized = resize_array(masked_crop, (args.patch_size, args.patch_size))

        row_profile = mask.astype(np.float32).mean(axis=1)
        col_profile = mask.astype(np.float32).mean(axis=0)
        row_profile = resize_array(
            row_profile[np.newaxis, :], (1, args.patch_size)
        ).reshape(-1)
        col_profile = resize_array(
            col_profile[np.newaxis, :], (1, args.patch_size)
        ).reshape(-1)

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

        patch_mask = resize_array(
            mask.astype(np.float32), tuple(int(v) for v in record.patch_scores.shape)
        )
        patch_mask = patch_mask >= 0.25
        if not np.any(patch_mask):
            flat_index = int(np.argmax(record.patch_scores))
            patch_mask = np.zeros_like(record.patch_scores, dtype=bool)
            patch_mask.flat[flat_index] = True

        region_patch_scores = record.patch_scores[patch_mask].astype(np.float32)
        region_patch_embeddings = record.patch_embeddings[patch_mask].astype(np.float32)

        shifted_scores = region_patch_scores - region_patch_scores.min()
        if float(shifted_scores.max()) > 0:
            shifted_scores = shifted_scores / shifted_scores.max()
        weights = shifted_scores + 1e-6

        mean_embedding = np.average(
            region_patch_embeddings, axis=0, weights=weights
        ).astype(np.float32)
        mean_norm = float(np.linalg.norm(mean_embedding))
        if mean_norm > 0:
            mean_embedding = mean_embedding / mean_norm

        base_feature = np.concatenate(
            [
                stats,
                map_resized.reshape(-1),
                raw_resized.reshape(-1),
                masked_resized.reshape(-1),
                row_profile,
                col_profile,
            ]
        ).astype(np.float32)

        base_features.append(base_feature)
        mean_embeddings.append(mean_embedding)
        patch_regions.append(region_patch_embeddings)
        patch_weights.append(weights.astype(np.float32))
        metadata_rows.append(
            {
                "dataset_name": record.dataset_name,
                "sample_index": record.sample_index,
                "true_label": record.true_label,
                "score": record.score,
                "prediction": record.prediction,
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
        )
        montage_rows.append(
            {
                "dataset_name": record.dataset_name,
                "sample_index": record.sample_index,
                "score": record.score,
                "raw_image": record.raw_image,
                "anomaly_map_norm": normalized_map,
                "largest_mask": mask,
            }
        )
        if idx == 1 or idx % 500 == 0 or idx == len(anomaly_records):
            log(f"Cached mode-comparison features for {idx}/{len(anomaly_records)} anomalies")

    return {
        "base_features": np.stack(base_features).astype(np.float32),
        "mean_embeddings": np.stack(mean_embeddings).astype(np.float32),
        "patch_regions": patch_regions,
        "patch_weights": patch_weights,
        "metadata_df": pd.DataFrame(metadata_rows),
        "montage_lookup": {
            (row["dataset_name"], row["sample_index"]): row for row in montage_rows
        },
    }


def fit_gram_projector(patch_regions: list[np.ndarray], gram_dim: int, seed: int):
    all_patches = np.concatenate(patch_regions, axis=0).astype(np.float32)
    gram_dim = min(gram_dim, all_patches.shape[0], all_patches.shape[1])
    projector = PCA(n_components=gram_dim, random_state=seed)
    projector.fit(all_patches)
    return projector


def compute_gram_feature(
    patch_region: np.ndarray, patch_weight: np.ndarray, projector: PCA
) -> np.ndarray:
    projected = projector.transform(patch_region.astype(np.float32))
    weights = patch_weight.astype(np.float32)
    weights = weights / max(float(weights.sum()), 1e-6)

    weighted_mean = np.sum(projected * weights[:, None], axis=0, keepdims=True)
    centered = projected - weighted_mean
    weighted_centered = centered * np.sqrt(weights[:, None])
    gram = weighted_centered.T @ weighted_centered

    upper_indices = np.triu_indices(gram.shape[0])
    gram_vector = gram[upper_indices].astype(np.float32)
    gram_norm = float(np.linalg.norm(gram_vector))
    if gram_norm > 0:
        gram_vector = gram_vector / gram_norm
    return gram_vector


def assemble_feature_matrix(mode: str, cache: dict, gram_projector: PCA) -> np.ndarray:
    gram_features = [
        compute_gram_feature(region, weight, gram_projector)
        for region, weight in zip(cache["patch_regions"], cache["patch_weights"])
    ]
    gram_features = np.stack(gram_features).astype(np.float32)

    if mode == "mean":
        return np.concatenate(
            [cache["base_features"], cache["mean_embeddings"]], axis=1
        ).astype(np.float32)
    if mode == "gram":
        return np.concatenate([cache["base_features"], gram_features], axis=1).astype(
            np.float32
        )
    if mode == "mean+gram":
        return np.concatenate(
            [cache["base_features"], cache["mean_embeddings"], gram_features], axis=1
        ).astype(np.float32)
    raise ValueError(f"Unsupported feature mode: {mode}")


def evaluate_config(features: np.ndarray, pca_dim: int, min_cluster_size: int, min_samples: int, seed: int):
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    pca_dim = min(pca_dim, features_scaled.shape[0], features_scaled.shape[1])
    pca = PCA(n_components=pca_dim, random_state=seed)
    features_pca = pca.fit_transform(features_scaled)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=False,
    )
    cluster_labels = clusterer.fit_predict(features_pca)

    valid_mask = cluster_labels != -1
    unique_clusters = sorted(label for label in np.unique(cluster_labels) if label != -1)
    noise_points = int((cluster_labels == -1).sum())
    clustered_points = int(valid_mask.sum())
    noise_ratio = noise_points / float(cluster_labels.shape[0])

    silhouette = None
    davies_bouldin = None
    if len(unique_clusters) >= 2 and clustered_points >= 2:
        silhouette = float(
            silhouette_score(features_pca[valid_mask], cluster_labels[valid_mask])
        )
        davies_bouldin = float(
            davies_bouldin_score(features_pca[valid_mask], cluster_labels[valid_mask])
        )

    return {
        "pca_dim": int(pca_dim),
        "min_cluster_size": int(min_cluster_size),
        "min_samples": int(min_samples),
        "num_clusters": int(len(unique_clusters)),
        "noise_points": noise_points,
        "clustered_points": clustered_points,
        "noise_ratio": noise_ratio,
        "silhouette": silhouette,
        "davies_bouldin": davies_bouldin,
        "pca_explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
        "cluster_labels": cluster_labels,
        "features_pca": features_pca,
    }


def result_sort_key(row: dict):
    valid = row["num_clusters"] >= 2 and row["silhouette"] is not None
    return (
        0 if valid else 1,
        -(row["silhouette"] if row["silhouette"] is not None else -1e9),
        row["davies_bouldin"] if row["davies_bouldin"] is not None else 1e9,
        row["noise_ratio"],
        -row["clustered_points"],
        -row["num_clusters"],
    )


def materialize_best_result(
    mode: str,
    args,
    metadata_df: pd.DataFrame,
    montage_lookup: dict,
    best_result: dict,
    best_dir: str,
):
    os.makedirs(best_dir, exist_ok=True)
    cluster_labels = best_result["cluster_labels"]
    features_pca = best_result["features_pca"]

    reducer = umap.UMAP(
        n_neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
        n_components=2,
        metric="euclidean",
        random_state=args.seed,
    )
    projection = reducer.fit_transform(features_pca)

    result_df = metadata_df.copy()
    result_df["cluster_label"] = cluster_labels.astype(int)
    result_df["umap_x"] = projection[:, 0]
    result_df["umap_y"] = projection[:, 1]
    result_df.to_csv(os.path.join(best_dir, "cluster_assignments.csv"), index=False)

    save_projection_plot(
        projection=projection,
        cluster_labels=cluster_labels,
        dataset_names=result_df["dataset_name"].tolist(),
        output_path=os.path.join(best_dir, "umap_cluster_projection.png"),
    )

    summary_rows = []
    for cluster_id in sorted(label for label in np.unique(cluster_labels) if label != -1):
        cluster_df = result_df[result_df["cluster_label"] == cluster_id].copy()
        cluster_features = features_pca[cluster_labels == cluster_id]
        centroid = cluster_features.mean(axis=0, keepdims=True)
        distances = np.linalg.norm(cluster_features - centroid, axis=1)
        cluster_df["centroid_distance"] = distances
        cluster_df = cluster_df.sort_values(
            by=["centroid_distance", "score"], ascending=[True, False]
        )

        summary_rows.append(
            {
                "cluster_label": int(cluster_id),
                "count": int(cluster_df.shape[0]),
                "mean_score": float(cluster_df["score"].mean()),
                "std_score": float(cluster_df["score"].std(ddof=0)),
                "dataset_counts": cluster_df["dataset_name"].value_counts().sort_index().to_dict(),
            }
        )

        representative_rows = [
            montage_lookup[(row["dataset_name"], int(row["sample_index"]))]
            for _, row in cluster_df.head(10).iterrows()
        ]
        save_cluster_montage(
            cluster_id=int(cluster_id),
            rows=representative_rows,
            output_path=os.path.join(best_dir, f"cluster_{int(cluster_id):02d}_montage.png"),
            num_images=10,
        )

    with open(os.path.join(best_dir, "cluster_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2)

    metrics = {
        "mode": mode,
        "datasets": args.datasets,
        "threshold_percentile": args.threshold_percentile,
        "patch_size": args.patch_size,
        "gram_dim": args.gram_dim,
        "num_anomaly_records": int(metadata_df.shape[0]),
        "feature_dim": int(best_result["features_pca"].shape[1]),
        "best_config": {
            "pca_dim": int(best_result["pca_dim"]),
            "min_cluster_size": int(best_result["min_cluster_size"]),
            "min_samples": int(best_result["min_samples"]),
        },
        "num_clusters": int(best_result["num_clusters"]),
        "noise_points": int(best_result["noise_points"]),
        "clustered_points": int(best_result["clustered_points"]),
        "noise_ratio": float(best_result["noise_ratio"]),
        "silhouette": best_result["silhouette"],
        "davies_bouldin": best_result["davies_bouldin"],
        "pca_explained_variance_ratio_sum": float(best_result["pca_explained_variance_ratio_sum"]),
    }
    with open(os.path.join(best_dir, "clustering_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    pca_dims = parse_int_list(args.pca_dims)
    min_cluster_sizes = parse_int_list(args.min_cluster_sizes)
    min_samples_list = parse_int_list(args.min_samples_list)

    cache = build_region_feature_cache(args)
    gram_projector = fit_gram_projector(
        patch_regions=cache["patch_regions"],
        gram_dim=args.gram_dim,
        seed=args.seed,
    )

    mode_best_summary = []
    total_runs = len(args.feature_modes) * len(pca_dims) * len(min_cluster_sizes) * len(min_samples_list)
    run_index = 0

    for mode in args.feature_modes:
        mode_dir = os.path.join(args.output_dir, mode.replace("+", "_plus_"))
        os.makedirs(mode_dir, exist_ok=True)
        features = assemble_feature_matrix(mode=mode, cache=cache, gram_projector=gram_projector)
        log(f"Starting sweep for mode={mode} with feature_dim={features.shape[1]}")

        mode_rows = []
        best_result = None
        for pca_dim in pca_dims:
            for min_cluster_size in min_cluster_sizes:
                for min_samples in min_samples_list:
                    run_index += 1
                    result = evaluate_config(
                        features=features,
                        pca_dim=pca_dim,
                        min_cluster_size=min_cluster_size,
                        min_samples=min_samples,
                        seed=args.seed,
                    )
                    row = {
                        "mode": mode,
                        "feature_dim": int(features.shape[1]),
                        **{
                            key: value
                            for key, value in result.items()
                            if key not in {"cluster_labels", "features_pca"}
                        },
                    }
                    mode_rows.append(row)

                    if best_result is None or result_sort_key(row) < result_sort_key(
                        {
                            "mode": mode,
                            "feature_dim": int(features.shape[1]),
                            **{
                                key: value
                                for key, value in best_result.items()
                                if key not in {"cluster_labels", "features_pca"}
                            },
                        }
                    ):
                        best_result = result

                    log(
                        f"Sweep {run_index}/{total_runs}: mode={mode}, pca={pca_dim}, "
                        f"min_cluster_size={min_cluster_size}, min_samples={min_samples}, "
                        f"clusters={row['num_clusters']}, noise_ratio={row['noise_ratio']:.4f}, "
                        f"silhouette={row['silhouette']}, db={row['davies_bouldin']}"
                    )

        mode_df = pd.DataFrame(mode_rows).sort_values(
            by=[
                "silhouette",
                "davies_bouldin",
                "noise_ratio",
                "clustered_points",
                "num_clusters",
            ],
            ascending=[False, True, True, False, False],
            na_position="last",
        )
        mode_df.to_csv(os.path.join(mode_dir, "scan_results.csv"), index=False)
        mode_df.head(args.top_k).to_csv(os.path.join(mode_dir, "top_configs.csv"), index=False)

        best_dir = os.path.join(mode_dir, "best_config_artifacts")
        materialize_best_result(
            mode=mode,
            args=args,
            metadata_df=cache["metadata_df"],
            montage_lookup=cache["montage_lookup"],
            best_result=best_result,
            best_dir=best_dir,
        )

        best_row = {
            "mode": mode,
            "feature_dim": int(features.shape[1]),
            **{
                key: value
                for key, value in best_result.items()
                if key not in {"cluster_labels", "features_pca"}
            },
        }
        mode_best_summary.append(best_row)
        log(
            f"Best for mode={mode}: pca={best_row['pca_dim']}, "
            f"min_cluster_size={best_row['min_cluster_size']}, min_samples={best_row['min_samples']}, "
            f"clusters={best_row['num_clusters']}, noise_ratio={best_row['noise_ratio']:.4f}, "
            f"silhouette={best_row['silhouette']}, db={best_row['davies_bouldin']}"
        )

    summary_df = pd.DataFrame(mode_best_summary).sort_values(
        by=["silhouette", "davies_bouldin", "noise_ratio"],
        ascending=[False, True, True],
        na_position="last",
    )
    summary_df.to_csv(os.path.join(args.output_dir, "mode_comparison_summary.csv"), index=False)

    with open(os.path.join(args.output_dir, "comparison_config.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "datasets": args.datasets,
                "feature_modes": args.feature_modes,
                "threshold_percentile": args.threshold_percentile,
                "patch_size": args.patch_size,
                "gram_dim": args.gram_dim,
                "pca_dims": pca_dims,
                "min_cluster_sizes": min_cluster_sizes,
                "min_samples_list": min_samples_list,
                "seed": args.seed,
            },
            f,
            indent=2,
        )

    log(f"Mode comparison finished. Summary saved to {args.output_dir}")


if __name__ == "__main__":
    main()
