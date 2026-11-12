import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import umap
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler
import hdbscan

from cluster_patchcore_anomalies import (
    extract_feature_vector,
    load_records,
    log,
    normalize_map,
    save_cluster_montage,
    save_projection_plot,
)


def parse_int_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Systematic hyperparameter sweep for semantic HDBSCAN clustering"
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
        help="Directory where sweep results will be saved",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["16QAM", "CHIRP", "GMSK", "QPSK"],
        help="Datasets to merge for clustering",
    )
    parser.add_argument("--threshold-percentile", type=float, default=90.0)
    parser.add_argument("--patch-size", type=int, default=8)
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
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top configs to export to a separate CSV",
    )
    return parser.parse_args()


def build_feature_cache(args):
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
                "anomaly_map_norm": normalize_map(record.anomaly_map),
                "largest_mask": largest_mask,
            }
        )
        if idx == 1 or idx % 500 == 0 or idx == len(anomaly_records):
            log(f"Cached features for {idx}/{len(anomaly_records)} anomalies")

    features = np.stack(feature_rows).astype(np.float32)
    metadata_df = pd.DataFrame(metadata_rows)
    montage_lookup = {
        (row["dataset_name"], row["sample_index"]): row for row in montage_rows
    }
    return features, metadata_df, montage_lookup


def evaluate_config(features_scaled, pca_dim, min_cluster_size, min_samples, seed):
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


def result_sort_key(row):
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

    best_metrics = {
        "datasets": args.datasets,
        "threshold_percentile": args.threshold_percentile,
        "patch_size": args.patch_size,
        "num_anomaly_records": int(metadata_df.shape[0]),
        "feature_dim": int(features_pca.shape[1]),
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
        json.dump(best_metrics, f, indent=2)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    pca_dims = parse_int_list(args.pca_dims)
    min_cluster_sizes = parse_int_list(args.min_cluster_sizes)
    min_samples_list = parse_int_list(args.min_samples_list)

    features, metadata_df, montage_lookup = build_feature_cache(args)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    scan_rows = []
    best_result = None
    total_runs = len(pca_dims) * len(min_cluster_sizes) * len(min_samples_list)
    run_index = 0

    for pca_dim in pca_dims:
        for min_cluster_size in min_cluster_sizes:
            for min_samples in min_samples_list:
                run_index += 1
                result = evaluate_config(
                    features_scaled=features_scaled,
                    pca_dim=pca_dim,
                    min_cluster_size=min_cluster_size,
                    min_samples=min_samples,
                    seed=args.seed,
                )
                scan_row = {
                    key: value
                    for key, value in result.items()
                    if key not in {"cluster_labels", "features_pca"}
                }
                scan_rows.append(scan_row)

                if best_result is None or result_sort_key(scan_row) < result_sort_key(
                    {
                        key: value
                        for key, value in best_result.items()
                        if key not in {"cluster_labels", "features_pca"}
                    }
                ):
                    best_result = result

                log(
                    f"Sweep {run_index}/{total_runs}: pca={pca_dim}, "
                    f"min_cluster_size={min_cluster_size}, min_samples={min_samples}, "
                    f"clusters={scan_row['num_clusters']}, noise_ratio={scan_row['noise_ratio']:.4f}, "
                    f"silhouette={scan_row['silhouette']}, db={scan_row['davies_bouldin']}"
                )

    results_df = pd.DataFrame(scan_rows)
    results_df = results_df.sort_values(
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
    results_df.to_csv(os.path.join(args.output_dir, "scan_results.csv"), index=False)
    results_df.head(args.top_k).to_csv(
        os.path.join(args.output_dir, "top_configs.csv"), index=False
    )

    best_dir = os.path.join(args.output_dir, "best_config_artifacts")
    materialize_best_result(args, metadata_df, montage_lookup, best_result, best_dir)

    with open(os.path.join(args.output_dir, "scan_config.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "datasets": args.datasets,
                "threshold_percentile": args.threshold_percentile,
                "patch_size": args.patch_size,
                "pca_dims": pca_dims,
                "min_cluster_sizes": min_cluster_sizes,
                "min_samples_list": min_samples_list,
                "seed": args.seed,
            },
            f,
            indent=2,
        )

    best_row = {
        key: value for key, value in best_result.items() if key not in {"cluster_labels", "features_pca"}
    }
    log(
        "Best config: "
        f"pca={best_row['pca_dim']}, min_cluster_size={best_row['min_cluster_size']}, "
        f"min_samples={best_row['min_samples']}, clusters={best_row['num_clusters']}, "
        f"noise_ratio={best_row['noise_ratio']:.4f}, silhouette={best_row['silhouette']}, "
        f"db={best_row['davies_bouldin']}"
    )
    log(f"Sweep artifacts saved to {args.output_dir}")


if __name__ == "__main__":
    main()
