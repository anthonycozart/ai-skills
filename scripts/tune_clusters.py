"""Evaluate HDBSCAN min_cluster_size across a range of values.

Reuses the embedding and UMAP reduction from cluster_labels.py, then
sweeps min_cluster_size and reports metrics to help choose a value.

Metrics reported:
    - Number of clusters
    - Noise ratio (% of points not assigned to any cluster)
    - DBCV score (density-based cluster validation; higher is better)
    - Largest cluster share (% of non-noise points in the biggest cluster)

Usage:
    python scripts/tune_clusters.py
    python scripts/tune_clusters.py --input data/pilot_classifications.jsonl
    python scripts/tune_clusters.py --min-sizes 3 5 8 10 15 20 30 50
    python scripts/tune_clusters.py --output data/cluster_tuning.txt
"""

import argparse
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import DATA_DIR
from scripts.cluster_labels import (
    cluster_embeddings,
    embed_labels,
    extract_labels,
    reduce_dimensions,
)
from src.parsing import load_jsonl


def compute_dbcv(reduced: np.ndarray, cluster_ids: np.ndarray) -> float:
    """Compute DBCV score. Returns NaN if fewer than 2 clusters."""
    import hdbscan

    n_clusters = len(set(cluster_ids)) - (1 if -1 in cluster_ids else 0)
    if n_clusters < 2:
        return float("nan")

    return hdbscan.validity.validity_index(reduced.astype(np.float64), cluster_ids)


def run_sweep(args):
    """Run the parameter sweep and print results."""
    # Extract and embed
    print("1. Loading and extracting labels...")
    records = load_jsonl(args.input)
    print(f"  Loaded {len(records)} records")

    unique_labels, label_to_indices = extract_labels(records, args.field)
    print(f"  Found {len(unique_labels)} unique '{args.field}' labels")

    if len(unique_labels) < 10:
        print("  Too few unique labels for meaningful clustering.")
        sys.exit(1)

    print("\n2. Embedding labels...")
    embeddings = embed_labels(unique_labels, args.field)

    print("\n3. Reducing dimensionality...")
    reduced_cluster, _ = reduce_dimensions(
        embeddings,
        cluster_dims=args.cluster_dims,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        field=args.field,
    )

    # Sweep
    print(f"\n4. Sweeping min_cluster_size: {args.min_sizes}")
    print()

    header = (
        f"  {'min_size':>8}  {'clusters':>8}  {'noise %':>8}  "
        f"{'DBCV':>8}  {'largest %':>9}"
    )
    print(header)
    print(f"  {'-' * 8}  {'-' * 8}  {'-' * 8}  {'-' * 8}  {'-' * 9}")

    results = []
    for size in args.min_sizes:
        import hdbscan

        clusterer = hdbscan.HDBSCAN(min_cluster_size=size)
        cluster_ids = clusterer.fit_predict(reduced_cluster)

        n_labels = len(unique_labels)
        n_clusters = len(set(cluster_ids)) - (1 if -1 in cluster_ids else 0)
        n_noise = int(np.sum(cluster_ids == -1))
        noise_pct = n_noise / n_labels * 100

        # Largest cluster share (of non-noise points)
        non_noise = cluster_ids[cluster_ids != -1]
        if len(non_noise) > 0:
            from collections import Counter
            counts = Counter(non_noise)
            largest = counts.most_common(1)[0][1]
            largest_pct = largest / len(non_noise) * 100
        else:
            largest_pct = 0.0

        dbcv = compute_dbcv(reduced_cluster, cluster_ids)

        row = {
            "min_cluster_size": size,
            "n_clusters": n_clusters,
            "noise_pct": noise_pct,
            "dbcv": dbcv,
            "largest_pct": largest_pct,
        }
        results.append(row)

        dbcv_str = f"{dbcv:.4f}" if not np.isnan(dbcv) else "N/A"
        print(
            f"  {size:>8}  {n_clusters:>8}  {noise_pct:>7.1f}%  "
            f"{dbcv_str:>8}  {largest_pct:>8.1f}%"
        )

    # Guidance
    print()
    print("Interpretation:")
    print("  - More clusters = finer granularity, fewer = broader groupings")
    print("  - Noise % too high (>30%) means many labels are unassigned")
    print("  - Higher DBCV = better density-separated clusters")
    print("  - Largest % too high (>40%) suggests one cluster is absorbing too much")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate HDBSCAN min_cluster_size across a range."
    )
    parser.add_argument(
        "--input", type=Path, default=DATA_DIR / "classifications.jsonl",
        help="Input classified JSONL (default: data/classifications.jsonl)",
    )
    parser.add_argument(
        "--field", default="object",
        help="Field to cluster on (default: object)",
    )
    parser.add_argument(
        "--min-sizes", type=int, nargs="+", default=[3, 5, 8, 10, 15, 20, 30, 50],
        help="min_cluster_size values to test (default: 3 5 8 10 15 20 30 50)",
    )
    parser.add_argument(
        "--cluster-dims", type=int, default=10,
        help="UMAP dimensions for clustering (default: 10)",
    )
    parser.add_argument(
        "--n-neighbors", type=int, default=15,
        help="UMAP n_neighbors (default: 15)",
    )
    parser.add_argument(
        "--min-dist", type=float, default=0.1,
        help="UMAP min_dist (default: 0.1)",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Write output to file (default: print to stdout)",
    )
    args = parser.parse_args()

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            with redirect_stdout(f):
                run_sweep(args)
        print(f"Saved to {args.output}")
    else:
        run_sweep(args)


if __name__ == "__main__":
    main()
