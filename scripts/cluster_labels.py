"""Embed, reduce, cluster, and label free-text classification labels.

Reads classified JSONL records, embeds the unique label strings via OpenAI,
reduces dimensionality with UMAP, clusters with HDBSCAN, and labels each
cluster with an LLM call.

Usage:
    python scripts/cluster_labels.py
    python scripts/cluster_labels.py --input data/pilot_classifications.jsonl --field object
    python scripts/cluster_labels.py --field primary_intent --min-cluster-size 3
    python scripts/cluster_labels.py --cluster-dims 15 --n-neighbors 20
"""

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import DATA_DIR, PROMPTS_DIR
from src.api_clients import call_model
from src.parsing import load_jsonl, load_prompt


# ---------------------------------------------------------------------------
# 1. Extract
# ---------------------------------------------------------------------------

def extract_labels(records: list[dict], field: str) -> tuple[list[str], dict[str, list[int]]]:
    """Pull the target field from each record and deduplicate.

    Returns:
        unique_labels: Ordered list of unique label strings.
        label_to_indices: Mapping from label string to list of record indices.
    """
    label_to_indices: dict[str, list[int]] = {}
    for idx, record in enumerate(records):
        label = record.get(field)
        if label is None:
            continue
        label = str(label).strip()
        if label not in label_to_indices:
            label_to_indices[label] = []
        label_to_indices[label].append(idx)

    unique_labels = list(label_to_indices.keys())
    return unique_labels, label_to_indices


# ---------------------------------------------------------------------------
# 2. Embed
# ---------------------------------------------------------------------------

def embed_labels(
    unique_labels: list[str],
    field: str,
    batch_size: int = 2000,
) -> np.ndarray:
    """Embed unique labels via OpenAI, with file-based caching.

    Returns:
        numpy array of shape (len(unique_labels), embedding_dim).
    """
    cache_array_path = DATA_DIR / f"embeddings_{field}.npy"
    cache_labels_path = DATA_DIR / f"embeddings_{field}_labels.json"

    # Check cache
    if cache_array_path.exists() and cache_labels_path.exists():
        cached_labels = json.loads(cache_labels_path.read_text())
        cached_array = np.load(cache_array_path)
        if set(unique_labels).issubset(set(cached_labels)):
            # Reorder cached embeddings to match unique_labels order
            label_to_row = {lbl: i for i, lbl in enumerate(cached_labels)}
            indices = [label_to_row[lbl] for lbl in unique_labels]
            print(f"  Loaded {len(unique_labels)} embeddings from cache")
            return cached_array[indices]

    # Call OpenAI
    import openai

    client = openai.OpenAI()  # uses OPENAI_API_KEY from env
    all_embeddings = []

    for start in range(0, len(unique_labels), batch_size):
        batch = unique_labels[start : start + batch_size]
        print(f"  Embedding batch {start // batch_size + 1} "
              f"({len(batch)} labels)...")
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=batch,
        )
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)

    embeddings = np.array(all_embeddings, dtype=np.float32)

    # Save cache
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.save(cache_array_path, embeddings)
    cache_labels_path.write_text(json.dumps(unique_labels, ensure_ascii=False))
    print(f"  Cached {len(unique_labels)} embeddings to {cache_array_path.name}")

    return embeddings


# ---------------------------------------------------------------------------
# 3. Reduce dimensionality
# ---------------------------------------------------------------------------

def reduce_dimensions(
    embeddings: np.ndarray,
    cluster_dims: int,
    n_neighbors: int,
    min_dist: float,
    field: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Run UMAP twice: once for clustering, once for 2D visualization.

    Returns:
        reduced_cluster: array of shape (n, cluster_dims)
        reduced_2d: array of shape (n, 2)
    """
    import umap

    print(f"  Reducing to {cluster_dims}D for clustering...")
    reducer_cluster = umap.UMAP(
        n_components=cluster_dims,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=42,
    )
    reduced_cluster = reducer_cluster.fit_transform(embeddings)

    print("  Reducing to 2D for visualization...")
    reducer_2d = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=42,
    )
    reduced_2d = reducer_2d.fit_transform(embeddings)

    # Save 2D coordinates
    umap_2d_path = DATA_DIR / f"umap_2d_{field}.npy"
    np.save(umap_2d_path, reduced_2d)
    print(f"  Saved 2D coordinates to {umap_2d_path.name}")

    return reduced_cluster, reduced_2d


# ---------------------------------------------------------------------------
# 4. Cluster
# ---------------------------------------------------------------------------

def cluster_embeddings(
    reduced: np.ndarray,
    min_cluster_size: int,
) -> np.ndarray:
    """Run HDBSCAN on the reduced embeddings.

    Returns:
        Array of cluster labels (int), -1 for noise.
    """
    import hdbscan

    print(f"  Running HDBSCAN (min_cluster_size={min_cluster_size})...")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    cluster_ids = clusterer.fit_predict(reduced)

    n_clusters = len(set(cluster_ids)) - (1 if -1 in cluster_ids else 0)
    n_noise = int(np.sum(cluster_ids == -1))
    print(f"  Found {n_clusters} clusters, {n_noise} noise points")

    return cluster_ids


# ---------------------------------------------------------------------------
# 5. Label clusters
# ---------------------------------------------------------------------------

def parse_cluster_label(response_text: str) -> dict | None:
    """Parse JSON response for cluster labeling (find JSON boundaries)."""
    text = response_text.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [line for line in lines if not line.strip().startswith("```")]
        text = "\n".join(lines).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        return None

    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None

    required = {"label", "coherent", "rationale"}
    if not required.issubset(parsed.keys()):
        return None

    return {
        "label": parsed["label"],
        "coherent": bool(parsed["coherent"]),
        "rationale": parsed["rationale"],
    }


def label_clusters(
    cluster_ids: np.ndarray,
    unique_labels: list[str],
    delay: float,
    field: str,
) -> dict[int, dict]:
    """Label each cluster by calling Haiku with the prompt template.

    Caches results to data/cluster_labels_{field}.json so labeling can
    resume if interrupted.

    Returns:
        Dict mapping cluster_id -> {label, coherent, rationale}.
    """
    system_prompt = load_prompt(PROMPTS_DIR / "system.md")
    prompt_template = load_prompt(PROMPTS_DIR / "label_clusters.md")

    # Group unique-label indices by cluster
    cluster_to_labels: dict[int, list[str]] = {}
    for label_idx, cid in enumerate(cluster_ids):
        cid = int(cid)
        if cid == -1:
            continue
        if cid not in cluster_to_labels:
            cluster_to_labels[cid] = []
        label = unique_labels[label_idx]
        if label not in cluster_to_labels[cid]:
            cluster_to_labels[cid].append(label)

    # Load cached labels if available
    cache_path = DATA_DIR / f"cluster_labels_{field}.json"
    cluster_info: dict[int, dict] = {}
    if cache_path.exists():
        cached = json.loads(cache_path.read_text())
        # Keys are strings in JSON, convert to int
        cluster_info = {int(k): v for k, v in cached.items()}
        print(f"  Loaded {len(cluster_info)} cached cluster labels")

    sorted_ids = sorted(cluster_to_labels.keys())
    to_label = [cid for cid in sorted_ids if cid not in cluster_info]

    if to_label:
        print(f"  {len(to_label)} clusters to label ({len(sorted_ids) - len(to_label)} cached)")
    else:
        print(f"  All {len(sorted_ids)} clusters already labeled (cached)")
        return cluster_info

    for i, cid in enumerate(to_label):
        members = cluster_to_labels[cid]
        members_text = "\n".join(members)
        user_prompt = prompt_template.replace("{{cluster_members}}", members_text)

        print(f"  Labeling cluster {cid} ({len(members)} unique labels)...")
        response = call_model(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model="claude-haiku-4-5-20251001",
        )

        parsed = parse_cluster_label(response.text)
        if parsed is None:
            print(f"    WARNING: Failed to parse response for cluster {cid}")
            parsed = {
                "label": f"cluster_{cid}",
                "coherent": False,
                "rationale": "Failed to parse LLM response.",
            }

        cluster_info[cid] = parsed

        # Save cache after each label
        cache_path.write_text(json.dumps(cluster_info, ensure_ascii=False, indent=2))

        # Delay between calls (skip after last)
        if i < len(to_label) - 1:
            time.sleep(delay)

    return cluster_info


# ---------------------------------------------------------------------------
# 6. Write output
# ---------------------------------------------------------------------------

def write_output(
    records: list[dict],
    field: str,
    unique_labels: list[str],
    label_to_indices: dict[str, list[int]],
    cluster_ids: np.ndarray,
    cluster_info: dict[int, dict],
):
    """Write per-record JSONL and cluster summary JSONL."""
    # Build mapping: unique label -> cluster_id
    label_to_cluster: dict[str, int] = {}
    for label_idx, cid in enumerate(cluster_ids):
        label_to_cluster[unique_labels[label_idx]] = int(cid)

    # Assign cluster info to each record
    record_cluster_ids = [None] * len(records)
    for label, indices in label_to_indices.items():
        cid = label_to_cluster.get(label, -1)
        for idx in indices:
            record_cluster_ids[idx] = cid

    # Write per-record output
    output_path = DATA_DIR / f"clusters_{field}.jsonl"
    with open(output_path, "w") as f:
        for idx, record in enumerate(records):
            cid = record_cluster_ids[idx]
            if cid is None:
                cid = -1
            out = dict(record)
            out["cluster_id"] = cid
            if cid != -1 and cid in cluster_info:
                out["cluster_label"] = cluster_info[cid]["label"]
                out["cluster_coherent"] = cluster_info[cid]["coherent"]
            else:
                out["cluster_label"] = None
                out["cluster_coherent"] = None
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(records)} records to {output_path.name}")

    # Build cluster summary
    summary_path = DATA_DIR / f"cluster_summary_{field}.jsonl"
    with open(summary_path, "w") as f:
        for cid in sorted(cluster_info.keys()):
            info = cluster_info[cid]

            # Count records per label within this cluster
            label_counts: Counter = Counter()
            for label_idx, c in enumerate(cluster_ids):
                if int(c) == cid:
                    lbl = unique_labels[label_idx]
                    label_counts[lbl] = len(label_to_indices[lbl])

            total_count = sum(label_counts.values())

            # Top 5 unique labels by frequency
            top_5 = label_counts.most_common(5)
            top_members = [
                {"label": lbl, "share": round(cnt / total_count, 2)}
                for lbl, cnt in top_5
            ]

            summary = {
                "cluster_id": cid,
                "label": info["label"],
                "coherent": info["coherent"],
                "rationale": info["rationale"],
                "count": total_count,
                "top_members": top_members,
            }
            f.write(json.dumps(summary, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(cluster_info)} cluster summaries to {summary_path.name}")

    # Print summary table to stdout
    print()
    print(f"{'ID':>4}  {'Label':<30}  {'Count':>5}  {'Coh':>3}  Top members")
    print("-" * 90)
    for cid in sorted(cluster_info.keys()):
        info = cluster_info[cid]

        label_counts: Counter = Counter()
        for label_idx, c in enumerate(cluster_ids):
            if int(c) == cid:
                lbl = unique_labels[label_idx]
                label_counts[lbl] = len(label_to_indices[lbl])

        total_count = sum(label_counts.values())
        top_3 = label_counts.most_common(3)
        top_str = ", ".join(f"{lbl} ({cnt})" for lbl, cnt in top_3)
        coh = "Y" if info["coherent"] else "N"

        print(f"{cid:>4}  {info['label']:<30}  {total_count:>5}  {coh:>3}  {top_str}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Embed, reduce, cluster, and label classification labels."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DATA_DIR / "classifications.jsonl",
        help="Path to input JSONL file (default: data/classifications.jsonl)",
    )
    parser.add_argument(
        "--field",
        default="object",
        help="Field to cluster on (default: object)",
    )
    parser.add_argument(
        "--cluster-dims",
        type=int,
        default=10,
        help="UMAP dimensions for clustering (default: 10)",
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=5,
        help="HDBSCAN min_cluster_size (default: 5)",
    )
    parser.add_argument(
        "--n-neighbors",
        type=int,
        default=15,
        help="UMAP n_neighbors (default: 15)",
    )
    parser.add_argument(
        "--min-dist",
        type=float,
        default=0.1,
        help="UMAP min_dist (default: 0.1)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Seconds between LLM labeling calls (default: 0.5)",
    )
    args = parser.parse_args()

    # --- 1. Extract ---
    print("1. Extracting labels...")
    records = load_jsonl(args.input)
    print(f"  Loaded {len(records)} records from {args.input.name}")

    unique_labels, label_to_indices = extract_labels(records, args.field)
    print(f"  Found {len(unique_labels)} unique '{args.field}' labels")

    if len(unique_labels) == 0:
        print(f"  No labels found for field '{args.field}'. Exiting.")
        sys.exit(1)

    # --- 2. Embed ---
    print("\n2. Embedding labels...")
    embeddings = embed_labels(unique_labels, args.field)
    print(f"  Embedding shape: {embeddings.shape}")

    # --- 3. Reduce ---
    print("\n3. Reducing dimensionality...")
    reduced_cluster, reduced_2d = reduce_dimensions(
        embeddings,
        cluster_dims=args.cluster_dims,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        field=args.field,
    )

    # --- 4. Cluster ---
    print("\n4. Clustering...")
    cluster_ids = cluster_embeddings(reduced_cluster, args.min_cluster_size)

    # --- 5. Label ---
    print("\n5. Labeling clusters...")
    cluster_info = label_clusters(cluster_ids, unique_labels, args.delay, args.field)

    # --- 6. Write output ---
    print("\n6. Writing output...")
    write_output(
        records,
        args.field,
        unique_labels,
        label_to_indices,
        cluster_ids,
        cluster_info,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
