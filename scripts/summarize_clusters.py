"""Summarize cluster distributions from clustered classification output.

Produces two tables:
1. Top 10 clusters by size, with top 5 object labels within each.
2. Top 20 noise labels (cluster_id = -1) by frequency.

Usage:
    python scripts/summarize_clusters.py
    python scripts/summarize_clusters.py --input data/clusters_object.jsonl
    python scripts/summarize_clusters.py --output data/cluster_distribution.txt
"""

import argparse
import json
import sys
from collections import Counter
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import DATA_DIR


def main():
    parser = argparse.ArgumentParser(description="Summarize cluster distributions.")
    parser.add_argument(
        "--input", type=Path, default=DATA_DIR / "clusters_object.jsonl",
        help="Input clustered JSONL (default: data/clusters_object.jsonl)",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Write output to file (default: print to stdout)",
    )
    args = parser.parse_args()

    records = []
    with open(args.input) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    total = len(records)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            with redirect_stdout(f):
                print_report(records, total)
        print(f"Saved to {args.output}")
    else:
        print_report(records, total)


def print_report(records: list[dict], total: int):
    # --- Table 1: Top 10 clusters with top 5 labels ---
    cluster_records: dict[int, list[dict]] = {}
    for r in records:
        cid = r.get("cluster_id", -1)
        cluster_records.setdefault(cid, []).append(r)

    # Remove noise from cluster ranking
    clusters_only = {k: v for k, v in cluster_records.items() if k != -1}
    sorted_clusters = sorted(clusters_only.items(), key=lambda x: -len(x[1]))[:10]

    print("=" * 80)
    print("TOP 10 CLUSTERS BY SIZE")
    print(f"(out of {total:,} total instructions)")
    print("=" * 80)

    for rank, (cid, recs) in enumerate(sorted_clusters, 1):
        cluster_label = recs[0].get("cluster_label", f"Cluster {cid}")
        cluster_pct = len(recs) / total * 100

        print(f"\n{rank}. {cluster_label}")
        print(f"   {len(recs):,} instructions ({cluster_pct:.1f}% of all)")
        print()

        # Top 5 labels within this cluster
        label_counts = Counter(r.get("object", "") for r in recs)
        top_5 = label_counts.most_common(5)

        print(f"   {'Label':<60} {'Count':>6} {'% cluster':>10}")
        print(f"   {'-' * 60} {'-' * 6} {'-' * 10}")
        for label, count in top_5:
            label_pct = count / len(recs) * 100
            display = label[:58] + ".." if len(label) > 60 else label
            print(f"   {display:<60} {count:>6} {label_pct:>9.1f}%")

    # --- Table 2: Top 20 noise labels ---
    noise_records = cluster_records.get(-1, [])
    noise_total = len(noise_records)

    print()
    print("=" * 80)
    print("TOP 20 NOISE LABELS (unassigned to any cluster)")
    print(f"({noise_total:,} noise instructions, {noise_total / total * 100:.1f}% of all)")
    print("=" * 80)
    print()

    noise_label_counts = Counter(r.get("object", "") for r in noise_records)
    top_20_noise = noise_label_counts.most_common(20)

    print(f"{'Label':<65} {'Count':>6} {'% of all':>9}")
    print(f"{'-' * 65} {'-' * 6} {'-' * 9}")
    for label, count in top_20_noise:
        pct = count / total * 100
        display = label[:63] + ".." if len(label) > 65 else label
        print(f"{display:<65} {count:>6} {pct:>8.2f}%")


if __name__ == "__main__":
    main()
