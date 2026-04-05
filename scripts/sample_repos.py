"""Step 1: Sample repos from GHArchive via BigQuery.

Queries the GHArchive public dataset for repos with push activity in a
recent time window, ordered by a deterministic hash for pseudo-random sampling.

Usage:
    python scripts/sample_repos.py --output data/sampled_repos.csv
    python scripts/sample_repos.py --output data/sampled_repos.csv --sql src/sample_repos.sql
"""

import argparse
import csv
import sys
from pathlib import Path

from dotenv import load_dotenv
from google.cloud import bigquery

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import GCP_PROJECT_ID, DATA_DIR, SRC_DIR


def main():
    parser = argparse.ArgumentParser(description="Sample repos from GHArchive via BigQuery.")
    parser.add_argument("--sql", type=Path, default=SRC_DIR / "sample_repos.sql",
                        help="Path to SQL query file (default: src/sample_repos.sql)")
    parser.add_argument("--output", type=Path, default=DATA_DIR / "sampled_repos.csv",
                        help="Output CSV file (default: data/sampled_repos.csv)")
    args = parser.parse_args()

    query = args.sql.read_text()

    client = bigquery.Client(project=GCP_PROJECT_ID)
    print(f"Running query against project: {GCP_PROJECT_ID}")
    print(f"SQL:\n{query}\n")

    result = client.query(query).result()
    rows = [dict(row) for row in result]
    print(f"Returned {len(rows)} repos")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["repo_id"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
