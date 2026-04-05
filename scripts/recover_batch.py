"""Recover results from an in-flight or completed batch.

Polls for completion, retrieves results, and writes the classifications file.
Use this when the main pipeline crashes after submitting a batch.

Usage:
    python scripts/recover_batch.py BATCH_ID --input data/pilot_corpus.jsonl --output data/pilot_classifications_sample_50.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.api_clients import poll_batch, retrieve_batch_results
from src.parsing import CLASSIFICATION_FIELDS, parse_classification


def main():
    parser = argparse.ArgumentParser(description="Recover results from a submitted batch.")
    parser.add_argument("batch_id", help="Anthropic batch ID")
    parser.add_argument("--input", type=Path, required=True, help="Input corpus JSONL (same one used for the batch)")
    parser.add_argument("--output", type=Path, required=True, help="Output classifications JSONL")
    parser.add_argument("--poll-interval", type=int, default=300, help="Seconds between polls (default: 300)")
    args = parser.parse_args()

    # Load the corpus records in order (custom_id = index)
    records = []
    with open(args.input) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    print(f"Corpus: {len(records)} records")
    print(f"Batch: {args.batch_id}")
    print(f"Polling for completion (every {args.poll_interval}s)...\n")

    status = poll_batch(args.batch_id, interval=args.poll_interval)
    print(f"\nBatch ended: {status['succeeded']} succeeded, {status['failed']} failed")

    print("Retrieving results...")
    results = retrieve_batch_results(args.batch_id)
    results_by_id = {r.custom_id: r for r in results}

    classified = 0
    errors = 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as out_f:
        for i, record in enumerate(records):
            result = results_by_id.get(str(i))
            if result is None or not result.succeeded:
                errors += 1
                continue

            classification = parse_classification(result.text)
            if classification is None:
                errors += 1
                continue

            output_record = {
                "repo": record["repo"],
                "database_id": record["database_id"],
                "path": record["path"],
                "matched_dirs": record["matched_dirs"],
                **classification,
            }
            out_f.write(json.dumps(output_record) + "\n")
            classified += 1

    print(f"\nDone. Classified {classified}, errors {errors}.")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
