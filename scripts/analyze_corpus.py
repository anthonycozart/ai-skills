"""Classify SKILL.md files using an LLM.

Reads the corpus JSONL, sends each record's content through the classification
prompt, and writes results with the original metadata plus classification fields.

Supports regular API calls (--mode regular) or the Anthropic Batch API
(--mode batch) for 50% cost savings on large runs.

Usage:
    python scripts/analyze_corpus.py
    python scripts/analyze_corpus.py --mode batch
    python scripts/analyze_corpus.py --model gemini-2.5-flash --limit 10
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import DATA_DIR, PROMPTS_DIR
from src.api_clients import (
    ANTHROPIC_MODELS,
    call_model,
    poll_batch,
    retrieve_batch_results,
    submit_batch,
)
from src.parsing import CLASSIFICATION_FIELDS, load_prompt, parse_classification


def load_already_classified(output_path: Path) -> set[tuple[str, str]]:
    """Load (repo, path) pairs from existing output for resume support.

    Only counts records that have the current CLASSIFICATION_FIELDS,
    so a prompt change invalidates stale results.
    """
    seen = set()
    if not output_path.exists():
        return seen
    with open(output_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if all(field in record for field in CLASSIFICATION_FIELDS):
                seen.add((record["repo"], record["path"]))
    return seen


def main():
    parser = argparse.ArgumentParser(
        description="Classify SKILL.md files using an LLM."
    )
    parser.add_argument(
        "--input", type=Path, default=DATA_DIR / "corpus.jsonl",
        help="Input corpus NDJSON (default: data/corpus.jsonl)",
    )
    parser.add_argument(
        "--output", type=Path, default=DATA_DIR / "classifications.jsonl",
        help="Output classifications NDJSON (default: data/classifications.jsonl)",
    )
    parser.add_argument(
        "--model", type=str, default="claude-sonnet-4-6",
        help="Model to use (default: claude-sonnet-4-6)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit to first N records (for testing)",
    )
    parser.add_argument(
        "--max-file-tokens", type=int, default=5000,
        help="Max tokens per file content, truncated if longer (default: 5000)",
    )
    parser.add_argument(
        "--mode", type=str, choices=["regular", "batch"], default="regular",
        help="API mode: 'regular' for sequential calls, 'batch' for Anthropic Batch API (default: regular)",
    )
    parser.add_argument(
        "--delay", type=float, default=0.5,
        help="Seconds between API calls in regular mode (default: 0.5)",
    )
    parser.add_argument(
        "--poll-interval", type=int, default=300,
        help="Seconds between status checks in batch mode (default: 300)",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    # Load prompts from files
    system_prompt = load_prompt(PROMPTS_DIR / "system.md")
    classify_template = load_prompt(PROMPTS_DIR / "classify.md")

    # Load corpus
    records = []
    with open(args.input) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if args.limit:
        records = records[:args.limit]

    # Resume support: skip already-classified records
    already_done = load_already_classified(args.output)
    to_process = [
        r for r in records
        if (r["repo"], r["path"]) not in already_done
    ]

    total = len(records)
    skipped = len(already_done)
    remaining = len(to_process)

    print(f"Corpus: {total} records")
    if skipped:
        print(f"Already classified: {skipped} (resuming)")
    print(f"To process: {remaining}")
    print(f"Model: {args.model}")
    print()

    if remaining == 0:
        print("Nothing to do.")
        return

    if args.mode == "batch" and args.model not in ANTHROPIC_MODELS:
        print(f"Error: Batch mode only supports Anthropic models, got: {args.model}")
        sys.exit(1)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Prepare prompts for all records (shared by both modes)
    def build_user_prompt(record):
        content = record["content"]
        if args.max_file_tokens:
            max_chars = args.max_file_tokens * 4
            if len(content) > max_chars:
                content = content[:max_chars]
        return classify_template.replace("{{content}}", content)

    if args.mode == "batch":
        _run_batch(args, to_process, system_prompt, build_user_prompt, remaining)
    else:
        _run_regular(args, to_process, system_prompt, build_user_prompt, remaining)


def _run_regular(args, to_process, system_prompt, build_user_prompt, remaining):
    """Classify records one at a time using the regular API."""
    use_cache = args.model in ANTHROPIC_MODELS

    total_input_tokens = 0
    total_output_tokens = 0
    classified = 0
    errors = 0

    with open(args.output, "a") as out_f:
        for i, record in enumerate(to_process):
            repo = record["repo"]
            path = record["path"]
            user_prompt = build_user_prompt(record)

            try:
                response = call_model(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    model=args.model,
                    cache_system_prompt=use_cache,
                )
            except Exception as e:
                print(f"  ERROR [{i+1}/{remaining}] {repo}/{path}: {e}")
                errors += 1
                time.sleep(args.delay)
                continue

            total_input_tokens += response.input_tokens
            total_output_tokens += response.output_tokens

            classification = parse_classification(response.text)
            if classification is None:
                print(
                    f"  PARSE ERROR [{i+1}/{remaining}] {repo}/{path}: "
                    f"could not parse response"
                )
                errors += 1
                time.sleep(args.delay)
                continue

            output_record = {
                "repo": repo,
                "database_id": record["database_id"],
                "path": path,
                "matched_dirs": record["matched_dirs"],
                **classification,
            }
            out_f.write(json.dumps(output_record) + "\n")
            out_f.flush()
            classified += 1

            print(
                f"  [{i+1}/{remaining}] {repo}/{path} -> "
                f"{classification['object']} "
                f"(in={response.input_tokens}, out={response.output_tokens})"
            )

            if i < remaining - 1:
                time.sleep(args.delay)

    print(f"\nDone. Classified {classified}, errors {errors}.")
    print(
        f"Tokens used: {total_input_tokens:,} input, "
        f"{total_output_tokens:,} output"
    )
    print(f"Saved to {args.output}")


def _run_batch(args, to_process, system_prompt, build_user_prompt, remaining):
    """Classify records using the Anthropic Batch API (50% cheaper)."""
    # Build batch requests, using index as custom_id
    print(f"Preparing {remaining} batch requests...")
    batch_requests = []
    for i, record in enumerate(to_process):
        batch_requests.append({
            "custom_id": str(i),
            "user_prompt": build_user_prompt(record),
        })

    # Submit
    print(f"Submitting batch to {args.model}...")
    batch_id = submit_batch(
        batch_requests,
        model=args.model,
        system_prompt=system_prompt,
    )
    print(f"Batch submitted: {batch_id}")

    # Poll
    print(f"\nPolling for completion (every {args.poll_interval}s)...")
    status = poll_batch(batch_id, interval=args.poll_interval)
    print(f"\nBatch ended: {status['succeeded']} succeeded, {status['failed']} failed")

    # Retrieve and write results
    print("Retrieving results...")
    results = retrieve_batch_results(batch_id)

    # Index results by custom_id
    results_by_id = {r.custom_id: r for r in results}

    total_input_tokens = 0
    total_output_tokens = 0
    classified = 0
    errors = 0

    with open(args.output, "a") as out_f:
        for i, record in enumerate(to_process):
            repo = record["repo"]
            path = record["path"]
            result = results_by_id.get(str(i))

            if result is None or not result.succeeded:
                error_msg = result.error if result else "no result returned"
                print(f"  ERROR {repo}/{path}: {error_msg}")
                errors += 1
                continue

            total_input_tokens += result.input_tokens
            total_output_tokens += result.output_tokens

            classification = parse_classification(result.text)
            if classification is None:
                print(f"  PARSE ERROR {repo}/{path}: could not parse response")
                errors += 1
                continue

            output_record = {
                "repo": repo,
                "database_id": record["database_id"],
                "path": path,
                "matched_dirs": record["matched_dirs"],
                **classification,
            }
            out_f.write(json.dumps(output_record) + "\n")
            classified += 1

    print(f"\nDone. Classified {classified}, errors {errors}.")
    print(
        f"Tokens used: {total_input_tokens:,} input, "
        f"{total_output_tokens:,} output"
    )
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
