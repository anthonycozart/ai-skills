"""Step 2: Check which sampled repos have AI tool directories (.claude, .cursor, .codex, .agents).

Uses batched GitHub GraphQL queries to check for root-level directories.
Repos with at least one match are saved for Step 3.

Automatically resumes from previous progress. Use --fresh to start over.

Usage:
    python scripts/check_dirs.py
    python scripts/check_dirs.py --batch-size 10
    python scripts/check_dirs.py --fresh
"""

import argparse
import base64
import csv
import json
import os
import struct
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import DATA_DIR

GRAPHQL_URL = "https://api.github.com/graphql"
TARGET_DIRS = [".claude", ".cursor", ".codex"]


def encode_repo_id_legacy(numeric_id: int) -> str:
    """Convert using the legacy format: base64('010:Repository{id}')."""
    raw = f"010:Repository{numeric_id}"
    return base64.b64encode(raw.encode()).decode()


def encode_repo_id_new(numeric_id: int) -> str:
    """Convert using the newer format: 'R_kgDO' + base64(id as 32-bit big-endian)."""
    packed = struct.pack(">I", numeric_id)
    return "R_kgDO" + base64.b64encode(packed).decode()


def build_directory_check_query(repo_ids: list[tuple[int, str]]) -> str:
    """Build a batched GraphQL query to check for AI tool directories.

    Args:
        repo_ids: list of (numeric_id, encoded_node_id) tuples.
    """
    fragments = []
    for i, (_, node_id) in enumerate(repo_ids):
        dir_checks = "\n      ".join(
            f'dir_{d.lstrip(".")}: object(expression: "HEAD:{d}") {{ __typename }}'
            for d in TARGET_DIRS
        )
        fragments.append(
            f'  repo_{i}: node(id: "{node_id}") {{\n'
            f"    ... on Repository {{\n"
            f"      nameWithOwner\n"
            f"      databaseId\n"
            f"      stargazerCount\n"
            f"      forkCount\n"
            f"      primaryLanguage {{ name }}\n"
            f"      pushedAt\n"
            f"      createdAt\n"
            f"      isArchived\n"
            f"      isFork\n"
            f"      {dir_checks}\n"
            f"    }}\n"
            f"  }}"
        )
    return "query {\n" + "\n".join(fragments) + "\n}"


def parse_response(data: dict) -> list[dict]:
    """Extract repos that have at least one AI tool directory."""
    results = []
    for key, repo in data.get("data", {}).items():
        if repo is None:
            continue

        matched_dirs = []
        for d in TARGET_DIRS:
            field = f"dir_{d.lstrip('.')}"
            if repo.get(field) is not None:
                matched_dirs.append(d)

        if matched_dirs and not repo.get("isFork"):
            primary_lang = repo.get("primaryLanguage")
            results.append({
                "nameWithOwner": repo["nameWithOwner"],
                "databaseId": repo["databaseId"],
                "stargazerCount": repo.get("stargazerCount"),
                "forkCount": repo.get("forkCount"),
                "primaryLanguage": primary_lang["name"] if primary_lang else None,
                "pushedAt": repo.get("pushedAt"),
                "createdAt": repo.get("createdAt"),
                "isArchived": repo.get("isArchived"),
                "isFork": repo.get("isFork"),
                "matched_dirs": matched_dirs,
            })
    return results


def run_query(query: str, token: str, max_retries: int = 6) -> dict:
    """Execute a GraphQL query with retry logic."""
    headers = {"Authorization": f"Bearer {token}"}

    for attempt in range(max_retries):
        try:
            resp = requests.post(GRAPHQL_URL, json={"query": query}, headers=headers,
                                 timeout=30)
        except (requests.ConnectionError, requests.Timeout) as e:
            wait = min(2 ** attempt, 60)
            print(f"  Connection error: {e}. Retry {attempt+1}/{max_retries} in {wait}s...")
            time.sleep(wait)
            continue

        if resp.status_code == 200:
            return resp.json()

        # Rate limit: sleep until reset
        if resp.status_code == 403:
            retry_after = resp.headers.get("retry-after")
            if retry_after:
                wait = int(retry_after) + 1
                print(f"  Secondary rate limit hit. Sleeping {wait}s...")
                time.sleep(wait)
                continue

            reset_time = resp.headers.get("X-RateLimit-Reset")
            if reset_time:
                wait = max(int(reset_time) - int(time.time()), 1) + 1
                print(f"  Rate limit exceeded. Sleeping {wait}s...")
                time.sleep(wait)
                continue

        # Server error: short backoff
        if resp.status_code in (502, 503):
            wait = min(2 ** attempt, 60)
            print(f"  Server error ({resp.status_code}). Retry {attempt+1}/{max_retries} in {wait}s...")
            time.sleep(wait)
            continue

        # Unexpected error: exponential backoff
        wait = min(2 ** attempt, 60)
        print(f"  HTTP {resp.status_code}. Retry {attempt+1}/{max_retries} in {wait}s...")
        time.sleep(wait)

    resp.raise_for_status()
    return {}


def main():
    parser = argparse.ArgumentParser(description="Check repos for AI tool directories via GitHub GraphQL.")
    parser.add_argument("--input", type=Path, default=DATA_DIR / "sampled_repos.csv",
                        help="CSV file with repo_id column (default: data/sampled_repos.csv)")
    parser.add_argument("--output", type=Path, default=DATA_DIR / "repos_with_dirs.jsonl",
                        help="Output JSONL file (default: data/repos_with_dirs.jsonl)")
    parser.add_argument("--batch-size", type=int, default=25,
                        help="Repos per GraphQL query (default: 25)")
    parser.add_argument("--delay", type=float, default=0.0,
                        help="Seconds between requests (default: 0.0)")
    parser.add_argument("--fresh", action="store_true",
                        help="Start fresh, ignoring existing progress")
    args = parser.parse_args()

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("Error: GITHUB_TOKEN not set in environment or .env")
        sys.exit(1)

    with open(args.input) as f:
        reader = csv.DictReader(f)
        all_repo_ids = [int(row["repo_id"]) for row in reader]

    total_in_sample = len(all_repo_ids)

    # Always resume unless --fresh is passed
    progress_file = args.output.with_suffix(".progress")
    already_processed = set()
    if not args.fresh and progress_file.exists():
        with open(progress_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    already_processed.add(int(line))
        repo_ids = [rid for rid in all_repo_ids if rid not in already_processed]
        print(f"Resuming: {len(already_processed)} already processed, {len(repo_ids)} remaining")
    else:
        repo_ids = all_repo_ids
        if args.fresh:
            # Clear progress and output for a true fresh start
            if progress_file.exists():
                progress_file.unlink()
            if args.output.exists():
                args.output.unlink()
            print("Starting fresh (cleared progress and output)")

    print(f"Loaded {len(repo_ids)} repo IDs to check ({total_in_sample} total in sample)")

    seen_ids = set(already_processed)

    def run_pass(ids_with_encoded, pass_name, output_file):
        """Run batched GraphQL queries, return (hits, null_repo_ids)."""
        batches = [ids_with_encoded[i:i + args.batch_size]
                   for i in range(0, len(ids_with_encoded), args.batch_size)]
        total_hits = 0
        null_repo_ids = []

        with open(output_file, "a") as f, open(progress_file, "a") as pf:
            for i, batch in enumerate(batches):
                query = build_directory_check_query(batch)
                result = run_query(query, token)

                if "errors" in result:
                    for err in result["errors"]:
                        # Extract which repo index had the error
                        path = err.get("path", [])
                        if path:
                            idx_str = path[0].replace("repo_", "")
                            if idx_str.isdigit():
                                idx = int(idx_str)
                                if idx < len(batch):
                                    null_repo_ids.append(batch[idx][0])

                # Also catch null data responses (deleted repos)
                for key, val in result.get("data", {}).items():
                    if val is None:
                        idx_str = key.replace("repo_", "")
                        if idx_str.isdigit():
                            idx = int(idx_str)
                            if idx < len(batch) and batch[idx][0] not in null_repo_ids:
                                null_repo_ids.append(batch[idx][0])

                hits = parse_response(result)
                new_hits = 0

                for hit in hits:
                    if hit["databaseId"] not in seen_ids:
                        seen_ids.add(hit["databaseId"])
                        f.write(json.dumps(hit) + "\n")
                        new_hits += 1

                # Record all checked repo IDs for resume
                for repo_id, _ in batch:
                    pf.write(f"{repo_id}\n")
                pf.flush()

                total_hits += new_hits

                print(f"  {pass_name} batch {i+1}/{len(batches)}: "
                      f"{new_hits} hits (running total: {total_hits})")

                if i < len(batches) - 1:
                    time.sleep(args.delay)

        return total_hits, null_repo_ids

    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Pass 1: legacy encoding
    legacy_encoded = [(rid, encode_repo_id_legacy(rid)) for rid in repo_ids]
    print(f"\nPass 1: legacy encoding ({len(repo_ids)} repos)")
    hits_1, failed_ids = run_pass(legacy_encoded, "Legacy", args.output)

    # Pass 2: retry failed IDs with new encoding
    hits_2 = 0
    still_failed = 0
    if failed_ids:
        new_encoded = [(rid, encode_repo_id_new(rid)) for rid in failed_ids]
        print(f"\nPass 2: new encoding ({len(failed_ids)} repos that failed legacy)")
        hits_2, still_failed_ids = run_pass(new_encoded, "New", args.output)
        still_failed = len(still_failed_ids)

    new_hits = hits_1 + hits_2
    total_processed = len(already_processed) + len(repo_ids)
    print(f"\nDone. {new_hits} new hits this run ({total_processed}/{total_in_sample} repos processed).")
    print(f"  Legacy encoding: {len(repo_ids) - len(failed_ids)} resolved, {hits_1} hits")
    if failed_ids:
        print(f"  New encoding: {len(failed_ids) - still_failed} resolved, {hits_2} hits")
    if still_failed:
        print(f"  {still_failed} repos could not be resolved (likely deleted)")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
