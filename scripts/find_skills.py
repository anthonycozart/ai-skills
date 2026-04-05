"""Step 3: Find SKILL.md file paths in repos that have AI tool directories.

Uses the GitHub REST Trees API to fetch recursive file trees and filter
for files named SKILL.md (case-insensitive).

Automatically resumes from previous progress. Use --fresh to start over.

Usage:
    python scripts/find_skills.py
    python scripts/find_skills.py --fresh
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import DATA_DIR

REST_BASE = "https://api.github.com"


def handle_rate_limit(resp: requests.Response):
    """Sleep if rate-limited, return True if the request should be retried."""
    if resp.status_code == 403:
        retry_after = resp.headers.get("retry-after")
        if retry_after:
            wait = int(retry_after) + 1
            print(f"  Secondary rate limit. Sleeping {wait}s...")
            time.sleep(wait)
            return True

        remaining = resp.headers.get("X-RateLimit-Remaining")
        if remaining == "0":
            reset_time = int(resp.headers.get("X-RateLimit-Reset", time.time()))
            wait = max(reset_time - int(time.time()), 1) + 1
            print(f"  Rate limit exceeded. Sleeping {wait}s...")
            time.sleep(wait)
            return True

    if resp.status_code in (502, 503):
        return True

    return False


def find_skill_files(owner: str, repo: str, token: str, max_retries: int = 3) -> tuple[list[str], bool]:
    """Fetch the recursive tree and return paths of all SKILL.md files.

    Returns:
        (list of SKILL.md paths, whether tree was truncated)
    """
    url = f"{REST_BASE}/repos/{owner}/{repo}/git/trees/HEAD?recursive=1"
    headers = {"Authorization": f"Bearer {token}"}

    for attempt in range(max_retries):
        try:
            resp = requests.get(url, headers=headers, timeout=30)
        except (requests.ConnectionError, requests.Timeout, requests.exceptions.ChunkedEncodingError) as e:
            wait = min(2 ** attempt, 60)
            print(f"  Connection error: {e}. Retry {attempt+1}/{max_retries} in {wait}s...")
            time.sleep(wait)
            continue

        if resp.status_code == 200:
            tree = resp.json()
            truncated = tree.get("truncated", False)
            paths = [
                entry["path"]
                for entry in tree.get("tree", [])
                if entry["type"] == "blob"
                and entry["path"].lower().endswith("skill.md")
            ]
            return paths, truncated

        if resp.status_code == 404:
            return [], False

        if handle_rate_limit(resp):
            backoff = 2 ** attempt
            time.sleep(backoff)
            continue

        resp.raise_for_status()

    return [], False


def main():
    parser = argparse.ArgumentParser(description="Find SKILL.md files in repos with AI tool directories.")
    parser.add_argument("--input", type=Path, default=DATA_DIR / "repos_with_dirs.jsonl",
                        help="JSONL from Step 2 (default: data/repos_with_dirs.jsonl)")
    parser.add_argument("--output", type=Path, default=DATA_DIR / "repos_with_skills.jsonl",
                        help="Output JSONL (default: data/repos_with_skills.jsonl)")
    parser.add_argument("--delay", type=float, default=0.1,
                        help="Seconds between requests (default: 0.1)")
    parser.add_argument("--fresh", action="store_true",
                        help="Start fresh, ignoring existing progress")
    args = parser.parse_args()

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("Error: GITHUB_TOKEN not set in environment or .env")
        sys.exit(1)

    all_repos = []
    with open(args.input) as f:
        for line in f:
            all_repos.append(json.loads(line))

    total_in_input = len(all_repos)

    # Always resume unless --fresh is passed
    progress_file = args.output.with_suffix(".progress")
    already_processed = set()
    if not args.fresh and progress_file.exists():
        with open(progress_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    already_processed.add(int(line))
        repos = [r for r in all_repos if r["databaseId"] not in already_processed]
        print(f"Resuming: {len(already_processed)} already processed, {len(repos)} remaining")
    else:
        repos = all_repos
        if args.fresh:
            if progress_file.exists():
                progress_file.unlink()
            if args.output.exists():
                args.output.unlink()
            print("Starting fresh (cleared progress and output)")

    print(f"Loaded {len(repos)} repos to check ({total_in_input} total in input)")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    total_skills = 0
    truncated_repos = []

    with open(args.output, "a") as f, open(progress_file, "a") as pf:
        for i, repo in enumerate(repos):
            name_with_owner = repo["nameWithOwner"]
            owner, repo_name = name_with_owner.split("/", 1)

            paths, truncated = find_skill_files(owner, repo_name, token)

            if truncated:
                truncated_repos.append(name_with_owner)

            if paths:
                record = {
                    "nameWithOwner": name_with_owner,
                    "databaseId": repo["databaseId"],
                    "matched_dirs": repo["matched_dirs"],
                    "skill_paths": paths,
                }
                f.write(json.dumps(record) + "\n")
                total_skills += len(paths)

            pf.write(f"{repo['databaseId']}\n")
            pf.flush()

            print(f"  [{i+1}/{len(repos)}] {name_with_owner}: {len(paths)} SKILL.md files")

            if i < len(repos) - 1:
                time.sleep(args.delay)

    total_processed = len(already_processed) + len(repos)
    print(f"\nDone. Found {total_skills} SKILL.md files across repos ({total_processed}/{total_in_input} processed).")
    if truncated_repos:
        print(f"  {len(truncated_repos)} repos had truncated trees: {truncated_repos[:5]}")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
