"""Step 4: Fetch SKILL.md file contents from GitHub.

Uses the GraphQL API to batch-fetch all SKILL.md files per repo in a single
query (one GraphQL point per repo). Saves results as NDJSON (one record per file).

Automatically resumes from previous progress. Use --fresh to start over.

Usage:
    python scripts/fetch_content.py
    python scripts/fetch_content.py --fresh
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import DATA_DIR

GRAPHQL_URL = "https://api.github.com/graphql"


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


def build_query(owner: str, repo_name: str, paths: list[str]) -> str:
    """Build a GraphQL query that fetches all files and repo metadata."""
    fields = []
    for i, path in enumerate(paths):
        escaped = path.replace("\\", "\\\\").replace('"', '\\"')
        fields.append(
            f'  file{i}: object(expression: "HEAD:{escaped}") {{\n'
            f'    ... on Blob {{ text byteSize }}\n'
            f'  }}'
        )
    joined = "\n".join(fields)
    escaped_owner = owner.replace("\\", "\\\\").replace('"', '\\"')
    escaped_repo = repo_name.replace("\\", "\\\\").replace('"', '\\"')
    return (
        f'query {{\n'
        f'  repository(owner: "{escaped_owner}", name: "{escaped_repo}") {{\n'
        f'    defaultBranchRef {{\n'
        f'      target {{\n'
        f'        ... on Commit {{\n'
        f'          history {{ totalCount }}\n'
        f'        }}\n'
        f'      }}\n'
        f'    }}\n'
        f'    mentionableUsers {{ totalCount }}\n'
        f'{joined}\n'
        f'  }}\n'
        f'}}'
    )


@dataclass
class RepoResult:
    """Results from a single repo GraphQL query."""
    files: dict[int, tuple[str, int]]  # index -> (content, byte_size)
    commit_count: int | None
    contributor_count: int | None


def fetch_repo_files(owner: str, repo_name: str, paths: list[str],
                     token: str, max_retries: int = 3) -> RepoResult:
    """Fetch all SKILL.md files and repo metadata in one GraphQL call.

    Returns:
        RepoResult with file contents and enrichment metadata.
    """
    query = build_query(owner, repo_name, paths)
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    for attempt in range(max_retries):
        try:
            resp = requests.post(GRAPHQL_URL, headers=headers,
                                 json={"query": query}, timeout=30)
        except (requests.ConnectionError, requests.Timeout, requests.exceptions.ChunkedEncodingError) as e:
            wait = min(2 ** attempt, 60)
            print(f"  Connection error: {e}. Retry {attempt+1}/{max_retries} in {wait}s...")
            time.sleep(wait)
            continue

        if handle_rate_limit(resp):
            backoff = 2 ** attempt
            time.sleep(backoff)
            continue

        if resp.status_code != 200:
            resp.raise_for_status()

        data = resp.json()

        if "errors" in data and not data.get("data"):
            print(f"  GraphQL errors: {data['errors']}")
            return RepoResult(files={}, commit_count=None, contributor_count=None)

        repo_data = data.get("data", {}).get("repository")
        if repo_data is None:
            return RepoResult(files={}, commit_count=None, contributor_count=None)

        # Extract file contents
        files = {}
        for i in range(len(paths)):
            file_data = repo_data.get(f"file{i}")
            if file_data and "text" in file_data:
                files[i] = (file_data["text"], file_data["byteSize"])

        # Extract enrichment metadata
        branch_ref = repo_data.get("defaultBranchRef")
        if branch_ref and branch_ref.get("target"):
            commit_count = branch_ref["target"].get("history", {}).get("totalCount")
        else:
            commit_count = None

        contributor_count = repo_data.get("mentionableUsers", {}).get("totalCount")

        return RepoResult(
            files=files,
            commit_count=commit_count,
            contributor_count=contributor_count,
        )

    return RepoResult(files={}, commit_count=None, contributor_count=None)


def main():
    parser = argparse.ArgumentParser(description="Fetch SKILL.md file contents from GitHub.")
    parser.add_argument("--input", type=Path, default=DATA_DIR / "repos_with_skills.jsonl",
                        help="JSONL from Step 3 (default: data/repos_with_skills.jsonl)")
    parser.add_argument("--output", type=Path, default=DATA_DIR / "corpus.jsonl",
                        help="Output NDJSON corpus (default: data/corpus.jsonl)")
    parser.add_argument("--delay", type=float, default=0.0,
                        help="Seconds between requests (default: 0.0)")
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

    total_files = sum(len(r["skill_paths"]) for r in repos)
    print(f"Loaded {len(repos)} repos with {total_files} SKILL.md files to fetch ({total_in_input} total in input)")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fetched = 0
    skipped = 0

    with open(args.output, "a") as f, open(progress_file, "a") as pf:
        for repo in repos:
            name_with_owner = repo["nameWithOwner"]
            owner, repo_name = name_with_owner.split("/", 1)
            paths = repo["skill_paths"]

            result = fetch_repo_files(owner, repo_name, paths, token)

            for i, path in enumerate(paths):
                if i not in result.files:
                    print(f"  SKIP {name_with_owner}/{path} (not found or binary)")
                    skipped += 1
                    continue

                content, byte_size = result.files[i]
                record = {
                    "repo": name_with_owner,
                    "database_id": repo["databaseId"],
                    "path": path,
                    "matched_dirs": repo["matched_dirs"],
                    "content": content,
                    "byte_size": byte_size,
                    "commit_count": result.commit_count,
                    "contributor_count": result.contributor_count,
                }
                f.write(json.dumps(record) + "\n")
                fetched += 1

                print(f"  [{fetched}/{total_files}] {name_with_owner}/{path} ({byte_size} bytes)")

            pf.write(f"{repo['databaseId']}\n")
            pf.flush()

            time.sleep(args.delay)

    total_processed = len(already_processed) + len(repos)
    print(f"\nDone. Fetched {fetched} files, skipped {skipped} ({total_processed}/{total_in_input} repos processed).")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
