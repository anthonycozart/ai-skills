"""Summarize the sampling funnel from GHArchive to final corpus.

Reads the data files from each step and prints counts and percentages.

Usage:
    python scripts/summarize_funnel.py
    python scripts/summarize_funnel.py --output data/funnel_summary.txt
"""

import argparse
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import DATA_DIR


def count_csv_rows(path: Path) -> int:
    """Count data rows in a CSV file (excludes header)."""
    with open(path) as f:
        return sum(1 for _ in f) - 1


def count_jsonl_rows(path: Path) -> int:
    """Count non-empty lines in a JSONL file."""
    with open(path) as f:
        return sum(1 for line in f if line.strip())


def count_skill_files(path: Path) -> int:
    """Sum skill_paths lengths across repos in the JSONL."""
    total = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                total += len(json.loads(line)["skill_paths"])
    return total


def print_funnel(data_dir: Path):
    """Print the sampling funnel summary."""
    sampled = data_dir / "sampled_repos.csv"
    dirs = data_dir / "repos_with_dirs.jsonl"
    skills = data_dir / "repos_with_skills.jsonl"
    corpus = data_dir / "corpus.jsonl"

    # (label, count, show_pct) — show_pct True for repo-level steps only
    steps = []

    if sampled.exists():
        n = count_csv_rows(sampled)
        steps.append(("1. Sampled repos (GHArchive)", n, True))

    if dirs.exists():
        n = count_jsonl_rows(dirs)
        steps.append(("2. Repos with AI tool dirs", n, True))

    if skills.exists():
        n_repos = count_jsonl_rows(skills)
        n_files = count_skill_files(skills)
        steps.append(("3. Repos with SKILL.md files", n_repos, True))
        steps.append(("   SKILL.md files found", n_files, False))

    if corpus.exists():
        n = count_jsonl_rows(corpus)
        steps.append(("4. Corpus (fetched files)", n, False))

    if not steps:
        print("No data files found.")
        return

    base = steps[0][1]

    print("=" * 55)
    print("Sampling Funnel")
    print("=" * 55)
    print()
    print(f"  {'Step':<35} {'Count':>8}  {'%':>7}")
    print(f"  {'-' * 35} {'-' * 8}  {'-' * 7}")

    for label, count, show_pct in steps:
        if show_pct:
            pct = count / base * 100
            print(f"  {label:<35} {count:>8,}  {pct:>6.1f}%")
        else:
            print(f"  {label:<35} {count:>8,}")

    print()


def main():
    parser = argparse.ArgumentParser(description="Summarize the sampling funnel.")
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Write output to file (default: print to stdout)",
    )
    args = parser.parse_args()

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            with redirect_stdout(f):
                print_funnel(DATA_DIR)
        print(f"Saved to {args.output}")
    else:
        print_funnel(DATA_DIR)


if __name__ == "__main__":
    main()
