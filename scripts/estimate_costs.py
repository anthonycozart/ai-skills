"""Estimate API cost of classifying the SKILL.md corpus.

Calculates token counts and costs using Sonnet 4.6 Batch API with prompt
caching. Can run on the pilot subsample or the full corpus, and optionally
projects costs to different corpus sizes.

Usage:
    python scripts/estimate_costs.py
    python scripts/estimate_costs.py --input data/pilot_corpus.jsonl
    python scripts/estimate_costs.py --input data/corpus.jsonl --project-to 5000 10000
"""

import argparse
import json
import statistics
import sys
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import DATA_DIR, PROMPTS_DIR

# Sonnet 4.6 Batch API + prompt caching rates ($ per million tokens)
RATE_CACHE_WRITE = 3.75
RATE_CACHE_READ = 0.15
RATE_INPUT = 1.50
RATE_OUTPUT = 7.50

OUTPUT_TOKENS_PER_FILE = 100
CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    """Approximate token count at ~4 characters per token."""
    return len(text) // CHARS_PER_TOKEN


def load_corpus(input_path: Path) -> list[dict]:
    """Load JSONL corpus records."""
    records = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def calculate_costs(
    num_files: int,
    prompt_tokens: int,
    total_file_tokens: int,
    output_tokens_per_file: int,
) -> dict:
    """Calculate cost breakdown for a given number of files."""
    cache_write_cost = prompt_tokens * RATE_CACHE_WRITE / 1_000_000
    cache_read_cost = prompt_tokens * (num_files - 1) * RATE_CACHE_READ / 1_000_000
    input_cost = total_file_tokens * RATE_INPUT / 1_000_000
    output_cost = output_tokens_per_file * num_files * RATE_OUTPUT / 1_000_000
    total = cache_write_cost + cache_read_cost + input_cost + output_cost

    return {
        "cache_write": cache_write_cost,
        "cache_read": cache_read_cost,
        "input": input_cost,
        "output": output_cost,
        "total": total,
    }


def format_cost_table(label: str, num_files: int, total_file_tokens: int, costs: dict) -> str:
    """Format a cost breakdown as a readable table."""
    lines = [
        f"  {label} ({num_files:,} files, {total_file_tokens:,} file tokens)",
        f"  {'Component':<30} {'Cost':>10}",
        f"  {'-' * 30} {'-' * 10}",
        f"  {'Cache write (1 request)':<30} ${costs['cache_write']:>9.4f}",
        f"  {'Cache read':<30} ${costs['cache_read']:>9.4f}",
        f"  {'Non-cached input (files)':<30} ${costs['input']:>9.4f}",
        f"  {'Output':<30} ${costs['output']:>9.4f}",
        f"  {'-' * 30} {'-' * 10}",
        f"  {'TOTAL':<30} ${costs['total']:>9.4f}",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Estimate API costs for SKILL.md classification")
    parser.add_argument(
        "--input",
        type=Path,
        default=DATA_DIR / "corpus.jsonl",
        help="Path to corpus JSONL file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write output to file (default: print to stdout)",
    )
    parser.add_argument(
        "--project-to",
        type=int,
        nargs="+",
        default=None,
        help="Project costs to these corpus sizes (e.g., --project-to 5000 10000)",
    )
    args = parser.parse_args()

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        out_file = open(args.output, "w")
    else:
        out_file = None

    def run():
        _run_report(args)

    if out_file:
        with redirect_stdout(out_file):
            run()
        out_file.close()
        print(f"Saved to {args.output}")
    else:
        run()


def _run_report(args):
    # Load prompts from files at runtime
    system_prompt = (PROMPTS_DIR / "system.md").read_text()
    classify_prompt = (PROMPTS_DIR / "classify.md").read_text()
    combined_prompt = system_prompt + "\n\n" + classify_prompt
    prompt_tokens = estimate_tokens(combined_prompt)

    # Load corpus and compute file token stats
    records = load_corpus(args.input)
    num_files = len(records)
    file_token_counts = [estimate_tokens(r["content"]) for r in records]
    total_file_tokens = sum(file_token_counts)

    mean_tokens = statistics.mean(file_token_counts)
    median_tokens = statistics.median(file_token_counts)
    p95_tokens = statistics.quantiles(file_token_counts, n=20)[18]  # 95th percentile
    max_tokens = max(file_token_counts)
    min_tokens = min(file_token_counts)

    # Print corpus statistics
    print("=" * 60)
    print("SKILL.md Classification Cost Estimate")
    print("=" * 60)
    print()
    print("Corpus Statistics")
    print("-" * 40)
    print(f"  Files in corpus:        {num_files:>10,}")
    print(f"  Total file tokens:      {total_file_tokens:>10,}")
    print(f"  Mean tokens/file:       {mean_tokens:>10,.0f}")
    print(f"  Median tokens/file:     {median_tokens:>10,.0f}")
    print(f"  P95 tokens/file:        {p95_tokens:>10,.0f}")
    print(f"  Max tokens/file:        {max_tokens:>10,}")
    print(f"  Min tokens/file:        {min_tokens:>10,}")
    print()
    print("Prompt Statistics")
    print("-" * 40)
    print(f"  Cached prefix tokens:   {prompt_tokens:>10,}")
    print(f"  Output tokens/file:     {OUTPUT_TOKENS_PER_FILE:>10,}")
    print()
    print("Rates (Sonnet 4.6 Batch API + Caching)")
    print("-" * 40)
    print(f"  Cache write:            ${RATE_CACHE_WRITE}/MTok  (regular rate; applied once)")
    print(f"  Cache read:             ${RATE_CACHE_READ}/MTok  (batch rate; best-effort in batch mode)")
    print(f"  Non-cached input:       ${RATE_INPUT}/MTok  (batch rate; 50% of regular $3.00)")
    print(f"  Output:                 ${RATE_OUTPUT}/MTok  (batch rate; 50% of regular $15.00)")
    print()
    print("Notes")
    print("-" * 40)
    print("  - Token counts are approximate (~4 chars/token); actual costs")
    print("    may be 10-15% higher.")
    print("  - Prompt caching saves ~$6 on the system prompt across all files.")
    print("    The bulk of cost is file content, which is not cached.")
    print("  - Regular (non-batch) API would cost roughly 2x these estimates.")
    print()

    # Calculate costs for input corpus
    corpus_costs = calculate_costs(num_files, prompt_tokens, total_file_tokens, OUTPUT_TOKENS_PER_FILE)

    # Print cost breakdowns
    print("Cost Estimates")
    print("=" * 60)
    print()
    print(format_cost_table(f"Input corpus ({args.input.name})", num_files, total_file_tokens, corpus_costs))

    # Project to other sizes if requested
    if args.project_to:
        mean_file_tokens = total_file_tokens / num_files
        for size in args.project_to:
            projected_tokens = int(mean_file_tokens * size)
            projected_costs = calculate_costs(size, prompt_tokens, projected_tokens, OUTPUT_TOKENS_PER_FILE)
            print()
            print(format_cost_table(f"Projected", size, projected_tokens, projected_costs))

    print()


if __name__ == "__main__":
    main()

