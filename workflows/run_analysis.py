"""Analysis workflow: classify, validate, and cluster skill files.

Can run on a random subsample (pilot) or the full corpus.

Usage:
    python workflows/run_pilot.py                        # pilot: subsample n=100
    python workflows/run_pilot.py --n 200 --seed 123     # pilot: custom subsample
    python workflows/run_pilot.py --skip-subsample        # full corpus
    python workflows/run_pilot.py --skip-validation       # skip 4-model validation
"""

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from config.settings import DATA_DIR

# Always use the project venv Python, regardless of how this script was invoked
VENV_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"
if VENV_PYTHON.exists():
    PYTHON = str(VENV_PYTHON)
else:
    PYTHON = sys.executable


def subsample_corpus(input_path: Path, output_path: Path, n: int, seed: int):
    """Draw a random subsample from the corpus and write to a new JSONL file."""
    records = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(line)

    if n >= len(records):
        sample = records
        print(f"  Corpus has {len(records)} records, using all of them.")
    else:
        random.seed(seed)
        sample = random.sample(records, n)
        print(f"  Sampled {n} records from {len(records)} (seed={seed}).")

    with open(output_path, "w") as f:
        for line in sample:
            f.write(line + "\n")

    print(f"  Wrote {len(sample)} records to {output_path.name}")
    return len(sample)


def run_step(description: str, cmd: list[str]):
    """Run a subprocess, printing its output in real time."""
    print(f"\n{'=' * 60}")
    print(f"  {description}")
    print(f"{'=' * 60}\n")

    result = subprocess.run(cmd, cwd=Path(__file__).resolve().parent.parent)

    if result.returncode != 0:
        print(f"\nERROR: '{description}' failed with exit code {result.returncode}")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(description="Run pilot pipeline on a corpus subsample.")
    parser.add_argument(
        "--n", type=int, default=100,
        help="Subsample size (default: 100)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--model", type=str, default="claude-sonnet-4-6",
        help="Model for classification (default: claude-sonnet-4-6)",
    )
    parser.add_argument(
        "--validation-n", type=int, default=None,
        help="Subsample size for validation (default: same as --n)",
    )
    parser.add_argument(
        "--delay", type=float, default=0.5,
        help="Seconds between API calls (default: 0.5)",
    )
    parser.add_argument(
        "--skip-subsample", action="store_true",
        help="Skip subsampling and run on the full corpus",
    )
    parser.add_argument(
        "--skip-validation", action="store_true",
        help="Skip the validation step",
    )
    parser.add_argument(
        "--skip-clustering", action="store_true",
        help="Skip the clustering step",
    )
    parser.add_argument(
        "--tag", type=str, default=None,
        help="Version tag appended to output filenames (e.g., --tag v3 produces pilot_classifications_v3.jsonl)",
    )
    parser.add_argument(
        "--validation-models", type=str, nargs="+", default=None,
        help="Models for validation step (display names, e.g. --validation-models Sonnet GPT-5.4)",
    )
    args = parser.parse_args()

    corpus_path = DATA_DIR / "corpus.jsonl"

    if not corpus_path.exists():
        print(f"Error: Corpus not found at {corpus_path}")
        sys.exit(1)

    # Set input/output paths based on whether we're subsampling
    suffix = f"_{args.tag}" if args.tag else ""
    if args.skip_subsample:
        input_corpus = corpus_path
        classifications_path = DATA_DIR / f"classifications{suffix}.jsonl"
        validation_path = DATA_DIR / f"validation_results{suffix}.jsonl"
    else:
        input_corpus = DATA_DIR / "pilot_corpus.jsonl"
        classifications_path = DATA_DIR / f"pilot_classifications{suffix}.jsonl"
        validation_path = DATA_DIR / f"pilot_validation{suffix}.jsonl"

    # Step 1: Subsample
    if not args.skip_subsample:
        print(f"\n{'=' * 60}")
        print(f"  Step 1: Subsample corpus (n={args.n}, seed={args.seed})")
        print(f"{'=' * 60}\n")

        subsample_corpus(corpus_path, input_corpus, args.n, args.seed)
    else:
        print("\n  Skipping subsample (--skip-subsample), using full corpus")

    # Step 2: Classify
    run_step(
        f"Step 2: Classify with {args.model}",
        [
            PYTHON, "scripts/analyze_corpus.py",
            "--input", str(input_corpus),
            "--output", str(classifications_path),
            "--model", args.model,
            "--delay", str(args.delay),
        ],
    )

    # Step 3: Validate
    if not args.skip_validation:
        validation_n = args.validation_n or args.n
        if args.skip_subsample:
            report_path = DATA_DIR / f"validation_report{suffix}.txt"
        else:
            report_path = DATA_DIR / f"pilot_validation_report{suffix}.txt"
        validate_cmd = [
            PYTHON, "scripts/validate_models.py",
            "--input", str(input_corpus),
            "--output", str(validation_path),
            "--seed", str(args.seed),
            "--delay", str(args.delay),
            "--classifications", str(classifications_path),
            "--report", str(report_path),
        ]
        if not args.skip_subsample:
            validate_cmd += ["--n", str(validation_n)]
        if args.validation_models:
            validate_cmd += ["--models"] + args.validation_models
        run_step(
            f"Step 3: Validate across models (n={validation_n})",
            validate_cmd,
        )
    else:
        print("\n  Skipping validation (--skip-validation)")

    # Step 4: Cluster
    if not args.skip_clustering:
        run_step(
            "Step 4: Cluster object labels",
            [
                PYTHON, "scripts/cluster_labels.py",
                "--input", str(classifications_path),
                "--field", "object",
            ],
        )
    else:
        print("\n  Skipping clustering (--skip-clustering)")

    print(f"\n{'=' * 60}")
    print("  Pipeline complete!")
    print(f"{'=' * 60}")
    print(f"\n  Outputs:")
    if not args.skip_subsample:
        print(f"    Subsample:       {input_corpus.name}")
    print(f"    Classifications: {classifications_path.name}")
    if not args.skip_validation:
        print(f"    Validation:      {validation_path.name}")
        print(f"    Report:          {report_path.name}")
    if not args.skip_clustering:
        print(f"    Clusters:        clusters_object.jsonl")
        print(f"    Cluster summary: cluster_summary_object.jsonl")


if __name__ == "__main__":
    main()
