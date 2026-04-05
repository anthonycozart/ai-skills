"""Inter-model agreement validation for SKILL.md classification.

Sends a random subsample of the corpus to 4 models and computes pairwise
agreement rates and Cohen's kappa to assess classification reliability.

Usage:
    python scripts/validate_models.py
    python scripts/validate_models.py --n 20 --seed 42
    python scripts/validate_models.py --input data/corpus.jsonl --output data/validation_results.jsonl
"""

import argparse
import json
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import redirect_stdout
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import DATA_DIR, PROMPTS_DIR
from src.api_clients import ANTHROPIC_MODELS, OPENAI_MODELS, call_model
from src.parsing import (
    CLASSIFICATION_FIELDS,
    FIXED_CATEGORY_FIELDS,
    load_prompt,
    parse_classification,
)

ALL_MODELS = [
    "claude-opus-4-6",
    "claude-sonnet-4-6",
    "claude-haiku-4-5-20251001",
    "gpt-5.4",
]

# Short display names for the summary table
MODEL_NAMES = {
    "claude-opus-4-6": "Opus",
    "claude-sonnet-4-6": "Sonnet",
    "claude-haiku-4-5-20251001": "Haiku",
    "gpt-5.4": "GPT-5.4",
}

# Active models for this run (may be overridden by --models flag)
MODELS = list(ALL_MODELS)


def compute_pairwise_agreement(results: list[dict]) -> dict:
    """Compute percent agreement between each model pair for each field.

    Returns:
        Dict mapping (model_a, model_b) -> {field: agreement_pct}.
    """
    agreements = {}
    for model_a, model_b in combinations(MODELS, 2):
        name_a = MODEL_NAMES[model_a]
        name_b = MODEL_NAMES[model_b]
        field_agreements = {}

        for field in FIXED_CATEGORY_FIELDS:
            matches = 0
            total = 0
            for record in results:
                val_a = record.get(name_a, {}).get(field)
                val_b = record.get(name_b, {}).get(field)
                if val_a is not None and val_b is not None:
                    total += 1
                    if val_a == val_b:
                        matches += 1
            field_agreements[field] = (matches / total * 100) if total > 0 else 0.0

        agreements[(name_a, name_b)] = field_agreements

    return agreements


def compute_cohens_kappa(results: list[dict], model_a: str, model_b: str) -> dict:
    """Compute Cohen's kappa between two models for each fixed-category field.

    Args:
        results: List of result records with per-model classifications.
        model_a: Display name of first model.
        model_b: Display name of second model.

    Returns:
        Dict mapping field -> kappa score.
    """
    from sklearn.metrics import cohen_kappa_score

    kappas = {}
    for field in FIXED_CATEGORY_FIELDS:
        labels_a = []
        labels_b = []
        for record in results:
            val_a = record.get(model_a, {}).get(field)
            val_b = record.get(model_b, {}).get(field)
            if val_a is not None and val_b is not None:
                # Convert None values (e.g. secondary_intent) to string
                labels_a.append(str(val_a))
                labels_b.append(str(val_b))

        if len(labels_a) < 2:
            kappas[field] = float("nan")
        else:
            kappas[field] = cohen_kappa_score(labels_a, labels_b)

    return kappas


def print_agreement_table(agreements: dict):
    """Print pairwise agreement percentages."""
    print("\n" + "=" * 70)
    print("PAIRWISE AGREEMENT (% exact match)")
    print("=" * 70)

    header = f"{'Model Pair':<25}"
    for field in FIXED_CATEGORY_FIELDS:
        header += f"{field:>18}"
    print(header)
    print("-" * 70)

    for (name_a, name_b), field_vals in agreements.items():
        row = f"{name_a} vs {name_b:<15}"
        for field in FIXED_CATEGORY_FIELDS:
            row += f"{field_vals[field]:>17.1f}%"
        print(row)


def print_kappa_table(kappas: dict, model_a: str, model_b: str):
    """Print Cohen's kappa scores."""
    print(f"\n{'=' * 50}")
    print(f"COHEN'S KAPPA: {model_a} vs {model_b}")
    print(f"{'=' * 50}")

    for field in FIXED_CATEGORY_FIELDS:
        k = kappas[field]
        print(f"  {field:<25} {k:>8.3f}")


def print_summary_table(results: list[dict]):
    """Print a summary table comparing all models' classification distributions."""
    print("\n" + "=" * 70)
    print("SUMMARY: Classification distributions by model")
    print("=" * 70)

    for field in FIXED_CATEGORY_FIELDS:
        print(f"\n  {field}:")
        # Collect all unique values across models
        all_values = set()
        model_counts = {}
        for model in MODELS:
            name = MODEL_NAMES[model]
            counts = {}
            for record in results:
                val = record.get(name, {}).get(field)
                if val is not None:
                    val_str = str(val)
                    all_values.add(val_str)
                    counts[val_str] = counts.get(val_str, 0) + 1
            model_counts[name] = counts

        sorted_values = sorted(all_values)
        # Header
        header = f"    {'Value':<25}"
        for model in MODELS:
            header += f"{MODEL_NAMES[model]:>10}"
        print(header)
        print("    " + "-" * (25 + 10 * len(MODELS)))

        for val in sorted_values:
            row = f"    {val:<25}"
            for model in MODELS:
                name = MODEL_NAMES[model]
                count = model_counts[name].get(val, 0)
                row += f"{count:>10}"
            print(row)


def print_discretion_by_intent(results: list[dict]):
    """Print cross-tab of Sonnet vs GPT-5.4 discretion pairings by primary_intent."""
    print("\n" + "=" * 70)
    print("DISCRETION DISAGREEMENT: Sonnet vs GPT-5.4 by primary_intent")
    print("=" * 70)

    # Collect all discretion values and intents
    discretion_vals = sorted({
        str(r.get("Sonnet", {}).get("discretion", ""))
        for r in results if r.get("Sonnet")
    } | {
        str(r.get("GPT-5.4", {}).get("discretion", ""))
        for r in results if r.get("GPT-5.4")
    } - {""})

    intent_vals = sorted({
        str(r.get("Sonnet", {}).get("primary_intent", ""))
        for r in results if r.get("Sonnet")
    } - {""})

    # For each Sonnet x GPT-5.4 discretion pairing, count by primary_intent (using Sonnet's)
    for s_disc in discretion_vals:
        for g_disc in discretion_vals:
            counts = {}
            total = 0
            for r in results:
                s = r.get("Sonnet", {})
                g = r.get("GPT-5.4", {})
                if s and g and str(s.get("discretion")) == s_disc and str(g.get("discretion")) == g_disc:
                    intent = str(s.get("primary_intent", "unknown"))
                    counts[intent] = counts.get(intent, 0) + 1
                    total += 1
            if total == 0:
                continue

            label = f"Sonnet={s_disc}, GPT-5.4={g_disc}"
            print(f"\n  {label}  (n={total})")
            for intent in intent_vals:
                c = counts.get(intent, 0)
                if c > 0:
                    print(f"    {intent:<25} {c:>5}")


def main():
    parser = argparse.ArgumentParser(
        description="Validate inter-model agreement on SKILL.md classification."
    )
    parser.add_argument(
        "--input", type=Path, default=DATA_DIR / "corpus.jsonl",
        help="Input corpus NDJSON (default: data/corpus.jsonl)",
    )
    parser.add_argument(
        "--output", type=Path, default=DATA_DIR / "validation_results.jsonl",
        help="Output results NDJSON (default: data/validation_results.jsonl)",
    )
    parser.add_argument(
        "--n", type=int, default=50,
        help="Number of records to subsample (default: 50)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--delay", type=float, default=0.5,
        help="Seconds between API calls (default: 0.5)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from existing output, only calling models with missing results",
    )
    parser.add_argument(
        "--report", type=Path, default=None,
        help="Save summary report to a text file (default: print to stdout)",
    )
    parser.add_argument(
        "--classifications", type=Path, default=None,
        help="Pre-populate Sonnet results from a classifications JSONL (avoids duplicate API calls)",
    )
    parser.add_argument(
        "--models", type=str, nargs="+", default=None,
        help="Models to validate (display names, e.g. --models Sonnet GPT-5.4). Default: all 4.",
    )
    args = parser.parse_args()

    # Override active model list if --models specified
    if args.models:
        name_to_id = {v: k for k, v in MODEL_NAMES.items()}
        global MODELS
        MODELS = []
        for name in args.models:
            if name not in name_to_id:
                print(f"Error: Unknown model name '{name}'. Valid: {', '.join(name_to_id)}")
                sys.exit(1)
            MODELS.append(name_to_id[name])
        print(f"Using models: {', '.join(args.models)}")

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

    # Draw subsample
    if args.n >= len(records):
        sample = records
        print(f"Corpus has {len(records)} records, using all of them.")
    else:
        random.seed(args.seed)
        sample = random.sample(records, args.n)
        print(f"Sampled {args.n} records from {len(records)} (seed={args.seed}).")

    print(f"Models: {', '.join(MODEL_NAMES[m] for m in MODELS)}")
    print()

    # Load existing results if resuming
    existing = {}
    if args.resume and args.output.exists():
        with open(args.output) as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    key = (rec["repo"], rec["path"])
                    existing[key] = rec
        print(f"Resuming: loaded {len(existing)} existing records from {args.output}")

    # Initialize result records from existing data or fresh
    results_by_key: dict[tuple[str, str], dict] = {}
    for record in sample:
        key = (record["repo"], record["path"])
        if args.resume and key in existing:
            results_by_key[key] = existing[key]
        else:
            results_by_key[key] = {"repo": record["repo"], "path": record["path"]}

    # Pre-populate Sonnet results from classifications file if provided
    if args.classifications and args.classifications.exists():
        seeded = 0
        with open(args.classifications) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                key = (rec["repo"], rec["path"])
                if key in results_by_key and results_by_key[key].get("Sonnet") is None:
                    classification = {
                        field: rec[field]
                        for field in CLASSIFICATION_FIELDS
                        if field in rec
                    }
                    if len(classification) == len(CLASSIFICATION_FIELDS):
                        results_by_key[key]["Sonnet"] = classification
                        seeded += 1
        if seeded:
            print(f"Seeded {seeded} Sonnet results from {args.classifications.name}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    total_tokens = {m: {"input": 0, "output": 0} for m in MODELS}
    errors = 0
    skipped = 0
    write_lock = threading.Lock()

    def save_results():
        """Save current results to disk (caller must hold write_lock)."""
        res = [results_by_key[(r["repo"], r["path"])] for r in sample]
        with open(args.output, "w") as f:
            for rec in res:
                f.write(json.dumps(rec) + "\n")

    def process_model(model):
        """Process all records for a single model. Thread-safe."""
        nonlocal errors, skipped
        name = MODEL_NAMES[model]
        use_cache = model in ANTHROPIC_MODELS

        to_run = [
            r for r in sample
            if results_by_key[(r["repo"], r["path"])].get(name) is None
        ]
        if not to_run:
            print(f"\n  {name}: all {len(sample)} records already done, skipping")
            with write_lock:
                skipped += len(sample)
            return

        print(f"\n  {name}: {len(to_run)} records to process ({len(sample) - len(to_run)} cached)")

        local_errors = 0
        for i, record in enumerate(to_run):
            repo = record["repo"]
            path = record["path"]
            key = (repo, path)

            user_prompt = classify_template.replace("{{content}}", record["content"])

            try:
                response = call_model(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    model=model,
                    cache_system_prompt=use_cache,
                )
            except Exception as e:
                print(f"    ERROR [{name}] [{i+1}/{len(to_run)}] {repo}/{path}: {e}")
                local_errors += 1
                with write_lock:
                    results_by_key[key][name] = None
                time.sleep(args.delay)
                continue

            total_tokens[model]["input"] += response.input_tokens
            total_tokens[model]["output"] += response.output_tokens

            classification = parse_classification(response.text)
            if classification is None:
                print(f"    PARSE ERROR [{name}] [{i+1}/{len(to_run)}] {repo}/{path}")
                local_errors += 1
                with write_lock:
                    results_by_key[key][name] = None
            else:
                with write_lock:
                    results_by_key[key][name] = classification

            if (i + 1) % 50 == 0:
                print(f"    [{name}] [{i+1}/{len(to_run)}] done")
                with write_lock:
                    save_results()

            if i < len(to_run) - 1:
                time.sleep(args.delay)

        with write_lock:
            errors += local_errors
            save_results()
        print(f"  {name}: done. Saved intermediate results.")

    # Group models by API provider so different APIs run in parallel,
    # but models sharing a rate limit run sequentially within their group.
    anthropic_models = [m for m in MODELS if m in ANTHROPIC_MODELS]
    openai_models = [m for m in MODELS if m in OPENAI_MODELS]

    def run_model_group(models):
        """Run models sequentially within a group (shared rate limit)."""
        for model in models:
            process_model(model)

    groups = [g for g in [anthropic_models, openai_models] if g]

    if len(groups) > 1:
        print(f"\nRunning {len(groups)} API groups in parallel...")
        with ThreadPoolExecutor(max_workers=len(groups)) as executor:
            futures = [executor.submit(run_model_group, g) for g in groups]
            for future in as_completed(futures):
                future.result()  # propagate exceptions
    else:
        for g in groups:
            run_model_group(g)

    results = [results_by_key[(r["repo"], r["path"])] for r in sample]

    if skipped:
        print(f"\nSkipped {skipped} model calls (already had results)")

    # Save raw results
    with open(args.output, "w") as f:
        for record in results:
            f.write(json.dumps(record) + "\n")
    print(f"\nRaw results saved to {args.output}")

    # Filter to records where all models succeeded
    complete = [
        r for r in results
        if all(r.get(MODEL_NAMES[m]) is not None for m in MODELS)
    ]

    def print_report():
        model_names = [MODEL_NAMES[m] for m in MODELS]
        print(f"Complete records (all {len(MODELS)} models): {len(complete)}/{len(results)}")

        if len(complete) < 2:
            print("Too few complete records for agreement analysis.")
            return

        # Pairwise agreement
        if len(MODELS) >= 2:
            agreements = compute_pairwise_agreement(complete)
            print_agreement_table(agreements)

        # Cohen's kappa for each pair
        for name_a, name_b in combinations(model_names, 2):
            kappas = compute_cohens_kappa(complete, name_a, name_b)
            print_kappa_table(kappas, name_a, name_b)

        # Summary table
        print_summary_table(complete)

        # Discretion cross-tab (only if both Sonnet and GPT-5.4 are active)
        if "Sonnet" in model_names and "GPT-5.4" in model_names:
            print_discretion_by_intent(complete)

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with open(args.report, "w") as f:
            with redirect_stdout(f):
                print_report()
        print(f"Report saved to {args.report}")
    else:
        print_report()


if __name__ == "__main__":
    main()
