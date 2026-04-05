"""Generate all visualizations for the AI skills write-up.

Reads clustered classifications and corpus data, outputs PNG charts to figures/.

Usage:
    python scripts/make_charts.py
    python scripts/make_charts.py --classifications data/clusters_object.jsonl --corpus data/pilot_corpus.jsonl
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import DATA_DIR

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"

# NYT-style defaults
FONT_FAMILY = [
    "-apple-system", "BlinkMacSystemFont", "Segoe UI", "Roboto",
    "Helvetica Neue", "Arial", "sans-serif",
]
COLOR_TEXT = "#1a1a1a"
COLOR_SECONDARY = "#666"
COLOR_BORDER = "#d4d4d4"
COLOR_FILL = "#1a1a1a"
COLOR_FILL_LIGHT = "#e0ddd8"
COLOR_MUTED = "#bbb"
COLOR_ACCENT = "#c44e52"
BG_COLOR = "#ffffff"

SOURCE_LINE = "Source: GitHub SKILL.md corpus, n=15,805"

# Primary intent ordering (bottom to top on charts)
INTENT_ORDER = [
    "tool-orchestration",
    "context-provision",
    "process-specification",
    "preference-alignment",
    "risk-mitigation",
]

INTENT_COLORS = {
    "risk-mitigation": "#c44e52",
    "preference-alignment": "#8c8c8c",
    "process-specification": "#4c72b0",
    "context-provision": "#dd8452",
    "tool-orchestration": "#55a868",
}


def setup_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": FONT_FAMILY,
        "text.color": COLOR_TEXT,
        "axes.labelcolor": COLOR_TEXT,
        "xtick.color": COLOR_SECONDARY,
        "ytick.color": COLOR_SECONDARY,
        "axes.edgecolor": COLOR_BORDER,
        "figure.facecolor": BG_COLOR,
        "axes.facecolor": BG_COLOR,
        "axes.grid": False,
    })


def clean_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLOR_BORDER)
    ax.spines["bottom"].set_color(COLOR_BORDER)
    ax.tick_params(length=0)


def add_subtitle(ax, text):
    ax.text(0, 1.02, text, transform=ax.transAxes, fontsize=11,
            color=COLOR_SECONDARY, va="bottom", ha="left")


def add_source(fig):
    fig.text(0.02, 0.005, SOURCE_LINE, fontsize=7.5, color=COLOR_MUTED,
             ha="left", va="bottom")


def load_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# =========================================================================
# 1.1 Histogram: skill file counts per repo
# =========================================================================

def chart_1_1(corpus):
    repo_counts = Counter(r["repo"] for r in corpus)
    counts = list(repo_counts.values())

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.arange(0, min(max(counts) + 2, 102), 1)
    weights = np.ones_like(counts, dtype=float) / len(counts) * 100
    ax.hist(counts, bins=bins, weights=weights, color=COLOR_MUTED, alpha=0.7,
            edgecolor="none")

    ax.set_title("Number of instruction files per repository",
                 fontsize=16, fontweight=700, loc="left", pad=20)
    add_subtitle(ax, "Most repositories contain fewer than 10 instruction files")
    ax.set_xlabel("Instruction files per repo", fontsize=12, labelpad=8)
    ax.set_ylabel("% of repositories", fontsize=12, labelpad=8)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
    clean_axes(ax)

    median_val = np.median(counts)
    ax.axvline(median_val, color=COLOR_ACCENT, linewidth=1.5, linestyle="--", alpha=0.8)
    ax.text(median_val + 1, ax.get_ylim()[1] * 0.9, f"Median: {median_val:.0f}",
            fontsize=11, color=COLOR_ACCENT, va="top")

    add_source(fig)
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(FIGURES_DIR / "1_1_skill_file_histogram.png", dpi=200, bbox_inches="tight",
                facecolor="white")
    plt.close()
    print("  1.1 Skill file histogram")


# =========================================================================
# 1.2 Semi-log: commit count vs skill file count
# =========================================================================

def chart_1_2(corpus):
    repo_data = {}
    for r in corpus:
        repo = r["repo"]
        if repo not in repo_data:
            repo_data[repo] = {"files": 0, "commits": r.get("commit_count", 0)}
        repo_data[repo]["files"] += 1

    # Bin repos by commit tier
    tiers = [
        ("1–10", 1, 10),
        ("11–100", 11, 100),
        ("101–1K", 101, 1_000),
        ("1K+", 1_001, float("inf")),
    ]

    tier_data = {label: [] for label, _, _ in tiers}
    for d in repo_data.values():
        commits = max(d["commits"], 1)
        for label, lo, hi in tiers:
            if lo <= commits <= hi:
                tier_data[label].append(d["files"])
                break

    tier_labels = [label for label, _, _ in tiers]
    data = [tier_data[label] for label in tier_labels]
    n_per_tier = [len(d) for d in data]

    fig, ax = plt.subplots(figsize=(8, 4))

    for i, tier_vals in enumerate(data, 1):
        arr = np.array(tier_vals)
        q1, median, q3 = np.percentile(arr, [25, 50, 75])
        ax.hlines(i, q1, q3, color=COLOR_MUTED, linewidth=2, alpha=0.8)
        ax.scatter(median, i, color=COLOR_ACCENT, s=70, zorder=4)
        ax.text(q3 + 0.5, i, f"median: {median:.0f}", va="center",
                fontsize=10, color=COLOR_SECONDARY)

    ax.set_yticks(range(1, len(tier_labels) + 1))
    ax.set_yticklabels([f"{l} commits (n={n:,})" for l, n in zip(tier_labels, n_per_tier)],
                       fontsize=11)
    ax.set_xlabel("Instruction files per repo", fontsize=12, labelpad=8)
    ax.set_title("Instruction file count by repository activity",
                 fontsize=16, fontweight=700, loc="left", pad=20)
    add_subtitle(ax, "Median file count is similar regardless of repository maturity")
    ax.set_xlim(0, None)
    clean_axes(ax)

    add_source(fig)
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(FIGURES_DIR / "1_2_files_by_commit_tier.png", dpi=200, bbox_inches="tight",
                facecolor="white")
    plt.close()
    print("  1.2 Files by commit tier dot plot")


# =========================================================================
# 2.1 Cleveland dot plot: object cluster + intent frequencies
# =========================================================================

def chart_2_1(classifications):
    total = len(classifications)

    # Top 10 clusters + other + unclustered
    cluster_counts = Counter()
    for r in classifications:
        cid = r.get("cluster_id", -1)
        label = r.get("cluster_label") or "Unclustered"
        if cid == -1:
            cluster_counts["Unclustered"] += 1
        else:
            cluster_counts[label] += 1

    top_10 = cluster_counts.most_common(12)
    # Separate unclustered
    top_labels = [(l, c) for l, c in top_10 if l != "Unclustered"][:10]
    unclustered_count = cluster_counts.get("Unclustered", 0)
    other_count = total - sum(c for _, c in top_labels) - unclustered_count

    categories = [l for l, _ in reversed(top_labels)] + ["All other clusters", "Unclustered"]
    values = [c / total * 100 for _, c in reversed(top_labels)] + [
        other_count / total * 100, unclustered_count / total * 100
    ]

    fig, ax = plt.subplots(figsize=(9, 6))
    y_pos = range(len(categories))

    ax.barh(y_pos, values, height=0.6, color=COLOR_FILL, alpha=0.8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(categories, fontsize=11)
    ax.set_xlabel("% of all instructions", fontsize=13, labelpad=8)
    ax.set_title("Instruction share by object cluster\n(top 10 clusters shown)",
                 fontsize=16, fontweight=700, loc="left", pad=12)
    clean_axes(ax)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))

    for i, v in enumerate(values):
        ax.text(v + 0.3, i, f"{v:.1f}%", va="center", fontsize=10, color=COLOR_SECONDARY)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "2_1_object_cluster_share.png", dpi=200, bbox_inches="tight",
                facecolor="white")
    plt.close()
    print("  2.1 Object cluster share")


# =========================================================================
# 2.1b Intent frequency bar chart
# =========================================================================

def chart_2_1b(classifications):
    total = len(classifications)

    intent_counts = Counter(r["primary_intent"] for r in classifications if r.get("primary_intent"))
    # INTENT_ORDER is bottom-to-top; for horizontal bar, first entry is bottom
    categories = INTENT_ORDER
    values = [intent_counts.get(c, 0) / total * 100 for c in categories]

    fig, ax = plt.subplots(figsize=(8, 4))
    y_pos = range(len(categories))

    # Highlight process-specification, mute everything else
    colors = [INTENT_COLORS["process-specification"] if c == "process-specification"
              else COLOR_MUTED for c in categories]
    ax.barh(y_pos, values, height=0.6, color=colors, alpha=0.85)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(categories, fontsize=11)
    ax.set_xlabel("% of all instructions", fontsize=12, labelpad=8)
    ax.set_title("Instructions by primary intent",
                 fontsize=16, fontweight=700, loc="left", pad=20)
    add_subtitle(ax, "Process specification dominates \u2014 3 in 5 instructions define how to do a task")
    clean_axes(ax)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))

    for i, v in enumerate(values):
        ax.text(v + 0.3, i, f"{v:.1f}%", va="center", fontsize=10, color=COLOR_SECONDARY)

    add_source(fig)
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(FIGURES_DIR / "2_1b_intent_frequency.png", dpi=200, bbox_inches="tight",
                facecolor="white")
    plt.close()
    print("  2.1b Intent frequency")


# =========================================================================
# 2.2 Bubble matrix: object cluster x primary intent
# =========================================================================

def chart_2_2(classifications):
    total = len(classifications)

    # Get top 10 cluster labels
    cluster_counts = Counter()
    for r in classifications:
        cid = r.get("cluster_id", -1)
        if cid != -1:
            label = r.get("cluster_label", f"Cluster {cid}")
            cluster_counts[label] += 1

    top_10_labels_raw = [l for l, _ in cluster_counts.most_common(10)]
    top_10_labels = top_10_labels_raw  # keep raw for matrix lookups

    # Build matrix
    matrix = {}
    for r in classifications:
        cid = r.get("cluster_id", -1)
        label = r.get("cluster_label")
        intent = r.get("primary_intent")
        if cid == -1 or label not in top_10_labels or intent not in INTENT_ORDER:
            continue
        key = (label, intent)
        matrix[key] = matrix.get(key, 0) + 1

    # Columns follow INTENT_ORDER (risk-mitigation leftmost)
    x_labels = list(reversed(INTENT_ORDER))
    y_labels_raw = list(reversed(top_10_labels))
    y_labels_display = [l.title() for l in y_labels_raw]

    # Compute totals per intent (for % within intent)
    intent_totals = Counter()
    for r in classifications:
        intent = r.get("primary_intent")
        if intent in INTENT_ORDER:
            intent_totals[intent] += 1

    fig, ax = plt.subplots(figsize=(10, 6))

    max_pct = 0
    for ylabel in y_labels_raw:
        for xlabel in x_labels:
            count = matrix.get((ylabel, xlabel), 0)
            if count > 0 and intent_totals[xlabel] > 0:
                pct = count / intent_totals[xlabel] * 100
                if pct > max_pct:
                    max_pct = pct

    for yi, ylabel in enumerate(y_labels_raw):
        for xi, xlabel in enumerate(x_labels):
            count = matrix.get((ylabel, xlabel), 0)
            if count == 0:
                continue
            pct = count / intent_totals[xlabel] * 100
            size = (pct / max_pct) * 800
            ax.scatter(xi, yi, s=size, color=INTENT_COLORS.get(xlabel, COLOR_FILL),
                      alpha=0.7, edgecolors="white", linewidth=0.5)
            if pct >= 3:
                ax.text(xi, yi, f"{pct:.0f}%", ha="center", va="center",
                       fontsize=8, color="white", fontweight=600)

    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, fontsize=10, rotation=30, ha="right")
    ax.set_yticks(range(len(y_labels_display)))
    ax.set_yticklabels(y_labels_display, fontsize=10)
    ax.set_title("Object cluster by primary intent",
                 fontsize=16, fontweight=700, loc="left", pad=20)
    add_subtitle(ax, "Bubble size = % within each intent")
    clean_axes(ax)
    ax.set_xlim(-0.5, len(x_labels) - 0.5)
    ax.set_ylim(-0.5, len(y_labels_raw) - 0.5)

    # Light grid
    for yi in range(len(y_labels_raw)):
        ax.axhline(yi, color=COLOR_BORDER, linewidth=0.5, zorder=0)
    for xi in range(len(x_labels)):
        ax.axvline(xi, color=COLOR_BORDER, linewidth=0.5, zorder=0)

    add_source(fig)
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(FIGURES_DIR / "2_2_object_intent_bubble.png", dpi=200, bbox_inches="tight",
                facecolor="white")
    plt.close()
    print("  2.2 Object x intent bubble matrix")


# =========================================================================
# 2.3 Matrix: primary intent x discretion
# =========================================================================

def chart_2_3(classifications):
    total = len(classifications)

    matrix = Counter()
    for r in classifications:
        intent = r.get("primary_intent")
        disc = r.get("discretion")
        if intent in INTENT_ORDER and disc:
            matrix[(intent, disc)] += 1

    disc_cols = ["prescribed", "adaptive"]
    y_labels = list(reversed(INTENT_ORDER))

    fig, ax = plt.subplots(figsize=(6, 4.5))

    cell_data = []
    for intent in y_labels:
        row = []
        for disc in disc_cols:
            count = matrix.get((intent, disc), 0)
            row.append(count)
        cell_data.append(row)

    cell_data = np.array(cell_data)
    im = ax.imshow(cell_data, cmap="Blues", aspect="auto")

    ax.set_xticks(range(len(disc_cols)))
    ax.set_xticklabels(disc_cols, fontsize=11)
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels, fontsize=11)

    for yi in range(len(y_labels)):
        for xi in range(len(disc_cols)):
            val = cell_data[yi, xi]
            pct = val / total * 100
            color = "white" if val > cell_data.max() * 0.5 else COLOR_TEXT
            ax.text(xi, yi, f"{val}\n({pct:.1f}%)", ha="center", va="center",
                   fontsize=10, color=color)

    ax.set_title("Primary intent by discretion",
                 fontsize=16, fontweight=700, loc="left", pad=20)
    add_subtitle(ax, "Nearly all instructions grant adaptive discretion")

    add_source(fig)
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(FIGURES_DIR / "2_3_intent_discretion_matrix.png", dpi=200, bbox_inches="tight",
                facecolor="white")
    plt.close()
    print("  2.3 Intent x discretion matrix")


# =========================================================================
# 2.4 Diverging stacked bar: intent (rows) x discretion (bar composition)
# =========================================================================

def chart_2_4(classifications):
    intent_disc = {}
    for r in classifications:
        intent = r.get("primary_intent")
        disc = r.get("discretion")
        if intent in INTENT_ORDER and disc:
            intent_disc.setdefault(intent, Counter())[disc] += 1

    # INTENT_ORDER is bottom-to-top
    y_labels = INTENT_ORDER
    y_pos = np.arange(len(y_labels))

    prescribed_pcts = []
    for intent in y_labels:
        counts = intent_disc.get(intent, Counter())
        total = sum(counts.values())
        prescribed_pcts.append(counts.get("prescribed", 0) / total * 100 if total else 0)

    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.scatter(prescribed_pcts, y_pos, s=80, color=COLOR_ACCENT, zorder=3)
    ax.hlines(y_pos, 0, prescribed_pcts, color=COLOR_ACCENT, linewidth=1.5, alpha=0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels, fontsize=11)
    ax.set_xlabel("% prescribed (within intent)", fontsize=12, labelpad=8)
    ax.set_title("Prescribed share by primary intent",
                 fontsize=16, fontweight=700, loc="left", pad=20)
    add_subtitle(ax, "Risk-mitigation instructions are the most frequently prescribed")
    clean_axes(ax)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))

    for i, pct in enumerate(prescribed_pcts):
        ax.text(pct + 0.5, i, f"{pct:.1f}%", va="center", fontsize=10, color=COLOR_SECONDARY)

    add_source(fig)
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(FIGURES_DIR / "2_4_intent_discretion_dotplot.png", dpi=200, bbox_inches="tight",
                facecolor="white")
    plt.close()
    print("  2.4 Intent x discretion dot plot")


# =========================================================================
# 2.5 Diverging stacked bar: intent x decision_count quartiles
# =========================================================================

def _compute_word_count(classifications, corpus):
    """Join word counts from corpus onto classifications."""
    corpus_wc = {}
    for r in corpus:
        key = (r["repo"], r["path"])
        corpus_wc[key] = len(r.get("content", "").split())

    for r in classifications:
        key = (r["repo"], r["path"])
        r["word_count"] = corpus_wc.get(key, 1)


def _quartile_bar(classifications, field, title, subtitle, footnote, filename):
    """Stacked bar: intent rows x quartiles of field/word_count."""
    # Compute normalized values
    vals = []
    for r in classifications:
        wc = r.get("word_count", 1)
        v = r.get(field, 0)
        if wc > 0 and v is not None:
            vals.append(v / wc)
        else:
            vals.append(0)

    if not vals:
        return

    quartiles = np.percentile(vals, [25, 50, 75])
    q_labels = ["1st Quartile (Lowest)", "2nd Quartile", "3rd Quartile", "4th Quartile"]

    def get_quartile(v):
        if v <= quartiles[0]:
            return 0
        elif v <= quartiles[1]:
            return 1
        elif v <= quartiles[2]:
            return 2
        else:
            return 3

    # Count by intent x quartile
    intent_q = {}
    for r, v in zip(classifications, vals):
        intent = r.get("primary_intent")
        if intent not in INTENT_ORDER:
            continue
        q = get_quartile(v)
        intent_q.setdefault(intent, Counter())[q] += 1

    y_labels = list(reversed(INTENT_ORDER))
    colors = ["#d4e6f1", "#7fb3d3", "#2980b9", "#1a5276"]

    fig, ax = plt.subplots(figsize=(8, 5))
    y_pos = np.arange(len(y_labels))

    for i, intent in enumerate(y_labels):
        counts = intent_q.get(intent, Counter())
        total = sum(counts.values())
        if total == 0:
            continue
        left = 0
        for q in range(4):
            pct = counts.get(q, 0) / total * 100
            ax.barh(i, pct, left=left, height=0.6, color=colors[q], alpha=0.9)
            if pct > 8:
                ax.text(left + pct / 2, i, f"{pct:.0f}%", ha="center", va="center",
                       fontsize=9, color="white" if q >= 2 else COLOR_TEXT)
            left += pct

    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels, fontsize=11)
    ax.set_xlabel("% of instructions within intent", fontsize=12, labelpad=8)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
    clean_axes(ax)

    # Title, subtitle, and legend as fig text
    fig.text(0.02, 0.97, title, fontsize=16, fontweight=700, ha="left", va="top")
    fig.text(0.02, 0.91, subtitle, fontsize=11, color=COLOR_SECONDARY, ha="left", va="top")
    handles = [plt.Rectangle((0, 0), 1, 1, fc=colors[q]) for q in range(4)]
    fig.legend(handles, q_labels, loc="upper center", ncol=4,
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.86))

    # Footnote and source
    fig.text(0.02, 0.035, footnote, fontsize=8, color=COLOR_SECONDARY,
             ha="left", va="bottom")
    fig.text(0.02, 0.005, SOURCE_LINE, fontsize=7.5, color=COLOR_MUTED,
             ha="left", va="bottom")

    fig.subplots_adjust(top=0.73, bottom=0.15)
    plt.savefig(FIGURES_DIR / filename, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()


def chart_2_5(classifications):
    _quartile_bar(
        classifications,
        "decision_count",
        "Decision density by primary intent",
        "Risk-mitigation and process-specification skew toward higher decision density",
        "Note: Decision counts are first divided by instruction word count, and then binned in corpus-wide quartiles.",
        "2_5_intent_decision_quartiles.png",
    )
    print("  2.5 Intent x decision quartiles")


def _quartile_bar_by_discretion(classifications, field, title, subtitle, footnote, filename):
    """Stacked bar: (intent x discretion) rows x quartiles of field/word_count."""
    vals = []
    for r in classifications:
        wc = r.get("word_count", 1)
        v = r.get(field, 0)
        if wc > 0 and v is not None:
            vals.append(v / wc)
        else:
            vals.append(0)

    if not vals:
        return

    quartiles = np.percentile(vals, [25, 50, 75])
    q_labels = ["1st Quartile (Lowest)", "2nd Quartile", "3rd Quartile", "4th Quartile"]

    def get_quartile(v):
        if v <= quartiles[0]:
            return 0
        elif v <= quartiles[1]:
            return 1
        elif v <= quartiles[2]:
            return 2
        else:
            return 3

    disc_labels = ["prescribed", "adaptive"]
    row_keys = []
    for intent in INTENT_ORDER:
        for disc in disc_labels:
            row_keys.append((intent, disc))

    row_q = {}
    for r, v in zip(classifications, vals):
        intent = r.get("primary_intent")
        disc = r.get("discretion")
        if intent not in INTENT_ORDER or disc not in disc_labels:
            continue
        q = get_quartile(v)
        row_q.setdefault((intent, disc), Counter())[q] += 1

    row_keys = [k for k in row_keys if sum(row_q.get(k, Counter()).values()) >= 5]
    y_labels = [f"{intent} / {disc}" for intent, disc in row_keys]

    colors = ["#d4e6f1", "#7fb3d3", "#2980b9", "#1a5276"]

    fig, ax = plt.subplots(figsize=(9, 6))
    y_pos = np.arange(len(y_labels))

    for i, key in enumerate(row_keys):
        counts = row_q.get(key, Counter())
        total = sum(counts.values())
        if total == 0:
            continue
        left = 0
        for q in range(4):
            pct = counts.get(q, 0) / total * 100
            ax.barh(i, pct, left=left, height=0.6, color=colors[q], alpha=0.9)
            if pct > 8:
                ax.text(left + pct / 2, i, f"{pct:.0f}%", ha="center", va="center",
                       fontsize=9, color="white" if q >= 2 else COLOR_TEXT)
            left += pct

        ax.text(101, i, f"n={total}", va="center", fontsize=8, color=COLOR_MUTED)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels, fontsize=10)
    ax.set_xlabel("% of instructions within group", fontsize=12, labelpad=8)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
    clean_axes(ax)

    fig.text(0.02, 0.97, title, fontsize=16, fontweight=700, ha="left", va="top")
    fig.text(0.02, 0.91, subtitle, fontsize=11, color=COLOR_SECONDARY, ha="left", va="top")
    handles = [plt.Rectangle((0, 0), 1, 1, fc=colors[q]) for q in range(4)]
    fig.legend(handles, q_labels, loc="upper center", ncol=4,
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.86))

    fig.text(0.02, 0.035, footnote, fontsize=8, color=COLOR_SECONDARY,
             ha="left", va="bottom")
    fig.text(0.02, 0.005, SOURCE_LINE, fontsize=7.5, color=COLOR_MUTED,
             ha="left", va="bottom")

    fig.subplots_adjust(top=0.73, bottom=0.15)
    plt.savefig(FIGURES_DIR / filename, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()


def chart_2_5b(classifications):
    _quartile_bar_by_discretion(
        classifications,
        "decision_count",
        "Decision density by primary intent and discretion",
        "Prescribed instructions tend toward lower decision density within the same intent",
        "Note: Decision counts are first divided by instruction word count, and then binned in corpus-wide quartiles. Groups with n < 5 omitted.",
        "2_5b_intent_discretion_decision_quartiles.png",
    )
    print("  2.5b Intent x discretion x decision quartiles")


# =========================================================================
# 2.6 Diverging stacked bar: intent x constraint_count quartiles
# =========================================================================

def chart_2_6(classifications):
    _quartile_bar(
        classifications,
        "constraint_count",
        "Constraint density by primary intent",
        "Risk-mitigation instructions carry the highest constraint density",
        "Note: Constraint counts are first divided by instruction word count, and then binned in corpus-wide quartiles.",
        "2_6_intent_constraint_quartiles.png",
    )
    print("  2.6 Intent x constraint quartiles")


def chart_2_6b(classifications):
    _quartile_bar_by_discretion(
        classifications,
        "constraint_count",
        "Constraint density by primary intent and discretion",
        "Prescribed instructions tend toward lower constraint density within the same intent",
        "Note: Constraint counts are first divided by instruction word count, and then binned in corpus-wide quartiles. Groups with n < 5 omitted.",
        "2_6b_intent_discretion_constraint_quartiles.png",
    )
    print("  2.6b Intent x discretion x constraint quartiles")


# =========================================================================
# 2.7 Horizontal bar: top 20 object labels + noise
# =========================================================================

def chart_2_7(classifications):
    total = len(classifications)

    # Top 20 cluster labels by count
    cluster_counts = Counter()
    for r in classifications:
        cid = r.get("cluster_id", -1)
        label = r.get("cluster_label")
        if cid != -1 and label:
            cluster_counts[label] += 1

    unclustered = sum(1 for r in classifications if r.get("cluster_id", -1) == -1)

    top_20 = cluster_counts.most_common(20)

    unclustered_pct = unclustered / total * 100
    categories = [l.title() for l, _ in reversed(top_20)]
    values = [c / total * 100 for _, c in reversed(top_20)]

    fig, ax = plt.subplots(figsize=(9, 8))
    y_pos = range(len(categories))

    ax.barh(y_pos, values, height=0.65, color=COLOR_FILL, alpha=0.85)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(categories, fontsize=10)
    ax.set_xlabel("% of all instructions", fontsize=13, labelpad=8)
    ax.set_title("Instruction share by object cluster (top 20)",
                 fontsize=16, fontweight=700, loc="left", pad=20)
    add_subtitle(ax, "UI/UX design is the most common object cluster")
    clean_axes(ax)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))

    for i, v in enumerate(values):
        ax.text(v + 0.2, i, f"{v:.1f}%", va="center", fontsize=9, color=COLOR_SECONDARY)

    # Footnote with unclustered %
    fig.text(0.02, 0.03, f"Note: {unclustered_pct:.1f}% of instructions were not assigned to any cluster.",
             fontsize=8, color=COLOR_SECONDARY, ha="left", va="bottom")
    fig.text(0.02, 0.005, SOURCE_LINE, fontsize=7.5, color=COLOR_MUTED,
             ha="left", va="bottom")

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(FIGURES_DIR / "2_7_object_label_share.png", dpi=200, bbox_inches="tight",
                facecolor="white")
    plt.close()
    print("  2.7 Object label share (top 20)")


# =========================================================================
# 2.8 Heatmap: decision quartile x constraint quartile for process-specification
#     faceted by discretion
# =========================================================================

def chart_2_8(classifications):
    """Heatmap of decision vs constraint quartiles for process-specification, faceted by discretion."""
    # Compute corpus-wide normalized values and quartile breakpoints
    decision_vals, constraint_vals = [], []
    for r in classifications:
        wc = r.get("word_count", 1)
        dv = r.get("decision_count", 0)
        cv = r.get("constraint_count", 0)
        decision_vals.append(dv / wc if wc > 0 and dv is not None else 0)
        constraint_vals.append(cv / wc if wc > 0 and cv is not None else 0)

    d_quartiles = np.percentile(decision_vals, [25, 50, 75])
    c_quartiles = np.percentile(constraint_vals, [25, 50, 75])

    def get_q(v, breaks):
        if v <= breaks[0]:
            return 0
        elif v <= breaks[1]:
            return 1
        elif v <= breaks[2]:
            return 2
        else:
            return 3

    # Filter to process-specification and compute quartile pairs
    disc_data = {"adaptive": Counter(), "prescribed": Counter()}
    disc_totals = {"adaptive": 0, "prescribed": 0}
    for r, dv, cv in zip(classifications, decision_vals, constraint_vals):
        if r.get("primary_intent") != "process-specification":
            continue
        disc = r.get("discretion")
        if disc not in disc_data:
            continue
        dq = get_q(dv, d_quartiles)
        cq = get_q(cv, c_quartiles)
        disc_data[disc][(dq, cq)] += 1
        disc_totals[disc] += 1

    q_tick_labels = ["Q1\n(lowest)", "Q2", "Q3", "Q4\n(highest)"]
    facets = ["prescribed", "adaptive"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

    for ax, disc in zip(axes, facets):
        total = disc_totals[disc]
        if total == 0:
            continue

        matrix = np.zeros((4, 4))
        for (dq, cq), count in disc_data[disc].items():
            matrix[dq, cq] = count / total * 100

        im = ax.imshow(matrix, cmap="Blues", aspect="auto", origin="lower",
                       vmin=0, vmax=max(np.max(m) for m in
                           [np.array([[disc_data[d].get((dq, cq), 0) / max(disc_totals[d], 1) * 100
                                       for cq in range(4)] for dq in range(4)])
                            for d in facets]))

        for dq in range(4):
            for cq in range(4):
                val = matrix[dq, cq]
                if val > 0:
                    color = "white" if val > np.max(matrix) * 0.5 else COLOR_TEXT
                    ax.text(cq, dq, f"{val:.1f}%", ha="center", va="center",
                           fontsize=9, color=color)

        ax.set_xticks(range(4))
        ax.set_xticklabels(q_tick_labels, fontsize=9)
        ax.set_yticks(range(4))
        if disc == facets[0]:
            ax.set_yticklabels(q_tick_labels, fontsize=9)
        ax.set_xlabel("Constraint quartile", fontsize=11, labelpad=6)
        ax.set_title(f"{disc.title()} (n={total:,})",
                     fontsize=13, fontweight=600, pad=8)
        ax.tick_params(length=0)

    axes[0].set_ylabel("Decision quartile", fontsize=11, labelpad=6)

    fig.suptitle("Decision vs. constraint density for process-specification",
                 fontsize=16, fontweight=700, x=0.02, ha="left", y=0.98)
    fig.text(0.02, 0.91, "Cell values show % of instructions within each discretion group",
             fontsize=11, color=COLOR_SECONDARY, ha="left", va="top")

    footnote = "Note: Both metrics are normalized by instruction word count and binned in corpus-wide quartiles."
    fig.text(0.02, 0.01, footnote, fontsize=8, color=COLOR_SECONDARY,
             ha="left", va="bottom")
    fig.text(0.98, 0.01, SOURCE_LINE, fontsize=7.5, color=COLOR_MUTED,
             ha="right", va="bottom")

    fig.subplots_adjust(top=0.82, bottom=0.15, wspace=0.08)
    plt.savefig(FIGURES_DIR / "2_8_process_spec_decision_constraint_matrix.png",
                dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print("  2.8 Process-specification decision x constraint matrix")


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate all visualizations.")
    parser.add_argument(
        "--classifications", type=Path,
        default=DATA_DIR / "clusters_object.jsonl",
        help="Clustered classifications JSONL",
    )
    parser.add_argument(
        "--corpus", type=Path,
        default=DATA_DIR / "pilot_corpus.jsonl",
        help="Corpus JSONL (for commit counts, word counts)",
    )
    args = parser.parse_args()

    setup_style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    classifications = load_jsonl(args.classifications)
    corpus = load_jsonl(args.corpus)
    print(f"  {len(classifications)} classifications, {len(corpus)} corpus records")

    # Join word counts onto classifications
    _compute_word_count(classifications, corpus)

    print("\nGenerating charts...")
    chart_1_1(corpus)
    chart_1_2(corpus)
    chart_2_1b(classifications)
    chart_2_2(classifications)
    chart_2_3(classifications)
    chart_2_4(classifications)
    chart_2_5(classifications)
    chart_2_5b(classifications)
    chart_2_6(classifications)
    chart_2_6b(classifications)
    chart_2_7(classifications)
    chart_2_8(classifications)

    print(f"\nDone. All charts saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
