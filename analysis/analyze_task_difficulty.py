"""Analyze MARBLE task difficulty distribution.

Reads ``results/marble/difficulty_profile/difficulty_summary.csv``
and produces:
  - task_difficulty_ranking.csv  (sorted ascending by mean_reward)
  - figures/marble/task_difficulty_distribution.pdf
  - docs/audit/marble_task_difficulty_profile.md
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

EASY_THRESHOLD = 0.9       # reward > 0.9 → easy
HARD_THRESHOLD = 0.5       # reward <= 0.5 → hard
# medium: 0.5 < reward <= 0.9


# ---------------------------------------------------------------------------
# Load & rank
# ---------------------------------------------------------------------------

def load_summary(csv_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["mean_reward"] = float(row["mean_reward"])
            row["std_reward"] = float(row["std_reward"])
            row["success_rate"] = float(row["success_rate"])
            row["failure_rate"] = float(row["failure_rate"])
            row["n_episodes"] = int(row["n_episodes"])
            row["n_real_engine"] = int(row["n_real_engine"])
            rows.append(row)
    return rows


def rank_tasks(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort ascending by mean_reward (hardest first)."""
    return sorted(rows, key=lambda r: (r["mean_reward"], r["domain"], r["task_id"]))


def classify_difficulty(row: dict[str, Any]) -> str:
    r = row["mean_reward"]
    if r > EASY_THRESHOLD:
        return "easy"
    elif r > HARD_THRESHOLD:
        return "medium"
    else:
        return "hard"


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_ranking_csv(
    ranked: list[dict[str, Any]], output_path: Path
) -> None:
    fieldnames = [
        "rank", "task_id", "domain", "difficulty",
        "mean_reward", "std_reward", "success_rate", "failure_rate",
        "n_episodes", "n_real_engine",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, row in enumerate(ranked, 1):
            out = dict(row)
            out["rank"] = i
            out["difficulty"] = classify_difficulty(row)
            writer.writerow({k: out[k] for k in fieldnames})


def generate_distribution_figure(
    ranked: list[dict[str, Any]], output_path: Path
) -> None:
    """Generate task difficulty distribution histogram."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not available, skipping figure generation")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    domains = sorted(set(r["domain"] for r in ranked))
    fig, axes = plt.subplots(
        1, len(domains), figsize=(4 * len(domains), 4), sharey=True
    )
    if len(domains) == 1:
        axes = [axes]

    for ax, domain in zip(axes, domains):
        rewards = [r["mean_reward"] for r in ranked if r["domain"] == domain]
        ax.hist(rewards, bins=10, range=(0, 1), color="steelblue", edgecolor="white")
        ax.axvline(HARD_THRESHOLD, color="red", linestyle="--", alpha=0.7, label=f"hard ≤{HARD_THRESHOLD}")
        ax.axvline(EASY_THRESHOLD, color="green", linestyle="--", alpha=0.7, label=f"easy >{EASY_THRESHOLD}")
        ax.set_title(domain)
        ax.set_xlabel("mean_reward")
        ax.set_ylabel("n_tasks")
        ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Figure saved: {output_path}")


def generate_report(
    ranked: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Generate markdown difficulty profile report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    domains = sorted(set(r["domain"] for r in ranked))

    lines: list[str] = []
    lines.append("# MARBLE Task Difficulty Profile\n")
    lines.append(f"**Generated from**: {len(ranked)} tasks across {len(domains)} domains\n")
    lines.append(f"**Thresholds**: easy (>{EASY_THRESHOLD}), medium ({HARD_THRESHOLD}–{EASY_THRESHOLD}), hard (≤{HARD_THRESHOLD})\n")

    lines.append("\n## 1. Per-Domain Distribution\n")
    lines.append("| Domain | Total | Easy | Medium | Hard | Easy% | Med% | Hard% |")
    lines.append("|--------|-------|------|--------|------|-------|------|-------|")

    for domain in domains:
        d_rows = [r for r in ranked if r["domain"] == domain]
        easy = [r for r in d_rows if classify_difficulty(r) == "easy"]
        medium = [r for r in d_rows if classify_difficulty(r) == "medium"]
        hard = [r for r in d_rows if classify_difficulty(r) == "hard"]
        n = len(d_rows)
        lines.append(
            f"| {domain} | {n} | {len(easy)} | {len(medium)} | {len(hard)} "
            f"| {len(easy)/n*100:.0f}% | {len(medium)/n*100:.0f}% | {len(hard)/n*100:.0f}% |"
        )

    lines.append("\n## 2. Hardest Tasks (Top 20)\n")
    lines.append("| Rank | Domain | Task ID | Mean Reward | Std | Failure Rate |")
    lines.append("|------|--------|---------|-------------|-----|--------------|")
    for i, r in enumerate(ranked[:20], 1):
        lines.append(
            f"| {i} | {r['domain']} | {r['task_id']} "
            f"| {r['mean_reward']:.4f} | {r['std_reward']:.4f} "
            f"| {r['failure_rate']:.2%} |"
        )

    lines.append("\n## 3. Easiest Tasks (Top 10)\n")
    lines.append("| Rank | Domain | Task ID | Mean Reward | Success Rate |")
    lines.append("|------|--------|---------|-------------|--------------|")
    for i, r in enumerate(reversed(ranked[-10:]), 1):
        lines.append(
            f"| {i} | {r['domain']} | {r['task_id']} "
            f"| {r['mean_reward']:.4f} | {r['success_rate']:.2%} |"
        )

    lines.append("\n## 4. Implications\n")
    hard_total = sum(1 for r in ranked if classify_difficulty(r) == "hard")
    med_total = sum(1 for r in ranked if classify_difficulty(r) == "medium")
    easy_total = sum(1 for r in ranked if classify_difficulty(r) == "easy")
    lines.append(f"- **Hard tasks** ({hard_total}): memory opportunity exists — baseline fails")
    lines.append(f"- **Medium tasks** ({med_total}): partial opportunity — some improvement margin")
    lines.append(f"- **Easy tasks** ({easy_total}): ceiling effect — memory unlikely to help")
    lines.append(f"\n**Recommendation**: Select pilot tasks from hard + medium categories.")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze MARBLE task difficulty distribution"
    )
    parser.add_argument(
        "--input-dir", type=str,
        default=str(_PROJECT_ROOT / "results" / "marble" / "difficulty_profile"),
        help="Input directory with difficulty_summary.csv",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: same as input)",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load
    summary_path = input_dir / "difficulty_summary.csv"
    if not summary_path.exists():
        print(f"ERROR: {summary_path} not found. Run run_difficulty_profile.py first.")
        sys.exit(1)

    rows = load_summary(summary_path)
    ranked = rank_tasks(rows)

    # Write ranking
    ranking_path = output_dir / "task_difficulty_ranking.csv"
    write_ranking_csv(ranked, ranking_path)
    print(f"Written: {ranking_path} ({len(ranked)} tasks)")

    # Generate figure
    fig_path = _PROJECT_ROOT / "figures" / "marble" / "task_difficulty_distribution.pdf"
    generate_distribution_figure(ranked, fig_path)

    # Generate report
    report_path = _PROJECT_ROOT / "docs" / "audit" / "marble_task_difficulty_profile.md"
    generate_report(ranked, report_path)


if __name__ == "__main__":
    main()
