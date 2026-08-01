"""Generate diagnostic decision and report.

Reads diagnostic_pair_summary.json and produces:
- decision.json (PROCEED / ADJUST_AND_RERUN / STOP_CURRENT_FORMULATION)
- diagnostic_report.md
- representative_pairs.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def generate_diagnostic_decision(
    summary_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Generate decision and report from analysis summary."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if not summary_path.exists():
        return {"error": "diagnostic_pair_summary.json not found"}

    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    total = summary.get("total_pairs", 0)
    valid = summary.get("valid_pairs", 0)
    invalid = summary.get("invalid_pairs", 0)
    failed = summary.get("failed_pairs", 0)
    validity_rate = summary.get("validity_rate", 0.0)
    positive = summary.get("positive_count", 0)
    negative = summary.get("negative_count", 0)
    neutral = summary.get("neutral_count", 0)
    neutral_rate = summary.get("neutral_rate", 0.0)
    score_effect = summary.get("score_effect", {})
    epsilon = summary.get("epsilon", 0.0)

    # Check conditions
    issues: list[str] = []
    blocking: list[str] = []

    # Validity
    if validity_rate < 0.7:
        blocking.append(f"LOW_VALIDITY: {validity_rate:.1%} < 70%")
    elif validity_rate < 0.9:
        issues.append(f"MODERATE_VALIDITY: {validity_rate:.1%} < 90%")

    # Neutral collapse
    if neutral_rate > 0.8:
        blocking.append(f"NEUTRAL_COLLAPSE: {neutral_rate:.1%} pairs are neutral")

    # Ceiling effect
    task_summary = summary.get("task_summary", [])
    ceiling_tasks = [
        t for t in task_summary
        if t.get("all_share_score_1", 0) == t.get("pairs", 0) and t["pairs"] > 0
    ]
    if len(ceiling_tasks) > len(task_summary) * 0.5:
        blocking.append(
            f"CEILING_EFFECT: {len(ceiling_tasks)}/{len(task_summary)} tasks "
            f"score 1.0 in both branches"
        )

    # Positive/negative pair count
    if positive < 3:
        issues.append(f"FEW_POSITIVE: only {positive} positive pairs")
    if negative < 3:
        issues.append(f"FEW_NEGATIVE: only {negative} negative pairs")

    # Memory type differentiation
    mem_summary = summary.get("memory_type_summary", [])
    beneficial_mean = next(
        (m["mean_effect"] for m in mem_summary if m["memory_type"] == "beneficial"), 0
    )
    conflicting_mean = next(
        (m["mean_effect"] for m in mem_summary if m["memory_type"] == "conflicting"), 0
    )
    if beneficial_mean <= 0 and conflicting_mean >= 0:
        issues.append(
            "NO_MEMORY_TYPE_DIFFERENTIATION: beneficial not positive, "
            "conflicting not negative"
        )

    # Order effect
    order_effect = summary.get("order_effect", [])
    if len(order_effect) >= 2:
        order_means = [o["mean_effect"] for o in order_effect]
        if len(order_means) == 2 and abs(order_means[0] - order_means[1]) > 0.5:
            blocking.append(
                f"ORDER_EFFECT: execution order effect delta = "
                f"{abs(order_means[0] - order_means[1]):.2f}"
            )

    # Runtime visibility
    visibility_violations = sum(
        1 for t in task_summary
        if True  # placeholder - check from raw results
    )

    # Decision
    if blocking:
        if any("VALIDITY" in b or "CEILING" in b or "ORDER" in b for b in blocking):
            decision = "STOP_CURRENT_FORMULATION"
        else:
            decision = "ADJUST_AND_RERUN"
    elif neutral_rate > 0.8 or positive < 5 or negative < 5:
        decision = "ADJUST_AND_RERUN"
    else:
        decision = "PROCEED"

    decision_obj = {
        "decision": decision,
        "blocking_issues": blocking,
        "concerns": issues,
        "metrics": {
            "total_pairs": total,
            "valid_pairs": valid,
            "invalid_pairs": invalid,
            "failed_pairs": failed,
            "validity_rate": validity_rate,
            "positive_count": positive,
            "negative_count": negative,
            "neutral_count": neutral,
            "neutral_rate": neutral_rate,
            "mean_score_effect": score_effect.get("mean", 0.0),
            "median_score_effect": score_effect.get("median", 0.0),
        },
    }

    # Write decision.json
    (output_dir / "decision.json").write_text(
        json.dumps(decision_obj, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    # Write diagnostic_report.md
    md = _render_report(summary, decision_obj)
    (output_dir / "diagnostic_report.md").write_text(md, encoding="utf-8")

    return decision_obj


def _render_report(summary: dict, decision: dict) -> str:
    """Render the diagnostic report as Markdown."""
    lines: list[str] = []
    lines.append("# Experiment A: 64-Pair Transfer Diagnostic Report\n")

    lines.append("## 1. Overall Statistics\n")
    m = decision["metrics"]
    lines.append(f"- Total pairs: {m['total_pairs']}")
    lines.append(f"- Valid pairs: {m['valid_pairs']}")
    lines.append(f"- Invalid pairs: {m['invalid_pairs']}")
    lines.append(f"- Failed pairs: {m['failed_pairs']}")
    lines.append(f"- Validity rate: {m['validity_rate']:.1%}")
    lines.append(f"- Positive: {m['positive_count']}")
    lines.append(f"- Negative: {m['negative_count']}")
    lines.append(f"- Neutral: {m['neutral_count']}")
    lines.append(f"- Mean score effect: {m['mean_score_effect']:+.3f}")
    lines.append(f"- Median score effect: {m['median_score_effect']:+.3f}")
    lines.append("")

    se = summary.get("score_effect", {})
    if se:
        lines.append("### Score Effect Distribution\n")
        lines.append(f"- Min: {se.get('min', 0):+.3f}")
        lines.append(f"- Max: {se.get('max', 0):+.3f}")
        lines.append(f"- StDev: {se.get('stdev', 0):.3f}")
        lines.append("")

    # Token/round effects
    for metric in ("tokens", "rounds"):
        me = summary.get(f"{metric}_effect", {})
        if me:
            lines.append(f"### {metric.title()} Effect\n")
            lines.append(f"- Mean delta: {me.get('mean', 0):+.1f}")
            lines.append(f"- Median delta: {me.get('median', 0):+.1f}")
            lines.append(f"- Count with data: {me.get('count', 0)}")
            lines.append("")

    lines.append("## 2. Memory Type Analysis\n")
    lines.append("| Memory Type | Pairs | Mean Effect | Median | Positive | Neutral | Negative |")
    lines.append("|-------------|------:|------------:|-------:|---------:|--------:|---------:|")
    for mt in summary.get("memory_type_summary", []):
        lines.append(
            f"| {mt['memory_type']} | {mt['pairs']} | "
            f"{mt['mean_effect']:+.3f} | {mt['median_effect']:+.3f} | "
            f"{mt['positive']} | {mt['neutral']} | {mt['negative']} |"
        )
    lines.append("")

    lines.append("## 3. Task Analysis\n")
    lines.append("| Task | Pairs | Mean Effect | Positive | Negative | Neutral | Ceiling |")
    lines.append("|-----:|------:|------------:|---------:|---------:|--------:|--------:|")
    for t in summary.get("task_summary", []):
        ceiling = "YES" if t.get("all_share_score_1", 0) == t["pairs"] else ""
        lines.append(
            f"| {t['task_id']} | {t['pairs']} | "
            f"{t['mean_effect']:+.3f} | {t['positive']} | {t['negative']} | "
            f"{t['neutral']} | {ceiling} |"
        )
    lines.append("")

    lines.append("## 4. Order Effect\n")
    for o in summary.get("order_effect", []):
        lines.append(
            f"- {o['execution_order']}: {o['pairs']} pairs, "
            f"mean effect = {o['mean_effect']:+.3f}"
        )
    lines.append("")

    lines.append("## 5. Invalidity Analysis\n")
    inv = summary.get("invalidity", {})
    reasons = inv.get("reasons", {})
    if reasons:
        for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
            lines.append(f"- {reason}: {count}")
    else:
        lines.append("- No invalid pairs.")
    lines.append("")

    lines.append("## 6. Decision\n")
    lines.append(f"**{decision['decision']}**\n")
    if decision.get("blocking_issues"):
        lines.append("Blocking issues:")
        for issue in decision["blocking_issues"]:
            lines.append(f"- {issue}")
        lines.append("")
    if decision.get("concerns"):
        lines.append("Concerns:")
        for concern in decision["concerns"]:
            lines.append(f"- {concern}")
    lines.append("")

    return "\n".join(lines)


def select_representative_pairs(
    run_dir: Path,
    output_path: Path,
    n: int = 10,
) -> list[dict]:
    """Select representative pairs for case studies."""
    results_path = run_dir / "diagnostic_results.jsonl"
    if not results_path.exists():
        return []

    results = [
        json.loads(line)
        for line in results_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    valid = [r for r in results if r.get("status") == "valid_complete"]

    # Select by category
    selected: list[dict] = []
    categories = {
        "beneficial_positive": lambda r: (
            r.get("memory_type") == "beneficial"
            and r.get("treatment_effect", {}).get("score_delta", 0) > 0
        ),
        "conflicting_negative": lambda r: (
            r.get("memory_type") == "conflicting"
            and r.get("treatment_effect", {}).get("score_delta", 0) < 0
        ),
        "role_mismatched": lambda r: r.get("memory_type") == "role_mismatched",
        "irrelevant_neutral": lambda r: (
            r.get("memory_type") == "irrelevant"
            and abs(r.get("treatment_effect", {}).get("score_delta", 0)) < 0.01
        ),
        "largest_positive": lambda r: r.get("treatment_effect", {}).get("score_delta", 0) > 0,
        "largest_negative": lambda r: r.get("treatment_effect", {}).get("score_delta", 0) < 0,
    }

    for cat_name, predicate in categories.items():
        matches = [r for r in valid if predicate(r)]
        if matches:
            # Pick the one with largest absolute effect
            best = max(matches, key=lambda r: abs(r.get("treatment_effect", {}).get("score_delta", 0)))
            entry = {
                "category": cat_name,
                "pair_key": best.get("pair_key"),
                "task_id": best.get("task_id"),
                "memory_id": best.get("memory_id"),
                "memory_type": best.get("memory_type"),
                "seed": best.get("seed"),
                "score_delta": best.get("treatment_effect", {}).get("score_delta"),
                "share_success": best.get("share_success"),
                "withhold_success": best.get("withhold_success"),
                "share_metrics": best.get("share_metrics"),
                "withhold_metrics": best.get("withhold_metrics"),
            }
            selected.append(entry)

    # Deduplicate by pair_key
    seen = set()
    deduped = []
    for s in selected:
        if s["pair_key"] not in seen:
            seen.add(s["pair_key"])
            deduped.append(s)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(deduped[:n], indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return deduped[:n]


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Generate diagnostic decision")
    parser.add_argument(
        "--run-dir",
        default="artifacts/paper_experiments/diagnostic_64/run_output",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/paper_experiments/diagnostic_64",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir)

    summary_path = run_dir / "diagnostic_pair_summary.json"
    decision = generate_diagnostic_decision(summary_path, output_dir)
    print(json.dumps(decision, indent=2, sort_keys=True))

    # Also select representative pairs
    reps = select_representative_pairs(
        run_dir, output_dir / "representative_pairs.json"
    )
    print(f"\nSelected {len(reps)} representative pairs")


if __name__ == "__main__":
    main()
