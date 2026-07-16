"""Analyze paired causal pilot results.

Reads pilot_results.jsonl and pilot_status.json to produce
CSV/JSON summaries of treatment effects, validity rates,
and category-level breakdowns.
"""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Any


def analyze_paired_pilot(
    pilot_dir: Path,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Analyze paired pilot results.

    Parameters
    ----------
    pilot_dir:
        Directory containing pilot_results.jsonl and pilot_status.json.
    output_dir:
        Output directory for summaries (defaults to pilot_dir).

    Returns
    -------
    Summary dict with effect sizes, validity rates, category breakdowns.
    """
    if output_dir is None:
        output_dir = pilot_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = pilot_dir / "pilot_results.jsonl"
    if not results_path.exists():
        return {"error": "pilot_results.jsonl not found", "pilot_dir": str(pilot_dir)}

    results: list[dict[str, Any]] = []
    for line in results_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        results.append(json.loads(line))

    if not results:
        return {"error": "no results found", "pilot_dir": str(pilot_dir)}

    # Basic counts
    total = len(results)
    valid = sum(1 for r in results if r.get("status") == "valid_complete")
    invalid = sum(1 for r in results if r.get("status") == "invalid_complete")
    failed_count = sum(1 for r in results if r.get("status") == "failed")

    # Label distribution (among valid pairs)
    label_counts: dict[str, int] = {}
    for r in results:
        label = r.get("paired_label")
        if label:
            label_counts[label] = label_counts.get(label, 0) + 1

    # Treatment effects
    score_deltas = [
        r["treatment_effect"]["score_delta"]
        for r in results
        if "treatment_effect" in r
    ]
    effect_summary = {
        "count": len(score_deltas),
        "mean_delta": statistics.mean(score_deltas) if score_deltas else 0.0,
        "stdev_delta": statistics.stdev(score_deltas) if len(score_deltas) > 1 else 0.0,
        "positive_count": sum(1 for d in score_deltas if d > 0),
        "negative_count": sum(1 for d in score_deltas if d < 0),
        "zero_count": sum(1 for d in score_deltas if d == 0),
    }

    # Category breakdown
    category_effects: dict[str, list[float]] = {}
    for r in results:
        if "treatment_effect" not in r:
            continue
        candidate = r.get("candidate_memory", {})
        category = candidate.get("category", "unknown")
        category_effects.setdefault(category, []).append(
            r["treatment_effect"]["score_delta"]
        )

    category_summary: dict[str, dict[str, float]] = {}
    for cat, deltas in sorted(category_effects.items()):
        category_summary[cat] = {
            "count": len(deltas),
            "mean_delta": statistics.mean(deltas) if deltas else 0.0,
            "positive_rate": sum(1 for d in deltas if d > 0) / max(len(deltas), 1),
        }

    # Runtime visibility
    visibility_verified = sum(
        1 for r in results
        if r.get("share_runtime_visibility_verified")
        and r.get("withhold_runtime_visibility_verified")
    )

    # Build summary
    summary = {
        "pilot_dir": str(pilot_dir),
        "total_pairs": total,
        "valid_complete": valid,
        "invalid_complete": invalid,
        "failed": failed_count,
        "validity_rate": valid / max(total, 1),
        "label_counts": label_counts,
        "treatment_effect": effect_summary,
        "category_summary": category_summary,
        "runtime_visibility_both_verified": visibility_verified,
    }

    # Write outputs
    summary_path = output_dir / "pilot_analysis.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    # CSV of per-pair results
    csv_path = output_dir / "pilot_pair_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "pair_key", "status", "paired_label",
            "share_runtime_visibility_verified",
            "withhold_runtime_visibility_verified",
            "score_delta", "invalid_reason",
        ])
        writer.writeheader()
        for r in results:
            effect = r.get("treatment_effect", {})
            writer.writerow({
                "pair_key": r.get("pair_key", ""),
                "status": r.get("status", ""),
                "paired_label": r.get("paired_label", ""),
                "share_runtime_visibility_verified": r.get("share_runtime_visibility_verified", ""),
                "withhold_runtime_visibility_verified": r.get("withhold_runtime_visibility_verified", ""),
                "score_delta": effect.get("score_delta", ""),
                "invalid_reason": r.get("invalid_reason", ""),
            })

    return summary


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Analyze paired pilot results")
    parser.add_argument("--pilot-dir", required=True, help="Pilot output directory")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    args = parser.parse_args()
    result = analyze_paired_pilot(
        pilot_dir=Path(args.pilot_dir),
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
