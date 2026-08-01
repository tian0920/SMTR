"""Generate experiment readiness report.

Produces experiment_readiness_report.json and .md that answer:
1. Is there a method difference? (SMTR vs B0 vs AllShare)
2. Is there a non-neutral paired signal?
3. Are there experiment design issues? (ceiling effect, neutral collapse)
4. Should the experiment scale up?
"""

from __future__ import annotations

import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def generate_readiness_report(
    *,
    acceptance_analysis_path: Path | None = None,
    pilot_analysis_path: Path | None = None,
    output_dir: Path,
) -> dict[str, Any]:
    """Generate the experiment readiness report.

    Parameters
    ----------
    acceptance_analysis_path:
        Path to acceptance_analysis.json from Phase 5.
    pilot_analysis_path:
        Path to pilot_analysis.json from Phase 6.4.
    output_dir:
        Directory for report outputs.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "schema_version": "readiness_report_v1",
    }

    # --- Section 1: Acceptance batch analysis ---
    acceptance: dict[str, Any] = {}
    if acceptance_analysis_path and acceptance_analysis_path.exists():
        acceptance = json.loads(
            acceptance_analysis_path.read_text(encoding="utf-8")
        )

    method_summary = acceptance.get("method_summary", {})
    completeness = acceptance.get("completeness", {})

    # Check ceiling effect
    ceiling_effect = completeness.get("ceiling_effect", False)
    # Check neutral collapse
    neutral_collapse = completeness.get("neutral_collapse", False)
    # Check missing methods
    methods_missing = completeness.get("methods_missing", [])
    # Score variance
    score_variance = completeness.get("score_variance", 0.0)

    # --- Section 2: Method difference analysis ---
    b0_stats = method_summary.get("b0_no_memory", {})
    all_share_stats = method_summary.get("all_share", {})
    smtr_stats = method_summary.get("smtr", {})

    b0_rate = b0_stats.get("success_rate", 0.0)
    all_share_rate = all_share_stats.get("success_rate", 0.0)
    smtr_rate = smtr_stats.get("success_rate", 0.0)

    method_difference = {
        "b0_success_rate": b0_rate,
        "all_share_success_rate": all_share_rate,
        "smtr_success_rate": smtr_rate,
        "smtr_minus_b0": smtr_rate - b0_rate,
        "smtr_minus_all_share": smtr_rate - all_share_rate,
        "has_method_difference": abs(smtr_rate - b0_rate) > 0.05,
    }

    # --- Section 3: Paired signal analysis ---
    pilot: dict[str, Any] = {}
    if pilot_analysis_path and pilot_analysis_path.exists():
        pilot = json.loads(
            pilot_analysis_path.read_text(encoding="utf-8")
        )

    treatment_effect = pilot.get("treatment_effect", {})
    mean_delta = treatment_effect.get("mean_delta", 0.0)
    positive_count = treatment_effect.get("positive_count", 0)
    negative_count = treatment_effect.get("negative_count", 0)
    total_effect = treatment_effect.get("count", 0)

    paired_signal = {
        "mean_treatment_effect": mean_delta,
        "positive_transfers": positive_count,
        "negative_transfers": negative_count,
        "neutral_count": total_effect - positive_count - negative_count,
        "has_non_neutral_signal": abs(mean_delta) > 0.05,
        "signal_direction": (
            "positive" if mean_delta > 0.05
            else "negative" if mean_delta < -0.05
            else "neutral"
        ),
    }

    # --- Section 4: Experiment design issues ---
    design_issues: list[str] = []
    if ceiling_effect:
        design_issues.append(
            "CEILING_EFFECT: All runs scored 1.0; "
            "tasks may be too easy to discriminate methods."
        )
    if neutral_collapse:
        design_issues.append(
            "NEUTRAL_COLLAPSE: SMTR and B0 produce identical results "
            "for every task; no treatment effect detectable."
        )
    if methods_missing:
        design_issues.append(
            f"MISSING_METHODS: {', '.join(methods_missing)} not present in results."
        )
    if score_variance < 0.01 and not ceiling_effect:
        design_issues.append(
            "LOW_VARIANCE: Score variance is very low; "
            "may lack statistical power for paired tests."
        )

    validity_rate = pilot.get("validity_rate", 1.0)
    if validity_rate < 0.8:
        design_issues.append(
            f"LOW_VALIDITY_RATE: Only {validity_rate:.1%} of pilot pairs "
            "passed validity checks; investigate engine/runtime issues."
        )

    # --- Section 5: Readiness verdict ---
    ready_for_scale = (
        not ceiling_effect
        and not neutral_collapse
        and not methods_missing
        and abs(mean_delta) > 0.05
        and validity_rate >= 0.8
    )

    verdict = {
        "ready_for_scaled_experiment": ready_for_scale,
        "blocking_issues": design_issues,
        "recommendations": [],
    }

    if ceiling_effect:
        verdict["recommendations"].append(
            "Add harder tasks where B0 fails but SMTR/AllShare succeed."
        )
    if neutral_collapse:
        verdict["recommendations"].append(
            "Use more diverse candidate memories with stronger signal."
        )
    if not ready_for_scale and not design_issues:
        verdict["recommendations"].append(
            "Increase sample size; current effect may need more pairs."
        )
    if ready_for_scale:
        verdict["recommendations"].append(
            "Proceed with scaled experiment using current task/memory set."
        )

    # --- Assemble report ---
    report["acceptance_batch"] = {
        "total_runs": acceptance.get("total_runs", 0),
        "methods_analyzed": list(method_summary.keys()),
        "ceiling_effect": ceiling_effect,
        "neutral_collapse": neutral_collapse,
    }
    report["method_difference"] = method_difference
    report["paired_signal"] = paired_signal
    report["design_issues"] = design_issues
    report["verdict"] = verdict

    # Write JSON
    json_path = output_dir / "experiment_readiness_report.json"
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    # Write Markdown
    md_path = output_dir / "experiment_readiness_report.md"
    md_path.write_text(_render_markdown(report), encoding="utf-8")

    return report


def _render_markdown(report: dict[str, Any]) -> str:
    """Render report as Markdown."""
    lines: list[str] = []
    lines.append("# Experiment Readiness Report\n")
    lines.append(f"*Generated: {report['generated_at']}*\n")

    lines.append("## 1. Acceptance Batch Summary\n")
    acc = report.get("acceptance_batch", {})
    lines.append(f"- Total runs: {acc.get('total_runs', 0)}")
    lines.append(f"- Methods: {', '.join(acc.get('methods_analyzed', []))}")
    lines.append(f"- Ceiling effect: {'YES' if acc.get('ceiling_effect') else 'No'}")
    lines.append(f"- Neutral collapse: {'YES' if acc.get('neutral_collapse') else 'No'}")
    lines.append("")

    lines.append("## 2. Method Difference\n")
    md = report.get("method_difference", {})
    lines.append(f"- B0 success rate: {md.get('b0_success_rate', 0):.3f}")
    lines.append(f"- AllShare success rate: {md.get('all_share_success_rate', 0):.3f}")
    lines.append(f"- SMTR success rate: {md.get('smtr_success_rate', 0):.3f}")
    lines.append(f"- SMTR - B0: {md.get('smtr_minus_b0', 0):+.3f}")
    lines.append(f"- Has method difference: {'YES' if md.get('has_method_difference') else 'No'}")
    lines.append("")

    lines.append("## 3. Paired Signal\n")
    ps = report.get("paired_signal", {})
    lines.append(f"- Mean treatment effect: {ps.get('mean_treatment_effect', 0):+.3f}")
    lines.append(f"- Positive transfers: {ps.get('positive_transfers', 0)}")
    lines.append(f"- Negative transfers: {ps.get('negative_transfers', 0)}")
    lines.append(f"- Signal direction: {ps.get('signal_direction', 'unknown')}")
    lines.append("")

    lines.append("## 4. Design Issues\n")
    issues = report.get("design_issues", [])
    if issues:
        for issue in issues:
            lines.append(f"- **{issue}**")
    else:
        lines.append("- No blocking design issues detected.")
    lines.append("")

    lines.append("## 5. Verdict\n")
    v = report.get("verdict", {})
    ready = v.get("ready_for_scaled_experiment", False)
    lines.append(
        f"**Ready for scaled experiment: {'YES' if ready else 'NO'}**\n"
    )
    if v.get("blocking_issues"):
        lines.append("Blocking issues:")
        for issue in v["blocking_issues"]:
            lines.append(f"- {issue}")
        lines.append("")
    if v.get("recommendations"):
        lines.append("Recommendations:")
        for rec in v["recommendations"]:
            lines.append(f"- {rec}")

    return "\n".join(lines) + "\n"


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Generate experiment readiness report")
    parser.add_argument(
        "--acceptance-analysis",
        type=Path,
        default=None,
        help="Path to acceptance_analysis.json",
    )
    parser.add_argument(
        "--pilot-analysis",
        type=Path,
        default=None,
        help="Path to pilot_analysis.json",
    )
    parser.add_argument("--output-dir", required=True, help="Output directory")
    args = parser.parse_args()

    report = generate_readiness_report(
        acceptance_analysis_path=args.acceptance_analysis,
        pilot_analysis_path=args.pilot_analysis,
        output_dir=Path(args.output_dir),
    )
    print(json.dumps(report["verdict"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
