"""Pilot summary builder + GO/NO-GO decision (Phase 34-38).

Reads the full analysis output from ``analyze_mechanism_pilot.py`` and
produces:
  - ``pilot_summary.csv``  — 6-method comparison table
  - ``pilot_summary.json`` — structured summary
  - ``pilot_mechanism_report.md`` — human-readable report
  - ``go_no_go.json`` — GO/YELLOW/NO-GO decision with criteria

Usage::

    python analysis/rima_transfer/build_pilot_summary.py \\
        --analysis-dir results/rima_transfer/pilot/analysis \\
        --output-dir results/rima_transfer/pilot
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))

logger = logging.getLogger("rima.pilot_summary")

__all__ = ["build_summary", "evaluate_go_no_go"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_analysis(analysis_dir: Path) -> dict[str, Any]:
    """Load the full analysis JSON."""
    path = analysis_dir / "full_analysis.json"
    if not path.exists():
        raise FileNotFoundError(f"Analysis not found: {path}")
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Summary table builder
# ---------------------------------------------------------------------------


def _method_row(
    method: str, analysis: dict[str, Any],
) -> dict[str, Any]:
    """Build one summary row for a method."""
    score = analysis.get("score", {})
    gamma = analysis.get("gamma_routing", {})
    cost = analysis.get("cost_decomposition", {})
    transfer = analysis.get("transfer_prediction", {})
    uncertainty = analysis.get("uncertainty_audit", {})
    search = analysis.get("search_transfer_sanity", {})

    # Known reuse late vs early.
    r_curve = analysis.get("known_reuse_curve", [])
    r_early = r_curve[0].get("known_reuse_rate", 0) if r_curve else 0
    r_late = r_curve[-1].get("known_reuse_rate", 0) if r_curve else 0

    # Global exploration late vs early.
    g_curve = analysis.get("global_exploration_curve", [])
    g_early = g_curve[0].get("global_retrieval_rate", 0) if g_curve else 0
    g_late = g_curve[-1].get("global_retrieval_rate", 0) if g_curve else 0

    return {
        "method": method,
        "n_runs": analysis.get("n_runs", 0),
        "n_tasks": analysis.get("n_tasks_total", 0),
        "score_early": round(score.get("score_early", 0) or 0, 4),
        "score_late": round(score.get("score_late", 0) or 0, 4),
        "delta_score": round(score.get("delta_score", 0) or 0, 4),
        "global_rate_early": round(g_early, 4),
        "global_rate_late": round(g_late, 4),
        "known_reuse_early": round(r_early, 4),
        "known_reuse_late": round(r_late, 4),
        "p_exploit_only": gamma.get("p_exploit_only", 0),
        "p_explore_only": gamma.get("p_explore_only", 0),
        "n_probes": analysis.get("n_probes_total", 0),
        "transfer_mae_early": transfer.get("mae_early"),
        "transfer_mae_late": transfer.get("mae_late"),
        "delta_mae": transfer.get("delta_mae"),
        "sign_acc_late": transfer.get("sign_acc_late"),
        "lcb_coverage": uncertainty.get("lcb_coverage"),
        "corr_sigma_error": uncertainty.get("corr_sigma_abs_error"),
        "cost_per_task": cost.get("total_per_task"),
        "n_critic_predictions": search.get("n_critic_total", 0),
        "selection_rate": search.get("selection_rate", 0),
    }


def build_summary_table(
    methods: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build the full pilot summary table."""
    rows: list[dict[str, Any]] = []
    for method in sorted(methods.keys()):
        rows.append(_method_row(method, methods[method]))
    return rows


# ---------------------------------------------------------------------------
# Sub-tables
# ---------------------------------------------------------------------------


def build_sub_tables(
    methods: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    """Build early/middle/late, routing, cost, uncertainty sub-tables."""
    result: dict[str, list[dict[str, Any]]] = {
        "score_by_phase": [],
        "routing": [],
        "cost": [],
        "uncertainty": [],
    }

    for method in sorted(methods.keys()):
        m = methods[method]
        score = m.get("score", {})

        # Score by phase.
        result["score_by_phase"].append({
            "method": method,
            "early": round(score.get("score_early", 0) or 0, 4),
            "late": round(score.get("score_late", 0) or 0, 4),
            "delta": round(score.get("delta_score", 0) or 0, 4),
        })

        # Routing.
        gamma = m.get("gamma_routing", {})
        result["routing"].append({
            "method": method,
            "exploit_only": gamma.get("p_exploit_only", 0),
            "exploit_explore": gamma.get("p_exploit_explore", 0),
            "explore_only": gamma.get("p_explore_only", 0),
            "warnings": "; ".join(gamma.get("warnings", [])),
        })

        # Cost.
        cost = m.get("cost_decomposition", {})
        result["cost"].append({
            "method": method,
            "retrieval": cost.get("retrieval_per_task", 0),
            "model": cost.get("model_per_task", 0),
            "env_learning": cost.get("env_learning_per_task", 0),
            "total": cost.get("total_per_task", 0),
        })

        # Uncertainty.
        unc = m.get("uncertainty_audit", {})
        result["uncertainty"].append({
            "method": method,
            "lcb_coverage": unc.get("lcb_coverage"),
            "corr_sigma_error": unc.get("corr_sigma_abs_error"),
            "mean_sigma": unc.get("mean_sigma"),
        })

    return result


# ---------------------------------------------------------------------------
# Phase 34: Cost-matched baseline integration
# ---------------------------------------------------------------------------


def check_cost_matched_baseline(
    methods: dict[str, Any],
) -> dict[str, Any]:
    """Verify static_same_probe_budget uses same budget as adaptive."""
    adaptive = methods.get("rima_transfer_adaptive", {})
    static = methods.get("rima_static_same_probe_budget", {})

    adaptive_probes = adaptive.get("n_probes_total", 0)
    static_probes = static.get("n_probes_total", 0)

    return {
        "adaptive_probe_count": adaptive_probes,
        "static_probe_count": static_probes,
        "budget_matched": (
            adaptive_probes == static_probes if static else None
        ),
    }


# ---------------------------------------------------------------------------
# Phase 35: No-uncertainty ablation integration
# ---------------------------------------------------------------------------


def check_no_uncertainty_ablation(
    methods: dict[str, Any],
) -> dict[str, Any]:
    """Verify β=0 routing but σ still logged."""
    no_unc = methods.get("rima_transfer_no_uncertainty", {})
    if not no_unc:
        return {"present": False}

    uncertainty = no_unc.get("uncertainty_audit", {})
    # Even with β=0, sigma should still be computed for logging.
    has_sigma = uncertainty.get("mean_sigma") is not None

    return {
        "present": True,
        "sigma_still_logged": has_sigma,
        "mean_sigma": uncertainty.get("mean_sigma"),
    }


# ---------------------------------------------------------------------------
# Phase 38: GO/YELLOW/NO-GO Decision
# ---------------------------------------------------------------------------


def evaluate_go_no_go(
    table: list[dict[str, Any]],
    methods: dict[str, Any],
    comparisons: list[dict[str, Any]],
) -> dict[str, Any]:
    """Apply decision criteria and return GO/YELLOW/NO-GO verdict.

    GO criteria:
      1. GlobalSearch decreases (G_late < G_early for adaptive)
      2. KnownReuse_late > KnownReuse_early (for adaptive)
      3. Score(adaptive) ≥ Score(static) - 0.02
      4. LateScore(adaptive) > LateScore(frozen)
      5. TransferMAE_late < TransferMAE_early (for adaptive)

    YELLOW: Frozen ≈ Adaptive, frozen reduces global retrieval.
    NO-GO: any strong negative signal.
    """
    # Find key methods.
    adaptive = next(
        (r for r in table if r["method"] == "rima_transfer_adaptive"), None,
    )
    frozen = next(
        (r for r in table if r["method"] == "rima_transfer_frozen"), None,
    )
    static = next(
        (r for r in table if r["method"] == "rima_static_same_probe_budget"),
        None,
    )
    receiver = next(
        (r for r in table if r["method"] == "rima_receiver"), None,
    )

    criteria: dict[str, dict[str, Any]] = {}
    go_signals = 0
    yellow_signals = 0
    no_go_signals = 0

    # Criterion 1: Global search decreases.
    if adaptive:
        g_decrease = adaptive["global_rate_late"] < adaptive["global_rate_early"]
        criteria["global_search_decrease"] = {
            "pass": g_decrease,
            "early": adaptive["global_rate_early"],
            "late": adaptive["global_rate_late"],
        }
        if g_decrease:
            go_signals += 1
    else:
        criteria["global_search_decrease"] = {"pass": None, "reason": "no data"}

    # Criterion 2: Known reuse increases.
    if adaptive:
        r_increase = adaptive["known_reuse_late"] > adaptive["known_reuse_early"]
        criteria["known_reuse_increase"] = {
            "pass": r_increase,
            "early": adaptive["known_reuse_early"],
            "late": adaptive["known_reuse_late"],
        }
        if r_increase:
            go_signals += 1
    else:
        criteria["known_reuse_increase"] = {"pass": None, "reason": "no data"}

    # Criterion 3: Score ≥ static - 0.02.
    if adaptive and static:
        margin = adaptive["score_late"] - static["score_late"]
        score_ok = margin >= -0.02
        criteria["score_vs_static"] = {
            "pass": score_ok,
            "adaptive_late": adaptive["score_late"],
            "static_late": static["score_late"],
            "margin": round(margin, 4),
        }
        if score_ok:
            go_signals += 1
        elif margin < -0.05:
            no_go_signals += 1
    else:
        criteria["score_vs_static"] = {"pass": None, "reason": "no data"}

    # Criterion 4: LateScore(adaptive) > LateScore(frozen).
    if adaptive and frozen:
        late_better = adaptive["score_late"] > frozen["score_late"]
        criteria["adaptive_late_gt_frozen"] = {
            "pass": late_better,
            "adaptive_late": adaptive["score_late"],
            "frozen_late": frozen["score_late"],
        }
        if late_better:
            go_signals += 1
        elif abs(adaptive["score_late"] - frozen["score_late"]) < 0.01:
            yellow_signals += 1
    else:
        criteria["adaptive_late_gt_frozen"] = {
            "pass": None, "reason": "no data",
        }

    # Criterion 5: TransferMAE_late < TransferMAE_early.
    if adaptive and adaptive.get("transfer_mae_early") is not None:
        mae_improve = (
            adaptive["transfer_mae_late"] is not None
            and adaptive["transfer_mae_late"] < adaptive["transfer_mae_early"]
        )
        criteria["transfer_mae_improves"] = {
            "pass": mae_improve,
            "mae_early": adaptive["transfer_mae_early"],
            "mae_late": adaptive["transfer_mae_late"],
        }
        if mae_improve:
            go_signals += 1
    else:
        criteria["transfer_mae_improves"] = {
            "pass": None, "reason": "no data",
        }

    # Final decision.
    if no_go_signals > 0:
        decision = "NO-GO"
    elif go_signals >= 4:
        decision = "GO"
    elif go_signals >= 2 or yellow_signals > 0:
        decision = "YELLOW"
    elif go_signals == 0 and not any(
        c.get("pass") for c in criteria.values()
    ):
        decision = "INSUFFICIENT_DATA"
    else:
        decision = "YELLOW"

    return {
        "decision": decision,
        "go_signals": go_signals,
        "yellow_signals": yellow_signals,
        "no_go_signals": no_go_signals,
        "criteria": criteria,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Markdown report builder
# ---------------------------------------------------------------------------


def build_markdown_report(
    table: list[dict[str, Any]],
    sub_tables: dict[str, list[dict[str, Any]]],
    go_result: dict[str, Any],
    cost_check: dict[str, Any],
    ablation_check: dict[str, Any],
    comparisons: list[dict[str, Any]],
) -> str:
    """Build human-readable mechanism report."""
    lines: list[str] = []
    lines.append("# RIMA-Transfer Pilot Mechanism Report")
    lines.append("")
    lines.append(f"Generated: {go_result.get('timestamp', 'N/A')}")
    lines.append("")

    # Summary table.
    lines.append("## 6-Method Summary")
    lines.append("")
    lines.append(
        "| Method | Score Early | Score Late | ΔScore | "
        "G_early | G_late | R_early | R_late | Transfer MAE |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in table:
        lines.append(
            f"| {r['method']} | {r['score_early']} | {r['score_late']} | "
            f"{r['delta_score']} | {r['global_rate_early']} | "
            f"{r['global_rate_late']} | {r['known_reuse_early']} | "
            f"{r['known_reuse_late']} | "
            f"{r.get('transfer_mae_late', 'N/A')} |"
        )
    lines.append("")

    # Routing sub-table.
    lines.append("## Routing Modes")
    lines.append("")
    lines.append("| Method | Exploit Only | Exploit-Explore | Explore Only |")
    lines.append("|---|---|---|---|")
    for r in sub_tables.get("routing", []):
        lines.append(
            f"| {r['method']} | {r['exploit_only']} | "
            f"{r['exploit_explore']} | {r['explore_only']} |"
        )
    lines.append("")

    # Cost sub-table.
    lines.append("## Cost Decomposition (per task)")
    lines.append("")
    lines.append("| Method | Retrieval | Model | Env-Learning | Total |")
    lines.append("|---|---|---|---|---|")
    for r in sub_tables.get("cost", []):
        lines.append(
            f"| {r['method']} | {r['retrieval']} | {r['model']} | "
            f"{r['env_learning']} | {r['total']} |"
        )
    lines.append("")

    # Uncertainty sub-table.
    lines.append("## Uncertainty Audit")
    lines.append("")
    lines.append("| Method | LCB Coverage | corr(σ, |err|) | Mean σ |")
    lines.append("|---|---|---|---|")
    for r in sub_tables.get("uncertainty", []):
        lines.append(
            f"| {r['method']} | {r.get('lcb_coverage', 'N/A')} | "
            f"{r.get('corr_sigma_error', 'N/A')} | "
            f"{r.get('mean_sigma', 'N/A')} |"
        )
    lines.append("")

    # Paired comparisons.
    if comparisons:
        lines.append("## Paired Comparisons (vs adaptive)")
        lines.append("")
        lines.append("| Method B | Metric | Mean Diff | CI Lower | CI Upper |")
        lines.append("|---|---|---|---|---|")
        for c in comparisons:
            lines.append(
                f"| {c['method_b']} | {c['metric']} | {c['mean_diff']} | "
                f"{c['ci_lower']} | {c['ci_upper']} |"
            )
        lines.append("")

    # Integration checks.
    lines.append("## Integration Checks")
    lines.append("")
    lines.append(f"- Cost-matched baseline: {cost_check}")
    lines.append(f"- No-uncertainty ablation: {ablation_check}")
    lines.append("")

    # GO/NO-GO.
    lines.append("## GO / YELLOW / NO-GO Decision")
    lines.append("")
    lines.append(f"**Decision: {go_result['decision']}**")
    lines.append("")
    lines.append(
        f"GO signals: {go_result['go_signals']}, "
        f"YELLOW: {go_result['yellow_signals']}, "
        f"NO-GO: {go_result['no_go_signals']}"
    )
    lines.append("")
    for name, detail in go_result.get("criteria", {}).items():
        status = "PASS" if detail.get("pass") else (
            "FAIL" if detail.get("pass") is False else "N/A"
        )
        lines.append(f"- **{name}**: {status} — {detail}")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------


def build_summary(
    analysis_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Build complete pilot summary and write all outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load analysis.
    full = _load_analysis(analysis_dir)
    methods = full.get("methods", {})
    comparisons = full.get("paired_comparisons", [])

    if not methods:
        logger.warning("No method results found in analysis.")
        return {}

    # Build summary table.
    table = build_summary_table(methods)

    # Build sub-tables.
    sub_tables = build_sub_tables(methods)

    # Integration checks (Phase 34-35).
    cost_check = check_cost_matched_baseline(methods)
    ablation_check = check_no_uncertainty_ablation(methods)

    # GO/NO-GO (Phase 38).
    go_result = evaluate_go_no_go(table, methods, comparisons)

    # --- Write outputs ---

    # pilot_summary.csv
    if table:
        csv_path = output_dir / "pilot_summary.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=table[0].keys())
            writer.writeheader()
            writer.writerows(table)
        logger.info("Wrote %s", csv_path)

    # pilot_summary.json
    summary_json = {
        "table": table,
        "sub_tables": sub_tables,
        "cost_matched_baseline": cost_check,
        "no_uncertainty_ablation": ablation_check,
        "go_no_go": go_result,
    }
    json_path = output_dir / "pilot_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary_json, f, indent=2)
    logger.info("Wrote %s", json_path)

    # go_no_go.json
    go_path = output_dir / "go_no_go.json"
    with open(go_path, "w") as f:
        json.dump(go_result, f, indent=2)
    logger.info("Wrote %s", go_path)

    # pilot_mechanism_report.md
    report_md = build_markdown_report(
        table, sub_tables, go_result,
        cost_check, ablation_check, comparisons,
    )
    md_path = output_dir / "pilot_mechanism_report.md"
    with open(md_path, "w") as f:
        f.write(report_md)
    logger.info("Wrote %s", md_path)

    return summary_json


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Pilot summary builder + GO/NO-GO (Phase 34-38).",
    )
    parser.add_argument(
        "--analysis-dir",
        default="results/rima_transfer/pilot/analysis",
    )
    parser.add_argument(
        "--output-dir",
        default="results/rima_transfer/pilot",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    project_root = _PROJECT_ROOT
    analysis_dir = project_root / args.analysis_dir
    output_dir = project_root / args.output_dir

    try:
        result = build_summary(analysis_dir, output_dir)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    if result:
        decision = result.get("go_no_go", {}).get("decision", "UNKNOWN")
        print(f"Decision: {decision}")
        return 0
    else:
        print("No summary produced.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
