"""Summary computation for SMTR comparison experiments."""

from collections import Counter
from typing import Any

from smtr.evaluation.paired_statistics import (
    compute_group_bootstrap_ci,
    compute_method_summary,
)
from smtr.experiment.schemas import (
    ComparisonRunRecord,
    ExperimentConfig,
    ExperimentSummary,
    MethodSummary,
)

CANONICAL_REASONS = frozenset({
    "shared",
    "tau_lcb_nonpositive",
    "tau_mean_nonpositive",
    "negative_risk_ucb_exceeded",
    "negative_risk_mean_exceeded",
    "low_support",
    "share_budget_exceeded",
    "missing_routing_card",
    "other",
})

_REASON_MAP: dict[str, str] = {
    "accepted": "shared",
    "critic_guided_share": "shared",
    "epsilon_exploration": "shared",
    "relevance_topk_selected": "shared",
    "baseline_no_memory_router": "other",
    "tau_lcb_nonpositive": "tau_lcb_nonpositive",
    "tau_lcb_below_threshold": "tau_lcb_nonpositive",
    "tau_below_threshold": "tau_lcb_nonpositive",
    "tau_mean_nonpositive": "tau_mean_nonpositive",
    "negative_risk_ucb_exceeds_epsilon": "negative_risk_ucb_exceeded",
    "negative_risk_ucb_exceeded": "negative_risk_ucb_exceeded",
    "negative_risk_veto": "negative_risk_ucb_exceeded",
    "negative_risk_mean_exceeded": "negative_risk_mean_exceeded",
    "budget_exhausted": "share_budget_exceeded",
    "relevance_topk_budget_exceeded": "share_budget_exceeded",
    "low_support": "low_support",
    "missing_routing_card": "missing_routing_card",
    "high_uncertainty": "other",
}


def canonicalize_reason(reason: str) -> str:
    if reason in CANONICAL_REASONS:
        return reason
    if reason in _REASON_MAP:
        return _REASON_MAP[reason]
    if reason.startswith("safety_guard_"):
        return "other"
    return "other"


def compute_transfer_label(method_success: bool, b0_success: bool) -> str:
    if method_success and not b0_success:
        return "positive_transfer"
    if not method_success and b0_success:
        return "negative_transfer"
    if method_success and b0_success:
        return "neutral_success"
    return "neutral_failure"


def compute_summary(
    runs: list[ComparisonRunRecord],
    config: ExperimentConfig,
) -> ExperimentSummary:
    runs_by_method: dict[str, list[ComparisonRunRecord]] = {}
    for run in runs:
        runs_by_method.setdefault(run.method, []).append(run)

    method_summaries: dict[str, MethodSummary] = {}
    b0_success_rate = compute_method_summary("B0", runs).success_rate
    for method_id in sorted(runs_by_method):
        summary = compute_method_summary(method_id, runs)
        if method_id != "B0":
            summary = summary.model_copy(
                update={"success_delta_vs_b0": summary.success_rate - b0_success_rate}
            )
        if method_id in {"SMTR", "EffectOnly-SMTR", "Robust-SMTR"}:
            summary = _add_rejection_metrics(summary, runs_by_method[method_id])
        method_summaries[method_id] = summary

    bootstrap_ci = compute_paired_bootstrap_ci(runs, config)
    base_episode_ids = {run.base_episode_id for run in runs}
    return ExperimentSummary(
        methods=method_summaries,
        bootstrap_ci=bootstrap_ci,
        n_base_episodes=len(base_episode_ids),
        n_traversal_runs=len(runs),
        n_runtime_executions_by_method={
            method: len(method_runs) for method, method_runs in runs_by_method.items()
        },
        experiment_valid=True,
    )


def _add_rejection_metrics(
    summary: MethodSummary,
    runs: list[ComparisonRunRecord],
) -> MethodSummary:
    total_decisions = 0
    counts: Counter[str] = Counter()
    other: Counter[str] = Counter()
    for run in runs:
        for invocation in run.invocations:
            for decision in invocation.decisions:
                total_decisions += 1
                canonical = canonicalize_reason(decision.reason)
                if decision.action == "share":
                    canonical = "shared"
                counts[canonical] += 1
                if canonical == "other":
                    other[decision.reason] += 1
    if total_decisions == 0:
        return summary
    accounted = sum(counts[reason] for reason in CANONICAL_REASONS)
    if accounted != total_decisions:
        raise ValueError("rejection reason accounting mismatch")
    return summary.model_copy(
        update={
            "share_decision_rate": counts["shared"] / total_decisions,
            "tau_mean_rejection_rate": counts["tau_mean_nonpositive"] / total_decisions,
            "negative_risk_mean_rejection_rate": (
                counts["negative_risk_mean_exceeded"] / total_decisions
            ),
            "tau_lcb_rejection_rate": counts["tau_lcb_nonpositive"] / total_decisions
            if summary.method == "Robust-SMTR"
            else None,
            "negative_risk_ucb_rejection_rate": (
                counts["negative_risk_ucb_exceeded"] / total_decisions
            )
            if summary.method == "Robust-SMTR"
            else None,
            "confidence_level": _confidence_level_from_runs(runs)
            if summary.method == "Robust-SMTR"
            else None,
            "share_budget_rejection_rate": (
                counts["share_budget_exceeded"] / total_decisions
            ),
            "low_support_rejection_rate": counts["low_support"] / total_decisions,
            "other_reason_counts": dict(other),
        }
    )


def compute_paired_bootstrap_ci(
    runs: list[ComparisonRunRecord],
    config: ExperimentConfig,
) -> dict[str, Any]:
    methods = {run.method for run in runs}
    default_pairs = [
        ("SMTR", "B0"),
        ("SMTR", "B1-Top1"),
        ("SMTR", "B1-Matched"),
        ("SMTR", "EffectOnly-SMTR"),
    ]
    method_pairs = [
        (a, b) for a, b in default_pairs if a in methods and b in methods
    ]
    return compute_group_bootstrap_ci(
        runs,
        method_pairs=method_pairs,
        bootstrap_seed=config.bootstrap_seed,
        bootstrap_n=config.bootstrap_n,
    )


def _confidence_level_from_runs(runs: list[ComparisonRunRecord]) -> float | None:
    for run in runs:
        for invocation in run.invocations:
            for decision in invocation.decisions:
                diagnostics = decision.robust_diagnostics or {}
                if "confidence_level" in diagnostics:
                    return float(diagnostics["confidence_level"])
    return None
