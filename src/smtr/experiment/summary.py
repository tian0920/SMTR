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
    "tau_mean_nonpositive",
    "negative_risk_mean_exceeded",
    "missing_routing_card",
    "factual_success_below_threshold",
    "other",
    # Robust and historical artifacts may still contain these reasons.
    "tau_lcb_nonpositive",
    "negative_risk_ucb_exceeded",
    "low_support",
    "share_budget_exceeded",
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
    b0_summary = compute_method_summary("B0", runs)
    b0_success_rate = b0_summary.success_rate
    b0_by_base = {run.base_episode_id: run.team_success for run in runs if run.method == "B0"}
    positive_opportunities = sum(1 for success in b0_by_base.values() if not success)
    negative_opportunities = sum(1 for success in b0_by_base.values() if success)
    base_count = len(b0_by_base)
    positive_opportunity_rate = (
        positive_opportunities / base_count if base_count else 0.0
    )
    negative_opportunity_rate = (
        negative_opportunities / base_count if base_count else 0.0
    )
    for method_id in sorted(runs_by_method):
        summary = compute_method_summary(method_id, runs)
        if method_id != "B0":
            summary = summary.model_copy(
                update={"success_delta_vs_b0": summary.success_rate - b0_success_rate}
            )
        summary = summary.model_copy(
            update={
                "opportunity_capture": (
                    (summary.positive_transfer_rate or 0.0) / positive_opportunity_rate
                    if positive_opportunity_rate > 0
                    and summary.positive_transfer_rate is not None
                    else None
                ),
                "safety_preservation": (
                    1.0
                    - (summary.negative_transfer_rate or 0.0)
                    / negative_opportunity_rate
                    if negative_opportunity_rate > 0
                    and summary.negative_transfer_rate is not None
                    else None
                ),
                "n_positive_transfer_opportunities": positive_opportunities,
                "n_negative_transfer_opportunities": negative_opportunities,
            "total_exposure_per_episode": summary.avg_selected_size,
            "mean_exposure_per_invocation": _mean_exposure_per_invocation(
                runs_by_method[method_id]
            ),
            "all_candidates_shared_rate": _all_candidates_shared_rate(
                runs_by_method[method_id]
            ),
            "payload_token_count": None,
            "mean_payload_tokens_per_invocation": None,
        }
        )
        if method_id in {
            "SMTR",
            "EffectOnly-SMTR",
            "Static-SMTR",
        }:
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
    effect_status: Counter[str] = Counter()
    risk_status: Counter[str] = Counter()
    divergence_count = 0
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
                if decision.effect_condition_status:
                    effect_status[decision.effect_condition_status] += 1
                if decision.risk_condition_status:
                    risk_status[decision.risk_condition_status] += 1
                if (
                    decision.selected_before_actual_digest
                    and decision.selected_before_critic_digest
                    and decision.selected_before_actual_digest
                    != decision.selected_before_critic_digest
                ):
                    divergence_count += 1
    if total_decisions == 0:
        return summary
    accounted = sum(counts[reason] for reason in CANONICAL_REASONS)
    if accounted != total_decisions:
        raise ValueError("rejection reason accounting mismatch")
    return summary.model_copy(
        update={
            "share_decision_rate": counts["shared"] / total_decisions,
            "tau_mean_rejection_rate": (
                counts["tau_mean_nonpositive"] / total_decisions
                if effect_status["not_applicable"] == 0
                else None
            ),
            "negative_risk_mean_rejection_rate": (
                counts["negative_risk_mean_exceeded"] / total_decisions
                if risk_status["not_applicable"] == 0
                else None
            ),
            "effect_condition_pass_rate": _condition_rate(effect_status, "passed"),
            "effect_condition_rejection_rate": _condition_rate(effect_status, "failed"),
            "risk_condition_pass_rate": _condition_rate(risk_status, "passed"),
            "risk_condition_rejection_rate": _condition_rate(risk_status, "failed"),
            "tau_lcb_rejection_rate": None,
            "negative_risk_ucb_rejection_rate": None,
            "confidence_level": None,
            "share_budget_rejection_rate": None,
            "low_support_rejection_rate": counts["low_support"] / total_decisions,
            "selected_set_conditioning_divergence_rate": (
                divergence_count / total_decisions if summary.method == "Static-SMTR" else None
            ),
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
        ("SMTR", "B1-AllCandidates"),
        ("SMTR", "B1-Matched"),
        ("SMTR", "EffectOnly-SMTR"),
        ("SMTR", "Static-SMTR"),
        ("SMTR", "FactualSuccess-SMTR"),
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


def _condition_rate(counts: Counter[str], status: str) -> float | None:
    applicable = counts["passed"] + counts["failed"]
    if applicable == 0:
        return None
    return counts[status] / applicable


def _mean_exposure_per_invocation(runs: list[ComparisonRunRecord]) -> float:
    exposures = []
    for run in runs:
        if run.number_of_invocations:
            exposures.append(run.total_memory_exposures / run.number_of_invocations)
    return sum(exposures) / len(exposures) if exposures else 0.0


def _all_candidates_shared_rate(runs: list[ComparisonRunRecord]) -> float | None:
    values = []
    for run in runs:
        for invocation in run.invocations:
            if invocation.candidate_memory_ids:
                values.append(
                    1.0
                    if set(invocation.selected_memory_ids)
                    == set(invocation.candidate_memory_ids)
                    else 0.0
                )
    return sum(values) / len(values) if values else None


def _confidence_level_from_runs(runs: list[ComparisonRunRecord]) -> float | None:
    for run in runs:
        for invocation in run.invocations:
            for decision in invocation.decisions:
                diagnostics = decision.robust_diagnostics or {}
                if "confidence_level" in diagnostics:
                    return float(diagnostics["confidence_level"])
    return None
