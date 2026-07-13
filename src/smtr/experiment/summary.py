"""Summary computation for B0/B1/M0 comparison experiment."""

from collections import Counter
from statistics import median
from typing import Any

from smtr.experiment.schemas import (
    ComparisonRunRecord,
    ExperimentConfig,
    ExperimentSummary,
    MethodSummary,
)

# Canonical rejection reason categories.
# Every router decision reason must map to exactly one of these.
CANONICAL_REASONS = frozenset({
    "shared",
    "tau_lcb_nonpositive",
    "negative_risk_ucb_exceeded",
    "low_support",
    "share_budget_exceeded",
    "no_critic_available",
    "other",
})

# Mapping from raw router reason strings to canonical categories.
_REASON_MAP: dict[str, str] = {
    # Causal gate reasons
    "accepted": "shared",
    "tau_lcb_nonpositive": "tau_lcb_nonpositive",
    "negative_risk_ucb_exceeds_epsilon": "negative_risk_ucb_exceeded",
    # Legacy gate reason (pre-fix)
    "tau_lcb_below_threshold": "tau_lcb_nonpositive",
    # Sequential router reasons
    "critic_guided_share": "shared",
    "epsilon_exploration": "shared",
    "budget_exhausted": "share_budget_exceeded",
    "low_support": "low_support",
    "no_critic_estimate": "no_critic_available",
    "no_critic_available": "no_critic_available",
    "negative_risk_veto": "negative_risk_ucb_exceeded",
    "high_uncertainty": "other",
    "tau_below_threshold": "tau_lcb_nonpositive",
    "missing_routing_card": "other",
    # B1 reasons
    "relevance_topk_selected": "shared",
    "relevance_topk_budget_exceeded": "share_budget_exceeded",
    # Baseline
    "baseline_no_memory_router": "shared",
    # Safety guard
    "safety_guard_veto": "other",
}


def canonicalize_reason(reason: str) -> str:
    """Map a raw router decision reason to a canonical category."""
    if reason in _REASON_MAP:
        return _REASON_MAP[reason]
    # Handle safety_guard_* prefixed reasons
    if reason.startswith("safety_guard_"):
        return "other"
    return "other"


def compute_summary(
    runs: list[ComparisonRunRecord],
    config: ExperimentConfig,
) -> ExperimentSummary:
    """Compute full experiment summary from run records."""
    # Group runs by method
    runs_by_method: dict[str, list[ComparisonRunRecord]] = {}
    for run in runs:
        runs_by_method.setdefault(run.method, []).append(run)

    # Compute per-method summaries
    method_summaries: dict[str, MethodSummary] = {}
    for method_id, method_runs in runs_by_method.items():
        summary = _method_summary(method_runs, method=method_id)
        # Add transfer metrics for non-B0 methods
        b0_runs = runs_by_method.get("B0", [])
        if b0_runs and method_id != "B0":
            b0_success_by_episode = _success_by_episode(b0_runs)
            summary = _add_transfer_metrics(summary, method_runs, b0_success_by_episode)
        # Add rejection metrics for M0-like methods (those using critic)
        if method_id in ("M0", "M0-Full", "A1-NoSet"):
            summary = _add_m0_rejection_metrics(summary, method_runs)
        method_summaries[method_id] = summary

    # Legacy: extract B0/B1/M0 for backward compatibility
    b0_summary = method_summaries.get("B0", MethodSummary(method="B0"))
    b1_summary = method_summaries.get("B1", MethodSummary(method="B1"))
    m0_summary = method_summaries.get("M0", MethodSummary(method="M0"))

    # Success delta vs B0 for all methods
    b0_sr = b0_summary.success_rate
    for method_id, summary in method_summaries.items():
        if method_id != "B0":
            method_summaries[method_id] = summary.model_copy(
                update={"success_delta_vs_b0": summary.success_rate - b0_sr}
            )
    # Update legacy references
    b1_summary = method_summaries.get("B1", b1_summary)
    m0_summary = method_summaries.get("M0", m0_summary)

    # M0 vs B1 comparison (legacy)
    m0_vs_b1 = _compute_m0_vs_b1(b1_summary, m0_summary)

    # Bootstrap CI
    bootstrap_ci = compute_bootstrap_ci(runs, config)

    # Check for invalid experiment (no_critic_available in M0-like methods)
    experiment_invalid = False
    invalid_reason = None
    for method_id in ("M0", "M0-Full", "A1-NoSet"):
        method_runs = runs_by_method.get(method_id, [])
        for run in method_runs:
            for trace in run.router_trace:
                for decision in trace.get("decisions", []):
                    raw_reason = decision.get("reason", "")
                    if canonicalize_reason(raw_reason) == "no_critic_available":
                        experiment_invalid = True
                        invalid_reason = (
                            "no_critic_available found in "
                            f"{method_id} run {run.episode_id}; "
                            "learned router has no critic"
                        )
                        break
                if experiment_invalid:
                    break
            if experiment_invalid:
                break
        if experiment_invalid:
            break

    return ExperimentSummary(
        b0=b0_summary,
        b1=b1_summary,
        m0=m0_summary,
        m0_vs_b1=m0_vs_b1,
        methods=method_summaries,
        bootstrap_ci=bootstrap_ci,
        experiment_invalid=experiment_invalid,
        invalid_reason=invalid_reason,
    )


def _method_summary(
    runs: list[ComparisonRunRecord], *, method: str
) -> MethodSummary:
    """Compute per-method summary statistics."""
    if not runs:
        return MethodSummary(method=method)

    n = len(runs)
    successes = sum(1 for r in runs if r.team_success)
    selected_sizes = [r.selected_count for r in runs]
    candidate_counts = [len(r.candidate_memory_ids) for r in runs]
    runtimes = [r.runtime_seconds for r in runs]
    all_withhold = sum(1 for r in runs if r.selected_count == 0)

    return MethodSummary(
        method=method,
        episode_count=n,
        success_rate=successes / n if n else 0.0,
        avg_selected_size=sum(selected_sizes) / n if n else 0.0,
        median_selected_size=float(median(selected_sizes)) if selected_sizes else 0.0,
        all_withhold_rate=all_withhold / n if n else 0.0,
        avg_candidate_count=sum(candidate_counts) / n if n else 0.0,
        mean_runtime=sum(runtimes) / n if n else 0.0,
    )


def _success_by_episode(
    runs: list[ComparisonRunRecord],
) -> dict[tuple[str, int], bool]:
    """Map (task_instance_id, generation_seed) → team_success for B0 runs."""
    result: dict[tuple[str, int], bool] = {}
    for run in runs:
        key = (run.task_instance_id, run.generation_seed)
        # B0 is deterministic, so just take the last/only value
        result[key] = run.team_success
    return result


def _add_transfer_metrics(
    summary: MethodSummary,
    runs: list[ComparisonRunRecord],
    b0_success_by_episode: dict[tuple[str, int], bool],
) -> MethodSummary:
    """Add policy-level transfer metrics to a method summary."""
    if not runs:
        return summary

    labels = [r.policy_level_transfer_label for r in runs if r.policy_level_transfer_label]
    if not labels:
        return summary

    n = len(labels)
    counts = Counter(labels)
    return summary.model_copy(
        update={
            "positive_transfer_rate": counts.get("positive_transfer", 0) / n,
            "negative_transfer_rate": counts.get("negative_transfer", 0) / n,
            "neutral_success_rate": counts.get("neutral_success", 0) / n,
            "neutral_failure_rate": counts.get("neutral_failure", 0) / n,
        }
    )


def _add_m0_rejection_metrics(
    summary: MethodSummary,
    runs: list[ComparisonRunRecord],
) -> MethodSummary:
    """Add M0-specific rejection reason metrics.

    Uses canonical reason mapping to ensure:
    - share_count + rejection_count == total_decisions
    - share_rate + sum(rejection_rates) == 1 (within float tolerance)
    - tau_lcb_nonpositive is never in other_reason_counts
    """
    if not runs:
        return summary

    total_decisions = 0
    share_count = 0
    tau_lcb_reject = 0
    neg_risk_reject = 0
    budget_reject = 0
    low_support_reject = 0
    no_critic_reject = 0
    other_reasons: Counter[str] = Counter()

    for run in runs:
        for trace in run.router_trace:
            for decision in trace.get("decisions", []):
                total_decisions += 1
                action = decision.get("action", "withhold")
                raw_reason = decision.get("reason", "")
                canonical = canonicalize_reason(raw_reason)

                if canonical == "shared":
                    share_count += 1
                elif canonical == "tau_lcb_nonpositive":
                    tau_lcb_reject += 1
                elif canonical == "negative_risk_ucb_exceeded":
                    neg_risk_reject += 1
                elif canonical == "share_budget_exceeded":
                    budget_reject += 1
                elif canonical == "low_support":
                    low_support_reject += 1
                elif canonical == "no_critic_available":
                    no_critic_reject += 1
                else:
                    other_reasons[raw_reason] += 1

    if total_decisions == 0:
        return summary

    return summary.model_copy(
        update={
            "share_decision_rate": share_count / total_decisions,
            "tau_lcb_rejection_rate": tau_lcb_reject / total_decisions,
            "negative_risk_ucb_rejection_rate": neg_risk_reject / total_decisions,
            "share_budget_rejection_rate": budget_reject / total_decisions,
            "low_support_rejection_rate": low_support_reject / total_decisions,
            "no_critic_rejection_rate": no_critic_reject / total_decisions,
            "other_reason_counts": dict(other_reasons),
        }
    )


def _compute_m0_vs_b1(
    b1: MethodSummary, m0: MethodSummary
) -> dict[str, float]:
    """Compute direct M0 vs B1 comparison metrics."""
    return {
        "success_difference": m0.success_rate - b1.success_rate,
        "negative_transfer_rate_difference": (
            (m0.negative_transfer_rate or 0.0) - (b1.negative_transfer_rate or 0.0)
        ),
        "positive_transfer_rate_difference": (
            (m0.positive_transfer_rate or 0.0) - (b1.positive_transfer_rate or 0.0)
        ),
        "average_selected_count_difference": (
            m0.avg_selected_size - b1.avg_selected_size
        ),
    }


def compute_bootstrap_ci(
    runs: list[ComparisonRunRecord],
    config: ExperimentConfig,
) -> dict[str, Any]:
    """Compute episode-level group bootstrap 95% CI.

    Bootstrap unit: (task_instance_id, generation_seed) group.
    M0 multiple traversal seeds belong to the same group.
    """
    import numpy as np

    # Group runs by (task_instance_id, generation_seed)
    groups: dict[tuple[str, int], list[ComparisonRunRecord]] = {}
    for run in runs:
        key = (run.task_instance_id, run.generation_seed)
        groups.setdefault(key, []).append(run)

    group_keys = sorted(groups.keys())
    if not group_keys:
        return {}

    rng = np.random.default_rng(config.bootstrap_seed)
    n_groups = len(group_keys)
    n_bootstrap = config.bootstrap_n

    b0_successes: list[list[float]] = [[] for _ in range(n_bootstrap)]
    b1_successes: list[list[float]] = [[] for _ in range(n_bootstrap)]
    m0_successes: list[list[float]] = [[] for _ in range(n_bootstrap)]

    # Methods that map to legacy B0/B1/M0 buckets for bootstrap CI
    b0_methods = {"B0"}
    b1_methods = {"B1", "B1-Top1", "B1-Top3", "B1-Matched"}
    m0_methods = {"M0", "M0-Full", "A1-NoSet"}

    for b in range(n_bootstrap):
        sampled_keys = rng.choice(n_groups, size=n_groups, replace=True)
        for idx in sampled_keys:
            key = group_keys[idx]
            for run in groups[key]:
                if run.failure_reason is not None:
                    continue
                success = float(run.team_success)
                if run.method in b0_methods:
                    b0_successes[b].append(success)
                elif run.method in b1_methods:
                    b1_successes[b].append(success)
                elif run.method in m0_methods:
                    m0_successes[b].append(success)

    def _percentiles(values_list: list[list[float]]) -> dict[str, float]:
        rates = [
            sum(v) / len(v) if v else 0.0 for v in values_list
        ]
        if not rates:
            return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0}
        return {
            "mean": float(np.mean(rates)),
            "ci_low": float(np.percentile(rates, 2.5)),
            "ci_high": float(np.percentile(rates, 97.5)),
        }

    return {
        "bootstrap_seed": config.bootstrap_seed,
        "n_bootstrap": n_bootstrap,
        "n_groups": n_groups,
        "b0_success_rate": _percentiles(b0_successes),
        "b1_success_rate": _percentiles(b1_successes),
        "m0_success_rate": _percentiles(m0_successes),
    }


def compute_transfer_label(
    method_success: bool, b0_success: bool
) -> str:
    """Compute policy-level transfer label.

    method=1, B0=0 → positive_transfer
    method=0, B0=1 → negative_transfer
    method=1, B0=1 → neutral_success
    method=0, B0=0 → neutral_failure
    """
    m = int(bool(method_success))
    b = int(bool(b0_success))
    if m == 1 and b == 0:
        return "positive_transfer"
    if m == 0 and b == 1:
        return "negative_transfer"
    if m == 1 and b == 1:
        return "neutral_success"
    return "neutral_failure"
