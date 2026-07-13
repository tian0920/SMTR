"""Rejection reason analysis with matched-case auditing.

For M0-Full and A1-NoSet:
1. Compute per-reason proportions and verify they sum to 1.
2. Find matched discordant cases (A1 share/M0 withhold, A1 withhold/M0 share).
3. Report candidate card metadata, predictions, and outcomes for each case.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from pydantic import BaseModel, Field

from smtr.experiment.candidate_diagnostics import (
    SCENARIO_TARGET_EFFECT,
    SCENARIO_TARGET_MEMORY,
)
from smtr.experiment.summary import canonicalize_reason


class ReasonProportions(BaseModel):
    """Per-method rejection reason proportions."""

    method: str
    total_decisions: int = 0
    shared: float = 0.0
    tau_lcb_nonpositive: float = 0.0
    negative_risk_ucb_exceeded: float = 0.0
    low_support: float = 0.0
    share_budget_exceeded: float = 0.0
    no_critic_available: float = 0.0
    other: float = 0.0
    sum_check: float = 0.0


class MatchedCase(BaseModel):
    """A discordant case where M0 and A1 disagree on a decision."""

    episode_id: str
    generation_seed: int
    memory_id: str
    case_type: str  # "A1_share_M0_withhold" or "A1_withhold_M0_share"
    candidate_memory_ids: list[str] = []
    selected_before_target_ids: list[str] = []
    a1_prediction: dict[str, Any] = {}
    m0_prediction: dict[str, Any] = {}
    ground_truth_effect: str = ""
    final_task_outcome: str = ""  # "success" or "failure"


class RejectionAnalysisResult(BaseModel):
    """Full rejection analysis for a scenario."""

    scenario: str
    m0_reasons: ReasonProportions = Field(default_factory=lambda: ReasonProportions(method="M0-Full"))
    a1_reasons: ReasonProportions = Field(default_factory=lambda: ReasonProportions(method="A1-NoSet"))
    matched_cases: list[MatchedCase] = []
    n_a1_share_m0_withhold: int = 0
    n_a1_withhold_m0_share: int = 0


def _compute_reason_proportions(
    runs: list[dict[str, Any]],
    *,
    method: str,
) -> ReasonProportions:
    """Compute per-reason proportions for a method."""
    method_runs = [r for r in runs if r.get("method") == method]
    counts: Counter[str] = Counter()
    total = 0

    for run in method_runs:
        for trace in run.get("router_trace", []):
            for dec in trace.get("decisions", []):
                total += 1
                raw_reason = dec.get("reason", "")
                canonical = canonicalize_reason(raw_reason)
                counts[canonical] += 1

    if total == 0:
        return ReasonProportions(method=method)

    props = {k: counts.get(k, 0) / total for k in [
        "shared", "tau_lcb_nonpositive", "negative_risk_ucb_exceeded",
        "low_support", "share_budget_exceeded", "no_critic_available", "other",
    ]}

    return ReasonProportions(
        method=method,
        total_decisions=total,
        sum_check=sum(props.values()),
        **props,
    )


def _extract_decision_for_memory(
    run: dict[str, Any],
    memory_id: str,
) -> dict[str, Any] | None:
    """Find the decision for a specific memory in a run."""
    for trace in run.get("router_trace", []):
        for dec in trace.get("decisions", []):
            if dec.get("memory_id") == memory_id:
                return dec
    return None


def _get_selected_before(
    run: dict[str, Any],
    target_mem_id: str,
) -> list[str]:
    """Get memory IDs selected before the target."""
    selected: list[str] = []
    for trace in run.get("router_trace", []):
        for dec in trace.get("decisions", []):
            mid = dec.get("memory_id", "")
            if mid == target_mem_id:
                break
            if dec.get("action") == "share":
                selected.append(mid)
    return selected


def compute_rejection_analysis(
    runs: list[dict[str, Any]],
    *,
    scenario: str,
    m0_method: str = "M0-Full",
    a1_method: str = "A1-NoSet",
) -> RejectionAnalysisResult:
    """Compute rejection reason analysis with matched-case audit.

    Args:
        runs: All run record dicts for the scenario.
        scenario: Scenario name.
        m0_method: Method ID for full SMTR.
        a1_method: Method ID for no-selected-set ablation.

    Returns:
        RejectionAnalysisResult with proportions and matched cases.
    """
    target_mem_id = SCENARIO_TARGET_MEMORY.get(scenario, "")
    target_effect = SCENARIO_TARGET_EFFECT.get(scenario, "neutral")

    m0_props = _compute_reason_proportions(runs, method=m0_method)
    a1_props = _compute_reason_proportions(runs, method=a1_method)

    # Index runs by (episode_id, generation_seed)
    m0_runs: dict[tuple[str, int], dict] = {}
    a1_runs: dict[tuple[str, int], dict] = {}
    for run in runs:
        key = (run.get("episode_id", ""), run.get("generation_seed", 0))
        if run.get("method") == m0_method:
            m0_runs[key] = run
        elif run.get("method") == a1_method:
            a1_runs[key] = run

    common_keys = sorted(set(m0_runs.keys()) & set(a1_runs.keys()))

    # Find matched discordant cases on the target memory
    matched_cases: list[MatchedCase] = []
    n_a1_share_m0_withhold = 0
    n_a1_withhold_m0_share = 0

    if not target_mem_id:
        return RejectionAnalysisResult(
            scenario=scenario,
            m0_reasons=m0_props,
            a1_reasons=a1_props,
        )

    for key in common_keys:
        m0_run = m0_runs[key]
        a1_run = a1_runs[key]

        m0_dec = _extract_decision_for_memory(m0_run, target_mem_id)
        a1_dec = _extract_decision_for_memory(a1_run, target_mem_id)

        if m0_dec is None or a1_dec is None:
            continue

        m0_action = m0_dec.get("action", "withhold")
        a1_action = a1_dec.get("action", "withhold")

        if a1_action == "share" and m0_action == "withhold":
            n_a1_share_m0_withhold += 1
            case = MatchedCase(
                episode_id=key[0],
                generation_seed=key[1],
                memory_id=target_mem_id,
                case_type="A1_share_M0_withhold",
                candidate_memory_ids=a1_run.get("candidate_memory_ids", []),
                selected_before_target_ids=_get_selected_before(a1_run, target_mem_id),
                a1_prediction={
                    "tau_mean": a1_dec.get("tau_mean"),
                    "tau_lcb": a1_dec.get("tau_lcb"),
                    "negative_risk_ucb": a1_dec.get("negative_risk_ucb"),
                    "action": a1_action,
                    "reason": a1_dec.get("reason"),
                },
                m0_prediction={
                    "tau_mean": m0_dec.get("tau_mean"),
                    "tau_lcb": m0_dec.get("tau_lcb"),
                    "negative_risk_ucb": m0_dec.get("negative_risk_ucb"),
                    "action": m0_action,
                    "reason": m0_dec.get("reason"),
                },
                ground_truth_effect=target_effect,
                final_task_outcome="success" if a1_run.get("team_success") else "failure",
            )
            matched_cases.append(case)

        elif a1_action == "withhold" and m0_action == "share":
            n_a1_withhold_m0_share += 1
            case = MatchedCase(
                episode_id=key[0],
                generation_seed=key[1],
                memory_id=target_mem_id,
                case_type="A1_withhold_M0_share",
                candidate_memory_ids=m0_run.get("candidate_memory_ids", []),
                selected_before_target_ids=_get_selected_before(m0_run, target_mem_id),
                a1_prediction={
                    "tau_mean": a1_dec.get("tau_mean"),
                    "tau_lcb": a1_dec.get("tau_lcb"),
                    "negative_risk_ucb": a1_dec.get("negative_risk_ucb"),
                    "action": a1_action,
                    "reason": a1_dec.get("reason"),
                },
                m0_prediction={
                    "tau_mean": m0_dec.get("tau_mean"),
                    "tau_lcb": m0_dec.get("tau_lcb"),
                    "negative_risk_ucb": m0_dec.get("negative_risk_ucb"),
                    "action": m0_action,
                    "reason": m0_dec.get("reason"),
                },
                ground_truth_effect=target_effect,
                final_task_outcome="success" if m0_run.get("team_success") else "failure",
            )
            matched_cases.append(case)

    return RejectionAnalysisResult(
        scenario=scenario,
        m0_reasons=m0_props,
        a1_reasons=a1_props,
        matched_cases=matched_cases,
        n_a1_share_m0_withhold=n_a1_share_m0_withhold,
        n_a1_withhold_m0_share=n_a1_withhold_m0_share,
    )


def compute_all_rejection_analyses(
    runs_by_scenario: dict[str, list[dict[str, Any]]],
) -> dict[str, RejectionAnalysisResult]:
    """Compute rejection analyses for all scenarios."""
    results: dict[str, RejectionAnalysisResult] = {}
    for scenario, runs in runs_by_scenario.items():
        results[scenario] = compute_rejection_analysis(runs, scenario=scenario)
    return results
