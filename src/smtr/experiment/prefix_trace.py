"""Prefix formation trace for ablation experiments.

For prefix-sensitive and flip scenarios, records per-target routing chain:
- Required prefix memories and their ranks in candidates
- Traversal order and whether prefix was selected before target
- Critic prediction on the target
- Target action and ground truth region

Summary metrics:
- prefix_candidate_recall
- prefix_order_success_rate
- prefix_selection_success_rate
- target_evaluated_under_correct_prefix_rate
- target_share_under_correct_prefix_rate
- success_given_correct_prefix
- success_without_correct_prefix
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from smtr.experiment.candidate_diagnostics import (
    SCENARIO_PREFIX_MEMORIES,
    SCENARIO_TARGET_EFFECT,
    SCENARIO_TARGET_MEMORY,
)


class PrefixTraceRecord(BaseModel):
    """Per-episode prefix formation trace."""

    episode_id: str
    receiver: str = ""
    target_memory_id: str = ""
    required_prefix_memory_ids: list[str] = []
    candidate_memory_ids: list[str] = []
    candidate_ranks: dict[str, int] = {}
    traversal_order: list[str] = []
    selected_before_target: list[str] = []
    required_prefix_in_candidates: bool = False
    required_prefix_traversed_before_target: bool = False
    required_prefix_selected: bool = False
    target_in_candidates: bool = False
    target_prediction: dict[str, float] = {}
    target_action: str = "withhold"
    target_ground_truth_region: str = "neutral"
    task_success: bool = False


class PrefixTraceSummary(BaseModel):
    """Aggregate prefix formation trace metrics."""

    scenario: str
    n_episodes: int = 0
    prefix_candidate_recall: float = 0.0
    """Fraction of episodes where all prefix memories are in candidates."""
    prefix_order_success_rate: float = 0.0
    """Fraction where prefix memories appear before target in traversal."""
    prefix_selection_success_rate: float = 0.0
    """Fraction where all prefix memories were selected before target."""
    target_evaluated_under_correct_prefix_rate: float = 0.0
    """Fraction where target was evaluated with correct prefix context."""
    target_share_under_correct_prefix_rate: float = 0.0
    """Fraction where target was shared given correct prefix."""
    success_given_correct_prefix: float = 0.0
    """Task success rate when prefix was correctly formed."""
    success_without_correct_prefix: float = 0.0
    """Task success rate when prefix was NOT correctly formed."""
    # Per-episode traces
    traces: list[PrefixTraceRecord] = []


def _build_prefix_trace(
    run: dict[str, Any],
    *,
    scenario: str,
) -> PrefixTraceRecord:
    """Build a prefix trace record for a single run."""
    target_mem_id = SCENARIO_TARGET_MEMORY.get(scenario, "")
    prefix_mems = SCENARIO_PREFIX_MEMORIES.get(scenario, [])
    target_effect = SCENARIO_TARGET_EFFECT.get(scenario, "neutral")

    candidate_ids = run.get("candidate_memory_ids", [])
    episode_id = run.get("episode_id", "")

    # Find candidate ranks
    candidate_ranks: dict[str, int] = {}
    for trace in run.get("router_trace", []):
        for dec in trace.get("decisions", []):
            mid = dec.get("memory_id", "")
            rank = dec.get("proposal_rank") or dec.get("candidate_position")
            if rank is not None and mid not in candidate_ranks:
                candidate_ranks[mid] = rank

    # Find traversal order and selected-before-target
    traversal_order: list[str] = []
    selected_before_target: list[str] = []
    target_prediction: dict[str, float] = {}
    target_action = "withhold"

    for trace in run.get("router_trace", []):
        decisions = trace.get("decisions", [])
        for dec in decisions:
            mid = dec.get("memory_id", "")
            if mid not in traversal_order:
                traversal_order.append(mid)
            if dec.get("action") == "share" and mid not in selected_before_target:
                if mid != target_mem_id:
                    selected_before_target.append(mid)
            if mid == target_mem_id:
                target_action = dec.get("action", "withhold")
                target_prediction = {
                    "tau_mean": dec.get("tau_mean") or 0.0,
                    "tau_lcb": dec.get("tau_lcb") or 0.0,
                    "negative_risk_ucb": dec.get("negative_risk_ucb") or 0.0,
                }

    # Check prefix conditions
    prefix_in_candidates = all(m in candidate_ids for m in prefix_mems) if prefix_mems else True
    prefix_traversed_before = all(
        m in traversal_order and traversal_order.index(m) < traversal_order.index(target_mem_id)
        for m in prefix_mems
    ) if prefix_mems and target_mem_id in traversal_order else False
    prefix_selected = all(m in selected_before_target for m in prefix_mems) if prefix_mems else True

    return PrefixTraceRecord(
        episode_id=episode_id,
        receiver=run.get("router_trace", [{}])[0].get("agent", "") if run.get("router_trace") else "",
        target_memory_id=target_mem_id,
        required_prefix_memory_ids=prefix_mems,
        candidate_memory_ids=candidate_ids,
        candidate_ranks=candidate_ranks,
        traversal_order=traversal_order,
        selected_before_target=selected_before_target,
        required_prefix_in_candidates=prefix_in_candidates,
        required_prefix_traversed_before_target=prefix_traversed_before,
        required_prefix_selected=prefix_selected,
        target_in_candidates=target_mem_id in candidate_ids,
        target_prediction=target_prediction,
        target_action=target_action,
        target_ground_truth_region=target_effect,
        task_success=run.get("team_success", False),
    )


def compute_prefix_trace(
    runs: list[dict[str, Any]],
    *,
    scenario: str,
    method: str = "M0-Full",
) -> PrefixTraceSummary:
    """Compute prefix formation trace for a scenario + method.

    Args:
        runs: List of run record dicts.
        scenario: Counterfactual scenario name.
        method: Method name to filter.

    Returns:
        PrefixTraceSummary with per-episode traces and aggregate metrics.
    """
    prefix_mems = SCENARIO_PREFIX_MEMORIES.get(scenario, [])
    if not prefix_mems:
        return PrefixTraceSummary(scenario=scenario)

    method_runs = [r for r in runs if r.get("method") == method]
    if not method_runs:
        return PrefixTraceSummary(scenario=scenario)

    traces: list[PrefixTraceRecord] = []
    for run in method_runs:
        trace = _build_prefix_trace(run, scenario=scenario)
        traces.append(trace)

    n = len(traces)

    # Compute aggregate metrics
    prefix_in_cands = sum(1 for t in traces if t.required_prefix_in_candidates)
    prefix_order_ok = sum(1 for t in traces if t.required_prefix_traversed_before_target)
    prefix_selected = sum(1 for t in traces if t.required_prefix_selected)

    correct_prefix_traces = [t for t in traces if t.required_prefix_selected]
    incorrect_prefix_traces = [t for t in traces if not t.required_prefix_selected]

    target_shared_correct = sum(
        1 for t in correct_prefix_traces if t.target_action == "share"
    )
    success_correct = sum(1 for t in correct_prefix_traces if t.task_success)
    success_incorrect = sum(1 for t in incorrect_prefix_traces if t.task_success)

    return PrefixTraceSummary(
        scenario=scenario,
        n_episodes=n,
        prefix_candidate_recall=prefix_in_cands / n if n else 0.0,
        prefix_order_success_rate=prefix_order_ok / n if n else 0.0,
        prefix_selection_success_rate=prefix_selected / n if n else 0.0,
        target_evaluated_under_correct_prefix_rate=len(correct_prefix_traces) / n if n else 0.0,
        target_share_under_correct_prefix_rate=(
            target_shared_correct / len(correct_prefix_traces)
            if correct_prefix_traces
            else 0.0
        ),
        success_given_correct_prefix=(
            success_correct / len(correct_prefix_traces)
            if correct_prefix_traces
            else 0.0
        ),
        success_without_correct_prefix=(
            success_incorrect / len(incorrect_prefix_traces)
            if incorrect_prefix_traces
            else 0.0
        ),
        traces=traces,
    )


def compute_all_prefix_traces(
    runs: list[dict[str, Any]],
    *,
    scenario: str,
) -> dict[str, PrefixTraceSummary]:
    """Compute prefix traces for all methods in a scenario."""
    methods = set(r.get("method") for r in runs)
    results: dict[str, PrefixTraceSummary] = {}
    for method in sorted(methods):
        results[method] = compute_prefix_trace(runs, scenario=scenario, method=method)
    return results
