"""5-stage bottleneck decomposition for ablation experiments.

For each positive-transfer or prefix-dependent scenario, decomposes the
pipeline into 5 stages and counts how many episodes survive each stage:

Stage 1 (Proposer recall): target/prefix in top-K candidates
Stage 2 (Traversal/order): prefix traversed before target
Stage 3 (Prefix selection): prefix selected for sharing before target
Stage 4 (Target routing): target correctly shared/withheld given prefix
Stage 5 (Execution): task succeeds given correct target selection
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from smtr.experiment.candidate_diagnostics import (
    SCENARIO_PREFIX_MEMORIES,
    SCENARIO_TARGET_EFFECT,
    SCENARIO_TARGET_MEMORY,
)


class FunnelStage(BaseModel):
    """A single stage in the bottleneck funnel."""

    name: str
    count: int = 0
    total: int = 0
    rate: float = 0.0


class BottleneckFunnelResult(BaseModel):
    """5-stage bottleneck funnel for a scenario + method."""

    scenario: str
    method: str
    stages: list[FunnelStage] = []
    ground_truth_opportunity: int = 0


def compute_bottleneck_funnel(
    runs: list[dict[str, Any]],
    *,
    scenario: str,
    method: str,
) -> BottleneckFunnelResult:
    """Compute 5-stage bottleneck funnel for a scenario + method.

    Args:
        runs: List of run record dicts (from runs.jsonl).
        scenario: Counterfactual scenario name.
        method: Method name to filter.

    Returns:
        BottleneckFunnelResult with per-stage counts and rates.
    """
    target_mem_id = SCENARIO_TARGET_MEMORY.get(scenario, "")
    prefix_mems = SCENARIO_PREFIX_MEMORIES.get(scenario, [])
    target_effect = SCENARIO_TARGET_EFFECT.get(scenario, "neutral")

    method_runs = [r for r in runs if r.get("method") == method]
    if not method_runs:
        return BottleneckFunnelResult(scenario=scenario, method=method)

    n = len(method_runs)

    # Stage 1: Proposer recall — target and prefix in candidates
    stage1_count = 0
    # Stage 2: Traversal order — prefix before target
    stage2_count = 0
    # Stage 3: Prefix selection — prefix selected before target
    stage3_count = 0
    # Stage 4: Target routing — correct decision on target
    stage4_count = 0
    # Stage 5: Execution — task success given correct routing
    stage5_count = 0

    for run in method_runs:
        candidate_ids = run.get("candidate_memory_ids", [])
        target_in_candidates = target_mem_id in candidate_ids
        prefix_in_candidates = all(m in candidate_ids for m in prefix_mems) if prefix_mems else True

        # Stage 1: target (and prefix if applicable) recalled
        stage1_ok = target_in_candidates and prefix_in_candidates
        if stage1_ok:
            stage1_count += 1

        # Build traversal order and selected-before-target
        traversal_order: list[str] = []
        selected_before_target: list[str] = []
        target_action = "withhold"
        target_decision_found = False

        for trace in run.get("router_trace", []):
            for dec in trace.get("decisions", []):
                mid = dec.get("memory_id", "")
                if mid not in traversal_order:
                    traversal_order.append(mid)
                if dec.get("action") == "share" and mid != target_mem_id:
                    selected_before_target.append(mid)
                if mid == target_mem_id:
                    target_action = dec.get("action", "withhold")
                    target_decision_found = True

        # Stage 2: prefix traversed before target
        if stage1_ok and prefix_mems and target_mem_id in traversal_order:
            prefix_before = all(
                m in traversal_order and traversal_order.index(m) < traversal_order.index(target_mem_id)
                for m in prefix_mems
            )
            if prefix_before:
                stage2_count += 1
        elif stage1_ok and not prefix_mems:
            # No prefix required, stage 2 passes automatically
            stage2_count += 1

        # Stage 3: prefix selected before target
        if stage2_count > 0 or (stage1_ok and not prefix_mems):
            if prefix_mems:
                prefix_selected = all(m in selected_before_target for m in prefix_mems)
            else:
                prefix_selected = True
            if prefix_selected:
                stage3_count += 1

        # Stage 4: target correctly routed
        # For positive targets: target should be shared
        # For negative targets: target should be withheld
        stage4_ok = False
        if target_effect == "positive" and target_action == "share":
            stage4_ok = True
        elif target_effect == "negative" and target_action == "withhold":
            stage4_ok = True
        elif target_effect == "neutral":
            # Neutral: any decision is "correct" from routing perspective
            stage4_ok = True

        if stage3_count > 0 and stage4_ok:
            stage4_count += 1
        elif stage3_count == 0 and not prefix_mems and stage1_ok and stage4_ok:
            # No prefix required, skip stage 3
            stage4_count += 1

        # Stage 5: task success given correct routing
        if stage4_ok and run.get("team_success", False):
            stage5_count += 1

    stages = [
        FunnelStage(name="ground_truth_opportunity", count=n, total=n, rate=1.0),
        FunnelStage(
            name="target_prefix_recalled",
            count=stage1_count, total=n,
            rate=stage1_count / n if n else 0.0,
        ),
        FunnelStage(
            name="correct_traversal_order",
            count=stage2_count, total=n,
            rate=stage2_count / n if n else 0.0,
        ),
        FunnelStage(
            name="correct_prefix_selected",
            count=stage3_count, total=n,
            rate=stage3_count / n if n else 0.0,
        ),
        FunnelStage(
            name="target_correctly_routed",
            count=stage4_count, total=n,
            rate=stage4_count / n if n else 0.0,
        ),
        FunnelStage(
            name="task_succeeds",
            count=stage5_count, total=n,
            rate=stage5_count / n if n else 0.0,
        ),
    ]

    return BottleneckFunnelResult(
        scenario=scenario,
        method=method,
        stages=stages,
        ground_truth_opportunity=n,
    )


def compute_all_bottleneck_funnels(
    runs: list[dict[str, Any]],
    *,
    scenario: str,
) -> dict[str, BottleneckFunnelResult]:
    """Compute bottleneck funnels for all methods in a scenario."""
    methods = sorted(set(r.get("method") for r in runs))
    results: dict[str, BottleneckFunnelResult] = {}
    for method in methods:
        results[method] = compute_bottleneck_funnel(runs, scenario=scenario, method=method)
    return results
