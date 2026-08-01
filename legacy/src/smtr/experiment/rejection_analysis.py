"""Rejection reason analysis with matched-case auditing.

For SMTR and EffectOnly-SMTR:
1. Compute per-reason proportions and verify they sum to 1.
2. Find matched discordant cases between the formal risk gate and effect-only gate.
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
    tau_mean_nonpositive: float = 0.0
    negative_risk_mean_exceeded: float = 0.0
    low_support: float = 0.0
    share_budget_exceeded: float = 0.0
    other: float = 0.0
    sum_check: float = 0.0


class MatchedCase(BaseModel):
    """A discordant case where SMTR and an ablation disagree on a decision."""

    episode_id: str
    generation_seed: int
    memory_id: str
    case_type: str
    candidate_memory_ids: list[str] = []
    selected_before_target_ids: list[str] = []
    ablation_prediction: dict[str, Any] = {}
    smtr_prediction: dict[str, Any] = {}
    ground_truth_effect: str = ""
    final_task_outcome: str = ""  # "success" or "failure"


class RejectionAnalysisResult(BaseModel):
    """Full rejection analysis for a scenario."""

    scenario: str
    smtr_reasons: ReasonProportions = Field(
        default_factory=lambda: ReasonProportions(method="SMTR")
    )
    ablation_reasons: ReasonProportions = Field(
        default_factory=lambda: ReasonProportions(method="EffectOnly-SMTR")
    )
    matched_cases: list[MatchedCase] = []
    n_ablation_share_smtr_withhold: int = 0
    n_ablation_withhold_smtr_share: int = 0


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
        "shared", "tau_mean_nonpositive", "negative_risk_mean_exceeded",
        "low_support", "share_budget_exceeded", "other",
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
    smtr_method: str = "SMTR",
    ablation_method: str = "EffectOnly-SMTR",
) -> RejectionAnalysisResult:
    """Compute rejection reason analysis with matched-case audit.

    Args:
        runs: All run record dicts for the scenario.
        scenario: Scenario name.
        smtr_method: Method ID for formal SMTR.
        ablation_method: Method ID for a formal ablation.

    Returns:
        RejectionAnalysisResult with proportions and matched cases.
    """
    target_mem_id = SCENARIO_TARGET_MEMORY.get(scenario, "")
    target_effect = SCENARIO_TARGET_EFFECT.get(scenario, "neutral")

    smtr_props = _compute_reason_proportions(runs, method=smtr_method)
    ablation_props = _compute_reason_proportions(runs, method=ablation_method)

    # Index runs by (episode_id, generation_seed)
    smtr_runs: dict[tuple[str, int], dict] = {}
    ablation_runs: dict[tuple[str, int], dict] = {}
    for run in runs:
        key = (run.get("episode_id", ""), run.get("generation_seed", 0))
        if run.get("method") == smtr_method:
            smtr_runs[key] = run
        elif run.get("method") == ablation_method:
            ablation_runs[key] = run

    common_keys = sorted(set(smtr_runs.keys()) & set(ablation_runs.keys()))

    # Find matched discordant cases on the target memory
    matched_cases: list[MatchedCase] = []
    n_ablation_share_smtr_withhold = 0
    n_ablation_withhold_smtr_share = 0

    if not target_mem_id:
        return RejectionAnalysisResult(
            scenario=scenario,
            smtr_reasons=smtr_props,
            ablation_reasons=ablation_props,
        )

    for key in common_keys:
        smtr_run = smtr_runs[key]
        ablation_run = ablation_runs[key]

        smtr_dec = _extract_decision_for_memory(smtr_run, target_mem_id)
        ablation_dec = _extract_decision_for_memory(ablation_run, target_mem_id)

        if smtr_dec is None or ablation_dec is None:
            continue

        smtr_action = smtr_dec.get("action", "withhold")
        ablation_action = ablation_dec.get("action", "withhold")

        if ablation_action == "share" and smtr_action == "withhold":
            n_ablation_share_smtr_withhold += 1
            case = MatchedCase(
                episode_id=key[0],
                generation_seed=key[1],
                memory_id=target_mem_id,
                case_type="ablation_share_smtr_withhold",
                candidate_memory_ids=ablation_run.get("candidate_memory_ids", []),
                selected_before_target_ids=_get_selected_before(
                    ablation_run,
                    target_mem_id,
                ),
                ablation_prediction={
                    "tau_mean": ablation_dec.get("tau_mean"),
                    "negative_risk_mean": ablation_dec.get("negative_risk_mean"),
                    "action": ablation_action,
                    "reason": ablation_dec.get("reason"),
                },
                smtr_prediction={
                    "tau_mean": smtr_dec.get("tau_mean"),
                    "negative_risk_mean": smtr_dec.get("negative_risk_mean"),
                    "action": smtr_action,
                    "reason": smtr_dec.get("reason"),
                },
                ground_truth_effect=target_effect,
                final_task_outcome=(
                    "success" if ablation_run.get("team_success") else "failure"
                ),
            )
            matched_cases.append(case)

        elif ablation_action == "withhold" and smtr_action == "share":
            n_ablation_withhold_smtr_share += 1
            case = MatchedCase(
                episode_id=key[0],
                generation_seed=key[1],
                memory_id=target_mem_id,
                case_type="ablation_withhold_smtr_share",
                candidate_memory_ids=smtr_run.get("candidate_memory_ids", []),
                selected_before_target_ids=_get_selected_before(smtr_run, target_mem_id),
                ablation_prediction={
                    "tau_mean": ablation_dec.get("tau_mean"),
                    "negative_risk_mean": ablation_dec.get("negative_risk_mean"),
                    "action": ablation_action,
                    "reason": ablation_dec.get("reason"),
                },
                smtr_prediction={
                    "tau_mean": smtr_dec.get("tau_mean"),
                    "negative_risk_mean": smtr_dec.get("negative_risk_mean"),
                    "action": smtr_action,
                    "reason": smtr_dec.get("reason"),
                },
                ground_truth_effect=target_effect,
                final_task_outcome="success" if smtr_run.get("team_success") else "failure",
            )
            matched_cases.append(case)

    return RejectionAnalysisResult(
        scenario=scenario,
        smtr_reasons=smtr_props,
        ablation_reasons=ablation_props,
        matched_cases=matched_cases,
        n_ablation_share_smtr_withhold=n_ablation_share_smtr_withhold,
        n_ablation_withhold_smtr_share=n_ablation_withhold_smtr_share,
    )


def compute_all_rejection_analyses(
    runs_by_scenario: dict[str, list[dict[str, Any]]],
) -> dict[str, RejectionAnalysisResult]:
    """Compute rejection analyses for all scenarios."""
    results: dict[str, RejectionAnalysisResult] = {}
    for scenario, runs in runs_by_scenario.items():
        results[scenario] = compute_rejection_analysis(runs, scenario=scenario)
    return results
