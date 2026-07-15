"""Diagnostics for Static-SMTR selected-set conditioning ablation."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from smtr.experiment.schemas import ComparisonRunRecord


def compute_static_set_diagnostics(
    runs: list[ComparisonRunRecord],
    *,
    smtr_method: str = "SMTR",
    static_method: str = "Static-SMTR",
) -> dict[str, Any]:
    smtr_runs = _index_runs(runs, smtr_method)
    static_runs = _index_runs(runs, static_method)
    common = sorted(set(smtr_runs) & set(static_runs))
    overall = _compute_overall(smtr_runs, static_runs, common)
    scenario_keys: dict[str, list[tuple[str, int | None]]] = defaultdict(list)
    for key in common:
        scenario_keys[smtr_runs[key].scenario or "default"].append(key)
    by_scenario = {
        scenario: _compute_overall(smtr_runs, static_runs, keys)
        for scenario, keys in scenario_keys.items()
    }
    return {"overall": overall, "by_scenario": by_scenario}


def _compute_overall(
    smtr_runs: dict[tuple[str, int | None], ComparisonRunRecord],
    static_runs: dict[tuple[str, int | None], ComparisonRunRecord],
    common: list[tuple[str, int | None]],
) -> dict[str, float | int | None]:
    action_pairs = 0
    action_agree = 0
    smtr_share_static_withhold = 0
    smtr_withhold_static_share = 0
    jaccards = []
    outcome_disagreements = 0
    divergence_decisions = 0
    static_decisions = 0

    for key in common:
        smtr = smtr_runs[key]
        static = static_runs[key]
        if smtr.team_success != static.team_success:
            outcome_disagreements += 1
        jaccards.append(
            _jaccard(
                smtr.unique_selected_memory_ids,
                static.unique_selected_memory_ids,
            )
        )
        for smtr_dec, static_dec in zip(
            _ordered_decisions(smtr),
            _ordered_decisions(static),
            strict=False,
        ):
            action_pairs += 1
            if smtr_dec.get("action") == static_dec.get("action"):
                action_agree += 1
            elif smtr_dec.get("action") == "share":
                smtr_share_static_withhold += 1
            else:
                smtr_withhold_static_share += 1
        for decision in _ordered_decisions(static):
            static_decisions += 1
            if decision.get("selected_before_actual_digest") != decision.get(
                "selected_before_critic_digest"
            ):
                divergence_decisions += 1

    return {
        "matched_run_count": len(common),
        "matched_decision_count": action_pairs,
        "action_agreement_rate": action_agree / action_pairs if action_pairs else None,
        "smtr_share_static_withhold_count": smtr_share_static_withhold,
        "smtr_withhold_static_share_count": smtr_withhold_static_share,
        "selected_set_jaccard": sum(jaccards) / len(jaccards) if jaccards else None,
        "policy_outcome_disagreement_rate": (
            outcome_disagreements / len(common) if common else None
        ),
        "selected_set_conditioning_divergence_rate": (
            divergence_decisions / static_decisions if static_decisions else None
        ),
    }


def _index_runs(
    runs: list[ComparisonRunRecord],
    method: str,
) -> dict[tuple[str, int | None], ComparisonRunRecord]:
    return {
        (run.base_episode_id, run.traversal_seed): run
        for run in runs
        if run.method == method
    }


def _ordered_decisions(run: ComparisonRunRecord) -> list[dict[str, Any]]:
    decisions = []
    for invocation_index, invocation in enumerate(run.invocations):
        for decision in invocation.decisions:
            decisions.append(
                {
                    "invocation_index": invocation_index,
                    "memory_id": decision.memory_id,
                    "action": decision.action,
                    "selected_before_actual_digest": decision.selected_before_actual_digest,
                    "selected_before_critic_digest": decision.selected_before_critic_digest,
                }
            )
    return decisions


def _jaccard(a: list[str], b: list[str]) -> float:
    left = set(a)
    right = set(b)
    if not left and not right:
        return 1.0
    return len(left & right) / len(left | right)
