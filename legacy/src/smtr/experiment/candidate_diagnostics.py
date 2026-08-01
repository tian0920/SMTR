"""Candidate-level ground-truth diagnostics for ablation experiments.

Uses counterfactual scenario metadata to compute per-decision-point metrics:
1. Candidate positive Recall@K
2. Candidate negative Recall@K
3. Router positive recall
4. Harmful-memory rejection rate
5. Positive transfer precision
6. Neutral exposure rate
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel


# Ground-truth transfer class for each scenario's target memory.
# This is the *causal effect* of sharing the target when no prefix interference.
SCENARIO_TARGET_EFFECT: dict[str, str] = {
    "positive": "positive",
    "negative": "negative",
    "neutral_success": "neutral",
    "neutral_failure": "neutral",
    "prefix_sensitive": "positive",  # positive when prefix is correct
    "flip_pos_to_neg": "negative",   # flipped by prefix → negative
    "flip_neg_to_pos": "positive",   # flipped by prefix → positive
    "flip_neu_to_neg": "negative",   # flipped by prefix → negative
    "flip_neu_to_pos": "positive",   # flipped by prefix → positive
}

# Target memory ID per scenario
SCENARIO_TARGET_MEMORY: dict[str, str] = {
    "positive": "mem_cf_positive",
    "negative": "mem_cf_negative",
    "neutral_success": "mem_cf_neutral_success",
    "neutral_failure": "mem_cf_neutral_failure",
    "prefix_sensitive": "mem_cf_prefix_recover",
    "flip_pos_to_neg": "mem_cf_positive",
    "flip_neg_to_pos": "mem_cf_negative",
    "flip_neu_to_neg": "mem_cf_neutral_success",
    "flip_neu_to_pos": "mem_cf_neutral_failure",
}

# Prefix memory IDs per scenario (empty for non-prefix scenarios)
SCENARIO_PREFIX_MEMORIES: dict[str, list[str]] = {
    "positive": [],
    "negative": [],
    "neutral_success": [],
    "neutral_failure": [],
    "prefix_sensitive": ["mem_prefix_lock"],
    "flip_pos_to_neg": ["mem_cf_block"],
    "flip_neg_to_pos": ["mem_cf_override"],
    "flip_neu_to_neg": ["mem_cf_block"],
    "flip_neu_to_pos": ["mem_cf_enable"],
}


class CandidateDiagnosticsSummary(BaseModel):
    """Aggregate candidate-level diagnostic metrics."""

    scenario: str
    # Recall@K: is the target in the proposer's top-K candidates?
    positive_target_recall_at_k: float | None = None
    negative_target_recall_at_k: float | None = None
    # Router metrics (conditional on target being in candidates)
    router_positive_recall: float | None = None
    """P(share target | target positive, target in candidates)"""
    harmful_memory_rejection: float | None = None
    """P(withhold target | target negative, target in candidates)"""
    # Precision metrics (among shared candidates)
    positive_transfer_precision: float | None = None
    """#positive_shared / #total_shared among target candidates"""
    neutral_exposure_rate: float | None = None
    """#neutral_shared / #total_shared among target candidates"""
    # Counts
    n_episodes: int = 0
    n_target_in_candidates: int = 0
    n_positive_targets: int = 0
    n_negative_targets: int = 0
    n_neutral_targets: int = 0


@dataclass
class _DecisionRecord:
    """Internal record for a single candidate decision."""
    memory_id: str
    action: str  # "share" or "withhold"
    is_target: bool
    ground_truth_effect: str  # "positive", "negative", "neutral"


def compute_candidate_diagnostics(
    runs: list[dict[str, Any]],
    *,
    scenario: str,
    method: str = "SMTR",
) -> CandidateDiagnosticsSummary:
    """Compute candidate-level diagnostics for a scenario + method.

    Args:
        runs: List of run record dicts (from runs.jsonl) for the given method.
        scenario: Counterfactual scenario name.
        method: Method name to filter (default: SMTR).

    Returns:
        CandidateDiagnosticsSummary with all metrics.
    """
    target_mem_id = SCENARIO_TARGET_MEMORY.get(scenario)
    target_effect = SCENARIO_TARGET_EFFECT.get(scenario)

    if target_mem_id is None or target_effect is None:
        return CandidateDiagnosticsSummary(scenario=scenario)

    # Filter runs for the given method
    method_runs = [r for r in runs if r.get("method") == method]
    if not method_runs:
        return CandidateDiagnosticsSummary(scenario=scenario)

    n_episodes = len(method_runs)

    # Track recall and decisions
    target_in_candidates_count = 0
    positive_in_candidates = 0
    negative_in_candidates = 0
    neutral_in_candidates = 0
    positive_shared = 0
    negative_shared = 0
    neutral_shared = 0
    target_shared_given_in_candidates = 0
    target_withheld_given_in_candidates = 0

    for run in method_runs:
        candidate_ids = run.get("candidate_memory_ids", [])
        target_in_cands = target_mem_id in candidate_ids

        if not target_in_cands:
            continue

        target_in_candidates_count += 1

        # Determine ground truth effect for this target
        effect = target_effect

        if effect == "positive":
            positive_in_candidates += 1
        elif effect == "negative":
            negative_in_candidates += 1
        else:
            neutral_in_candidates += 1

        # Check if target was shared in any invocation
        target_was_shared = False
        for trace in run.get("router_trace", []):
            for dec in trace.get("decisions", []):
                if dec.get("memory_id") == target_mem_id and dec.get("action") == "share":
                    target_was_shared = True

        if target_was_shared:
            target_shared_given_in_candidates += 1
            if effect == "positive":
                positive_shared += 1
            elif effect == "negative":
                negative_shared += 1
            else:
                neutral_shared += 1
        else:
            target_withheld_given_in_candidates += 1

    # Compute metrics
    summary = CandidateDiagnosticsSummary(
        scenario=scenario,
        n_episodes=n_episodes,
        n_target_in_candidates=target_in_candidates_count,
    )

    if target_in_candidates_count == 0:
        return summary

    # Recall@K: fraction of episodes where target is in candidates
    recall_rate = target_in_candidates_count / n_episodes

    if target_effect == "positive":
        summary.positive_target_recall_at_k = recall_rate
    elif target_effect == "negative":
        summary.negative_target_recall_at_k = recall_rate

    summary.n_positive_targets = positive_in_candidates
    summary.n_negative_targets = negative_in_candidates
    summary.n_neutral_targets = neutral_in_candidates

    # Router positive recall
    if positive_in_candidates > 0:
        summary.router_positive_recall = (
            target_shared_given_in_candidates / positive_in_candidates
            if target_effect == "positive"
            else None
        )
        # For negative targets: harmful rejection
        if target_effect == "negative":
            summary.harmful_memory_rejection = (
                target_withheld_given_in_candidates / negative_in_candidates
            )

    # For positive targets
    if target_effect == "positive" and positive_in_candidates > 0:
        summary.router_positive_recall = (
            target_shared_given_in_candidates / positive_in_candidates
        )

    # For negative targets
    if target_effect == "negative" and negative_in_candidates > 0:
        summary.harmful_memory_rejection = (
            target_withheld_given_in_candidates / negative_in_candidates
        )

    # Precision and exposure (among all runs where target was shared)
    total_target_shared = positive_shared + negative_shared + neutral_shared
    if total_target_shared > 0:
        summary.positive_transfer_precision = positive_shared / total_target_shared
        summary.neutral_exposure_rate = neutral_shared / total_target_shared

    return summary


def compute_all_candidate_diagnostics(
    runs: list[dict[str, Any]],
    *,
    scenario: str,
) -> dict[str, CandidateDiagnosticsSummary]:
    """Compute candidate diagnostics for all methods in a scenario.

    Returns dict of method_name -> CandidateDiagnosticsSummary.
    """
    methods = set(r.get("method") for r in runs)
    results: dict[str, CandidateDiagnosticsSummary] = {}
    for method in sorted(methods):
        results[method] = compute_candidate_diagnostics(
            runs, scenario=scenario, method=method
        )
    return results
