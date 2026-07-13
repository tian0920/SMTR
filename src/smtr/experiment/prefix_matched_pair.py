"""Prefix matched-pair audit for M0-Full vs A1-NoSet comparison.

For prefix-sensitive and flip scenarios, compares critic predictions
between M0-Full and A1-NoSet on the same base episodes.

Reports:
- delta-tau correlation with ground truth
- delta-tau MAE
- effect direction accuracy
- per-scenario flip accuracy
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel

from smtr.experiment.candidate_diagnostics import (
    SCENARIO_PREFIX_MEMORIES,
    SCENARIO_TARGET_EFFECT,
    SCENARIO_TARGET_MEMORY,
)


class PrefixMatchedPairResult(BaseModel):
    """Result of prefix matched-pair audit for a scenario."""

    scenario: str
    n_paired_episodes: int = 0
    delta_tau_correlation: float | None = None
    delta_tau_mae: float | None = None
    effect_direction_accuracy: float | None = None
    # Per flip-type accuracy
    positive_to_negative_accuracy: float | None = None
    negative_to_positive_accuracy: float | None = None
    neutral_to_negative_accuracy: float | None = None
    neutral_to_positive_accuracy: float | None = None


def _extract_target_prediction(
    run: dict[str, Any],
    target_mem_id: str,
) -> dict[str, float]:
    """Extract critic prediction for the target memory from a run."""
    for trace in run.get("router_trace", []):
        for dec in trace.get("decisions", []):
            if dec.get("memory_id") == target_mem_id:
                return {
                    "tau_mean": dec.get("tau_mean") or 0.0,
                    "tau_lcb": dec.get("tau_lcb") or 0.0,
                    "tau_ucb": dec.get("tau_ucb") or 0.0,
                    "negative_risk_ucb": dec.get("negative_risk_ucb") or 0.0,
                    "action": dec.get("action", "withhold"),
                }
    return {"tau_mean": 0.0, "tau_lcb": 0.0, "tau_ucb": 0.0, "negative_risk_ucb": 0.0, "action": "withhold"}


def compute_prefix_matched_pair_audit(
    runs: list[dict[str, Any]],
    *,
    scenario: str,
    method_a: str = "M0-Full",
    method_b: str = "A1-NoSet",
) -> PrefixMatchedPairResult:
    """Compute prefix matched-pair audit for M0 vs A1.

    Args:
        runs: All run record dicts for the scenario.
        scenario: Scenario name.
        method_a: First method (default M0-Full).
        method_b: Second method (default A1-NoSet).

    Returns:
        PrefixMatchedPairResult with delta-tau metrics.
    """
    target_mem_id = SCENARIO_TARGET_MEMORY.get(scenario, "")
    target_effect = SCENARIO_TARGET_EFFECT.get(scenario, "neutral")
    prefix_mems = SCENARIO_PREFIX_MEMORIES.get(scenario, [])

    if not target_mem_id:
        return PrefixMatchedPairResult(scenario=scenario)

    # Index runs by (episode_id, generation_seed) for pairing
    runs_a: dict[tuple[str, int], dict] = {}
    runs_b: dict[tuple[str, int], dict] = {}
    for run in runs:
        key = (run.get("episode_id", ""), run.get("generation_seed", 0))
        if run.get("method") == method_a:
            runs_a[key] = run
        elif run.get("method") == method_b:
            runs_b[key] = run

    common_keys = sorted(set(runs_a.keys()) & set(runs_b.keys()))
    if not common_keys:
        return PrefixMatchedPairResult(scenario=scenario)

    # Compute per-episode delta-tau
    delta_taus: list[float] = []
    ground_truth_taus: list[float] = []
    direction_correct: list[bool] = []

    # Ground truth tau: +1 for positive, -1 for negative, 0 for neutral
    gt_tau_map = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}
    gt_tau = gt_tau_map.get(target_effect, 0.0)

    for key in common_keys:
        pred_a = _extract_target_prediction(runs_a[key], target_mem_id)
        pred_b = _extract_target_prediction(runs_b[key], target_mem_id)

        delta_tau = pred_a["tau_mean"] - pred_b["tau_mean"]
        delta_taus.append(delta_tau)
        ground_truth_taus.append(gt_tau)

        # Direction: does sign(delta_tau) match sign(ground_truth)?
        if gt_tau != 0:
            direction_correct.append(np.sign(delta_tau) == np.sign(gt_tau))
        else:
            # For neutral, any small delta is acceptable
            direction_correct.append(abs(delta_tau) < 0.5)

    n = len(delta_taus)
    if n == 0:
        return PrefixMatchedPairResult(scenario=scenario)

    # Correlation
    delta_arr = np.array(delta_taus)
    gt_arr = np.array(ground_truth_taus)

    if np.std(gt_arr) > 0 and np.std(delta_arr) > 0:
        correlation = float(np.corrcoef(delta_arr, gt_arr)[0, 1])
    else:
        correlation = None

    # MAE
    mae = float(np.mean(np.abs(delta_arr - gt_arr)))

    # Direction accuracy
    dir_acc = sum(direction_correct) / n if direction_correct else None

    # Flip-type accuracy (per scenario)
    flip_accs: dict[str, float | None] = {
        "positive_to_negative_accuracy": None,
        "negative_to_positive_accuracy": None,
        "neutral_to_negative_accuracy": None,
        "neutral_to_positive_accuracy": None,
    }

    # Determine which flip type this scenario is
    # For flip scenarios, check if the method correctly identifies the flipped effect
    if scenario == "flip_pos_to_neg":
        # Target was positive, now negative -> A1 should withhold, M0 should withhold
        a_withhold = sum(1 for k in common_keys if _extract_target_prediction(runs_a[k], target_mem_id)["action"] == "withhold")
        flip_accs["positive_to_negative_accuracy"] = a_withhold / n
    elif scenario == "flip_neg_to_pos":
        # Target was negative, now positive -> should share
        a_share = sum(1 for k in common_keys if _extract_target_prediction(runs_a[k], target_mem_id)["action"] == "share")
        flip_accs["negative_to_positive_accuracy"] = a_share / n
    elif scenario == "flip_neu_to_neg":
        a_withhold = sum(1 for k in common_keys if _extract_target_prediction(runs_a[k], target_mem_id)["action"] == "withhold")
        flip_accs["neutral_to_negative_accuracy"] = a_withhold / n
    elif scenario == "flip_neu_to_pos":
        a_share = sum(1 for k in common_keys if _extract_target_prediction(runs_a[k], target_mem_id)["action"] == "share")
        flip_accs["neutral_to_positive_accuracy"] = a_share / n

    return PrefixMatchedPairResult(
        scenario=scenario,
        n_paired_episodes=n,
        delta_tau_correlation=correlation,
        delta_tau_mae=mae,
        effect_direction_accuracy=dir_acc,
        **flip_accs,
    )


def compute_all_prefix_matched_pair_audits(
    runs: list[dict[str, Any]],
    *,
    scenarios: list[str] | None = None,
) -> dict[str, PrefixMatchedPairResult]:
    """Compute prefix matched-pair audits for all relevant scenarios."""
    if scenarios is None:
        scenarios = [
            "prefix_sensitive",
            "flip_pos_to_neg", "flip_neg_to_pos",
            "flip_neu_to_neg", "flip_neu_to_pos",
        ]
    results: dict[str, PrefixMatchedPairResult] = {}
    for scenario in scenarios:
        results[scenario] = compute_prefix_matched_pair_audit(runs, scenario=scenario)
    return results
