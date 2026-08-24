"""Unified online behavioral outcome abstraction for MARBLE episodes.

This module provides a layer between raw MARBLE engine output and the
TCI (Transfer Causal Inference) admission rule. It extracts online-observable
performance signals without modifying TCI's core logic.

The TCI still receives two outcomes (expose, withhold) and computes:
    delta = expose.performance_score - withhold.performance_score

Usage::

    evaluator = BehavioralOutcomeEvaluator(scenario="minecraft")
    outcome = evaluator.extract(raw_output)
    # outcome.performance_score is a float suitable for TCI delta computation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class BehavioralOutcome:
    """Unified online-observable behavioral outcome from a MARBLE episode.

    Attributes:
        success: Binary task completion signal (True/False/None if unknown).
        performance_score: Primary continuous score for TCI delta computation.
            Must be online-observable (no ground-truth leakage).
            None if no valid signal could be extracted.
        iteration_scores: Per-iteration performance signals (may be empty).
        final_score: Native MARBLE evaluator final score (if available).
        signal_type: Name of the signal used for performance_score.
        metadata: Additional diagnostic fields (token_usage, etc.).
    """

    success: bool | None = None
    performance_score: float | None = None
    iteration_scores: list[float] = field(default_factory=list)
    final_score: float | None = None
    signal_type: str = "none"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "performance_score": self.performance_score,
            "iteration_scores": list(self.iteration_scores),
            "final_score": self.final_score,
            "signal_type": self.signal_type,
            "metadata": dict(self.metadata),
        }


class BehavioralOutcomeEvaluator:
    """Extract online-observable performance signals from MARBLE raw engine output.

    This evaluator does NOT modify TCI admission rules. It only extracts
    signals from raw engine output and packages them into a BehavioralOutcome.

    Signal priority (highest quality first):
    1. native_final_score: Domain-specific evaluator output (e.g., block_hit_rate)
    2. iteration_improvement: Summary length delta between first and last iteration
    3. binary_success: Fallback to team_success (0.0 or 1.0)

    Parameters:
        scenario: MARBLE domain name (e.g., "minecraft", "database").
    """

    def __init__(self, scenario: str) -> None:
        self.scenario = scenario

    def extract(self, raw_output: dict[str, Any]) -> BehavioralOutcome:
        """Extract BehavioralOutcome from raw MARBLE engine output.

        Parameters:
            raw_output: Parsed JSONL output from MARBLE engine.

        Returns:
            BehavioralOutcome with the best available performance signal.
        """
        # Extract basic signals
        success = _extract_team_success(raw_output)
        iterations = raw_output.get("iterations", [])
        if not isinstance(iterations, list):
            iterations = []

        # Extract iteration-level signals
        iteration_scores = _extract_iteration_summary_lengths(iterations)

        # Extract native final evaluator score
        native_final = _extract_native_final_score(raw_output, self.scenario)

        # Determine best performance signal
        metadata: dict[str, Any] = {
            "token_usage": raw_output.get("token_usage", 0),
            "n_iterations": len(iterations),
            "n_task_results_total": sum(
                len(it.get("task_results", []))
                for it in iterations
                if isinstance(it, dict)
            ),
        }

        # Priority 1: Native final evaluator score
        if native_final is not None:
            return BehavioralOutcome(
                success=success,
                performance_score=native_final,
                iteration_scores=iteration_scores,
                final_score=native_final,
                signal_type=f"native_final_{self.scenario}",
                metadata=metadata,
            )

        # Priority 2: Iteration improvement (summary length delta)
        if len(iteration_scores) >= 2:
            improvement = iteration_scores[-1] - iteration_scores[0]
            # Normalize to [0, 1] range (approximate)
            # Summary lengths typically 100-2000 chars
            normalized = max(0.0, min(1.0, improvement / 1000.0))
            metadata["iteration_improvement_raw"] = improvement
            return BehavioralOutcome(
                success=success,
                performance_score=normalized,
                iteration_scores=iteration_scores,
                final_score=None,
                signal_type="iteration_improvement",
                metadata=metadata,
            )

        # Priority 3: Binary success fallback
        return BehavioralOutcome(
            success=success,
            performance_score=1.0 if success else 0.0,
            iteration_scores=iteration_scores,
            final_score=None,
            signal_type="binary_success",
            metadata=metadata,
        )


def _extract_team_success(raw: dict[str, Any]) -> bool | None:
    """Extract binary team success from raw output."""
    if "team_success" in raw:
        return bool(raw["team_success"])
    if "success" in raw:
        return bool(raw["success"])
    iterations = raw.get("iterations", [])
    if isinstance(iterations, list) and iterations:
        last = iterations[-1]
        if isinstance(last, dict):
            tr = last.get("task_results", [])
            if isinstance(tr, list) and tr:
                return True
            summary = last.get("summary", "")
            if summary and len(str(summary)) > 20:
                return True
    return False


def _extract_iteration_summary_lengths(iterations: list[Any]) -> list[float]:
    """Extract per-iteration summary lengths as proxy signals."""
    scores: list[float] = []
    for it in iterations:
        if not isinstance(it, dict):
            continue
        summary = it.get("summary", "")
        scores.append(float(len(str(summary))))
    return scores


def _extract_native_final_score(
    raw: dict[str, Any],
    scenario: str,
) -> float | None:
    """Extract native MARBLE evaluator final score.

    Returns None if no valid native signal is available.
    """
    task_eval = raw.get("task_evaluation")

    if scenario == "minecraft":
        # task_evaluation is block_hit_rate * 5 (float)
        if isinstance(task_eval, (int, float)):
            # Normalize to [0, 1]
            return float(task_eval) / 5.0
        return None

    if scenario == "research" and isinstance(task_eval, dict):
        # Research evaluator outputs {innovation, safety, feasibility} (1-5 each)
        vals = []
        for key in ("innovation", "safety", "feasibility"):
            v = task_eval.get(key)
            if isinstance(v, (int, float)) and 1 <= v <= 5:
                vals.append(float(v))
        if vals:
            # Average and normalize to [0, 1]
            avg = sum(vals) / len(vals)
            return (avg - 1.0) / 4.0  # Map 1-5 → 0-1
        return None

    if scenario == "database" and isinstance(task_eval, dict):
        # Database evaluator outputs {root_causes, predicted}
        # SMTR fine-grained already computes recall/precision/F1
        # For TCI, use recall as the signal
        predicted = task_eval.get("predicted", "")
        expected = task_eval.get("root_causes", [])
        if isinstance(expected, list) and expected:
            if isinstance(predicted, str):
                # Count how many expected root causes appear in predicted
                matches = sum(1 for rc in expected if str(rc) in predicted)
                recall = matches / len(expected)
                return float(recall)
            if isinstance(predicted, list):
                pred_set = {str(p) for p in predicted}
                exp_set = {str(e) for e in expected}
                if exp_set:
                    tp = len(pred_set & exp_set)
                    return float(tp) / len(exp_set)
        return None

    # Bargaining: no native evaluator
    # Coding: code_quality only in star mode
    return None


def compute_delta(
    expose: BehavioralOutcome,
    withhold: BehavioralOutcome,
) -> float:
    """Compute TCI causal effect delta.

    delta = expose.performance_score - withhold.performance_score

    Returns 0.0 if either outcome has no performance score.
    """
    if expose.performance_score is None or withhold.performance_score is None:
        return 0.0
    return expose.performance_score - withhold.performance_score
