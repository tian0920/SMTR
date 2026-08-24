"""Official MultiAgentBench metric outcome adapter.

This module reads the official ``task_evaluation`` field from MARBLE engine
output and computes the official Task Score (TS) as defined in the
MultiAgentBench paper (ACL 2025).

**IMPORTANT**: This adapter does NOT create new metrics. It reads the official
evaluator output directly, without human partial-credit or ground-truth labels.

Official Metrics (from paper Table 1):
- **database**: Root cause recall (subset match) → [0, 1]
- **research**: Avg(innovation, safety, feasibility) → [1, 5] → scale to [0, 1]
- **minecraft**: block_hit_rate → [0, 1]
- **coding**: Avg(instruction_following, executability, consistency, quality) → [1, 5] → scale to [0, 1]
- **bargaining**: Avg(buyer/seller × effectiveness/progress/interaction) → [1, 5] → scale to [0, 1]

Usage::

    from smtr.marble.outcome.official_metric_outcome import OfficialMetricOutcomeEvaluator

    evaluator = OfficialMetricOutcomeEvaluator(scenario="database")
    outcome = evaluator.evaluate(task=task_dict, run_result=run_dict)

    # For TCI delta computation:
    delta = outcome.oriented_delta  # expose - withhold (for maximize metrics)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReceiverTCIOutcome:
    """Standardized outcome for TCI delta computation.

    Attributes:
        raw_metric_name: Official metric name (e.g., "task_score", "block_hit_rate")
        raw_expose_score: Raw score for expose branch (before orientation)
        raw_withhold_score: Raw score for withhold branch (before orientation)
        oriented_delta: Delta with correct sign (positive = better for expose)
        receiver_id: Receiver agent identifier
        metric_type: "maximize" or "minimize" (all official metrics are maximize)
    """

    raw_metric_name: str
    raw_expose_score: float | None
    raw_withhold_score: float | None
    oriented_delta: float
    receiver_id: str
    metric_type: str = "maximize"

    @property
    def is_valid(self) -> bool:
        """Check if both scores are available."""
        return (
            self.raw_expose_score is not None
            and self.raw_withhold_score is not None
        )

    @property
    def has_positive_delta(self) -> bool:
        """Check if expose > withhold."""
        return self.is_valid and self.oriented_delta > 0

    @property
    def has_negative_delta(self) -> bool:
        """Check if expose < withhold."""
        return self.is_valid and self.oriented_delta < 0


class OfficialMetricOutcomeEvaluator:
    """Read official MultiAgentBench task_evaluation and compute TS score.

    This evaluator reads the ``task_evaluation`` field from MARBLE engine
    output and scales it to [0, 1] using official formulas.

    Parameters:
        scenario: Domain name (database, research, minecraft, coding, bargaining)

    Example:
        >>> evaluator = OfficialMetricOutcomeEvaluator(scenario="minecraft")
        >>> outcome = evaluator.evaluate(task={}, run_result={"task_evaluation": 3.5})
        >>> print(outcome)
        OfficialMetricOutcome(
            scenario='minecraft',
            raw_metric_name='block_hit_rate',
            raw_score=3.5,
            normalized_score=0.7,
            is_valid=True
        )
    """

    def __init__(self, scenario: str) -> None:
        self.scenario = scenario

    def evaluate(
        self,
        *,
        task: dict[str, Any],
        run_result: dict[str, Any],
    ) -> OfficialMetricOutcome:
        """Evaluate one run using official MultiAgentBench metrics.

        Args:
            task: Task configuration dict (may contain ground truth for database)
            run_result: MARBLE engine output dict (must contain task_evaluation)

        Returns:
            OfficialMetricOutcome with normalized score in [0, 1]
        """
        task_eval = run_result.get("task_evaluation")

        if task_eval is None:
            logger.warning(
                f"task_evaluation is None for scenario={self.scenario}, "
                f"task_id={task.get('task_id', '?')}"
            )
            return OfficialMetricOutcome(
                scenario=self.scenario,
                raw_metric_name=self._get_metric_name(),
                raw_score=None,
                normalized_score=None,
                is_valid=False,
                failure_reason="task_evaluation_not_available",
            )

        try:
            raw_score, normalized_score = self._compute_score(
                task_eval=task_eval, task=task
            )
            return OfficialMetricOutcome(
                scenario=self.scenario,
                raw_metric_name=self._get_metric_name(),
                raw_score=raw_score,
                normalized_score=normalized_score,
                is_valid=True,
                failure_reason=None,
            )
        except Exception as e:
            logger.exception(
                f"Failed to compute official score for scenario={self.scenario}"
            )
            return OfficialMetricOutcome(
                scenario=self.scenario,
                raw_metric_name=self._get_metric_name(),
                raw_score=None,
                normalized_score=None,
                is_valid=False,
                failure_reason=f"computation_error: {e}",
            )

    def _get_metric_name(self) -> str:
        """Return the official metric name for this scenario."""
        metric_names = {
            "database": "root_cause_recall",
            "research": "avg_innovation_safety_feasibility",
            "minecraft": "block_hit_rate",
            "coding": "avg_code_quality",
            "bargaining": "avg_negotiation_quality",
        }
        return metric_names.get(self.scenario, "unknown")

    def _compute_score(
        self,
        task_eval: Any,
        task: dict[str, Any],
    ) -> tuple[float, float]:
        """Compute raw and normalized scores from task_evaluation.

        Returns:
            (raw_score, normalized_score) where normalized_score ∈ [0, 1]
        """
        if self.scenario == "minecraft":
            return self._compute_minecraft(task_eval)
        elif self.scenario == "database":
            return self._compute_database(task_eval, task)
        elif self.scenario == "research":
            return self._compute_research(task_eval)
        elif self.scenario == "coding":
            return self._compute_coding(task_eval)
        elif self.scenario == "bargaining":
            return self._compute_bargaining(task_eval)
        else:
            raise ValueError(f"Unknown scenario: {self.scenario}")

    def _compute_minecraft(self, task_eval: Any) -> tuple[float, float]:
        """Minecraft: task_evaluation = block_hit_rate * 5 → scale to [0, 1].

        Official metric: block_hit_rate ∈ [0, 1]
        Engine output: block_hit_rate * 5 ∈ [0, 5]
        Normalization: divide by 5
        """
        if not isinstance(task_eval, (int, float)):
            raise TypeError(
                f"minecraft task_evaluation should be numeric, got {type(task_eval)}"
            )
        raw_score = float(task_eval)
        normalized_score = raw_score / 5.0
        return raw_score, normalized_score

    def _compute_database(
        self, task_eval: Any, task: dict[str, Any]
    ) -> tuple[float, float]:
        """Database: Compute root cause recall from predicted vs ground_truth.

        Official metric: recall = |predicted ∩ ground_truth| / |ground_truth|
        Engine output: {root_cause: [...], predicted: ...}
        """
        if not isinstance(task_eval, dict):
            raise TypeError(
                f"database task_evaluation should be dict, got {type(task_eval)}"
            )

        ground_truth = task_eval.get("root_cause", [])
        predicted_raw = task_eval.get("predicted", "")

        # Parse predicted root causes (may be string or list)
        if isinstance(predicted_raw, str):
            # Try to extract root causes from text
            predicted = _extract_root_causes_from_text(predicted_raw)
        elif isinstance(predicted_raw, list):
            predicted = predicted_raw
        else:
            predicted = []

        if not ground_truth:
            # No ground truth → cannot compute recall
            return 0.0, 0.0

        # Compute recall (subset match)
        ground_truth_set = set(str(rc).lower().strip() for rc in ground_truth)
        predicted_set = set(str(rc).lower().strip() for rc in predicted)
        intersection = ground_truth_set & predicted_set
        recall = len(intersection) / len(ground_truth_set) if ground_truth_set else 0.0

        return recall, recall

    def _compute_research(self, task_eval: Any) -> tuple[float, float]:
        """Research: Average(innovation, safety, feasibility) → scale [1,5] to [0,1].

        Official metric: {innovation: 1-5, safety: 1-5, feasibility: 1-5}
        Normalization: (avg - 1) / 4
        """
        if not isinstance(task_eval, dict):
            raise TypeError(
                f"research task_evaluation should be dict, got {type(task_eval)}"
            )

        innovation = task_eval.get("innovation")
        safety = task_eval.get("safety")
        feasibility = task_eval.get("feasibility")

        if not all(isinstance(v, (int, float)) for v in [innovation, safety, feasibility]):
            raise ValueError(
                f"research ratings must be numeric: innovation={innovation}, "
                f"safety={safety}, feasibility={feasibility}"
            )

        avg_rating = (innovation + safety + feasibility) / 3.0
        normalized = (avg_rating - 1) / 4.0  # Scale [1, 5] → [0, 1]

        return avg_rating, normalized

    def _compute_coding(self, task_eval: Any) -> tuple[float, float]:
        """Coding: Average(instruction_following, executability, consistency, quality) → scale [1,5] to [0,1].

        Official metric: 4 dimensions, each 1-5
        Normalization: (avg - 1) / 4
        """
        if not isinstance(task_eval, dict):
            raise TypeError(
                f"coding task_evaluation should be dict, got {type(task_eval)}"
            )

        instruction = task_eval.get("instruction_following")
        executability = task_eval.get("executability")
        consistency = task_eval.get("consistency")
        quality = task_eval.get("quality")

        dimensions = [instruction, executability, consistency, quality]
        if not all(isinstance(v, (int, float)) for v in dimensions):
            raise ValueError(
                f"coding ratings must be numeric: {task_eval}"
            )

        avg_rating = sum(dimensions) / 4.0
        normalized = (avg_rating - 1) / 4.0  # Scale [1, 5] → [0, 1]

        return avg_rating, normalized

    def _compute_bargaining(self, task_eval: Any) -> tuple[float, float]:
        """Bargaining: Average(buyer/seller × effectiveness/progress/interaction) → scale [1,5] to [0,1].

        Official metric: 6 dimensions (2 roles × 3 aspects), each 1-5
        Normalization: (avg - 1) / 4
        """
        if not isinstance(task_eval, dict):
            raise TypeError(
                f"bargaining task_evaluation should be dict, got {type(task_eval)}"
            )

        buyer = task_eval.get("buyer", {})
        seller = task_eval.get("seller", {})

        if not isinstance(buyer, dict) or not isinstance(seller, dict):
            raise ValueError(
                f"bargaining task_evaluation should have buyer/seller dicts: {task_eval}"
            )

        buyer_eff = buyer.get("effectiveness")
        buyer_prog = buyer.get("progress")
        buyer_int = buyer.get("interaction")
        seller_eff = seller.get("effectiveness")
        seller_prog = seller.get("progress")
        seller_int = seller.get("interaction")

        dimensions = [buyer_eff, buyer_prog, buyer_int, seller_eff, seller_prog, seller_int]
        if not all(isinstance(v, (int, float)) for v in dimensions):
            raise ValueError(
                f"bargaining ratings must be numeric: {task_eval}"
            )

        avg_rating = sum(dimensions) / 6.0
        normalized = (avg_rating - 1) / 4.0  # Scale [1, 5] → [0, 1]

        return avg_rating, normalized

    def compute_delta(
        self,
        *,
        expose_result: dict[str, Any],
        withhold_result: dict[str, Any],
        receiver_id: str = "receiver",
    ) -> ReceiverTCIOutcome:
        """Compute oriented delta between expose and withhold branches.

        For maximize metrics (all official metrics):
            delta = score_expose - score_withhold

        Args:
            expose_result: Run result dict for expose branch
            withhold_result: Run result dict for withhold branch
            receiver_id: Receiver agent identifier

        Returns:
            ReceiverTCIOutcome with oriented_delta
        """
        expose_outcome = self.evaluate(task={}, run_result=expose_result)
        withhold_outcome = self.evaluate(task={}, run_result=withhold_result)

        if not expose_outcome.is_valid or not withhold_outcome.is_valid:
            return ReceiverTCIOutcome(
                raw_metric_name=self._get_metric_name(),
                raw_expose_score=expose_outcome.raw_score,
                raw_withhold_score=withhold_outcome.raw_score,
                oriented_delta=0.0,
                receiver_id=receiver_id,
                metric_type="maximize",
            )

        # All official metrics are "maximize" (higher = better)
        delta = expose_outcome.normalized_score - withhold_outcome.normalized_score

        return ReceiverTCIOutcome(
            raw_metric_name=self._get_metric_name(),
            raw_expose_score=expose_outcome.raw_score,
            raw_withhold_score=withhold_outcome.raw_score,
            oriented_delta=delta,
            receiver_id=receiver_id,
            metric_type="maximize",
        )


@dataclass(frozen=True)
class OfficialMetricOutcome:
    """Outcome from official metric evaluation.

    Attributes:
        scenario: Domain name
        raw_metric_name: Official metric name
        raw_score: Raw score (before normalization)
        normalized_score: Score in [0, 1]
        is_valid: Whether score was successfully computed
        failure_reason: Reason if is_valid=False
    """

    scenario: str
    raw_metric_name: str
    raw_score: float | None
    normalized_score: float | None
    is_valid: bool
    failure_reason: str | None


def _extract_root_causes_from_text(text: str) -> list[str]:
    """Extract root cause names from free-text prediction.

    This is a heuristic parser. For production use, the database evaluator
    should return structured predictions.
    """
    # Try common patterns
    import re

    # Pattern 1: "Root cause: X" or "Root causes: X, Y"
    matches = re.findall(r"[Rr]oot\s+[Cc]ause[s]?\s*[:\-]\s*(.+)", text)
    if matches:
        causes = []
        for match in matches:
            # Split by comma, semicolon, or "and"
            parts = re.split(r"[,;]|\band\b", match)
            causes.extend(p.strip() for p in parts if p.strip())
        return causes

    # Pattern 2: Numbered list "1. X\n2. Y"
    numbered = re.findall(r"^\s*\d+\.\s*(.+)$", text, re.MULTILINE)
    if numbered:
        return [n.strip() for n in numbered]

    # Fallback: Return empty (will result in recall=0)
    return []
