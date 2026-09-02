"""RIMA canonical outcome definition (Phase 1).

The formal RIMA estimand (paper Eq. 4) is::

    tau(m, a_r | x_t) = E[Y(1) - Y(0) | m, a_r, x_t]

Phase 1 pins down the outcome variable ``Y``:

* **Primary causal outcome** = normalized official MultiAgentBench Task
  Score, in ``[0, 1]``.
* ``team_success`` is FORBIDDEN as a primary causal outcome. It is
  diagnostic metadata only.

Per-scenario official metrics (MultiAgentBench ACL 2025, Table 1):

* ``database``: official root-cause score (root cause recall, subset match)
* ``research``: official task-evaluation score
* ``minecraft``: official ``block_hit_rate``
* ``coding``: official coding task score
* ``bargaining``: official bargaining task score

This module wraps the existing
:class:`smtr.marble.outcome.official_metric_outcome.OfficialMetricOutcomeEvaluator`
and exposes the single canonical entry point used by the RIMA runner.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from smtr.marble.outcome.official_metric_outcome import (
    OfficialMetricOutcome,
    OfficialMetricOutcomeEvaluator,
    ReceiverTCIOutcome,
)

__all__ = [
    "RimaOfficialOutcome",
    "RimaOutcomeEvaluator",
    "PRIMARY_OUTCOME_NAME",
    "TEAM_SUCCESS_ROLE",
    "SCENARIO_OFFICIAL_METRICS",
]

#: The single formal outcome name.
PRIMARY_OUTCOME_NAME = "official_task_score"

#: team_success is diagnostic metadata only — never a primary causal outcome.
TEAM_SUCCESS_ROLE = "diagnostic_metadata_only"

#: Official per-scenario metric mapping (source of truth).
SCENARIO_OFFICIAL_METRICS = {
    "database": "root_cause_recall",
    "research": "avg_innovation_safety_feasibility",
    "minecraft": "block_hit_rate",
    "coding": "avg_code_quality",
    "bargaining": "avg_negotiation_quality",
}


@dataclass(frozen=True)
class RimaOfficialOutcome:
    """Canonical RIMA outcome record.

    Attributes:
        scenario: MultiAgentBench domain.
        task_score: normalized official Task Score in [0, 1] (primary outcome Y).
        is_valid: whether the official score was computed.
        failure_reason: reason when ``is_valid`` is False (fail-closed).
        team_success: optional diagnostic metadata (never used for admission).
    """

    scenario: str
    task_score: float | None
    is_valid: bool
    failure_reason: str | None = None
    team_success: bool | None = None

    @property
    def normalized_score(self) -> float | None:
        """Alias for the primary outcome Y."""
        return self.task_score


class RimaOutcomeEvaluator:
    """Canonical evaluator producing the official Task Score for RIMA.

    Thin, validated wrapper over the official metric adapter. Guarantees:

    * Outcome is always the official Task Score in [0, 1].
    * Invalid outcome -> ``is_valid=False`` and ``task_score=None``
      (fail-closed; never a silent zero).
    * ``team_success`` is carried only as diagnostic metadata.
    """

    def __init__(self, scenario: str) -> None:
        if scenario not in SCENARIO_OFFICIAL_METRICS:
            raise ValueError(
                f"Unknown scenario for RIMA official outcome: {scenario!r}. "
                f"Supported: {sorted(SCENARIO_OFFICIAL_METRICS)}"
            )
        self.scenario = scenario
        self._evaluator = OfficialMetricOutcomeEvaluator(scenario=scenario)

    @property
    def official_metric_name(self) -> str:
        return SCENARIO_OFFICIAL_METRICS[self.scenario]

    def evaluate(
        self,
        *,
        task: dict[str, Any],
        run_result: dict[str, Any],
    ) -> RimaOfficialOutcome:
        """Compute the official Task Score for one run."""
        outcome: OfficialMetricOutcome = self._evaluator.evaluate(
            task=task, run_result=run_result
        )
        team_success = self._extract_team_success(run_result)
        return RimaOfficialOutcome(
            scenario=self.scenario,
            task_score=outcome.normalized_score,
            is_valid=outcome.is_valid,
            failure_reason=outcome.failure_reason,
            team_success=team_success,
        )

    def compute_delta(
        self,
        *,
        expose_result: dict[str, Any],
        withhold_result: dict[str, Any],
        receiver_id: str,
        task: dict[str, Any] | None = None,
    ) -> ReceiverTCIOutcome:
        """Oriented official-score delta between expose and withhold.

        Invalid on either branch -> delta is ``None`` (fail-closed).
        """
        return self._evaluator.compute_delta(
            expose_result=expose_result,
            withhold_result=withhold_result,
            receiver_id=receiver_id,
        )

    @staticmethod
    def _extract_team_success(run_result: dict[str, Any]) -> bool | None:
        """Diagnostic only; never feeds admission."""
        if "team_success" in run_result:
            value = run_result["team_success"]
            return bool(value) if value is not None else None
        return None
