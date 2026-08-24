"""Online TCI evaluator: real expose/withhold rollouts per (memory, receiver).

Replaces the offline ``ReceiverInterventionEvaluator`` (which consumed
synthetic paired records) with real MARBLE Engine rollouts.

For each (candidate memory, receiver agent) pair:

1. **Branch A (expose)**: inject the candidate memory into the receiver,
   run a MARBLE continuation episode, observe ``expose_outcome``.
2. **Branch B (withhold)**: run the *same* continuation **without** the
   memory, observe ``withhold_outcome``.
3. ``delta = expose_normalized_score - withhold_normalized_score``

The outcome signal is the **official MultiAgentBench Task Score** (normalized
to [0, 1]) — NOT binary team_success.

Decision rule (threshold-free, identical to offline):

* ``delta > 0``  -> ``"validated"`` for this receiver
* ``delta <= 0`` -> ``"rejected"`` for this receiver

If either branch has ``official_metric_valid=False``:

* ``validation_status = "INVALID_OUTCOME"``
* delta is undefined (NaN)
* The record is NOT admitted or rejected — it is excluded.

Safety constraint
-----------------
The evaluator is forbidden from reading:

* task labels / ground-truth answers
* future trajectory data
* any information that would not be available at the intervention point

Only the MARBLE Engine official evaluator output is used as the reward signal.
``team_success`` is recorded for diagnostic logging only.

Usage::

    evaluator = OnlineReceiverInterventionEvaluator(
        collector=TrajectoryCollector(),
    )
    record = evaluator.validate(
        candidate=candidate_memory,
        receiver_id="agent1",
        task=marble_task,
        seed=0,
    )
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from smtr.baselines.base_memory_controller import CandidateMemory
from smtr.marble.trajectory_collector import Trajectory, TrajectoryCollector
from smtr.marble.task_loader import MarbleTask
from smtr.memory.render import render_procedure_payload

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

# Validation status constants
VALIDATION_STATUS_VALIDATED = "validated"
VALIDATION_STATUS_REJECTED = "rejected"
VALIDATION_STATUS_INVALID = "INVALID_OUTCOME"


@dataclass(frozen=True)
class OnlineValidationRecord:
    """One online TCI validation for a (memory, receiver) pair.

    Uses official MultiAgentBench normalized Task Score as the outcome.
    ``team_success`` fields are kept for diagnostic logging only.

    Attributes:
        metric_name: Official metric name (e.g., "root_cause_recall").
        raw_expose_score: Raw official score for expose branch.
        raw_withhold_score: Raw official score for withhold branch.
        normalized_expose_score: Normalized [0,1] expose score.
        normalized_withhold_score: Normalized [0,1] withhold score.
        delta: normalized_expose - normalized_withhold (NaN if invalid).
        decision: "validated" | "rejected" | "INVALID_OUTCOME".
        expose_metric_valid: Whether expose branch metric is valid.
        withhold_metric_valid: Whether withhold branch metric is valid.
        expose_success: DIAGNOSTIC ONLY — binary team_success for expose.
        withhold_success: DIAGNOSTIC ONLY — binary team_success for withhold.
    """

    memory_id: str
    receiver_id: str
    task_id: str
    scenario: str
    seed: int

    # Official metric fields
    metric_name: str = "unknown"
    raw_expose_score: float | None = None
    raw_withhold_score: float | None = None
    normalized_expose_score: float | None = None
    normalized_withhold_score: float | None = None
    delta: float = 0.0
    decision: str = VALIDATION_STATUS_REJECTED

    # Metric validity
    expose_metric_valid: bool = False
    withhold_metric_valid: bool = False

    # Diagnostic (legacy)
    expose_success: bool = False
    withhold_success: bool = False
    expose_real_engine: bool = False
    withhold_real_engine: bool = False

    expose_duration_seconds: float = 0.0
    withhold_duration_seconds: float = 0.0

    validation_source: str = "online_counterfactual_rollout"
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "receiver_id": self.receiver_id,
            "task_id": self.task_id,
            "scenario": self.scenario,
            "seed": self.seed,
            "metric_name": self.metric_name,
            "raw_expose_score": self.raw_expose_score,
            "raw_withhold_score": self.raw_withhold_score,
            "normalized_expose_score": self.normalized_expose_score,
            "normalized_withhold_score": self.normalized_withhold_score,
            "delta": self.delta,
            "decision": self.decision,
            "expose_metric_valid": self.expose_metric_valid,
            "withhold_metric_valid": self.withhold_metric_valid,
            "expose_success": self.expose_success,
            "withhold_success": self.withhold_success,
            "expose_real_engine": self.expose_real_engine,
            "withhold_real_engine": self.withhold_real_engine,
            "expose_duration_seconds": self.expose_duration_seconds,
            "withhold_duration_seconds": self.withhold_duration_seconds,
            "validation_source": self.validation_source,
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class OnlineReceiverInterventionEvaluator:
    """Online TCI: real expose/withhold rollouts per (memory, receiver).

    Uses **official MultiAgentBench normalized Task Score** as the outcome
    signal. Binary ``team_success`` is recorded for diagnostic logging
    only and is NOT used for delta computation.

    Parameters
    ----------
    collector:
        ``TrajectoryCollector`` used to execute both branches.
    """

    def __init__(
        self,
        *,
        collector: TrajectoryCollector | None = None,
    ) -> None:
        self._collector = collector or TrajectoryCollector()
        self._records: list[OnlineValidationRecord] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(
        self,
        candidate: CandidateMemory,
        receiver_id: str,
        task: MarbleTask,
        *,
        seed: int = 0,
        episode_id: int | str = "unknown",
        extra_memory_payloads: list[str] | None = None,
    ) -> OnlineValidationRecord:
        """Run expose/withhold rollouts and return the validation record.

        Parameters
        ----------
        candidate:
            The candidate memory to validate.
        receiver_id:
            Agent ID that receives (or does not receive) the memory.
        task:
            The MARBLE task to execute.
        seed:
            Generation seed for reproducibility.
        episode_id:
            Episode identifier for logging.
        extra_memory_payloads:
            Additional memory payloads already in the pool (injected
            into *both* branches to keep the comparison fair).
        """
        # Render the candidate into an injectable text payload
        candidate_payload = _render_candidate(candidate)

        # Build the full payload lists for each branch
        base_payloads = list(extra_memory_payloads or [])

        expose_payloads = base_payloads + [candidate_payload]
        withhold_payloads = list(base_payloads)  # no candidate

        receiver_ids = [receiver_id]

        # -- Branch A: expose (inject candidate) ----------------------------
        expose_traj = self._collector.collect(
            task,
            seed=seed,
            method="online_expose",
            memory_payloads=expose_payloads,
            receiver_agent_ids=receiver_ids,
        )

        # -- Branch B: withhold (no candidate) ------------------------------
        withhold_traj = self._collector.collect(
            task,
            seed=seed,
            method="online_withhold",
            memory_payloads=withhold_payloads if withhold_payloads else None,
            receiver_agent_ids=receiver_ids if withhold_payloads else None,
        )

        # -- Compute delta from official metric ----------------------------
        expose_metric_valid = expose_traj.official_metric_valid
        withhold_metric_valid = withhold_traj.official_metric_valid
        metric_name = expose_traj.official_metric_name

        both_valid = expose_metric_valid and withhold_metric_valid

        if both_valid:
            # Both branches have valid official metric → compute delta
            expose_score = expose_traj.official_metric_normalized
            withhold_score = withhold_traj.official_metric_normalized
            assert expose_score is not None and withhold_score is not None
            delta = expose_score - withhold_score
            decision = (
                VALIDATION_STATUS_VALIDATED
                if delta > 0
                else VALIDATION_STATUS_REJECTED
            )
            error_msg = None
        else:
            # At least one branch has invalid metric → INVALID_OUTCOME
            expose_score = expose_traj.official_metric_normalized
            withhold_score = withhold_traj.official_metric_normalized
            delta = 0.0
            decision = VALIDATION_STATUS_INVALID
            reasons = []
            if not expose_metric_valid:
                reasons.append(f"expose: {expose_traj.official_metric_error}")
            if not withhold_metric_valid:
                reasons.append(f"withhold: {withhold_traj.official_metric_error}")
            error_msg = "; ".join(reasons)
            logger.warning(
                "INVALID_OUTCOME: memory=%s receiver=%s task=%s seed=%d "
                "reason=%s",
                candidate.memory_id, receiver_id, task.task_id, seed,
                error_msg,
            )

        record = OnlineValidationRecord(
            memory_id=candidate.memory_id,
            receiver_id=receiver_id,
            task_id=task.task_id,
            scenario=task.scenario,
            seed=seed,
            metric_name=metric_name,
            raw_expose_score=expose_traj.official_metric_raw,
            raw_withhold_score=withhold_traj.official_metric_raw,
            normalized_expose_score=expose_score,
            normalized_withhold_score=withhold_score,
            delta=delta,
            decision=decision,
            expose_metric_valid=expose_metric_valid,
            withhold_metric_valid=withhold_metric_valid,
            # Diagnostic only — NOT used for delta
            expose_success=expose_traj.team_success,
            withhold_success=withhold_traj.team_success,
            expose_real_engine=expose_traj.real_engine_executed,
            withhold_real_engine=withhold_traj.real_engine_executed,
            expose_duration_seconds=expose_traj.engine_duration_seconds,
            withhold_duration_seconds=withhold_traj.engine_duration_seconds,
            error=error_msg,
        )
        self._records.append(record)

        logger.info(
            "online_tci: memory=%s receiver=%s task=%s seed=%d "
            "metric=%s expose=%.3f withhold=%.3f delta=%.3f -> %s",
            candidate.memory_id,
            receiver_id,
            task.task_id,
            seed,
            metric_name,
            expose_score if expose_score is not None else float("nan"),
            withhold_score if withhold_score is not None else float("nan"),
            delta,
            decision,
        )
        return record

    def validate_batch(
        self,
        candidates: list[CandidateMemory],
        receiver_ids: list[str],
        task: MarbleTask,
        *,
        seed: int = 0,
        extra_memory_payloads: list[str] | None = None,
    ) -> list[OnlineValidationRecord]:
        """Validate multiple candidates across multiple receivers.

        Returns one ``OnlineValidationRecord`` per (candidate, receiver)
        pair — ``len(candidates) * len(receiver_ids)`` records total.
        """
        records: list[OnlineValidationRecord] = []
        for candidate in candidates:
            for rid in receiver_ids:
                rec = self.validate(
                    candidate,
                    rid,
                    task,
                    seed=seed,
                    extra_memory_payloads=extra_memory_payloads,
                )
                records.append(rec)
        return records

    @property
    def records(self) -> list[OnlineValidationRecord]:
        """All validation records collected so far."""
        return list(self._records)

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _render_candidate(candidate: CandidateMemory) -> str:
    """Render a CandidateMemory into an injectable text payload.

    Falls back to raw content if the memory does not carry a structured
    ``payload`` dict (which ``render_procedure_payload`` requires).
    """
    # If the candidate has a structured payload in metadata, use it
    payload = candidate.metadata.get("payload") if candidate.metadata else None
    if isinstance(payload, dict):
        try:
            return render_procedure_payload({"payload": payload})
        except (ValueError, KeyError):
            pass

    # Fallback: use the raw content directly
    return candidate.content
