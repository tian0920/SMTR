"""Online TCI evaluator: real expose/withhold rollouts per (memory, receiver).

Replaces the offline ``ReceiverInterventionEvaluator`` (which consumed
synthetic paired records) with real MARBLE Engine rollouts.

For each (candidate memory, receiver agent) pair:

1. **Branch A (expose)**: inject the candidate memory into the receiver,
   run a MARBLE continuation episode, observe ``expose_outcome``.
2. **Branch B (withhold)**: run the *same* continuation **without** the
   memory, observe ``withhold_outcome``.
3. ``delta = expose_outcome - withhold_outcome``

Decision rule (threshold-free, identical to offline):

* ``delta > 0``  -> ``"validated"`` for this receiver
* ``delta <= 0`` -> ``"rejected"`` for this receiver

Safety constraint
-----------------
The evaluator is forbidden from reading:

* task labels / ground-truth answers
* future trajectory data
* any information that would not be available at the intervention point

Only the MARBLE Engine interaction outcome (``team_success`` / ``score``)
is used as the reward signal.

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

@dataclass(frozen=True)
class OnlineValidationRecord:
    """One online TCI validation for a (memory, receiver) pair.

    Compatible with the existing ``ReceiverValidationRecord`` in
    ``memory_schema.py`` — carries the same ``delta`` / ``decision``
    semantics plus provenance fields for the online rollout.
    """

    memory_id: str
    receiver_id: str
    task_id: str
    scenario: str
    seed: int

    expose_outcome: float
    withhold_outcome: float
    delta: float
    decision: str  # "validated" | "rejected"

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
            "expose_outcome": self.expose_outcome,
            "withhold_outcome": self.withhold_outcome,
            "delta": self.delta,
            "decision": self.decision,
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

    Parameters
    ----------
    collector:
        ``TrajectoryCollector`` used to execute both branches.
    use_score:
        When ``True``, use the MARBLE numeric ``score`` as the reward
        signal.  When ``False`` (default), use the binary
        ``team_success`` (0.0 or 1.0) which is more robust across
        heterogeneous domains.
    """

    def __init__(
        self,
        *,
        collector: TrajectoryCollector | None = None,
        use_score: bool = False,
    ) -> None:
        self._collector = collector or TrajectoryCollector()
        self._use_score = use_score
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

        # -- Compute delta --------------------------------------------------
        expose_reward = self._extract_reward(expose_traj)
        withhold_reward = self._extract_reward(withhold_traj)
        delta = expose_reward - withhold_reward
        decision = "validated" if delta > 0 else "rejected"

        record = OnlineValidationRecord(
            memory_id=candidate.memory_id,
            receiver_id=receiver_id,
            task_id=task.task_id,
            scenario=task.scenario,
            seed=seed,
            expose_outcome=expose_reward,
            withhold_outcome=withhold_reward,
            delta=delta,
            decision=decision,
            expose_success=expose_traj.team_success,
            withhold_success=withhold_traj.team_success,
            expose_real_engine=expose_traj.real_engine_executed,
            withhold_real_engine=withhold_traj.real_engine_executed,
            expose_duration_seconds=expose_traj.engine_duration_seconds,
            withhold_duration_seconds=withhold_traj.engine_duration_seconds,
        )
        self._records.append(record)

        logger.info(
            "online_tci: memory=%s receiver=%s task=%s seed=%d "
            "expose=%.2f withhold=%.2f delta=%.2f -> %s",
            candidate.memory_id,
            receiver_id,
            task.task_id,
            seed,
            expose_reward,
            withhold_reward,
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

    def summary(self) -> dict[str, Any]:
        """Aggregate statistics across all validations."""
        if not self._records:
            return {"total": 0}
        total = len(self._records)
        validated = sum(1 for r in self._records if r.decision == "validated")
        rejected = total - validated
        deltas = [r.delta for r in self._records]
        mean_delta = sum(deltas) / len(deltas) if deltas else 0.0

        # Per-receiver breakdown
        per_receiver: dict[str, dict[str, int]] = {}
        for r in self._records:
            if r.receiver_id not in per_receiver:
                per_receiver[r.receiver_id] = {"validated": 0, "rejected": 0, "total": 0}
            per_receiver[r.receiver_id][r.decision] += 1  # type: ignore[operator]
            per_receiver[r.receiver_id]["total"] += 1

        # Engine execution stats
        real_engine_count = sum(
            1 for r in self._records
            if r.expose_real_engine and r.withhold_real_engine
        )
        total_engine_time = sum(
            r.expose_duration_seconds + r.withhold_duration_seconds
            for r in self._records
        )

        return {
            "total": total,
            "validated": validated,
            "rejected": rejected,
            "validation_rate": validated / max(total, 1),
            "mean_delta": round(mean_delta, 4),
            "real_engine_executions": real_engine_count,
            "total_engine_time_seconds": round(total_engine_time, 2),
            "per_receiver": per_receiver,
        }


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
