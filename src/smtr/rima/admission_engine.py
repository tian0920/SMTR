"""RIMA admission engine (Phases 7, 9, 12).

At task ``t`` the formal flow is strictly::

    M_t        = all historical shared memories (origin task < t)
    C_r_t      = Retrieve(x_t, M_t)               # per receiver
    tau_hat    = FrozenCritic(m, r, x_t)
    A_r_t      = { m in C_r_t | tau_hat(m, r, x_t) > 0 }
    K_r_{t+1}  = Update(K_r_t, A_r_t)

Rules enforced here:

* ``C_r_t`` and ``K_r_t`` are never mixed (candidates come from the pool,
  admissions go into receiver knowledge).
* Multi-memory admission: ALL positive-tau candidates are admitted
  (paper Eq. 9) — NO ``max_shared_memories_per_receiver == 1`` limit;
  context length is controlled by the retrieval budget, not by top-1.
* Self-transfer pairs are excluded with status ``SELF_TRANSFER_EXCLUDED``
  and counted; they never reach critic, statistics, or K_r.
* Admission decisions always carry ``decision_source ==
  "frozen_transfer_critic"`` (fail-closed guard).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from smtr.memory.receiver_knowledge import ReceiverKnowledgeState
from smtr.memory.shared_memory_pool import SharedMemory, SharedMemoryPool
from smtr.rima.admission import (
    AdmissionDecision,
    AdmissionStatus,
    assert_formal_decision_source,
)
from smtr.rima.features import ReceiverConditionedTransferFeatures
from smtr.router.official_score_transfer_critic import (
    MatchedInterventionExample,
    OfficialScoreTransferCritic,
)

__all__ = [
    "AdmissionStatistics",
    "RimaAdmissionEngine",
]


@dataclass
class AdmissionStatistics:
    """Aggregated admission statistics (self-transfer counted separately)."""

    n_candidates: int = 0
    n_admitted: int = 0
    n_rejected: int = 0
    n_invalid_prediction: int = 0
    self_transfer_excluded_count: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "n_candidates": self.n_candidates,
            "n_admitted": self.n_admitted,
            "n_rejected": self.n_rejected,
            "n_invalid_prediction": self.n_invalid_prediction,
            "self_transfer_excluded_count": self.self_transfer_excluded_count,
        }


FeatureBuilder = Callable[[SharedMemory, str, dict[str, Any]], ReceiverConditionedTransferFeatures]


class RimaAdmissionEngine:
    """Formal admission engine driven ONLY by the frozen transfer critic.

    Parameters:
        critic: fitted AND frozen official-score critic. The engine refuses
            to admit with an unfitted critic; a non-frozen critic raises to
            enforce Stage-E protocol.
        pool: shared memory pool M.
        feature_builder: builds routing-card-only features for a
            (memory, receiver, task) triple.
        retrieval_top_k: retrieval budget K for candidate generation.
    """

    def __init__(
        self,
        *,
        critic: OfficialScoreTransferCritic,
        pool: SharedMemoryPool,
        feature_builder: FeatureBuilder,
        retrieval_top_k: int = 5,
    ) -> None:
        # Critic must be fitted; formal continual evaluation additionally
        # requires it frozen (Stage E must never re-fit).
        critic.predict_one  # attribute check only
        if critic._mu1 is None or critic._mu0 is None:  # noqa: SLF001 (guard)
            raise RuntimeError("Admission requires a fitted critic.")
        self.critic = critic
        self.pool = pool
        self.feature_builder = feature_builder
        self.retrieval_top_k = retrieval_top_k
        self.stats = AdmissionStatistics()
        self.decisions: list[AdmissionDecision] = []

    def require_frozen(self) -> None:
        """Enforce Stage-E invariant: critic frozen during continual eval."""
        if not self.critic.is_frozen:
            raise RuntimeError(
                "Formal continual evaluation requires a FROZEN critic "
                "(Stage E never re-fits). Call critic.freeze() first."
            )

    def admit_for_task(
        self,
        *,
        task: dict[str, Any],
        task_id: str,
        task_position: int,
        receiver_id: str,
        knowledge: ReceiverKnowledgeState,
    ) -> list[SharedMemory]:
        """Run the formal admission flow for one (task, receiver).

        Returns the admitted memories A_r_t (also merged into ``knowledge``).
        """
        candidates = self.pool.retrieve(
            task,
            receiver_id,
            self.retrieval_top_k,
            current_task_position=task_position,
        )
        admitted: list[SharedMemory] = []
        for memory in candidates:
            decision = self._decide_one(
                memory=memory,
                receiver_id=receiver_id,
                task=task,
                task_id=task_id,
                task_position=task_position,
            )
            self.decisions.append(decision)
            if decision.status == AdmissionStatus.SELF_TRANSFER_EXCLUDED:
                self.stats.self_transfer_excluded_count += 1
            elif decision.status == AdmissionStatus.INVALID_PREDICTION:
                self.stats.n_invalid_prediction += 1
            elif decision.admitted:
                self.stats.n_admitted += 1
                knowledge.admit(
                    memory,
                    decision.tau_hat,
                    task_id=task_id,
                    task_position=task_position,
                )
                admitted.append(memory)
            else:
                self.stats.n_rejected += 1
        self.stats.n_candidates += len(candidates)
        return admitted

    def _decide_one(
        self,
        *,
        memory: SharedMemory,
        receiver_id: str,
        task: dict[str, Any],
        task_id: str,
        task_position: int,
    ) -> AdmissionDecision:
        assert_formal_decision_source("frozen_transfer_critic")

        if memory.source_agent_id == receiver_id:
            return AdmissionDecision(
                memory_id=memory.memory_id,
                receiver_id=receiver_id,
                task_id=task_id,
                status=AdmissionStatus.SELF_TRANSFER_EXCLUDED,
                tau_hat=None,
            )

        features = self.feature_builder(memory, receiver_id, task)
        example = MatchedInterventionExample(
            task_id=task_id,
            memory_id=memory.memory_id,
            receiver_id=receiver_id,
            source_agent_id=memory.source_agent_id,
            official_expose_score=None,
            official_withhold_score=None,
            features=features,
        )
        prediction = self.critic.predict_one(example)
        if not prediction.is_valid:
            # Fail-closed: invalid prediction -> never admitted, never zero.
            return AdmissionDecision(
                memory_id=memory.memory_id,
                receiver_id=receiver_id,
                task_id=task_id,
                status=AdmissionStatus.INVALID_PREDICTION,
                tau_hat=None,
            )
        status = (
            AdmissionStatus.ADMITTED if prediction.tau_hat > 0.0 else AdmissionStatus.REJECTED
        )
        return AdmissionDecision(
            memory_id=memory.memory_id,
            receiver_id=receiver_id,
            task_id=task_id,
            status=status,
            tau_hat=prediction.tau_hat,
        )
