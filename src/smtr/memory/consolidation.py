"""TCI memory admission gate (long-term memory extension, Task 2).

Upgrades the pipeline from

    candidate memory -> retrieval/expose

to

    candidate memory -> TCI gate -> keep (validated) | reject

The gate is deliberately threshold-free: causal utility is
``delta = reward_expose - reward_withhold`` and the default (and only)
rule is ``delta > 0 -> validated, else rejected``. No new tunable
hyperparameters are introduced, and no existing memory interface is
modified.

Receiver-conditioned extension (Receiver=3 protocol):
    ``admit_for_receiver()`` adds per-receiver TCI validation.
    Same delta rule, but decision is recorded per receiver_id.
"""

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from smtr.memory.memory_schema import ReceiverValidationRecord
from smtr.memory.persistent_memory import PersistentMemoryBank

if TYPE_CHECKING:
    from smtr.analysis.cost_tracker import TCICostTracker


class AdmissionDecision(BaseModel):
    """One recorded TCI gate decision (validation result)."""

    model_config = ConfigDict(frozen=True)

    memory_id: str
    reward_expose: float
    reward_withhold: float
    delta: float
    decision: str  # "validated" | "rejected"
    timestamp: datetime


class ReceiverAdmissionDecision(BaseModel):
    """One receiver-conditioned TCI gate decision."""

    model_config = ConfigDict(frozen=True)

    memory_id: str
    receiver_id: str
    reward_expose: float
    reward_withhold: float
    delta: float
    decision: str  # "validated" | "rejected"
    validation_source: str = "receiver_counterfactual_rollout"
    timestamp: datetime


class MemoryAdmissionController:
    """Applies the TCI gate to candidate memories in a persistent bank."""

    def __init__(
        self,
        bank: PersistentMemoryBank,
        cost_tracker: "TCICostTracker | None" = None,
    ) -> None:
        self._bank = bank
        self._decisions: list[AdmissionDecision] = []
        self._cost_tracker = cost_tracker

    def admit(
        self, memory_id: str, *, reward_expose: float, reward_withhold: float,
        episode_id: int = -1,
    ) -> AdmissionDecision:
        """Run the TCI gate for one candidate memory.

        delta > 0 -> validate; otherwise reject. Every call is recorded
        in the decision log and updates the bank entry.
        """
        # Record cost (intervention: expose + withhold pair)
        if self._cost_tracker is not None:
            self._cost_tracker.record_intervention(
                memory_id=memory_id,
                expose_reward=reward_expose,
                withhold_reward=reward_withhold,
                episode=episode_id,
            )
        
        delta = reward_expose - reward_withhold
        if delta > 0:
            self._bank.validate_memory(
                memory_id, delta,
                episode_id=episode_id,
                expose_reward=reward_expose,
                withhold_reward=reward_withhold,
                decision="validated",
            )
            decision = "validated"
        else:
            self._bank.reject_memory(
                memory_id, delta,
                episode_id=episode_id,
                expose_reward=reward_expose,
                withhold_reward=reward_withhold,
                decision="rejected",
            )
            decision = "rejected"
        
        # Record cost (validation decision)
        if self._cost_tracker is not None:
            self._cost_tracker.record_validation(
                memory_id=memory_id,
                delta=delta,
                decision=decision,
                episode=episode_id,
            )
        
        record = AdmissionDecision(
            memory_id=memory_id,
            reward_expose=reward_expose,
            reward_withhold=reward_withhold,
            delta=delta,
            decision=decision,
            timestamp=datetime.now(UTC),
        )
        self._decisions.append(record)
        return record

    def admit_for_receiver(
        self,
        memory_id: str,
        *,
        receiver_id: str,
        reward_expose: float,
        reward_withhold: float,
        episode_id: int = -1,
        validation_source: str = "receiver_counterfactual_rollout",
    ) -> ReceiverAdmissionDecision:
        """Run receiver-conditioned TCI gate for one (memory, receiver) pair.

        Same threshold-free delta rule as ``admit()`` but records the
        decision per-receiver. The bank entry is updated with
        receiver-specific validation history.

        Does NOT modify the global status (validated/rejected) — only
        records per-receiver decisions in ``receiver_decisions``.
        """
        if self._cost_tracker is not None:
            self._cost_tracker.record_intervention(
                memory_id=memory_id,
                expose_reward=reward_expose,
                withhold_reward=reward_withhold,
                episode=episode_id,
            )

        delta = reward_expose - reward_withhold
        decision = "validated" if delta > 0 else "rejected"

        # Record per-receiver validation in bank
        rec = ReceiverValidationRecord(
            receiver_id=receiver_id,
            episode_id=episode_id,
            expose_reward=reward_expose,
            withhold_reward=reward_withhold,
            delta=delta,
            decision=decision,
            validation_source=validation_source,
        )
        entry = self._bank.get(memory_id)
        new_decisions = dict(entry.receiver_decisions)
        new_decisions[receiver_id] = decision
        updated = entry.model_copy(
            update={
                "receiver_id": receiver_id,
                "validation_target": receiver_id,
                "receiver_validation_history": entry.receiver_validation_history + (rec,),
                "receiver_decisions": new_decisions,
                "validation_source": validation_source,
                "validation_count": entry.validation_count + 1,
                "updated_at": datetime.now(UTC),
            }
        )
        self._bank._entries[memory_id] = updated

        if self._cost_tracker is not None:
            self._cost_tracker.record_validation(
                memory_id=memory_id,
                delta=delta,
                decision=decision,
                episode=episode_id,
            )

        record = ReceiverAdmissionDecision(
            memory_id=memory_id,
            receiver_id=receiver_id,
            reward_expose=reward_expose,
            reward_withhold=reward_withhold,
            delta=delta,
            decision=decision,
            validation_source=validation_source,
            timestamp=datetime.now(UTC),
        )
        self._decisions.append(record)
        return record

    def admit_from_pair_record(self, record: dict) -> AdmissionDecision:
        """Convenience adapter for shared-control paired records.

        Accepts any dict carrying the paired outcome fields used by the
        MARBLE feasibility pipeline (``Y_expose`` / ``Y_withhold`` style
        keys) plus ``candidate_memory_id``.
        """
        memory_id = str(record.get("candidate_memory_id", ""))
        expose = _outcome(record, ("reward_expose", "Y_expose", "share_team_success"))
        withhold = _outcome(record, ("reward_withhold", "Y_withhold", "control_team_success"))
        return self.admit(memory_id, reward_expose=expose, reward_withhold=withhold)

    @property
    def decisions(self) -> list[AdmissionDecision]:
        return list(self._decisions)

    def summary(self) -> dict[str, int]:
        return {
            "total": len(self._decisions),
            "validated": sum(1 for d in self._decisions if d.decision == "validated"),
            "rejected": sum(1 for d in self._decisions if d.decision == "rejected"),
        }

    def receiver_summary(self) -> dict[str, dict[str, int]]:
        """Per-receiver validation counts across all decisions."""
        result: dict[str, dict[str, int]] = {}
        for d in self._decisions:
            if isinstance(d, ReceiverAdmissionDecision):
                rid = d.receiver_id
                if rid not in result:
                    result[rid] = {"validated": 0, "rejected": 0}
                result[rid][d.decision] = result[rid].get(d.decision, 0) + 1
        return result


def _outcome(record: dict, keys: tuple[str, ...]) -> float:
    for key in keys:
        if key in record and record[key] is not None:
            return float(record[key])
    raise KeyError(f"none of {keys} present in paired record")
