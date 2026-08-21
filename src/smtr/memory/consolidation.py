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
"""

from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict

from smtr.memory.persistent_memory import PersistentMemoryBank


class AdmissionDecision(BaseModel):
    """One recorded TCI gate decision (validation result)."""

    model_config = ConfigDict(frozen=True)

    memory_id: str
    reward_expose: float
    reward_withhold: float
    delta: float
    decision: str  # "validated" | "rejected"
    timestamp: datetime


class MemoryAdmissionController:
    """Applies the TCI gate to candidate memories in a persistent bank."""

    def __init__(self, bank: PersistentMemoryBank) -> None:
        self._bank = bank
        self._decisions: list[AdmissionDecision] = []

    def admit(
        self, memory_id: str, *, reward_expose: float, reward_withhold: float,
        episode_id: int = -1,
    ) -> AdmissionDecision:
        """Run the TCI gate for one candidate memory.

        delta > 0 -> validate; otherwise reject. Every call is recorded
        in the decision log and updates the bank entry.
        """
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


def _outcome(record: dict, keys: tuple[str, ...]) -> float:
    for key in keys:
        if key in record and record[key] is not None:
            return float(record[key])
    raise KeyError(f"none of {keys} present in paired record")
