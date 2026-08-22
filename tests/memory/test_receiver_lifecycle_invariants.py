"""End-to-end invariant tests for receiver-conditioned memory lifecycle.

Prevents future regression of the receiver-conditioned lifecycle:

    Invariant 1: receiver1-validated memory cannot be retrieved by receiver2
    Invariant 2: receiver2-rejected memory cannot enter receiver2 context
    Invariant 3: same memory can have receiver1 validated, receiver2 rejected
    Invariant 4: retrieval result is deterministic
    Invariant 5: missing counterfactual outcome fails loudly
"""

import math

import pytest

from smtr.memory.consolidation import MemoryAdmissionController
from smtr.memory.memory_schema import ReceiverValidationRecord
from smtr.memory.persistent_memory import PersistentMemoryBank
from smtr.memory.receiver_intervention import (
    MissingCounterfactualOutcomeError,
    ReceiverInterventionEvaluator,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _bank_with_candidate() -> PersistentMemoryBank:
    bank = PersistentMemoryBank()
    bank.add_candidate(
        memory_id="m1",
        content="use index scan on large tables",
        source_episode=0,
        receiver="agent1",
        created_step=0,
    )
    return bank


def _bank_with_two_receivers() -> tuple[PersistentMemoryBank, MemoryAdmissionController]:
    bank = _bank_with_candidate()
    ctrl = MemoryAdmissionController(bank)
    # receiver1: validated (delta > 0)
    ctrl.admit_for_receiver(
        "m1", receiver_id="receiver1",
        reward_expose=1.0, reward_withhold=0.0, episode_id=1,
    )
    # receiver2: rejected (delta <= 0)
    ctrl.admit_for_receiver(
        "m1", receiver_id="receiver2",
        reward_expose=0.0, reward_withhold=1.0, episode_id=1,
    )
    return bank, ctrl


# ---------------------------------------------------------------------------
# Invariant 1: receiver1-validated memory cannot be retrieved by receiver2
# ---------------------------------------------------------------------------

class TestInvariant1:
    def test_receiver1_validated_not_retrieved_by_receiver2(self) -> None:
        bank, _ = _bank_with_two_receivers()
        # receiver1 should see m1
        r1_memories = bank.get_receiver_validated_memories("receiver1")
        assert [e.memory_id for e in r1_memories] == ["m1"]
        # receiver2 should NOT see m1
        r2_memories = bank.get_receiver_validated_memories("receiver2")
        assert r2_memories == []


# ---------------------------------------------------------------------------
# Invariant 2: receiver2-rejected memory cannot enter receiver2 context
# ---------------------------------------------------------------------------

class TestInvariant2:
    def test_rejected_memory_excluded_from_context(self) -> None:
        bank, _ = _bank_with_two_receivers()
        # Simulate context building for receiver2
        context_memories = bank.get_receiver_validated_memories("receiver2")
        memory_ids = [e.memory_id for e in context_memories]
        assert "m1" not in memory_ids

    def test_rejected_status_recorded(self) -> None:
        bank, _ = _bank_with_two_receivers()
        assert bank.get_receiver_status("m1", "receiver2") == "rejected"


# ---------------------------------------------------------------------------
# Invariant 3: same memory can have receiver1 validated, receiver2 rejected
# ---------------------------------------------------------------------------

class TestInvariant3:
    def test_divergent_receiver_decisions(self) -> None:
        bank, _ = _bank_with_two_receivers()
        entry = bank.get("m1")
        assert entry.receiver_status["receiver1"] == "validated"
        assert entry.receiver_status["receiver2"] == "rejected"
        assert entry.receiver_decisions["receiver1"] == "validated"
        assert entry.receiver_decisions["receiver2"] == "rejected"

    def test_validation_history_per_receiver(self) -> None:
        bank, _ = _bank_with_two_receivers()
        entry = bank.get("m1")
        r1_records = [r for r in entry.receiver_validation_history
                      if r.receiver_id == "receiver1"]
        r2_records = [r for r in entry.receiver_validation_history
                      if r.receiver_id == "receiver2"]
        assert len(r1_records) == 1
        assert len(r2_records) == 1
        assert r1_records[0].delta > 0
        assert r2_records[0].delta < 0

    def test_global_status_unchanged_by_receiver_admission(self) -> None:
        """Legacy global status must remain 'candidate' after receiver admissions."""
        bank, _ = _bank_with_two_receivers()
        entry = bank.get("m1")
        assert entry.status == "candidate"  # Not modified by admit_for_receiver


# ---------------------------------------------------------------------------
# Invariant 4: retrieval result is deterministic
# ---------------------------------------------------------------------------

class TestInvariant4:
    def test_retrieval_deterministic_across_calls(self) -> None:
        bank, _ = _bank_with_two_receivers()
        # Add a second memory
        bank.add_candidate(
            memory_id="m2", content="check pg_stat_statements",
            source_episode=1, receiver="agent2", created_step=1,
        )
        ctrl = MemoryAdmissionController(bank)
        ctrl.admit_for_receiver(
            "m2", receiver_id="receiver1",
            reward_expose=0.8, reward_withhold=0.2, episode_id=2,
        )
        # Call retrieval multiple times
        results = [
            [e.memory_id for e in bank.get_receiver_validated_memories("receiver1")]
            for _ in range(10)
        ]
        # All results must be identical
        assert all(r == results[0] for r in results)
        assert results[0] == ["m1", "m2"]

    def test_retrieval_deterministic_after_save_load(self, tmp_path) -> None:
        bank, _ = _bank_with_two_receivers()
        path = tmp_path / "bank.jsonl"
        bank.save(path)
        loaded = PersistentMemoryBank.load(path)
        original = [e.memory_id for e in bank.get_receiver_validated_memories("receiver1")]
        reloaded = [e.memory_id for e in loaded.get_receiver_validated_memories("receiver1")]
        assert original == reloaded


# ---------------------------------------------------------------------------
# Invariant 5: missing counterfactual outcome fails loudly
# ---------------------------------------------------------------------------

class TestInvariant5:
    def test_no_outcome_source_raises(self) -> None:
        ev = ReceiverInterventionEvaluator()
        with pytest.raises(MissingCounterfactualOutcomeError) as exc_info:
            ev.evaluate(
                memory_id="m1",
                receiver_ids=["agent1"],
                episode_id=42,
            )
        assert "m1" in str(exc_info.value)
        assert "agent1" in str(exc_info.value)

    def test_nan_outcome_raises(self) -> None:
        ev = ReceiverInterventionEvaluator()
        with pytest.raises(MissingCounterfactualOutcomeError) as exc_info:
            ev.evaluate(
                memory_id="m1",
                receiver_ids=["agent1"],
                paired_outcomes={"agent1": (float("nan"), 0.0)},
            )
        assert "not finite" in str(exc_info.value)

    def test_none_outcome_raises(self) -> None:
        ev = ReceiverInterventionEvaluator()
        with pytest.raises(MissingCounterfactualOutcomeError):
            ev.evaluate(
                memory_id="m1",
                receiver_ids=["agent1"],
                paired_outcomes={"agent1": (None, 0.0)},
            )

    def test_outcome_fn_returns_none_raises(self) -> None:
        ev = ReceiverInterventionEvaluator(outcome_fn=lambda *a: None)
        with pytest.raises(MissingCounterfactualOutcomeError):
            ev.evaluate(
                memory_id="m1",
                receiver_ids=["agent1"],
                episode_id=1,
            )

    def test_valid_outcome_does_not_raise(self) -> None:
        """Sanity: valid outcome must NOT raise."""
        ev = ReceiverInterventionEvaluator()
        result = ev.evaluate(
            memory_id="m1",
            receiver_ids=["agent1"],
            paired_outcomes={"agent1": (1.0, 0.0)},
            episode_id=1,
        )
        assert result.n_validated == 1


# ---------------------------------------------------------------------------
# Cross-invariant: receiver_validation_summary consistency
# ---------------------------------------------------------------------------

class TestReceiverValidationSummary:
    def test_summary_reflects_divergent_decisions(self) -> None:
        bank, _ = _bank_with_two_receivers()
        summary = bank.receiver_validation_summary()
        assert "receiver1" in summary
        assert "receiver2" in summary
        assert summary["receiver1"]["validated"] == 1
        assert summary["receiver1"]["rejected"] == 0
        assert summary["receiver2"]["validated"] == 0
        assert summary["receiver2"]["rejected"] == 1

    def test_summary_latest_delta(self) -> None:
        bank, ctrl = _bank_with_two_receivers()
        # Re-validate receiver2 with positive delta
        ctrl.admit_for_receiver(
            "m1", receiver_id="receiver2",
            reward_expose=0.8, reward_withhold=0.1, episode_id=2,
        )
        summary = bank.receiver_validation_summary()
        assert summary["receiver2"]["latest_delta"] == pytest.approx(0.7)
        assert summary["receiver2"]["validation_count"] == 2
        # receiver2 now validated
        assert bank.get_receiver_status("m1", "receiver2") == "validated"
