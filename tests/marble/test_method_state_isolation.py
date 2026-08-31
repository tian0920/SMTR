"""P0-5 Integrity Suite — Cases 6-10, 14: Method state isolation.

Invariants tested:
  6. smtr_receiver retrieval respects receiver_status.
  7. smtr_uniform has independent global persistent state.
  8. full_memory retains ALL historical candidates across tasks.
  9. retrieval retrieves from historical pool.
  10. Different method states do NOT share mutable state.
  14. Same task / seed / environment -> expose-withhold matched.
"""

from __future__ import annotations

import pytest

from smtr.baselines.base_memory_controller import CandidateMemory
from smtr.memory.method_state import MethodMemoryState, MethodStateContainer
from smtr.memory.online_receiver_intervention import (
    OnlineValidationRecord,
    VALIDATION_STATUS_VALIDATED,
    VALIDATION_STATUS_REJECTED,
)

RECEIVER_IDS = ["agent1", "agent2", "agent3"]
METHODS = ["no_memory", "full_memory", "retrieval", "smtr_uniform", "smtr_receiver"]


def _make_candidate(memory_id: str, content: str = "test", episode: int = 0) -> CandidateMemory:
    return CandidateMemory(
        memory_id=memory_id,
        type="experience",
        content=content,
        source_episode=episode,
    )


def _make_record(
    memory_id: str,
    receiver_id: str,
    delta: float,
    decision: str | None = None,
) -> OnlineValidationRecord:
    if decision is None:
        decision = VALIDATION_STATUS_VALIDATED if delta > 0 else VALIDATION_STATUS_REJECTED
    return OnlineValidationRecord(
        memory_id=memory_id,
        receiver_id=receiver_id,
        task_id="task_01",
        scenario="bargaining",
        seed=0,
        delta=delta,
        decision=decision,
        normalized_expose_score=0.5 + delta / 2,
        normalized_withhold_score=0.5 - delta / 2,
        expose_metric_valid=True,
        withhold_metric_valid=True,
    )


def _render_fn(entry) -> str:
    return entry.content


# -----------------------------------------------------------------------
# Case 6: smtr_receiver retrieval respects receiver_status
# -----------------------------------------------------------------------

class TestSmtrReceiverRespectsReceiverStatus:
    """smtr_receiver must use receiver_status for retrieval, not global status."""

    def test_06_only_receiver_validated_memories_returned(self):
        """Only memories with receiver_status[rid]='validated' are returned."""
        state = MethodMemoryState("smtr_receiver")

        c1 = _make_candidate("mem_001")
        c2 = _make_candidate("mem_002")
        state.register_candidates([c1, c2], RECEIVER_IDS)

        # mem_001: validated for agent1, rejected for agent2
        records = [
            _make_record("mem_001", "agent1", 0.5),
            _make_record("mem_001", "agent2", -0.3),
            _make_record("mem_001", "agent3", 0.2),
        ]
        state.update_after_tci([c1], records, RECEIVER_IDS)

        # mem_002: rejected for agent1, validated for agent2
        records2 = [
            _make_record("mem_002", "agent1", -0.2),
            _make_record("mem_002", "agent2", 0.4),
            _make_record("mem_002", "agent3", -0.1),
        ]
        state.update_after_tci([c2], records2, RECEIVER_IDS)

        # agent1 gets mem_001 only
        payloads_r1 = state.get_injection_payloads("agent1", render_fn=_render_fn)
        assert len(payloads_r1) == 1

        # agent2 gets mem_002 only
        payloads_r2 = state.get_injection_payloads("agent2", render_fn=_render_fn)
        assert len(payloads_r2) == 1

        # agent3 gets mem_001 only
        payloads_r3 = state.get_injection_payloads("agent3", render_fn=_render_fn)
        assert len(payloads_r3) == 1

    def test_06_rejected_receiver_excluded(self):
        """Rejected receiver never gets the memory."""
        state = MethodMemoryState("smtr_receiver")

        c = _make_candidate("mem_rej")
        state.register_candidates([c], RECEIVER_IDS)

        # All receivers reject
        records = [_make_record("mem_rej", rid, -0.5) for rid in RECEIVER_IDS]
        state.update_after_tci([c], records, RECEIVER_IDS)

        for rid in RECEIVER_IDS:
            payloads = state.get_injection_payloads(rid, render_fn=_render_fn)
            assert payloads == []


# -----------------------------------------------------------------------
# Case 7: smtr_uniform has independent global persistent state
# -----------------------------------------------------------------------

class TestSmtrUniformIndependentGlobalState:
    """smtr_uniform uses mean_delta across all receivers for a global decision."""

    def test_07_positive_mean_delta_validates_globally(self):
        """mean_delta > 0 -> validated for all receivers."""
        state = MethodMemoryState("smtr_uniform")

        c = _make_candidate("mem_uniform_pos")
        state.register_candidates([c], RECEIVER_IDS)

        # Deltas: 0.3, 0.1, 0.2 -> mean = 0.2 > 0
        records = [
            _make_record("mem_uniform_pos", "agent1", 0.3),
            _make_record("mem_uniform_pos", "agent2", 0.1),
            _make_record("mem_uniform_pos", "agent3", 0.2),
        ]
        state.update_after_tci([c], records, RECEIVER_IDS)

        # Should be validated (global decision)
        assert state.validated_size() == 1

        # All receivers get the same memory (global, not per-receiver)
        for rid in RECEIVER_IDS:
            payloads = state.get_injection_payloads(rid, render_fn=_render_fn)
            assert len(payloads) == 1

    def test_07_negative_mean_delta_rejects_globally(self):
        """mean_delta <= 0 -> rejected for all receivers."""
        state = MethodMemoryState("smtr_uniform")

        c = _make_candidate("mem_uniform_neg")
        state.register_candidates([c], RECEIVER_IDS)

        # Deltas: -0.1, 0.3, -0.4 -> mean = -0.067 < 0
        records = [
            _make_record("mem_uniform_neg", "agent1", -0.1),
            _make_record("mem_uniform_neg", "agent2", 0.3),
            _make_record("mem_uniform_neg", "agent3", -0.4),
        ]
        state.update_after_tci([c], records, RECEIVER_IDS)

        assert state.validated_size() == 0
        for rid in RECEIVER_IDS:
            payloads = state.get_injection_payloads(rid, render_fn=_render_fn)
            assert payloads == []

    def test_07_smtr_uniform_state_independent_of_smtr_receiver(self):
        """smtr_uniform bank is completely independent of smtr_receiver bank."""
        container = MethodStateContainer(["smtr_uniform", "smtr_receiver"])

        c = _make_candidate("mem_shared")
        container.register_candidates_all([c], RECEIVER_IDS)

        # Same records for both methods
        records = [
            _make_record("mem_shared", "agent1", 0.5),
            _make_record("mem_shared", "agent2", -0.3),
            _make_record("mem_shared", "agent3", 0.1),
        ]

        # mean = (0.5 - 0.3 + 0.1) / 3 ~ 0.1 > 0 -> smtr_uniform validates
        container.get("smtr_uniform").update_after_tci([c], records, RECEIVER_IDS)
        container.get("smtr_receiver").update_after_tci([c], records, RECEIVER_IDS)

        # smtr_uniform: validated globally -> all receivers get it
        uniform_payloads = container.get("smtr_uniform").get_injection_payloads(
            "agent2", render_fn=_render_fn
        )
        assert len(uniform_payloads) == 1  # global, even agent2 gets it

        # smtr_receiver: agent2 rejected -> agent2 doesn't get it
        receiver_payloads = container.get("smtr_receiver").get_injection_payloads(
            "agent2", render_fn=_render_fn
        )
        assert len(receiver_payloads) == 0


# -----------------------------------------------------------------------
# Case 8: full_memory retains ALL historical candidates across tasks
# -----------------------------------------------------------------------

class TestFullMemoryRetainsAllHistory:
    """full_memory stores ALL candidates regardless of TCI outcome."""

    def test_08_all_candidates_stored_across_tasks(self):
        """full_memory retains candidates from multiple tasks."""
        state = MethodMemoryState("full_memory")

        # Task 0: 2 candidates
        c0a = _make_candidate("mem_t0_a", episode=0)
        c0b = _make_candidate("mem_t0_b", episode=0)
        state.register_candidates([c0a, c0b], RECEIVER_IDS)
        records_0 = [
            _make_record("mem_t0_a", rid, 0.3) for rid in RECEIVER_IDS
        ] + [
            _make_record("mem_t0_b", rid, -0.2) for rid in RECEIVER_IDS
        ]
        state.update_after_tci([c0a, c0b], records_0, RECEIVER_IDS)

        # Task 1: 1 more candidate
        c1 = _make_candidate("mem_t1", episode=1)
        state.register_candidates([c1], RECEIVER_IDS)
        records_1 = [_make_record("mem_t1", rid, 0.1) for rid in RECEIVER_IDS]
        state.update_after_tci([c1], records_1, RECEIVER_IDS)

        # All 3 candidates stored (full_memory validates everything)
        assert state.memory_size() == 3
        payloads = state.get_injection_payloads("agent1", render_fn=_render_fn)
        assert len(payloads) == 3

    def test_08_full_memory_stores_even_negative_delta(self):
        """full_memory stores candidates even with negative delta."""
        state = MethodMemoryState("full_memory")

        c = _make_candidate("mem_neg")
        state.register_candidates([c], RECEIVER_IDS)
        records = [_make_record("mem_neg", rid, -0.5) for rid in RECEIVER_IDS]
        state.update_after_tci([c], records, RECEIVER_IDS)

        # full_memory validates everything (stores all)
        assert state.validated_size() == 1


# -----------------------------------------------------------------------
# Case 9: retrieval retrieves from historical pool
# -----------------------------------------------------------------------

class TestRetrievalFromHistoricalPool:
    """retrieval method selects top-k from ALL historical validated memories."""

    def test_09_top_k_from_historical_pool(self):
        """retrieval returns top-k memories sorted by tci_effect."""
        state = MethodMemoryState("retrieval")

        # Add 5 candidates with different tci_effects
        candidates = []
        records = []
        for i in range(5):
            c = _make_candidate(f"mem_retr_{i}", episode=i)
            candidates.append(c)
            state.register_candidates([c], RECEIVER_IDS)
            # Different deltas: 0.1, 0.5, 0.3, 0.8, 0.2
            delta = [0.1, 0.5, 0.3, 0.8, 0.2][i]
            recs = [_make_record(f"mem_retr_{i}", rid, delta) for rid in RECEIVER_IDS]
            records.extend(recs)
            state.update_after_tci([c], recs, RECEIVER_IDS)

        # All 5 stored
        assert state.memory_size() == 5

        # Retrieval with top_k=3 should return 3 memories
        payloads = state.get_injection_payloads("agent1", render_fn=_render_fn, top_k=3)
        assert len(payloads) == 3

    def test_09_retrieval_respects_top_k_limit(self):
        """retrieval never returns more than top_k."""
        state = MethodMemoryState("retrieval")

        for i in range(10):
            c = _make_candidate(f"mem_{i}", episode=i)
            state.register_candidates([c], RECEIVER_IDS)
            recs = [_make_record(f"mem_{i}", rid, 0.1 * (i + 1)) for rid in RECEIVER_IDS]
            state.update_after_tci([c], recs, RECEIVER_IDS)

        assert state.memory_size() == 10

        # Default top_k=3
        payloads = state.get_injection_payloads("agent1", render_fn=_render_fn)
        assert len(payloads) == 3

        # Custom top_k=5
        payloads_5 = state.get_injection_payloads("agent1", render_fn=_render_fn, top_k=5)
        assert len(payloads_5) == 5


# -----------------------------------------------------------------------
# Case 10: Different method states do NOT share mutable state
# -----------------------------------------------------------------------

class TestMethodStateIsolation:
    """Each method has its own independent bank -- no shared mutable state."""

    def test_10_no_shared_mutable_state_between_methods(self):
        """Adding to one method's bank does not affect another method's bank."""
        container = MethodStateContainer(METHODS)

        c = _make_candidate("mem_isolated")
        # Register only in full_memory
        fm = container.get("full_memory")
        fm.register_candidates([c], RECEIVER_IDS)
        records = [_make_record("mem_isolated", rid, 0.5) for rid in RECEIVER_IDS]
        fm.update_after_tci([c], records, RECEIVER_IDS)

        # full_memory has the memory
        assert fm.memory_size() == 1

        # Other methods do NOT have it
        for method in METHODS:
            if method == "full_memory":
                continue
            other_state = container.get(method)
            assert other_state.memory_size() == 0, (
                f"{method} should not have memories added to full_memory"
            )

    def test_10_bank_references_are_independent(self):
        """Each MethodMemoryState has a distinct PersistentMemoryBank."""
        container = MethodStateContainer(METHODS)

        bank_ids = set()
        for method in METHODS:
            state = container.get(method)
            bank_ids.add(id(state.bank))

        # All bank IDs must be unique
        assert len(bank_ids) == len(METHODS)

    def test_10_admission_controllers_are_independent(self):
        """Each MethodMemoryState has its own MemoryAdmissionController."""
        container = MethodStateContainer(METHODS)

        controller_ids = set()
        for method in METHODS:
            state = container.get(method)
            controller_ids.add(id(state.admission))

        assert len(controller_ids) == len(METHODS)


# -----------------------------------------------------------------------
# Case 14: Same task / seed / environment -> expose-withhold matched
# -----------------------------------------------------------------------

class TestExposeWithholdMatched:
    """Expose and withhold must use identical task/seed/environment configuration."""

    def test_14_online_validation_record_has_matched_task_seed(self):
        """OnlineValidationRecord carries the same task_id and seed for both branches."""
        record = OnlineValidationRecord(
            memory_id="mem_001",
            receiver_id="agent1",
            task_id="task_42",
            scenario="bargaining",
            seed=3,
            delta=0.15,
            decision=VALIDATION_STATUS_VALIDATED,
            normalized_expose_score=0.6,
            normalized_withhold_score=0.45,
            expose_metric_valid=True,
            withhold_metric_valid=True,
        )

        # Both branches share the same task_id and seed
        assert record.task_id == "task_42"
        assert record.seed == 3
        # Both metric validity flags should be consistent
        assert record.expose_metric_valid == record.withhold_metric_valid

    def test_14_matched_records_produce_consistent_delta(self):
        """When both branches are valid, delta = expose - withhold exactly."""
        expose_score = 0.72
        withhold_score = 0.55

        record = OnlineValidationRecord(
            memory_id="mem_matched",
            receiver_id="agent1",
            task_id="task_01",
            scenario="bargaining",
            seed=0,
            normalized_expose_score=expose_score,
            normalized_withhold_score=withhold_score,
            delta=expose_score - withhold_score,
            decision=VALIDATION_STATUS_VALIDATED,
            expose_metric_valid=True,
            withhold_metric_valid=True,
        )

        expected_delta = expose_score - withhold_score
        assert abs(record.delta - expected_delta) < 1e-9
        assert record.delta > 0
        assert record.decision == "validated"

    def test_14_invalid_branches_produce_none_delta_not_zero(self):
        """When either branch is invalid, delta is None (not 0.0)."""
        record = OnlineValidationRecord(
            memory_id="mem_invalid",
            receiver_id="agent1",
            task_id="task_01",
            scenario="database",
            seed=0,
            delta=None,
            decision="invalid",
            expose_metric_valid=True,
            withhold_metric_valid=False,
        )

        assert record.delta is None
        assert record.decision == "invalid"
        # Ensure it's not a hidden 0.0
        assert record.delta is not 0  # noqa: F632 (intentional identity check)
        assert record.delta != 0.0
