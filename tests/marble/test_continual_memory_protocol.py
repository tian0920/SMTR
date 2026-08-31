"""P0-5 Integrity Suite — Cases 11-13: Continual memory protocol.

Invariants tested (causal pre-update protocol):
  11. task_t candidate MUST NOT be used for task_t formal evaluation.
  12. task_t candidate IS allowed to be reused from task_{t+1} onwards.
  13. Scenario boundary -> reset all method states.
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
    """Simple render function for tests."""
    return entry.content


# -----------------------------------------------------------------------
# Case 11: task_t candidate MUST NOT affect task_t evaluation
# -----------------------------------------------------------------------

class TestTaskCandidateNotUsedForCurrentEvaluation:
    """The continual protocol requires evaluation BEFORE discovery/TCI/update."""

    def test_11_evaluation_uses_only_historical_memory(self):
        """Before TCI update, method state contains only historical memories."""
        container = MethodStateContainer(METHODS)
        state = container.get("full_memory")

        # Task 0: register and validate a candidate
        c0 = _make_candidate("mem_task0", episode=0)
        state.register_candidates([c0], RECEIVER_IDS)
        records_0 = [_make_record("mem_task0", rid, delta=0.3) for rid in RECEIVER_IDS]
        state.update_after_tci([c0], records_0, RECEIVER_IDS)

        # Verify: after task 0 update, memory is available
        payloads_after_t0 = state.get_injection_payloads("agent1", render_fn=_render_fn)
        assert len(payloads_after_t0) == 1

        # Task 1: STEP 1 — Evaluate using K_{t-1} (memory from task 0 only)
        payloads_before_t1_discovery = state.get_injection_payloads(
            "agent1", render_fn=_render_fn
        )
        assert len(payloads_before_t1_discovery) == 1  # only task 0 memory

        # Task 1: STEP 2 — Discover new candidate
        c1 = _make_candidate("mem_task1", episode=1)
        state.register_candidates([c1], RECEIVER_IDS)

        # CRITICAL: After registration but BEFORE TCI update,
        # the new candidate is in the bank but NOT validated yet.
        # get_injection_payloads for full_memory returns validated only.
        payloads_during_t1 = state.get_injection_payloads(
            "agent1", render_fn=_render_fn
        )
        # mem_task1 is candidate (not validated) -> only mem_task0 returned
        assert len(payloads_during_t1) == 1

    def test_11_no_memory_never_injects(self):
        """no_memory method must always return empty payloads."""
        state = MethodMemoryState("no_memory")

        c = _make_candidate("mem_001")
        state.register_candidates([c], RECEIVER_IDS)
        records = [_make_record("mem_001", rid, 0.5) for rid in RECEIVER_IDS]
        state.update_after_tci([c], records, RECEIVER_IDS)

        payloads = state.get_injection_payloads("agent1", render_fn=_render_fn)
        assert payloads == []
        # no_memory never validates; candidate is registered but not validated
        assert state.validated_size() == 0


# -----------------------------------------------------------------------
# Case 12: task_t candidate reusable from task_{t+1}
# -----------------------------------------------------------------------

class TestTaskCandidateReusableFromNextTask:
    """After TCI update at task_t, the validated memory is available for task_{t+1}."""

    def test_12_validated_memory_available_for_next_task(self):
        """After TCI validation at task_t, memory is injected at task_{t+1}."""
        state = MethodMemoryState("full_memory")

        # Task 0: discover, validate
        c0 = _make_candidate("mem_task0", episode=0)
        state.register_candidates([c0], RECEIVER_IDS)
        records = [_make_record("mem_task0", rid, 0.3) for rid in RECEIVER_IDS]
        state.update_after_tci([c0], records, RECEIVER_IDS)

        # Task 1: evaluation step — memory from task 0 is available
        payloads = state.get_injection_payloads("agent1", render_fn=_render_fn)
        assert len(payloads) == 1
        assert "test" in payloads[0]  # content is "test"

    def test_12_smtr_receiver_validated_across_tasks(self):
        """smtr_receiver: validated memory at task_t available at task_{t+1}."""
        state = MethodMemoryState("smtr_receiver")

        c0 = _make_candidate("mem_t0", episode=0)
        state.register_candidates([c0], RECEIVER_IDS)

        # agent1: delta > 0 -> validated; agent2: delta < 0 -> rejected
        records = [
            _make_record("mem_t0", "agent1", 0.4),
            _make_record("mem_t0", "agent2", -0.3),
            _make_record("mem_t0", "agent3", 0.2),
        ]
        state.update_after_tci([c0], records, RECEIVER_IDS)

        # Task 1: agent1 gets the memory, agent2 does not
        payloads_r1 = state.get_injection_payloads("agent1", render_fn=_render_fn)
        payloads_r2 = state.get_injection_payloads("agent2", render_fn=_render_fn)

        assert len(payloads_r1) == 1
        assert len(payloads_r2) == 0

    def test_12_accumulation_across_multiple_tasks(self):
        """Memories from multiple past tasks accumulate for future evaluation."""
        state = MethodMemoryState("full_memory")

        # Task 0
        c0 = _make_candidate("mem_t0", episode=0)
        state.register_candidates([c0], RECEIVER_IDS)
        records_0 = [_make_record("mem_t0", rid, 0.3) for rid in RECEIVER_IDS]
        state.update_after_tci([c0], records_0, RECEIVER_IDS)

        # Task 1
        c1 = _make_candidate("mem_t1", episode=1)
        state.register_candidates([c1], RECEIVER_IDS)
        records_1 = [_make_record("mem_t1", rid, 0.2) for rid in RECEIVER_IDS]
        state.update_after_tci([c1], records_1, RECEIVER_IDS)

        # Task 2 evaluation: both memories available
        payloads = state.get_injection_payloads("agent1", render_fn=_render_fn)
        assert len(payloads) == 2


# -----------------------------------------------------------------------
# Case 13: Scenario boundary reset
# -----------------------------------------------------------------------

class TestScenarioBoundaryReset:
    """All method states MUST reset when the scenario changes."""

    def test_13_reset_clears_all_method_banks(self):
        """MethodStateContainer.reset_all() clears every method bank."""
        container = MethodStateContainer(METHODS)

        # Populate with task 0 memories
        c = _make_candidate("mem_bargaining_001")
        container.register_candidates_all([c], RECEIVER_IDS)

        records = [_make_record("mem_bargaining_001", rid, 0.5) for rid in RECEIVER_IDS]
        for method in METHODS:
            container.get(method).update_after_tci([c], records, RECEIVER_IDS)

        # Verify: at least some methods have memories
        assert container.get("full_memory").memory_size() > 0
        assert container.get("smtr_uniform").validated_size() > 0

        # Scenario boundary: reset
        container.reset_all()

        # Verify: ALL methods are empty
        for method in METHODS:
            state = container.get(method)
            assert state.memory_size() == 0, f"{method} should be empty after reset"
            payloads = state.get_injection_payloads("agent1", render_fn=_render_fn)
            assert payloads == [], f"{method} should have no payloads after reset"

    def test_13_reset_preserves_method_identity(self):
        """After reset, each method still has its own isolated bank."""
        container = MethodStateContainer(METHODS)

        container.reset_all()

        # Add to full_memory only
        c = _make_candidate("mem_post_reset")
        fm_state = container.get("full_memory")
        fm_state.register_candidates([c], RECEIVER_IDS)
        records = [_make_record("mem_post_reset", rid, 0.3) for rid in RECEIVER_IDS]
        fm_state.update_after_tci([c], records, RECEIVER_IDS)

        # full_memory has it, others don't
        assert fm_state.memory_size() == 1
        assert container.get("smtr_uniform").memory_size() == 0
        assert container.get("no_memory").memory_size() == 0

    def test_13_reset_creates_fresh_bank_instances(self):
        """reset() creates new PersistentMemoryBank instances (not stale refs)."""
        state = MethodMemoryState("full_memory")

        c = _make_candidate("mem_before_reset")
        state.register_candidates([c], RECEIVER_IDS)
        records = [_make_record("mem_before_reset", rid, 0.5) for rid in RECEIVER_IDS]
        state.update_after_tci([c], records, RECEIVER_IDS)

        assert state.memory_size() == 1

        old_bank_id = id(state.bank)
        state.reset()

        # New bank instance
        assert id(state.bank) != old_bank_id
        assert state.memory_size() == 0
