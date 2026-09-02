"""Tests for ReceiverTransferState (RIMA-v2 §2-3, tests in §40).

Covers:
    test_register_memory_is_idempotent
    test_transfer_state_does_not_store_payload
    test_negative_memory_remains_in_transfer_state
    test_same_memory_can_be_negative_then_positive_across_tasks
    test_transfer_state_is_receiver_specific
    test_self_transfer_cannot_be_registered
    test_prediction_history_is_task_conditioned
"""

from __future__ import annotations

import pytest

from smtr.rima.transfer_state import (
    ReceiverTransferState,
    ReceiverTransferStateContainer,
    TransferPredictionRecord,
    TransferStateEntry,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state(receiver_id: str = "r1") -> ReceiverTransferState:
    return ReceiverTransferState(receiver_id)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_register_memory_is_idempotent():
    """Registering the same memory twice must return the same entry."""
    state = _make_state()
    entry1 = state.register_memory("m1", "src1", "t1", 0)
    entry2 = state.register_memory("m1", "src1", "t2", 1)
    assert entry1 is entry2
    assert entry1.first_seen_task_id == "t1"
    assert entry1.first_seen_task_position == 0
    assert len(state) == 1


def test_transfer_state_does_not_store_payload():
    """TransferStateEntry must not contain any payload or SharedMemory."""
    state = _make_state()
    entry = state.register_memory("m1", "src1", "t1", 0)
    # Entry only has metadata fields — no payload, no SharedMemory
    attrs = set(vars(entry).keys())
    forbidden = {"payload", "procedure_payload", "shared_memory", "memory"}
    assert not (attrs & forbidden), f"Found forbidden payload fields: {attrs & forbidden}"


def test_negative_memory_remains_in_transfer_state():
    """Memory with LCB < 0 must stay in transfer state."""
    state = _make_state()
    state.register_memory("m1", "src1", "t1", 0)

    # Record a negative prediction (LCB < 0)
    state.record_prediction(
        "m1", "t1", 0,
        mu_tau=-0.3, sigma_tau=0.1, lcb=-0.464,
        status="negative", candidate_source="global",
    )

    # Memory must still be in the state
    assert state.contains("m1")
    entry = state.get_entry("m1")
    assert entry is not None
    assert entry.last_lcb == pytest.approx(-0.464)


def test_same_memory_can_be_negative_then_positive_across_tasks():
    """A memory negative on task 1 must be selectable on task 2 if LCB > 0."""
    state = _make_state()
    state.register_memory("m1", "src1", "t1", 0)

    # Task 1: LCB < 0 (negative)
    state.record_prediction(
        "m1", "t1", 0,
        mu_tau=-0.2, sigma_tau=0.1, lcb=-0.364,
        status="negative", candidate_source="global",
    )
    assert state.get_entry("m1").last_lcb < 0

    # Task 2: LCB > 0 (positive) — memory must be reconsiderable
    state.record_prediction(
        "m1", "t2", 1,
        mu_tau=0.5, sigma_tau=0.1, lcb=0.336,
        status="positive", candidate_source="known",
    )
    entry = state.get_entry("m1")
    assert entry is not None
    assert entry.last_lcb > 0
    assert entry.last_task_id == "t2"
    assert entry.times_considered == 2
    assert len(entry.prediction_history) == 2


def test_transfer_state_is_receiver_specific():
    """Different receivers must have independent transfer states."""
    container = ReceiverTransferStateContainer()
    state_r1 = container.ensure("r1")
    state_r2 = container.ensure("r2")

    state_r1.register_memory("m1", "src1", "t1", 0)
    state_r1.record_prediction(
        "m1", "t1", 0,
        mu_tau=0.5, sigma_tau=0.1, lcb=0.336,
        status="positive", candidate_source="global",
    )

    # r2 must not know about m1
    assert not state_r2.contains("m1")
    assert state_r1.contains("m1")

    # Register same memory for r2 with different prediction
    state_r2.register_memory("m1", "src1", "t1", 0)
    state_r2.record_prediction(
        "m1", "t1", 0,
        mu_tau=-0.1, sigma_tau=0.2, lcb=-0.428,
        status="negative", candidate_source="global",
    )

    assert state_r1.get_entry("m1").last_lcb > 0
    assert state_r2.get_entry("m1").last_lcb < 0


def test_self_transfer_cannot_be_registered():
    """source_agent_id == receiver_id must raise ValueError."""
    state = ReceiverTransferState("agent_a")
    with pytest.raises(ValueError, match="Self-transfer"):
        state.register_memory("m1", "agent_a", "t1", 0)


def test_prediction_history_is_task_conditioned():
    """Each prediction record must carry its own task_id and task_position."""
    state = _make_state()
    state.register_memory("m1", "src1", "t1", 0)

    state.record_prediction(
        "m1", "t1", 0,
        mu_tau=0.3, sigma_tau=0.1, lcb=0.136,
        status="positive", candidate_source="global",
    )
    state.record_prediction(
        "m1", "t2", 1,
        mu_tau=0.4, sigma_tau=0.15, lcb=0.154,
        status="positive", candidate_source="known",
    )
    state.record_prediction(
        "m1", "t3", 2,
        mu_tau=-0.1, sigma_tau=0.2, lcb=-0.428,
        status="negative", candidate_source="known",
    )

    entry = state.get_entry("m1")
    assert entry is not None
    assert len(entry.prediction_history) == 3

    # Each record must have its own task context
    assert entry.prediction_history[0].task_id == "t1"
    assert entry.prediction_history[0].task_position == 0
    assert entry.prediction_history[1].task_id == "t2"
    assert entry.prediction_history[1].task_position == 1
    assert entry.prediction_history[2].task_id == "t3"
    assert entry.prediction_history[2].task_position == 2


# ---------------------------------------------------------------------------
# Additional tests
# ---------------------------------------------------------------------------


def test_record_prediction_unknown_memory_raises():
    """Recording prediction for unregistered memory must raise KeyError."""
    state = _make_state()
    with pytest.raises(KeyError, match="not in transfer state"):
        state.record_prediction(
            "m_unknown", "t1", 0,
            mu_tau=0.5, sigma_tau=0.1, lcb=0.336,
            status="positive", candidate_source="global",
        )


def test_mark_selected_increments_counter():
    state = _make_state()
    state.register_memory("m1", "src1", "t1", 0)
    state.mark_selected("m1")
    state.mark_selected("m1")
    assert state.get_entry("m1").times_selected == 2


def test_known_memory_ids():
    state = _make_state()
    state.register_memory("m1", "src1", "t1", 0)
    state.register_memory("m2", "src2", "t1", 0)
    assert state.known_memory_ids() == {"m1", "m2"}


def test_to_dict_serialization():
    state = _make_state()
    state.register_memory("m1", "src1", "t1", 0)
    state.record_prediction(
        "m1", "t1", 0,
        mu_tau=0.3, sigma_tau=0.1, lcb=0.136,
        status="positive", candidate_source="global",
    )
    d = state.to_dict()
    assert d["receiver_id"] == "r1"
    assert d["n_entries"] == 1
    assert "m1" in d["entries"]
    assert d["entries"]["m1"]["times_considered"] == 1


def test_container_ensure_creates_once():
    container = ReceiverTransferStateContainer()
    s1 = container.ensure("r1")
    s2 = container.ensure("r1")
    assert s1 is s2
    assert container.receiver_ids() == {"r1"}


def test_container_get_returns_none_for_unknown():
    container = ReceiverTransferStateContainer()
    assert container.get("unknown") is None
