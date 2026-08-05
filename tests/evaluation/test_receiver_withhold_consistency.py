"""清单 Test 7: receiver withhold consistency (P0-14).

All candidates of one task/receiver/seed must share the identical
no-memory (withhold) outcome; conflicting values fail fast.
"""

from __future__ import annotations

import pytest

from smtr.evaluation.metrics import (
    InconsistentControlOutcomeError,
    check_receiver_withhold_consistency,
    compute_receiver_policy_metrics,
)


def _record(memory_id: str, seed: int, withhold_success: bool) -> dict:
    return {
        "task_id": "t1",
        "receiver_agent_id": "r1",
        "candidate_memory_id": memory_id,
        "generation_seed": seed,
        "valid": True,
        "label": "neutral_success" if withhold_success else "positive_transfer",
        "y_share": 1,
        "y_withhold": int(withhold_success),
        "share": {"team_success": True},
        "withhold": {"team_success": withhold_success},
    }


def test_identical_withhold_outcomes_are_legal():
    records = [
        _record("m1", 0, withhold_success=True),
        _record("m2", 0, withhold_success=True),
        _record("m3", 0, withhold_success=True),
    ]
    check_receiver_withhold_consistency(records)  # must not raise


def test_conflicting_withhold_outcomes_raise():
    records = [
        _record("m1", 0, withhold_success=True),
        _record("m2", 0, withhold_success=False),
    ]
    with pytest.raises(
        ValueError,
        match="inconsistent no-memory outcome across candidates",
    ):
        check_receiver_withhold_consistency(records)


def test_policy_metrics_raise_the_same_error():
    """compute_receiver_policy_metrics enforces the identical invariant."""
    records = [
        _record("m1", 0, withhold_success=True),
        _record("m2", 0, withhold_success=False),
    ]
    decisions = [
        {"task_id": "t1", "generation_seed": 0, "receiver_agent_id": "r1",
         "candidate_memory_id": "m1", "action": "withhold"},
        {"task_id": "t1", "generation_seed": 0, "receiver_agent_id": "r1",
         "candidate_memory_id": "m2", "action": "withhold"},
    ]
    with pytest.raises(InconsistentControlOutcomeError):
        compute_receiver_policy_metrics(
            method="test", decisions=decisions, paired_outcomes=records,
        )
