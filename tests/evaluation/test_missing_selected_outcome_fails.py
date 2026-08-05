"""清单 Test 6: missing selected outcome must fail fast.

When the router selects a memory that has no paired record for the current
task/receiver/seed, the policy metrics must raise instead of silently
skipping the episode and lowering ``policy_total``.
"""

from __future__ import annotations

import pytest

from smtr.evaluation.metrics import (
    compute_candidate_transfer_metrics,
    compute_receiver_policy_metrics,
)


def _record(memory_id: str, seed: int, label: str, y_share: int, y_withhold: int) -> dict:
    return {
        "task_id": "t1",
        "generation_seed": seed,
        "receiver_agent_id": "r1",
        "candidate_memory_id": memory_id,
        "valid": True,
        "label": label,
        "share": {"team_success": bool(y_share)},
        "withhold": {"team_success": bool(y_withhold)},
    }


def _decision(memory_id: str, seed: int, action: str) -> dict:
    return {
        "task_id": "t1",
        "generation_seed": seed,
        "receiver_agent_id": "r1",
        "candidate_memory_id": memory_id,
        "action": action,
    }


def test_selected_memory_without_record_raises():
    """Router selects memB but memB has no paired record for seed 0."""
    decisions = [
        _decision("memA", 0, "withhold"),
        _decision("memB", 0, "share"),
    ]
    paired_outcomes = [_record("memA", 0, "neutral_success", 1, 1)]
    with pytest.raises(ValueError, match="no paired outcome for receiver policy replay"):
        compute_receiver_policy_metrics(
            method="smtr", decisions=decisions, paired_outcomes=paired_outcomes
        )


def test_candidate_decision_without_record_raises():
    decisions = [_decision("memB", 0, "share")]
    paired_outcomes = [_record("memA", 0, "neutral_success", 1, 1)]
    with pytest.raises(ValueError, match="no matching core-valid paired record"):
        compute_candidate_transfer_metrics(
            method="smtr", decisions=decisions, paired_outcomes=paired_outcomes
        )


def test_missing_withhold_outcome_raises():
    """No-memory episode whose candidates all lack records must raise."""
    decisions = [
        _decision("memA", 0, "withhold"),
        _decision("memB", 0, "withhold"),
    ]
    with pytest.raises(ValueError, match="no withhold outcome"):
        compute_receiver_policy_metrics(
            method="b0_no_memory", decisions=decisions, paired_outcomes=[]
        )
