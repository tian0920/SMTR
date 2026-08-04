"""Receiver-episode risk-utility curve tests (第二轮清单第二章)."""

from __future__ import annotations

import pytest

from smtr.evaluation.metrics import (
    InconsistentControlOutcomeError,
    compute_receiver_episode_risk_utility_curve,
)


def _record(memory_id: str, *, share_success: bool, withhold_success: bool,
            task_id: str = "t1", receiver: str = "r1", seed: int = 0) -> dict:
    return {
        "task_id": task_id,
        "generation_seed": seed,
        "receiver_agent_id": receiver,
        "candidate_memory_id": memory_id,
        "share": {"team_success": share_success},
        "withhold": {"team_success": withhold_success},
    }


def _prediction(memory_id: str, tau_hat: float, eta_hat: float,
                task_id: str = "t1", receiver: str = "r1", seed: int = 0) -> dict:
    return {
        "task_id": task_id,
        "generation_seed": seed,
        "receiver_agent_id": receiver,
        "candidate_memory_id": memory_id,
        "tau_hat": tau_hat,
        "eta_hat": eta_hat,
    }


def test_risk_utility_counts_one_receiver_episode_once():
    """1 task, 1 receiver, 1 seed, 4 candidates, exactly 1 selected ->
    receiver_episode_count must be 1, not 4."""
    records = [
        _record("m1", share_success=True, withhold_success=False),
        _record("m2", share_success=False, withhold_success=False),
        _record("m3", share_success=False, withhold_success=False),
        _record("m4", share_success=False, withhold_success=False),
    ]
    predictions = [
        _prediction("m1", tau_hat=0.5, eta_hat=0.05),
        _prediction("m2", tau_hat=-0.3, eta_hat=0.4),
        _prediction("m3", tau_hat=0.2, eta_hat=0.6),  # risky: eta > epsilon
        _prediction("m4", tau_hat=-0.1, eta_hat=0.1),
    ]

    curve = compute_receiver_episode_risk_utility_curve(
        records, predictions, epsilons=(0.10,))

    assert len(curve) == 1
    point = curve[0]
    assert point["receiver_episode_count"] == 1
    assert point["share_coverage"] == 1.0
    assert point["candidate_share_rate"] == 0.25
    # Selected m1 share outcome succeeds.
    assert point["policy_success_rate"] == 1.0
    assert point["positive_transfer_recall"] == 1.0
    assert point["negative_transfer_exposure_rate"] == 0.0
    assert point["safe_exposure_precision"] == 1.0


def test_no_selected_memory_uses_one_common_withhold_outcome():
    """No eligible candidate -> the episode contributes the single common
    withhold outcome exactly once."""
    records = [
        _record("m1", share_success=True, withhold_success=True),
        _record("m2", share_success=True, withhold_success=True),
    ]
    predictions = [
        _prediction("m1", tau_hat=-0.2, eta_hat=0.05),
        _prediction("m2", tau_hat=-0.1, eta_hat=0.05),
    ]

    curve = compute_receiver_episode_risk_utility_curve(
        records, predictions, epsilons=(0.10,))

    point = curve[0]
    assert point["receiver_episode_count"] == 1
    assert point["share_coverage"] == 0.0
    assert point["policy_success_rate"] == 1.0  # common withhold outcome
    assert point["candidate_share_rate"] == 0.0


def test_inconsistent_withhold_outcomes_raise():
    """Conflicting Y_0 outcomes within one episode must fail fast."""
    records = [
        _record("m1", share_success=False, withhold_success=True),
        _record("m2", share_success=False, withhold_success=False),
    ]
    predictions = [
        _prediction("m1", tau_hat=-0.2, eta_hat=0.05),
        _prediction("m2", tau_hat=-0.1, eta_hat=0.05),
    ]

    with pytest.raises(InconsistentControlOutcomeError):
        compute_receiver_episode_risk_utility_curve(
            records, predictions, epsilons=(0.10,))
