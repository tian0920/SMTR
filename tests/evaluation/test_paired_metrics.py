"""Tests for paired evaluation metrics."""

from __future__ import annotations

from smtr.evaluation.metrics import compute_method_metrics


def _make_decisions():
    return [
        {"task_id": "t1", "generation_seed": 0, "candidate_memory_id": "m1",
         "receiver_agent_id": "r1", "receiver_role": "executor", "writer_role": "planner",
         "action": "share", "tau_hat": 0.3, "eta_hat": 0.1},
        {"task_id": "t1", "generation_seed": 0, "candidate_memory_id": "m2",
         "receiver_agent_id": "r1", "receiver_role": "executor", "writer_role": "executor",
         "action": "withhold", "tau_hat": -0.1, "eta_hat": 0.4},
        {"task_id": "t2", "generation_seed": 0, "candidate_memory_id": "m1",
         "receiver_agent_id": "r2", "receiver_role": "critic", "writer_role": "planner",
         "action": "withhold", "tau_hat": 0.1, "eta_hat": 0.3},
    ]


def _make_paired_outcomes():
    return [
        {"task_id": "t1", "generation_seed": 0, "candidate_memory_id": "m1",
         "receiver_agent_id": "r1", "label": "positive_transfer",
         "share": {"team_success": True}, "withhold": {"team_success": False}},
        {"task_id": "t1", "generation_seed": 0, "candidate_memory_id": "m2",
         "receiver_agent_id": "r1", "label": "negative_transfer",
         "share": {"team_success": False}, "withhold": {"team_success": True}},
        {"task_id": "t2", "generation_seed": 0, "candidate_memory_id": "m1",
         "receiver_agent_id": "r2", "label": "negative_transfer",
         "share": {"team_success": False}, "withhold": {"team_success": True}},
    ]


def test_policy_success_uses_correct_potential_outcome():
    """Policy success must use the potential outcome matching the action."""
    metrics = compute_method_metrics(
        method="test",
        decisions=_make_decisions(),
        paired_outcomes=_make_paired_outcomes(),
        negative_risk_budget=0.2,
    )
    # m1/r1: share -> share.team_success=True -> success
    # m2/r1: withhold -> withhold.team_success=True -> success
    # m1/r2: withhold -> withhold.team_success=True -> success
    assert metrics["paired_policy_success_rate"] == 1.0


def test_pair_key_includes_task_seed_receiver_memory():
    """Pair key must include task_id, generation_seed, receiver, memory."""
    # Same memory, different task -> different outcomes
    decisions = [
        {"task_id": "t1", "generation_seed": 0, "candidate_memory_id": "m1",
         "receiver_agent_id": "r1", "receiver_role": "executor", "writer_role": "planner",
         "action": "share", "tau_hat": 0.3, "eta_hat": 0.1},
        {"task_id": "t2", "generation_seed": 0, "candidate_memory_id": "m1",
         "receiver_agent_id": "r1", "receiver_role": "executor", "writer_role": "planner",
         "action": "share", "tau_hat": 0.3, "eta_hat": 0.1},
    ]
    outcomes = [
        {"task_id": "t1", "generation_seed": 0, "candidate_memory_id": "m1",
         "receiver_agent_id": "r1", "label": "positive_transfer",
         "share": {"team_success": True}, "withhold": {"team_success": False}},
        {"task_id": "t2", "generation_seed": 0, "candidate_memory_id": "m1",
         "receiver_agent_id": "r1", "label": "negative_transfer",
         "share": {"team_success": False}, "withhold": {"team_success": True}},
    ]
    metrics = compute_method_metrics(
        method="test", decisions=decisions, paired_outcomes=outcomes, negative_risk_budget=0.2,
    )
    # One positive, one negative -> policy success = 0.5
    assert metrics["paired_policy_success_rate"] == 0.5


def test_same_memory_flip_requires_different_receiver():
    """Same-memory flip must have different receivers."""
    decisions = [
        {"task_id": "t1", "generation_seed": 0, "candidate_memory_id": "m1",
         "receiver_agent_id": "r1", "receiver_role": "executor", "writer_role": "planner",
         "action": "share", "tau_hat": 0.3, "eta_hat": 0.1},
        {"task_id": "t2", "generation_seed": 0, "candidate_memory_id": "m1",
         "receiver_agent_id": "r2", "receiver_role": "critic", "writer_role": "planner",
         "action": "withhold", "tau_hat": -0.1, "eta_hat": 0.3},
    ]
    metrics = compute_method_metrics(
        method="test", decisions=decisions, paired_outcomes=[], negative_risk_budget=0.2,
    )
    assert metrics["same_memory_different_receiver_flip_count"] == 1


def test_risk_budget_not_hardcoded():
    """Quarantine threshold must use the negative_risk_budget parameter."""
    decisions = [
        {"task_id": "t1", "generation_seed": 0, "candidate_memory_id": "m1",
         "receiver_agent_id": "r1", "receiver_role": "executor", "writer_role": "planner",
         "action": "withhold", "tau_hat": 0.1, "eta_hat": 0.25},
    ]
    # With budget=0.2, eta=0.25 > 0.2 -> quarantine
    m1 = compute_method_metrics(method="t", decisions=decisions, paired_outcomes=[], negative_risk_budget=0.2)
    assert m1["receiver_specific_quarantine_pair_count"] == 1

    # With budget=0.3, eta=0.25 < 0.3 -> no quarantine
    m2 = compute_method_metrics(method="t", decisions=decisions, paired_outcomes=[], negative_risk_budget=0.3)
    assert m2["receiver_specific_quarantine_pair_count"] == 0


def test_negative_transfer_rejection():
    """Negative transfer rejection rate must be computed correctly."""
    decisions = [
        {"task_id": "t1", "generation_seed": 0, "candidate_memory_id": "m1",
         "receiver_agent_id": "r1", "receiver_role": "executor", "writer_role": "planner",
         "action": "withhold", "tau_hat": -0.1, "eta_hat": 0.4},
    ]
    outcomes = [
        {"task_id": "t1", "generation_seed": 0, "candidate_memory_id": "m1",
         "receiver_agent_id": "r1", "label": "negative_transfer",
         "share": {"team_success": False}, "withhold": {"team_success": True}},
    ]
    metrics = compute_method_metrics(method="t", decisions=decisions, paired_outcomes=outcomes, negative_risk_budget=0.2)
    assert metrics["negative_transfer_rejection_rate"] == 1.0
