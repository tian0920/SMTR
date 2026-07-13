"""Tests for current rejection reason mapping."""

import pytest

from smtr.experiment.schemas import (
    ComparisonRunRecord,
    DecisionRecord,
    ExperimentConfig,
    RoutingInvocationRecord,
)
from smtr.experiment.summary import CANONICAL_REASONS, canonicalize_reason, compute_summary


def test_removed_no_critic_reasons_are_not_formal_categories():
    assert "no_critic_available" not in CANONICAL_REASONS
    assert canonicalize_reason("no_critic_available") == "other"
    assert canonicalize_reason("no_critic_estimate") == "other"


@pytest.mark.parametrize(
    ("raw", "canonical"),
    [
        ("accepted", "shared"),
        ("tau_lcb_nonpositive", "tau_lcb_nonpositive"),
        ("negative_risk_ucb_exceeds_epsilon", "negative_risk_ucb_exceeded"),
        ("budget_exhausted", "share_budget_exceeded"),
        ("low_support", "low_support"),
        ("missing_routing_card", "missing_routing_card"),
    ],
)
def test_reason_mapping(raw, canonical):
    assert canonicalize_reason(raw) == canonical


def test_rejection_metrics_sum_to_one(tmp_path):
    run = ComparisonRunRecord(
        experiment_id="exp",
        base_episode_id="base",
        episode_id="base",
        task_instance_id="base",
        method="SMTR",
        router_name="ProductionSequentialRouter",
        task_seed=0,
        environment_seed=0,
        generation_seed=0,
        memory_snapshot_id="snap",
        environment_snapshot_digest="env",
        invocations=[
            RoutingInvocationRecord(
                invocation_id="inv",
                graph_node="pre_route_planner",
                receiver_agent_id="planner",
                receiver_role="planner",
                context_fingerprint_digest="ctx",
                candidate_request_digest="req",
                candidate_memory_ids=["m1", "m2", "m3", "m4"],
                candidate_scores=[1.0, 0.9, 0.8, 0.7],
                proposal_order=["m1", "m2", "m3", "m4"],
                traversal_order=["m1", "m2", "m3", "m4"],
                decisions=[
                    DecisionRecord(
                        decision_index=0,
                        memory_id="m1",
                        action="share",
                            reason="shared",
                        traversal_position=0,
                        selected_before_digest="empty",
                    ),
                    DecisionRecord(
                        decision_index=1,
                        memory_id="m2",
                        action="withhold",
                            reason="tau_mean_nonpositive",
                        traversal_position=1,
                        selected_before_digest="x",
                    ),
                    DecisionRecord(
                        decision_index=2,
                        memory_id="m3",
                        action="withhold",
                            reason="negative_risk_mean_exceeded",
                        traversal_position=2,
                        selected_before_digest="x",
                    ),
                    DecisionRecord(
                        decision_index=3,
                        memory_id="m4",
                        action="withhold",
                            reason="share_budget_exceeded",
                        traversal_position=3,
                        selected_before_digest="x",
                    ),
                ],
                selected_memory_ids=["m1"],
                visible_payload_memory_ids=["m1"],
            )
        ],
    )
    summary = compute_summary(
        [run],
        ExperimentConfig(db_path=":memory:", output_dir=str(tmp_path)),
    )
    method = summary.methods["SMTR"]
    total = (
        (method.share_decision_rate or 0.0)
        + (method.tau_mean_rejection_rate or 0.0)
        + (method.negative_risk_mean_rejection_rate or 0.0)
        + (method.share_budget_rejection_rate or 0.0)
        + (method.low_support_rejection_rate or 0.0)
    )
    assert total == pytest.approx(1.0)
