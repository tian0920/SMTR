"""Tests for current ablation analysis modules."""

import pytest

from smtr.evaluation.prefix_intervention_audit import (
    PrefixInterventionRecord,
    audit_prefix_interventions,
)
from smtr.experiment.bottleneck_funnel import compute_bottleneck_funnel
from smtr.experiment.schemas import ComparisonRunRecord, DecisionRecord, RoutingInvocationRecord


def _run(*, decisions, candidates=None, visible=None, success=True):
    candidates = candidates or [decision.memory_id for decision in decisions]
    visible = visible or [
        decision.memory_id for decision in decisions if decision.action == "share"
    ]
    return ComparisonRunRecord(
        experiment_id="exp",
        base_episode_id="base",
        episode_id="base",
        task_instance_id="base",
        method="M0-Full",
        router_name="ProductionSequentialRouter",
        task_seed=0,
        environment_seed=0,
        generation_seed=0,
        memory_snapshot_id="snap",
        environment_snapshot_digest="env",
        team_success=success,
        invocations=[
            RoutingInvocationRecord(
                invocation_id="inv",
                graph_node="pre_route_planner",
                receiver_agent_id="planner",
                receiver_role="planner",
                context_fingerprint_digest="ctx",
                candidate_request_digest="req",
                candidate_memory_ids=candidates,
                candidate_scores=[1.0 for _ in candidates],
                proposal_order=candidates,
                traversal_order=candidates,
                decisions=decisions,
                visible_payload_memory_ids=visible,
            )
        ],
    )


def _decision(memory_id, action="share", position=0):
    return DecisionRecord(
        decision_index=position,
        memory_id=memory_id,
        action=action,
        reason="accepted" if action == "share" else "tau_lcb_nonpositive",
        traversal_position=position,
        selected_before_digest="empty",
    )


def test_bottleneck_funnel_monotone_and_shuffle_invariant():
    runs = [
        _run(decisions=[_decision("target")], success=True),
        _run(decisions=[_decision("other")], candidates=["other"], success=True),
        _run(decisions=[_decision("target", action="withhold")], success=False),
    ]
    result = compute_bottleneck_funnel(runs, target_memory_id="target")
    shuffled = compute_bottleneck_funnel(list(reversed(runs)), target_memory_id="target")
    assert result == shuffled
    values = list(result.values())
    assert all(left >= right for left, right in zip(values, values[1:], strict=False))


def test_prefix_intervention_requires_four_branches():
    records = [
        PrefixInterventionRecord(
            prefix_intervention_group_id="g",
            base_decision_digest="d",
            receiver_agent_id="planner",
            target_memory_id="target",
            prefix_variant_id="S0",
            target_action="share",
            outcome_success=True,
            m0_pred_tau=0.2,
            a1_pred_tau=0.1,
        )
    ]
    with pytest.raises(ValueError, match="lacks four branches"):
        audit_prefix_interventions(records)


def test_prefix_intervention_delta_tau():
    records = []
    outcomes = {
        ("S0", "share"): True,
        ("S0", "withhold"): False,
        ("S1", "share"): False,
        ("S1", "withhold"): False,
    }
    preds = {"S0": 0.9, "S1": -0.1}
    for (variant, action), outcome in outcomes.items():
        records.append(
            PrefixInterventionRecord(
                prefix_intervention_group_id="g",
                base_decision_digest="d",
                receiver_agent_id="planner",
                target_memory_id="target",
                prefix_variant_id=variant,
                target_action=action,
                outcome_success=outcome,
                m0_pred_tau=preds[variant],
                a1_pred_tau=preds[variant],
            )
        )
    result = audit_prefix_interventions(records)
    assert result.n_groups == 1
    assert result.delta_tau_mae == 0.0
    assert result.direction_accuracy == 1.0
