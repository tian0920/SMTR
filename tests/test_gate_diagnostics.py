"""Tests for gate diagnostic funnels."""

from smtr.evaluation.gate_diagnostics import compute_gate_diagnostics
from smtr.experiment.schemas import ComparisonRunRecord, DecisionRecord, RoutingInvocationRecord


def _run(decisions, *, success=True, method="SMTR", base_id="base"):
    return ComparisonRunRecord(
        experiment_id="exp",
        base_episode_id=base_id,
        episode_id=base_id,
        task_instance_id=base_id,
        method=method,
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
                candidate_memory_ids=[decision.memory_id for decision in decisions],
                candidate_scores=[1.0 for _ in decisions],
                proposal_order=[decision.memory_id for decision in decisions],
                traversal_order=[decision.memory_id for decision in decisions],
                decisions=decisions,
            )
        ],
    )


def _decision(memory_id, transfer_class, *, action="share", tau_mean=0.5, tau_lcb=0.2):
    return DecisionRecord(
        decision_index=0,
        memory_id=memory_id,
        action=action,
        reason="accepted" if action == "share" else "tau_lcb_nonpositive",
        traversal_position=0,
        selected_before_digest="empty",
        tau_mean=tau_mean,
        tau_lcb=tau_lcb,
        negative_risk_mean=0.1,
        negative_risk_ucb=0.15,
        true_transfer_class=transfer_class,
        true_tau=1.0 if transfer_class == "positive" else -1.0,
    )


def test_gate_diagnostic_counts_and_shuffle_invariance():
    runs = [
        _run([_decision("p", "positive", action="share")], success=True),
        _run(
            [_decision("n", "negative", action="withhold", tau_mean=-0.2, tau_lcb=-0.3)],
            success=True,
            base_id="base2",
        ),
    ]
    result = compute_gate_diagnostics(runs, epsilon=0.2)
    shuffled = compute_gate_diagnostics(list(reversed(runs)), epsilon=0.2)
    assert result == shuffled
    funnel = result["SMTR"]
    assert funnel.positive_opportunity_count == 1
    assert funnel.shared_count == 1
    assert funnel.task_success_count == 1
    assert funnel.negative_opportunity_count == 1
    assert funnel.withheld_count == 1
    assert funnel.task_preserved_count == 1


def test_gate_diagnostic_skips_unknown_truth():
    unknown = _decision("u", None)
    result = compute_gate_diagnostics([_run([unknown])], epsilon=0.2)
    assert result["SMTR"].positive_opportunity_count == 0
    assert result["SMTR"].negative_opportunity_count == 0
