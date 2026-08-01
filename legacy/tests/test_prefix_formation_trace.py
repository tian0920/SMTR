"""Tests for invocation-local prefix traces."""

from smtr.experiment.prefix_trace import compute_prefix_traces
from smtr.experiment.schemas import ComparisonRunRecord, DecisionRecord, RoutingInvocationRecord


def _run(invocations):
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
        invocations=invocations,
    )


def test_prefix_trace_uses_same_invocation_selected_before():
    invocation = RoutingInvocationRecord(
        invocation_id="inv",
        graph_node="pre_route_planner",
        receiver_agent_id="planner",
        receiver_role="planner",
        context_fingerprint_digest="ctx",
        candidate_request_digest="req",
        candidate_memory_ids=["prefix", "target"],
        candidate_scores=[0.9, 0.8],
        proposal_order=["prefix", "target"],
        traversal_order=["prefix", "target"],
        decisions=[
            DecisionRecord(
                decision_index=0,
                memory_id="prefix",
                action="share",
                reason="accepted",
                traversal_position=0,
                selected_before_digest="empty",
            ),
            DecisionRecord(
                decision_index=1,
                memory_id="target",
                action="share",
                reason="accepted",
                traversal_position=1,
                selected_before_memory_ids=["prefix"],
                selected_before_digest="prefix",
            ),
        ],
        selected_memory_ids=["prefix", "target"],
        visible_payload_memory_ids=["prefix", "target"],
    )
    traces = compute_prefix_traces(
        [_run([invocation])],
        target_memory_id="target",
        required_prefix_memory_ids=["prefix"],
    )
    assert len(traces) == 1
    assert traces[0].required_prefix_in_same_invocation_candidates is True
    assert traces[0].required_prefix_selected_before_target is True
    assert traces[0].target_evaluated_under_required_prefix is True


def test_empty_required_prefix_is_not_marked_correct():
    invocation = RoutingInvocationRecord(
        invocation_id="inv",
        graph_node="pre_route_planner",
        receiver_agent_id="planner",
        receiver_role="planner",
        context_fingerprint_digest="ctx",
        candidate_request_digest="req",
        candidate_memory_ids=["target"],
        candidate_scores=[0.8],
        proposal_order=["target"],
        traversal_order=["target"],
        decisions=[
            DecisionRecord(
                decision_index=0,
                memory_id="target",
                action="withhold",
                reason="tau_lcb_nonpositive",
                traversal_position=0,
                selected_before_digest="empty",
            )
        ],
    )
    traces = compute_prefix_traces(
        [_run([invocation])],
        target_memory_id="target",
        required_prefix_memory_ids=[],
    )
    assert traces[0].required_prefix_in_same_invocation_candidates is False
    assert traces[0].target_evaluated_under_required_prefix is False
