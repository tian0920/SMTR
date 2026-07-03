from smtr.counterfactual.decision_points import (
    InMemoryDecisionPointRecorder,
    canonical_digest,
)
from smtr.memory.seed_memories import build_seed_memories
from smtr.runtime.graph import run_demo, run_episode


def test_recorder_captures_three_baseline_decision_points() -> None:
    recorder = InMemoryDecisionPointRecorder()

    run_episode(seed=7, decision_point_recorder=recorder)

    assert [point.graph_node for point in recorder.decision_points] == [
        "pre_route_planner",
        "pre_route_executor",
        "pre_route_critic",
    ]


def test_demo_without_recorder_has_no_decision_point_side_effect() -> None:
    state = run_demo(seed=7)

    assert "decision_points" not in state


def test_decision_point_is_before_router_and_injection() -> None:
    recorder = InMemoryDecisionPointRecorder()

    run_episode(seed=7, decision_point_recorder=recorder)
    point = recorder.decision_points[0]

    assert point.candidate_proposal.ranked_candidates
    assert point.graph_state_snapshot["router_trace"] == []
    assert point.graph_state_snapshot["agent_local_context"]["planner"]["visible_payloads"] == []


def test_decision_point_state_is_deep_copied_and_has_no_payload_steps() -> None:
    recorder = InMemoryDecisionPointRecorder()
    state = run_episode(seed=7, decision_point_recorder=recorder)
    point = recorder.decision_points[0]

    state["agent_local_context"]["planner"]["visible_payloads"].append({"steps": ["leak"]})
    assert point.graph_state_snapshot["agent_local_context"]["planner"]["visible_payloads"] == []

    payload_steps = [step for _, payload in build_seed_memories() for step in payload.steps]
    point_text = repr(point.model_dump())
    assert all(step not in point_text for step in payload_steps)


def test_snapshot_digest_is_deterministic_for_same_state() -> None:
    payload = {"b": [2, 1], "a": {"x": True}}

    assert canonical_digest(payload) == canonical_digest({"a": {"x": True}, "b": [2, 1]})
