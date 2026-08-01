"""Tests for multi-agent trajectory parsing."""

from __future__ import annotations

from smtr.marble.real_workflows import _build_agent_slices, _agent_id


def test_multi_agent_trace_splits_by_agent():
    """Multi-agent raw trace should be split into different agent slices."""
    raw = {
        "agents": [
            {"agent_id": "planner1", "role": "planner", "capabilities": ["plan"]},
            {"agent_id": "executor1", "role": "executor", "capabilities": ["sql"]},
        ],
        "actions": [
            {"agent_id": "planner1", "name": "create_plan"},
            {"agent_id": "executor1", "name": "run_query"},
            {"agent_id": "executor1", "name": "validate_result"},
        ],
        "tool_calls": [
            {"agent_id": "executor1", "tool": "sql_tool", "arguments": {"sql": "SELECT 1"}},
        ],
    }
    slices = _build_agent_slices(
        raw=raw,
        messages=[],
        actions=raw["actions"],
        tool_calls=raw["tool_calls"],
        sql=["SELECT 1"],
    )
    assert len(slices) == 2
    planner = next(s for s in slices if s.agent_id == "planner1")
    executor = next(s for s in slices if s.agent_id == "executor1")
    assert planner.agent_role == "planner"
    assert executor.agent_role == "executor"
    assert len(executor.actions) == 2
    assert len(executor.tool_calls) == 1


def test_no_agent_attribution_no_fabrication():
    """Actions without agent attribution should not be assigned to a fabricated writer."""
    raw = {
        "agents": [],
        "actions": [
            {"name": "do_something"},
            {"name": "do_other"},
        ],
        "tool_calls": [],
    }
    slices = _build_agent_slices(
        raw=raw,
        messages=[],
        actions=raw["actions"],
        tool_calls=[],
        sql=[],
    )
    # No agent attribution -> no slices
    assert len(slices) == 0


def test_valid_trajectory_must_have_agent_actions():
    """A valid trajectory must have agent-specific actions."""
    from smtr.marble.real_data import AgentTrajectorySlice, RealDatabaseTrajectory

    # Valid trajectory with agent slices
    traj = RealDatabaseTrajectory(
        trajectory_id="t1",
        task_id="task1",
        split="train",
        generation_seed=0,
        model_id="test",
        score=1.0,
        task_success=True,
        agents=(
            AgentTrajectorySlice(
                agent_id="a1",
                agent_role="executor",
                actions=({"name": "query"},),
            ),
        ),
    )
    assert traj.agents
    assert all(a.agent_id for a in traj.agents)
    assert any(a.actions or a.tool_calls for a in traj.agents)


def test_agent_id_extraction():
    """_agent_id should try multiple field names."""
    assert _agent_id({"agent_id": "x"}) == "x"
    assert _agent_id({"sender_id": "y"}) == "y"
    assert _agent_id({"source_agent": "z"}) == "z"
    assert _agent_id({"agent": "w"}) == "w"
    assert _agent_id({"other": "v"}) is None
