"""Tests for memory extraction from real trajectories."""

from __future__ import annotations

import json

from smtr.marble.real_data import (
    AgentTrajectorySlice,
    ExtractedMemory,
    RealDatabaseTrajectory,
    extract_procedural_memories,
    normalize_action_step,
    normalize_sql,
)


def _make_trajectory(
    *,
    actions: list[dict],
    agent_id: str = "agent1",
    agent_role: str = "executor",
    task_success: bool = True,
    split: str = "train",
) -> RealDatabaseTrajectory:
    return RealDatabaseTrajectory(
        trajectory_id="traj001",
        task_id="task1",
        split=split,
        generation_seed=0,
        model_id="test-model",
        score=1.0 if task_success else 0.0,
        task_success=task_success,
        agents=(
            AgentTrajectorySlice(
                agent_id=agent_id,
                agent_role=agent_role,
                actions=tuple(actions),
                tool_names=("sql_tool",),
            ),
        ),
    )


def test_procedure_from_actual_action_order():
    """Procedure must come from actual action order, not a fixed template."""
    traj = _make_trajectory(actions=[
        {"name": "inspect_health", "tool": "monitor"},
        {"name": "run_query", "tool": "sql_tool", "arguments": {"sql": "SELECT * FROM t"}},
        {"name": "cross_check", "tool": "validator"},
    ])
    memories = extract_procedural_memories([traj], min_actions=2)
    assert len(memories) == 1
    proc = memories[0].payload.procedure
    assert "inspect_health" in proc or "monitor" in proc
    assert "sql_tool" in proc or "query" in proc.lower()
    # Must NOT be the old fixed template
    assert "Inspect database health and workload evidence" not in proc


def test_different_trajectories_different_procedures():
    """Two different trajectories must produce different procedures."""
    traj1 = _make_trajectory(actions=[
        {"name": "check_index", "tool": "indexer"},
        {"name": "analyze_query", "tool": "analyzer"},
    ])
    traj2 = RealDatabaseTrajectory(
        trajectory_id="traj002",
        task_id="task2",
        split="train",
        generation_seed=0,
        model_id="test-model",
        score=1.0,
        task_success=True,
        agents=(
            AgentTrajectorySlice(
                agent_id="agent2",
                agent_role="critic",
                actions=(
                    {"name": "validate_schema", "tool": "schema_check"},
                    {"name": "run_explain", "tool": "explain_tool"},
                    {"name": "compare_plans", "tool": "plan_diff"},
                ),
                tool_names=("schema_check", "explain_tool"),
            ),
        ),
    )
    mem1 = extract_procedural_memories([traj1], min_actions=2)
    mem2 = extract_procedural_memories([traj2], min_actions=2)
    assert mem1[0].payload.procedure != mem2[0].payload.procedure


def test_routing_card_no_procedure():
    """Routing card must not contain procedure text (清单 Writer-Agnostic
    第三章: procedure lives in the payload, the card carries only
    requirement metadata plus coarse procedure type/length buckets)."""
    traj = _make_trajectory(actions=[
        {"name": "step_a", "tool": "tool_a"},
        {"name": "step_b", "tool": "tool_b"},
    ])
    memories = extract_procedural_memories([traj], min_actions=2)
    card_json = json.dumps(memories[0].routing_card.model_dump(mode="json")).lower()
    assert memories[0].payload.procedure.lower() not in card_json
    assert "ordered_steps" not in card_json


def test_only_train_split():
    """Memory extraction must only read train trajectories."""
    traj = _make_trajectory(
        actions=[{"name": "x", "tool": "y"}, {"name": "z", "tool": "w"}],
        split="test",
    )
    try:
        extract_procedural_memories([traj], min_actions=2)
        assert False, "should have raised"
    except ValueError as e:
        assert "train" in str(e)


def test_failed_trajectory_no_memory():
    """Failed trajectories must not produce memories."""
    traj = RealDatabaseTrajectory(
        trajectory_id="traj_fail",
        task_id="task_f",
        split="train",
        generation_seed=0,
        model_id="test",
        score=0.0,
        task_success=False,
        agents=(
            AgentTrajectorySlice(
                agent_id="a1",
                agent_role="executor",
                actions=({"name": "x"}, {"name": "y"}),
            ),
        ),
    )
    memories = extract_procedural_memories([traj], min_actions=2)
    assert len(memories) == 0


def test_provenance_is_real_agent():
    """Provenance must equal the real agent from the trajectory, and the
    routing card must carry no source-agent fields (清单 Writer-Agnostic
    第二、三章)."""
    traj = _make_trajectory(
        actions=[{"name": "a", "tool": "t1"}, {"name": "b", "tool": "t2"}],
        agent_id="real_writer_42",
        agent_role="planner",
    )
    memories = extract_procedural_memories([traj], min_actions=2)
    provenance = memories[0].payload.provenance
    assert provenance.source_agent_id == "real_writer_42"
    assert provenance.source_agent_role == "planner"
    assert provenance.source_task_id == "task1"
    assert provenance.source_trajectory_id == "traj001"
    card_keys = set(memories[0].routing_card.model_dump())
    assert not card_keys & {
        "writer", "source_agent_id", "source_agent_role",
        "source_task_id", "source_trajectory_id",
    }


def test_normalize_sql():
    """SQL normalization must replace constants."""
    sql = "SELECT * FROM users WHERE id = 123 AND name = 'alice' LIMIT 10"
    normalized = normalize_sql(sql)
    assert "123" not in normalized
    assert "alice" not in normalized
    assert "<STR>" in normalized
    assert "<LIMIT>" in normalized


def test_normalize_action_step():
    """normalize_action_step must extract tool/action info."""
    step = normalize_action_step({"tool": "sql_runner", "arguments": {"sql": "SELECT 1"}}, 1)
    assert "sql_runner" in step
    assert "select" in step.lower()

    step2 = normalize_action_step({"name": "check_health"}, 2)
    assert "check_health" in step2
