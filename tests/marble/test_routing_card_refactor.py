"""Routing-card refactor tests (清单 Writer-Agnostic 第三/六章).

The routing card must only carry outcome-independent, trajectory-observable
attributes: writer/source-agent identity, fixed compatible-receiver-role
lists, transfer hints and paired outcomes must never enter the card, the
feature encoder or the label-free baselines.
"""

from __future__ import annotations

import inspect
import json

import pytest

from smtr.core.types import AgentProfile, CandidateExposureInput, MemoryRoutingCard, ReceiverState
from smtr.marble.real_data import (
    AgentTrajectorySlice,
    RealDatabaseTrajectory,
    extract_procedural_memories,
)
from smtr.router.baselines import receiver_compatible_top1_score
from smtr.router.transfer_features import (
    HashingTransferFeatureEncoder,
    build_routing_card_from_pool_entry,
)

_LEGACY_CARD_FIELDS = (
    "writer",
    "source_agent",
    "source_agent_id",
    "source_agent_role",
    "source_task_id",
    "source_scenario",
    "source_trajectory_id",
    "positive_transfer_hints",
    "negative_transfer_hints",
    "compatible_receiver_roles",
    "incompatible_receiver_roles",
)


def _make_trajectory(
    *,
    trajectory_id: str = "traj001",
    task_id: str = "task1",
    actions: list[dict],
    sql_statements: tuple[str, ...] = (),
    agent_role: str = "executor",
) -> RealDatabaseTrajectory:
    return RealDatabaseTrajectory(
        trajectory_id=trajectory_id,
        task_id=task_id,
        split="train",
        generation_seed=0,
        model_id="test-model",
        score=1.0,
        task_success=True,
        agents=(
            AgentTrajectorySlice(
                agent_id="agent1",
                agent_role=agent_role,
                actions=tuple(actions),
                tool_names=("sql_tool",),
                sql_statements=sql_statements,
            ),
        ),
    )


def _receiver(role: str = "executor") -> ReceiverState:
    return ReceiverState(
        task_id="t1",
        scenario="database",
        task_instruction="diagnose database latency",
        receiver=AgentProfile(
            agent_id="r1", role=role, capabilities=("sql",), tool_names=("sql_tool",),
        ),
        environment_signature=("read-only SQL",),
    )


def test_routing_card_excludes_writer_and_hint_fields():
    """Extracted cards must not carry writer identity or legacy hint fields."""
    traj = _make_trajectory(actions=[
        {"name": "inspect_health", "tool": "monitor"},
        {"name": "run_query", "tool": "sql_tool", "arguments": {"sql": "SELECT 1"}},
    ])
    memories = extract_procedural_memories([traj], min_actions=2)
    card = memories[0].routing_card
    for field in _LEGACY_CARD_FIELDS:
        assert field not in card.model_dump(), f"legacy field {field} in routing card"


def test_routing_card_excludes_procedure_payload():
    traj = _make_trajectory(actions=[
        {"name": "unique_step_alpha", "tool": "tool_alpha"},
        {"name": "unique_step_beta", "tool": "tool_beta"},
    ])
    memories = extract_procedural_memories([traj], min_actions=2)
    memory = memories[0]
    card = memory.routing_card
    card_json = json.dumps(card.model_dump(mode="json")).lower()
    # The numbered procedure text and its ordered steps must never leak
    # into the card (action names as observable tags are allowed).
    assert memory.payload.procedure.lower() not in card_json
    assert "1. " not in card.goal_summary
    card_keys = set(card.model_dump())
    for forbidden in ("procedure", "ordered_steps", "raw_action_sequence"):
        assert forbidden not in card_keys


def test_routing_card_excludes_paired_outcomes():
    traj = _make_trajectory(actions=[
        {"name": "step_a", "tool": "tool_a"},
        {"name": "step_b", "tool": "tool_b"},
    ])
    memories = extract_procedural_memories([traj], min_actions=2)
    card_json = json.dumps(memories[0].routing_card.model_dump(mode="json")).lower()
    for forbidden in ("team_success", "y_share", "y_withhold", "ground_truth"):
        assert forbidden not in card_json


def test_different_trajectories_produce_distinct_cards():
    """Cards must differ beyond step count: same step counts but different
    operation mixes and action names must yield different cards."""
    traj_a = _make_trajectory(
        trajectory_id="trajA",
        actions=[
            {"name": "check_index", "tool": "indexer"},
            {"name": "run_query", "tool": "sql_tool", "arguments": {"sql": "SELECT 1"}},
        ],
        sql_statements=("SELECT count(*) FROM orders",),
    )
    traj_b = _make_trajectory(
        trajectory_id="trajB",
        task_id="task2",
        actions=[
            {"name": "compare_plans", "tool": "plan_diff"},
            {"name": "run_query", "tool": "sql_tool", "arguments": {"sql": "EXPLAIN SELECT 1"}},
        ],
        sql_statements=("EXPLAIN SELECT * FROM users",),
    )
    card_a = extract_procedural_memories([traj_a], min_actions=2)[0].routing_card
    card_b = extract_procedural_memories([traj_b], min_actions=2)[0].routing_card
    assert card_a.goal_summary != card_b.goal_summary
    assert set(card_a.task_tags) != set(card_b.task_tags)


def test_card_extraction_depends_only_on_source_trajectory():
    """Extraction interface must not accept paired records or outcomes."""
    params = set(inspect.signature(extract_procedural_memories).parameters)
    assert params == {"trajectories", "min_actions"}


def test_pool_builder_rejects_legacy_routing_cards():
    """Pool entries carrying writer identity or missing v3 requirements
    must fail closed instead of silently falling back."""
    legacy_writer = {
        "memory_id": "m1",
        "routing_card": {
            "goal_summary": "diagnose latency",
            "task_tags": ["database"],
            "writer": {"agent_id": "w1", "role": "executor"},
        },
    }
    with pytest.raises(ValueError, match="legacy routing-card schema"):
        build_routing_card_from_pool_entry(legacy_writer)

    missing_requirements = {
        "memory_id": "m2",
        "routing_card": {
            "goal_summary": "diagnose latency",
            "task_tags": ["database"],
        },
    }
    with pytest.raises(ValueError, match="legacy routing-card schema"):
        build_routing_card_from_pool_entry(missing_requirements)


def test_pool_builder_restores_explicit_requirements():
    pool_entry = {
        "memory_id": "m1",
        "routing_card": {
            "goal_summary": "diagnose latency",
            "task_tags": ["database"],
            "required_tools": ["sql_tool"],
            "required_capabilities": ["sql"],
            "execution_role_tags": ["executor"],
            "environment_constraints": ["read-only SQL"],
        },
    }
    card = build_routing_card_from_pool_entry(pool_entry)
    assert card.required_tools == ("sql_tool",)
    assert card.required_capabilities == ("sql",)
    assert card.execution_role_tags == ("executor",)
    assert card.environment_constraints == ("read-only SQL",)


def test_encoder_tokens_exclude_legacy_writer_prefixes():
    """Feature tokens must never carry writer/provenance legacy prefixes."""
    card = MemoryRoutingCard(
        memory_id="m1",
        goal_summary="diagnose database latency via select-based method",
        task_tags=("database", "latency"),
        required_tools=("sql_tool",),
        required_capabilities=("sql",),
        environment_constraints=("read-only SQL",),
    )
    encoder = HashingTransferFeatureEncoder(feature_block="full")
    tokens = encoder.tokens(CandidateExposureInput(
        receiver_state=_receiver(), candidate_card=card, selected_prefix_cards=(),
    ))
    forbidden_prefixes = (
        "writer_role:", "writer_capability", "writer_tool",
        "compatible_receiver_role:", "incompatible_receiver_role:",
        "positive_hint_token:", "negative_hint_token:",
        "source_agent", "source_task", "source_trajectory",
    )
    for token in tokens:
        assert not token.startswith(forbidden_prefixes), token


def test_receiver_compatible_top1_uses_explicit_requirements():
    """Baseline score must depend on explicit memory requirements, not on
    writer identity or legacy hint fields."""
    satisfied = MemoryRoutingCard(
        memory_id="m1",
        goal_summary="diagnose database latency",
        task_tags=("database", "latency"),
        required_tools=("sql_tool",),
        required_capabilities=("sql",),
    )
    unsatisfied = satisfied.model_copy(update={
        "required_tools": ("admin_console",),
        "required_capabilities": ("cluster_admin",),
    })
    rs = _receiver(role="executor")
    assert receiver_compatible_top1_score(rs, satisfied) > receiver_compatible_top1_score(
        rs, unsatisfied
    )
