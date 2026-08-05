"""Routing-card refactor tests (清单第九章 / Commit 7).

The routing card must only carry outcome-independent, trajectory-observable
attributes: human-authored transfer hints, fixed compatible-receiver-role
lists and paired outcomes must never enter the card, the feature encoder or
the label-free baselines.
"""

from __future__ import annotations

import inspect
import json

from smtr.core.types import AgentProfile, CandidateExposureInput, MemoryRoutingCard, ReceiverState
from smtr.marble.real_data import (
    AgentTrajectorySlice,
    RealDatabaseTrajectory,
    extract_procedural_memories,
)
from smtr.router.baselines import role_aware_top1_score
from smtr.router.transfer_features import (
    HashingTransferFeatureEncoder,
    build_routing_card_from_pool_entry,
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


def test_routing_card_excludes_transfer_hints():
    """Extracted cards must never populate deprecated human hint fields."""
    traj = _make_trajectory(actions=[
        {"name": "inspect_health", "tool": "monitor"},
        {"name": "run_query", "tool": "sql_tool", "arguments": {"sql": "SELECT 1"}},
    ])
    memories = extract_procedural_memories([traj], min_actions=2)
    card = memories[0].routing_card
    assert card.positive_transfer_hints == ()
    assert card.negative_transfer_hints == ()
    assert card.compatible_receiver_roles == ()
    assert card.incompatible_receiver_roles == ()


def test_routing_card_excludes_procedure_payload():
    traj = _make_trajectory(actions=[
        {"name": "unique_step_alpha", "tool": "tool_alpha"},
        {"name": "unique_step_beta", "tool": "tool_beta"},
    ])
    memories = extract_procedural_memories([traj], min_actions=2)
    memory = memories[0]
    card_json = json.dumps(memory.routing_card.model_dump(mode="json")).lower()
    # The numbered procedure text and its ordered steps must never leak
    # into the card (action names as observable tags are allowed).
    assert memory.payload.procedure.lower() not in card_json
    assert "1. " not in memory.routing_card.goal_summary
    for forbidden in ("procedure", "ordered_steps", "raw_action_sequence"):
        assert forbidden not in card_json


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


def test_pool_builder_never_restores_deprecated_hint_fields():
    """Even when an old pool entry stores hints, they must be dropped."""
    pool_entry = {
        "memory_id": "m1",
        "routing_card": {
            "goal_summary": "diagnose latency",
            "task_tags": ["database"],
            "positive_transfer_hints": ["helpful for executors"],
            "negative_transfer_hints": ["expensive query"],
            "compatible_receiver_roles": ["executor"],
            "incompatible_receiver_roles": ["planner"],
            "writer": {"agent_id": "w1", "role": "executor"},
        },
    }
    card = build_routing_card_from_pool_entry(pool_entry)
    assert card.positive_transfer_hints == ()
    assert card.negative_transfer_hints == ()
    assert card.compatible_receiver_roles == ()
    assert card.incompatible_receiver_roles == ()


def test_encoder_features_unaffected_by_deprecated_hints():
    """Feature tokens must be identical with or without legacy hint fields."""
    base_card = MemoryRoutingCard(
        memory_id="m1",
        goal_summary="diagnose database latency via select-based method",
        task_tags=("database", "latency"),
        environment_constraints=("read-only SQL",),
        writer=AgentProfile(agent_id="w1", role="executor", capabilities=("sql",),
                            tool_names=("sql_tool",)),
        source_task_id="src1",
        source_scenario="database",
    )
    hinted_card = base_card.model_copy(update={
        "positive_transfer_hints": ("great hint",),
        "negative_transfer_hints": ("bad hint",),
        "compatible_receiver_roles": ("executor",),
        "incompatible_receiver_roles": ("planner",),
    })
    encoder = HashingTransferFeatureEncoder(feature_block="full")

    def _tokens(card: MemoryRoutingCard) -> list[str]:
        return encoder.tokens(CandidateExposureInput(
            receiver_state=_receiver(), candidate_card=card, selected_prefix_cards=(),
        ))

    assert _tokens(base_card) == _tokens(hinted_card)
    forbidden_prefixes = (
        "compatible_receiver_role:", "incompatible_receiver_role:",
        "positive_hint_token:", "negative_hint_token:",
    )
    for token in _tokens(hinted_card):
        assert not token.startswith(forbidden_prefixes)


def test_role_aware_top1_ignores_hint_fields():
    """Baseline score must depend on observable writer/receiver roles, not on
    deprecated compatible-role hint fields."""
    card = MemoryRoutingCard(
        memory_id="m1",
        goal_summary="diagnose database latency",
        task_tags=("database", "latency"),
        writer=AgentProfile(agent_id="w1", role="executor", capabilities=("sql",),
                            tool_names=("sql_tool",)),
        source_task_id="src1",
        source_scenario="database",
    )
    hinted = card.model_copy(update={"compatible_receiver_roles": ("planner",)})
    rs = _receiver(role="executor")
    assert role_aware_top1_score(rs, card) == role_aware_top1_score(rs, hinted)
    # Same-role writer/receiver must score higher than mismatched role.
    mismatched = card.model_copy(update={
        "writer": card.writer.model_copy(update={"role": "critic"}),
    })
    assert role_aware_top1_score(rs, card) > role_aware_top1_score(rs, mismatched)
