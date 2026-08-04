"""Tests for the four precise feature ablation modes (清单第六章)."""

from __future__ import annotations

import pytest

from smtr.core.types import (
    AgentProfile,
    CandidateExposureInput,
    MemoryRoutingCard,
    ReceiverState,
)
from smtr.evaluation.feature_ablation import FEATURE_MODES, audit_feature_modes
from smtr.router.transfer_features import HashingTransferFeatureEncoder


def _make_input(receiver_id: str = "r1", receiver_role: str = "executor") -> CandidateExposureInput:
    writer = AgentProfile(
        agent_id="w1", role="planner", capabilities=("plan",), tool_names=("search",)
    )
    receiver = AgentProfile(
        agent_id=receiver_id, role=receiver_role, capabilities=("sql",), tool_names=("db_query",)
    )
    card = MemoryRoutingCard(
        memory_id="m1",
        goal_summary="Diagnose database issue",
        task_tags=("database",),
        environment_constraints=("read-only",),
        writer=writer,
        source_task_id="t1",
        source_scenario="database",
        compatible_receiver_roles=("executor",),
    )
    rs = ReceiverState(
        task_id="t2",
        scenario="database",
        task_instruction="Fix the slow query",
        receiver=receiver,
        environment_signature=("production",),
    )
    return CandidateExposureInput(receiver_state=rs, candidate_card=card)


def _token_set(feature_block: str, item: CandidateExposureInput) -> set[str]:
    return set(HashingTransferFeatureEncoder(feature_block=feature_block).tokens(item))


def _vector(feature_block: str, item: CandidateExposureInput):
    return HashingTransferFeatureEncoder(feature_block=feature_block).encode_one(item).toarray()


def test_no_receiver_features_are_receiver_invariant():
    """Same task/memory/writer but different receivers.

    full must differ across receivers; no_receiver and memory_task_only
    must be identical across receivers.
    """
    item_r1 = _make_input(receiver_id="r1", receiver_role="executor")
    item_r2 = _make_input(receiver_id="r2", receiver_role="planner")

    full_r1 = _token_set("full", item_r1)
    full_r2 = _token_set("full", item_r2)
    assert full_r1 != full_r2

    no_receiver_r1 = _token_set("no_receiver", item_r1)
    no_receiver_r2 = _token_set("no_receiver", item_r2)
    assert no_receiver_r1 == no_receiver_r2

    memory_task_r1 = _token_set("memory_task_only", item_r1)
    memory_task_r2 = _token_set("memory_task_only", item_r2)
    assert memory_task_r1 == memory_task_r2

    # Hashed vectors agree too (not just raw token lists).
    assert (_vector("no_receiver", item_r1) == _vector("no_receiver", item_r2)).all()
    assert (_vector("memory_task_only", item_r1) == _vector("memory_task_only", item_r2)).all()
    assert not (_vector("full", item_r1) == _vector("full", item_r2)).all()


def test_no_pair_interaction_keeps_writer_and_receiver_marginals():
    """no_pair_interaction keeps writer and receiver marginals, drops pair tokens."""
    tokens = _token_set("no_pair_interaction", _make_input())
    assert any(t.startswith("writer_role:") for t in tokens)
    assert any(t.startswith("writer_cap:") for t in tokens)
    assert any(t.startswith("receiver_role:") for t in tokens)
    assert any(t.startswith("receiver_cap:") for t in tokens)
    pair_prefixes = (
        "wr_pair:",
        "wr_same_role:",
        "wr_cap_overlap_bucket:",
        "wr_tool_overlap_bucket:",
        "writer_receiver_mismatch:",
    )
    for prefix in pair_prefixes:
        assert not any(t.startswith(prefix) for t in tokens), f"{prefix} must be dropped"


def test_no_receiver_keeps_task_memory_writer_but_not_receiver():
    """no_receiver keeps task/environment/memory/writer, drops receiver identity."""
    tokens = _token_set("no_receiver", _make_input())
    assert any(t.startswith("task_token:") for t in tokens)
    assert any(t.startswith("env:") for t in tokens)
    assert any(t.startswith("memory_goal_token:") for t in tokens)
    assert any(t.startswith("writer_role:") for t in tokens)
    assert not any(t.startswith("receiver_role:") for t in tokens)
    assert not any(t.startswith("receiver_cap:") for t in tokens)
    assert not any(t.startswith("receiver_tool:") for t in tokens)
    assert not any(t.startswith("wr_pair:") for t in tokens)


def test_memory_task_only_drops_writer_receiver_and_interaction():
    """memory_task_only keeps only task context, environment and memory card."""
    tokens = _token_set("memory_task_only", _make_input())
    assert any(t.startswith("task_token:") for t in tokens)
    assert any(t.startswith("env:") for t in tokens)
    assert any(t.startswith("memory_goal_token:") for t in tokens)
    assert not any(t.startswith("writer_role:") for t in tokens)
    assert not any(t.startswith("receiver_role:") for t in tokens)
    assert not any(t.startswith("wr_pair:") for t in tokens)
    assert any(t == "prefix_size:0" for t in tokens)


def test_unknown_feature_mode_raises():
    """Unknown feature_block must fail fast."""
    encoder = HashingTransferFeatureEncoder(feature_block="bogus_mode")
    with pytest.raises(ValueError, match="unknown feature_block"):
        encoder.tokens(_make_input())


def test_feature_modes_registry_matches_spec():
    """The ablation framework must expose exactly the four spec modes."""
    assert FEATURE_MODES == [
        "full",
        "no_pair_interaction",
        "no_receiver",
        "memory_task_only",
    ]


def test_audit_feature_modes_smoke():
    """audit_feature_modes trains all modes on an identical split."""
    labels = ["positive_transfer", "negative_transfer", "neutral_success", "neutral_failure"]
    data = []
    for task_index in range(4):
        for candidate_index in range(8):
            writer = AgentProfile(agent_id=f"w{candidate_index % 2}", role="planner")
            receiver = AgentProfile(agent_id=f"r{candidate_index % 3}", role="executor")
            card = MemoryRoutingCard(
                memory_id=f"m{candidate_index}",
                goal_summary=f"goal {candidate_index}",
                writer=writer,
                source_task_id=f"src{candidate_index}",
                source_scenario="database",
            )
            rs = ReceiverState(
                task_id=f"task{task_index}",
                scenario="database",
                task_instruction=f"fix query {task_index}",
                receiver=receiver,
            )
            item = CandidateExposureInput(receiver_state=rs, candidate_card=card)
            data.append((item, labels[(task_index + candidate_index) % 4]))

    report = audit_feature_modes(data, seed=0, n_bootstrap=3, n_features=64)
    assert set(report["modes"]) == set(FEATURE_MODES)
    for mode in FEATURE_MODES:
        assert report["modes"][mode]["n_test"] > 0
        assert report["modes"][mode]["accuracy"] is not None
    assert report["best_ablation_mode"] in FEATURE_MODES
    assert "full_gain_over_no_pair_interaction" in report
