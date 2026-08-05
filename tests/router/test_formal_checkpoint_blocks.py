"""Test 9 (清单 P1-1/P1-2): feature-block checkpoint separation.

GlobalTransferCritic may only consume a ``global_transfer`` checkpoint,
SMTR-no-pair only ``no_pair_interaction`` and SMTR only ``full``; the
three formal blocks must also keep their prescribed feature sets.
"""

from __future__ import annotations

import pytest

from smtr.core.types import (
    AgentProfile,
    CandidateExposureInput,
    MemoryRoutingCard,
    ReceiverState,
)
from smtr.marble.paired_evaluation import verify_formal_checkpoint_blocks
from smtr.router.transfer_critic import FourOutcomeTransferCritic
from smtr.router.transfer_features import HashingTransferFeatureEncoder


def _make_input() -> CandidateExposureInput:
    writer = AgentProfile(
        agent_id="w1", role="planner", capabilities=("plan",), tool_names=("search",)
    )
    receiver = AgentProfile(
        agent_id="r1", role="executor", capabilities=("sql",), tool_names=("db_query",)
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


def _tokens(block: str) -> set[str]:
    return set(HashingTransferFeatureEncoder(feature_block=block).tokens(_make_input()))


def test_global_transfer_block_drops_writer_receiver_and_interaction():
    """global_transfer keeps task/environment/memory semantics only (清单 P1-1)."""
    tokens = _tokens("global_transfer")
    assert not any(t.startswith("writer_") for t in tokens)
    assert not any(t.startswith("receiver_") for t in tokens)
    assert not any(t.startswith("wr_") for t in tokens)
    assert any(t.startswith("scenario:") or t.startswith("task_tag:") for t in tokens)
    assert any(t.startswith("memory_goal_token:") for t in tokens)
    # The formal block is token-identical to its legacy alias.
    assert tokens == _tokens("memory_task_only")


def test_no_pair_interaction_block_keeps_marginals_drops_interactions():
    """no_pair_interaction keeps writer/receiver marginals, drops pair tokens."""
    tokens = _tokens("no_pair_interaction")
    assert any(t.startswith("writer_role:") for t in tokens)
    assert any(t.startswith("receiver_role:") for t in tokens)
    assert not any(t.startswith("wr_pair:") for t in tokens)
    assert not any(t.startswith("wr_same_role:") for t in tokens)


def test_smtr_only_accepts_full_checkpoint():
    """SMTR must reject checkpoints trained with any other block."""
    wrong = FourOutcomeTransferCritic(feature_block="global_transfer")
    with pytest.raises(ValueError, match="feature_block='full'"):
        verify_formal_checkpoint_blocks(full_critic=wrong)


def test_global_transfer_critic_only_accepts_global_transfer_checkpoint():
    """GlobalTransferCritic must reject non-global_transfer checkpoints."""
    full = FourOutcomeTransferCritic(feature_block="full")
    ok = FourOutcomeTransferCritic(feature_block="full")
    with pytest.raises(ValueError, match="global_transfer"):
        verify_formal_checkpoint_blocks(full_critic=ok, global_critic=full)
    # The formal block and its legacy alias are both accepted.
    verify_formal_checkpoint_blocks(
        full_critic=ok,
        global_critic=FourOutcomeTransferCritic(feature_block="global_transfer"),
    )
    verify_formal_checkpoint_blocks(
        full_critic=ok,
        global_critic=FourOutcomeTransferCritic(feature_block="memory_task_only"),
    )


def test_smtr_no_pair_only_accepts_no_pair_interaction_checkpoint():
    """SMTR-no-pair must reject checkpoints trained with any other block."""
    ok = FourOutcomeTransferCritic(feature_block="full")
    with pytest.raises(ValueError, match="no_pair_interaction"):
        verify_formal_checkpoint_blocks(
            full_critic=ok,
            no_pair_critic=FourOutcomeTransferCritic(feature_block="full"),
        )
    verify_formal_checkpoint_blocks(
        full_critic=ok,
        no_pair_critic=FourOutcomeTransferCritic(feature_block="no_pair_interaction"),
    )
