"""Test 12 (清单 Writer-Agnostic 第十章): feature-block checkpoint separation.

GlobalTransferCritic may only consume a ``global_transfer`` checkpoint,
SMTR-no-compatibility-interaction only ``no_compatibility_interaction``
and SMTR only ``full``; the three formal blocks must also keep their
prescribed feature sets.
"""

from __future__ import annotations

import pytest

from smtr.core.types import (
    AgentProfile,
    CandidateExposureInput,
    MemoryRoutingCard,
    ReceiverState,
)
from smtr.marble.formal_protocol import verify_formal_checkpoint_blocks
from smtr.router.transfer_critic import FourOutcomeTransferCritic
from smtr.router.transfer_features import HashingTransferFeatureEncoder


def _make_input() -> CandidateExposureInput:
    receiver = AgentProfile(
        agent_id="r1", role="executor", capabilities=("sql",), tool_names=("db_query",)
    )
    card = MemoryRoutingCard(
        memory_id="m1",
        goal_summary="Diagnose database issue",
        task_tags=("database",),
        required_tools=("db_query",),
        required_capabilities=("sql",),
        execution_role_tags=("executor",),
        environment_constraints=("read-only",),
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


def test_global_transfer_block_drops_receiver_and_interaction():
    """global_transfer keeps task/environment/memory semantics only."""
    tokens = _tokens("global_transfer")
    assert not any(t.startswith("writer") for t in tokens)
    assert not any(t.startswith("receiver_") for t in tokens)
    assert not any(t.startswith("mr_") for t in tokens)
    assert any(t.startswith("scenario:") for t in tokens)
    assert any(t.startswith("memory_goal_token:") for t in tokens)
    # Legacy block names are rejected outright.
    with pytest.raises(ValueError, match="unknown feature_block"):
        _tokens("memory_task_only")


def test_no_compatibility_interaction_keeps_receiver_marginals():
    """no_compatibility_interaction keeps receiver marginals, drops mr_ tokens."""
    tokens = _tokens("no_compatibility_interaction")
    assert any(t.startswith("receiver_role:") for t in tokens)
    assert not any(t.startswith("mr_") for t in tokens)
    assert not any(t.startswith("writer") for t in tokens)


def test_smtr_only_accepts_full_checkpoint():
    """SMTR must reject checkpoints trained with any other block."""
    wrong = FourOutcomeTransferCritic(feature_block="global_transfer")
    with pytest.raises(ValueError, match="feature_block in"):
        verify_formal_checkpoint_blocks(
            full_critic=wrong,
            global_critic=None,
            no_compatibility_critic=None,
            methods=["smtr"],
            require_calibration=False,
        )


def test_global_transfer_critic_only_accepts_global_transfer_checkpoint():
    """GlobalTransferCritic must reject non-global_transfer checkpoints."""
    ok = FourOutcomeTransferCritic(feature_block="full")
    full = FourOutcomeTransferCritic(feature_block="full")
    with pytest.raises(ValueError, match="global_transfer"):
        verify_formal_checkpoint_blocks(
            full_critic=ok,
            global_critic=full,
            no_compatibility_critic=None,
            methods=["smtr", "global_transfer_critic"],
            require_calibration=False,
        )
    # The formal block is accepted; a different formal block is not.
    verify_formal_checkpoint_blocks(
        full_critic=ok,
        global_critic=FourOutcomeTransferCritic(feature_block="global_transfer"),
        no_compatibility_critic=None,
        methods=["smtr", "global_transfer_critic"],
        require_calibration=False,
    )
    with pytest.raises(ValueError, match="global_transfer"):
        verify_formal_checkpoint_blocks(
            full_critic=ok,
            global_critic=FourOutcomeTransferCritic(
                feature_block="no_compatibility_interaction"
            ),
            no_compatibility_critic=None,
            methods=["smtr", "global_transfer_critic"],
            require_calibration=False,
        )


def test_smtr_no_compatibility_only_accepts_matching_checkpoint():
    """SMTR-no-compatibility-interaction rejects any other block."""
    ok = FourOutcomeTransferCritic(feature_block="full")
    with pytest.raises(ValueError, match="no_compatibility_interaction"):
        verify_formal_checkpoint_blocks(
            full_critic=ok,
            global_critic=None,
            no_compatibility_critic=FourOutcomeTransferCritic(feature_block="full"),
            methods=["smtr", "smtr_no_compatibility_interaction"],
            require_calibration=False,
        )
    verify_formal_checkpoint_blocks(
        full_critic=ok,
        global_critic=None,
        no_compatibility_critic=FourOutcomeTransferCritic(
            feature_block="no_compatibility_interaction"
        ),
        methods=["smtr", "smtr_no_compatibility_interaction"],
        require_calibration=False,
    )
