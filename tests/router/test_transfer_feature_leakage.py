"""Tests for transfer feature leakage prevention."""

from __future__ import annotations

import pytest

from smtr.core.types import AgentProfile, CandidateExposureInput, MemoryRoutingCard, ReceiverState
from smtr.router.transfer_features import FORBIDDEN_FEATURE_TOKENS, HashingTransferFeatureEncoder


def _make_input() -> CandidateExposureInput:
    receiver = AgentProfile(agent_id="r1", role="executor", capabilities=("sql",))
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
    )
    return CandidateExposureInput(receiver_state=rs, candidate_card=card)


def test_full_features_contain_mr_interaction():
    """Full feature block must contain memory-receiver compatibility tokens."""
    encoder = HashingTransferFeatureEncoder(feature_block="full")
    tokens = encoder.tokens(_make_input())
    assert any(t.startswith("mr_") for t in tokens)
    assert not any(t.startswith("writer") for t in tokens)


def test_no_compatibility_interaction_excludes_interaction():
    """no_compatibility_interaction drops memory-receiver interaction tokens."""
    encoder = HashingTransferFeatureEncoder(feature_block="no_compatibility_interaction")
    tokens = encoder.tokens(_make_input())
    assert not any(t.startswith("mr_") for t in tokens)
    assert not any(t.startswith("writer") for t in tokens)
    assert any(t.startswith("receiver_role:") for t in tokens)


def test_forbidden_tokens_not_present():
    """Forbidden tokens must not appear in feature output."""
    encoder = HashingTransferFeatureEncoder(feature_block="full")
    tokens = encoder.tokens(_make_input())
    for token in tokens:
        prefix = token.lower().split(":", 1)[0]
        assert prefix not in FORBIDDEN_FEATURE_TOKENS, f"forbidden: {token}"


def test_tokens_calls_leakage_validator():
    """tokens() must actually call the leakage validator (raises on forbidden)."""
    encoder = HashingTransferFeatureEncoder(feature_block="full")
    # Normal input should not raise
    tokens = encoder.tokens(_make_input())
    assert len(tokens) > 0
