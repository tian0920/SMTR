"""Tests for A1-NoSet: SMTR without selected-set features."""

import json
from pathlib import Path

import pytest

from smtr.counterfactual.schemas import RoutingFeatureSnapshot
from smtr.memory.schemas import ContextFingerprint
from smtr.router.transfer_critic import FourOutcomeTransferCritic
from smtr.router.transfer_features import HashingTransferFeatureEncoder, TransferPredictionInput

A1_CHECKPOINT = Path("checkpoints/critic_no_selected_set_v1.joblib")
A1_METADATA = Path("checkpoints/critic_no_selected_set_v1.metadata.json")


def _make_prediction_input() -> TransferPredictionInput:
    """Create a minimal prediction input for testing."""
    ctx = ContextFingerprint(
        task_id="test_task",
        task_tags=["test"],
        receiver_agent_id="agent_1",
        receiver_role="planner",
        receiver_capabilities=["planning"],
        task_stage="planning",
        episode_id="ep0",
        environment_facts={"location": "workbench"},
        selected_set_signature="",
    )
    card = RoutingFeatureSnapshot(
        memory_id="test_mem",
        active_payload_version=1,
        goal_summary="test goal",
        task_tags=["test"],
        precondition_summary="none",
        postcondition_summary="none",
        required_environment_facts={},
        forbidden_environment_facts={},
        compatible_receiver_roles=["planner"],
        compatible_receiver_capabilities=["planning"],
        execution_success_alpha=1.0,
        execution_success_beta=1.0,
        execution_success_count=1,
        execution_failure_count=0,
        paired_positive_transfer_count=0,
        paired_negative_transfer_count=0,
        paired_neutral_transfer_count=0,
    )
    return TransferPredictionInput(context=ctx, candidate_card=card, selected_cards=[])


class TestA1FeatureBlock:
    """Test that A1 feature block excludes selected-set and interaction tokens."""

    def test_no_selected_set_tokens(self):
        """context_plus_candidate block has no selected_* tokens (except selected_count:)."""
        encoder = HashingTransferFeatureEncoder(feature_block="context_plus_candidate")
        item = _make_prediction_input()
        tokens = encoder.tokens(item)
        selected_tokens = [
            t for t in tokens
            if t.startswith("selected_") and not t.startswith("selected_count:")
        ]
        assert not selected_tokens

    def test_no_interaction_tokens(self):
        """context_plus_candidate block has no interaction_* tokens."""
        encoder = HashingTransferFeatureEncoder(feature_block="context_plus_candidate")
        item = _make_prediction_input()
        tokens = encoder.tokens(item)
        interaction_tokens = [t for t in tokens if t.startswith("interaction_")]
        assert not interaction_tokens

    def test_has_context_tokens(self):
        """context_plus_candidate block includes context tokens."""
        encoder = HashingTransferFeatureEncoder(feature_block="context_plus_candidate")
        item = _make_prediction_input()
        tokens = encoder.tokens(item)
        context_tokens = [
            t
            for t in tokens
            if t.startswith("task_tag:") or t.startswith("receiver_role:")
        ]
        assert len(context_tokens) > 0

    def test_has_candidate_tokens(self):
        """context_plus_candidate block includes candidate tokens."""
        encoder = HashingTransferFeatureEncoder(feature_block="context_plus_candidate")
        item = _make_prediction_input()
        tokens = encoder.tokens(item)
        cand_tokens = [t for t in tokens if t.startswith("cand_")]
        assert len(cand_tokens) > 0


@pytest.mark.skipif(not A1_CHECKPOINT.exists(), reason="A1 checkpoint not yet trained")
class TestA1Checkpoint:
    """Test A1 checkpoint loading and compatibility."""

    def test_checkpoint_loads(self):
        """A1 checkpoint can be loaded by ProductionSequentialRouter."""
        critic = FourOutcomeTransferCritic.load(A1_CHECKPOINT)
        assert critic is not None
        assert len(critic.models) > 0

    def test_metadata_consistent(self):
        """A1 metadata matches training requirements."""
        if not A1_METADATA.exists():
            pytest.skip("A1 metadata not found")
        metadata = json.loads(A1_METADATA.read_text())
        assert metadata["feature_block"] == "context_plus_candidate"
        assert metadata["selected_set_features_enabled"] is False
        assert metadata["pairwise_features_enabled"] is False
        assert metadata["train_record_count"] == 160  # Same as M0

    def test_feature_block_override_removed(self):
        """Factory no longer mutates checkpoint feature blocks."""
        from smtr.router.factory import build_router

        with pytest.raises(TypeError):
            build_router(
                mode="learned",
                critic_checkpoint=str(A1_CHECKPOINT),
                feature_block="context_plus_candidate",
            )
