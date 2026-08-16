"""Tests for ablation checkpoint separation."""

from __future__ import annotations

import pytest

from smtr.core.types import AgentProfile, CandidateExposureInput, MemoryRoutingCard, ReceiverState
from smtr.router.transfer_critic import FourOutcomeTransferCritic


def _make_inputs_and_labels(n: int = 20):
    inputs = []
    labels = []
    for i in range(n):
        receiver = AgentProfile(agent_id=f"r{i}", role="executor")
        card = MemoryRoutingCard(
            memory_id=f"m{i}", goal_summary=f"Goal {i}", task_tags=("database",),
        )
        rs = ReceiverState(task_id=f"target{i}", scenario="database", task_instruction="test", receiver=receiver)
        inputs.append(CandidateExposureInput(receiver_state=rs, candidate_card=card))
        labels.append(["positive_transfer", "negative_transfer", "neutral_success", "neutral_failure"][i % 4])
    return inputs, labels


def test_full_checkpoint_not_no_compatibility_interaction(tmp_path):
    """Full checkpoint must not be usable as no_compatibility_interaction checkpoint."""
    inputs, labels = _make_inputs_and_labels(20)
    critic = FourOutcomeTransferCritic(n_bootstrap=3, n_features=64, feature_block="full", seed=0)
    critic.fit(inputs, labels)
    path = tmp_path / "full.joblib"
    critic.save(path)
    loaded = FourOutcomeTransferCritic.load(path)
    assert loaded.feature_block == "full"
    assert loaded.feature_block != "no_compatibility_interaction"


def test_no_compatibility_interaction_checkpoint_not_full(tmp_path):
    """no_compatibility_interaction checkpoint must not be usable as full checkpoint."""
    inputs, labels = _make_inputs_and_labels(20)
    critic = FourOutcomeTransferCritic(
        n_bootstrap=3, n_features=64, feature_block="no_compatibility_interaction", seed=0
    )
    critic.fit(inputs, labels)
    path = tmp_path / "no_compat.joblib"
    critic.save(path)
    loaded = FourOutcomeTransferCritic.load(path)
    assert loaded.feature_block == "no_compatibility_interaction"
    assert loaded.feature_block != "full"


def test_smtr_no_compatibility_interaction_uses_separate_critic():
    """SMTR-no-compatibility-interaction must use its own critic and reject others."""
    from smtr.router.baselines import SMTRNoCompatibilityInteractionRouter

    inputs, labels = _make_inputs_and_labels(20)
    critic = FourOutcomeTransferCritic(
        n_bootstrap=3, n_features=64, feature_block="no_compatibility_interaction", seed=0
    )
    critic.fit(inputs, labels)
    SMTRNoCompatibilityInteractionRouter(critic=critic)

    # A full-block critic must be rejected (checkpoint separation).
    full_critic = FourOutcomeTransferCritic(n_bootstrap=3, n_features=64, feature_block="full", seed=0)
    full_critic.fit(inputs, labels)
    with pytest.raises(ValueError, match="feature block mismatch"):
        SMTRNoCompatibilityInteractionRouter(critic=full_critic)
