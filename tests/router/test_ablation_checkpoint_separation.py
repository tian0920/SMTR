"""Tests for ablation checkpoint separation."""

from __future__ import annotations

import pytest

from smtr.core.types import AgentProfile, CandidateExposureInput, MemoryRoutingCard, ReceiverState
from smtr.router.transfer_critic import FourOutcomeTransferCritic


def _make_inputs_and_labels(n: int = 20):
    inputs = []
    labels = []
    for i in range(n):
        writer = AgentProfile(agent_id=f"w{i}", role="planner" if i % 2 == 0 else "executor")
        receiver = AgentProfile(agent_id=f"r{i}", role="executor")
        card = MemoryRoutingCard(
            memory_id=f"m{i}", goal_summary=f"Goal {i}", task_tags=("database",),
            writer=writer, source_task_id=f"t{i}", source_scenario="database",
        )
        rs = ReceiverState(task_id=f"target{i}", scenario="database", task_instruction="test", receiver=receiver)
        inputs.append(CandidateExposureInput(receiver_state=rs, candidate_card=card))
        labels.append(["positive_transfer", "negative_transfer", "neutral_success", "neutral_failure"][i % 4])
    return inputs, labels


def test_full_checkpoint_not_no_writer_receiver(tmp_path):
    """Full checkpoint must not be usable as no_writer_receiver checkpoint."""
    inputs, labels = _make_inputs_and_labels(20)
    critic = FourOutcomeTransferCritic(n_bootstrap=3, n_features=64, feature_block="full", seed=0)
    critic.fit(inputs, labels)
    path = tmp_path / "full.joblib"
    critic.save(path)
    loaded = FourOutcomeTransferCritic.load(path)
    assert loaded.feature_block == "full"
    assert loaded.feature_block != "no_writer_receiver"


def test_no_writer_receiver_checkpoint_not_full(tmp_path):
    """no_writer_receiver checkpoint must not be usable as full checkpoint."""
    inputs, labels = _make_inputs_and_labels(20)
    critic = FourOutcomeTransferCritic(n_bootstrap=3, n_features=64, feature_block="no_writer_receiver", seed=0)
    critic.fit(inputs, labels)
    path = tmp_path / "no_wr.joblib"
    critic.save(path)
    loaded = FourOutcomeTransferCritic.load(path)
    assert loaded.feature_block == "no_writer_receiver"
    assert loaded.feature_block != "full"


def test_smtr_no_risk_uses_full_critic():
    """SMTR-no-risk must use full critic, only ignoring eta."""
    from smtr.router.baselines import SMTRNoRiskRouter

    inputs, labels = _make_inputs_and_labels(20)
    critic = FourOutcomeTransferCritic(n_bootstrap=3, n_features=64, feature_block="full", seed=0)
    critic.fit(inputs, labels)
    router = SMTRNoRiskRouter(critic=critic)
    assert router.critic.feature_block == "full"


def test_smtr_no_writer_receiver_uses_separate_critic():
    """SMTR-no-writer-receiver must use a separate no_writer_receiver critic."""
    from smtr.router.baselines import SMTRNoWriterReceiverRouter

    inputs, labels = _make_inputs_and_labels(20)
    critic = FourOutcomeTransferCritic(n_bootstrap=3, n_features=64, feature_block="no_writer_receiver", seed=0)
    critic.fit(inputs, labels)
    router = SMTRNoWriterReceiverRouter(critic=critic)
    assert router.critic.feature_block == "no_writer_receiver"
