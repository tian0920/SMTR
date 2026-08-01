"""Tests for critic bootstrap robustness."""

from __future__ import annotations

import numpy as np
import pytest

from smtr.core.types import AgentProfile, CandidateExposureInput, MemoryRoutingCard, ReceiverState
from smtr.router.transfer_critic import FourOutcomeTransferCritic, _stratified_bootstrap_indices


def _make_inputs_and_labels(n: int = 20):
    """Create synthetic inputs with multiple classes."""
    inputs = []
    labels = []
    for i in range(n):
        writer = AgentProfile(agent_id=f"w{i}", role="planner" if i % 2 == 0 else "executor")
        receiver = AgentProfile(agent_id=f"r{i}", role="executor")
        card = MemoryRoutingCard(
            memory_id=f"m{i}",
            goal_summary=f"Goal {i}",
            task_tags=("database",),
            writer=writer,
            source_task_id=f"t{i}",
            source_scenario="database",
        )
        rs = ReceiverState(task_id=f"target{i}", scenario="database", task_instruction="test", receiver=receiver)
        inputs.append(CandidateExposureInput(receiver_state=rs, candidate_card=card))
        labels.append(["positive_transfer", "negative_transfer", "neutral_success", "neutral_failure"][i % 4])
    return inputs, labels


def test_single_class_raises():
    """Single-class training data must raise a clear error."""
    inputs, _ = _make_inputs_and_labels(10)
    labels = ["positive_transfer"] * 10
    critic = FourOutcomeTransferCritic(n_bootstrap=3, n_features=64)
    with pytest.raises(ValueError, match="at least two"):
        critic.fit(inputs, labels)


def test_stratified_bootstrap_all_classes():
    """Stratified bootstrap must include all classes in every sample."""
    y = np.array([0, 0, 0, 1, 1, 2, 2, 3, 3, 3])
    rng = np.random.default_rng(42)
    for _ in range(50):
        idx = _stratified_bootstrap_indices(y, rng)
        y_boot = y[idx]
        assert set(np.unique(y_boot)) == {0, 1, 2, 3}


def test_predict_sums_to_one():
    """Predict output probabilities must sum to 1."""
    inputs, labels = _make_inputs_and_labels(40)
    critic = FourOutcomeTransferCritic(n_bootstrap=5, n_features=64, seed=0)
    critic.fit(inputs, labels)
    pred = critic.predict(inputs[0])
    total = pred.q00_neutral_failure + pred.q01_negative_transfer + pred.q10_positive_transfer + pred.q11_neutral_success
    assert abs(total - 1.0) < 1e-6


def test_checkpoint_feature_block_restored(tmp_path):
    """Checkpoint must correctly restore feature_block."""
    inputs, labels = _make_inputs_and_labels(20)
    critic = FourOutcomeTransferCritic(n_bootstrap=3, n_features=64, feature_block="no_writer_receiver", seed=0)
    critic.fit(inputs, labels)
    path = tmp_path / "test.joblib"
    critic.save(path)
    loaded = FourOutcomeTransferCritic.load(path)
    assert loaded.feature_block == "no_writer_receiver"
