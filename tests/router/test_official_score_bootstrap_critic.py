"""Tests for BootstrapOfficialScoreTransferCritic (RIMA-v2 §41).

Covers:
    test_bootstrap_returns_mu_and_nonnegative_sigma
    test_point_prediction_matches_distribution_mean
    test_cluster_bootstrap_uses_task_ids
    test_self_transfer_prediction_is_invalid
    test_invalid_training_examples_are_excluded
    test_frozen_critic_cannot_refit
    test_checkpoint_roundtrip_preserves_distribution
    test_same_seed_reproduces_same_predictions
    test_n_bootstrap_members_equals_requested_count
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from smtr.rima.features import (
    ReceiverConditionedTransferFeatures,
    RimaFeatureEncoder,
)
from smtr.router.official_score_transfer_critic import (
    BootstrapOfficialScoreTransferCritic,
    MatchedInterventionExample,
    TransferEffectDistribution,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_features(
    task_id: str = "t1",
    memory_id: str = "m1",
    receiver_id: str = "r1",
) -> ReceiverConditionedTransferFeatures:
    return ReceiverConditionedTransferFeatures(
        task_id=task_id,
        memory_id=memory_id,
        receiver_id=receiver_id,
        task_repr={"scenario": "test", "task_type": "test", "text": "test task"},
        receiver_repr={"role": "executor", "capabilities": ["coding"]},
        routing_card={
            "goal_summary": "goal",
            "task_tags": ["test"],
            "precondition_summary": "",
            "compatible_receiver_roles": ["executor"],
            "compatible_receiver_capabilities": ["coding"],
            "procedure_type": "experience",
        },
    )


def _make_example(
    task_id: str = "t1",
    memory_id: str = "m1",
    receiver_id: str = "r1",
    source_agent_id: str = "w1",
    expose_score: float | None = 0.8,
    withhold_score: float | None = 0.4,
) -> MatchedInterventionExample:
    features = _make_features(task_id, memory_id, receiver_id)
    return MatchedInterventionExample(
        task_id=task_id,
        memory_id=memory_id,
        receiver_id=receiver_id,
        source_agent_id=source_agent_id,
        official_expose_score=expose_score,
        official_withhold_score=withhold_score,
        features=features,
    )


def _make_training_set(n_tasks: int = 4, n_per_task: int = 3) -> list[MatchedInterventionExample]:
    """Create a small training set with multiple tasks and examples per task."""
    examples = []
    for t in range(n_tasks):
        for i in range(n_per_task):
            examples.append(
                _make_example(
                    task_id=f"task_{t}",
                    memory_id=f"mem_{t}_{i}",
                    receiver_id=f"recv_{i}",
                    source_agent_id=f"src_{t}_{i}",
                    expose_score=0.6 + 0.1 * i,
                    withhold_score=0.3 + 0.05 * i,
                )
            )
    return examples


def _fit_critic(
    examples: list[MatchedInterventionExample] | None = None,
    n_bootstrap: int = 5,
    seed: int = 0,
) -> BootstrapOfficialScoreTransferCritic:
    encoder = RimaFeatureEncoder(n_features=128, include_receiver=True)
    critic = BootstrapOfficialScoreTransferCritic(
        encoder=encoder, n_bootstrap=n_bootstrap, seed=seed
    )
    if examples is None:
        examples = _make_training_set()
    critic.fit(examples)
    return critic


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_bootstrap_returns_mu_and_nonnegative_sigma():
    """Distribution must return finite mu_tau and sigma_tau >= 0."""
    critic = _fit_critic()
    ex = _make_example(task_id="t_new", memory_id="m_new", receiver_id="recv_0")
    dist = critic.predict_distribution(ex)

    assert isinstance(dist, TransferEffectDistribution)
    assert dist.mu_tau is not None
    assert dist.sigma_tau is not None
    assert dist.sigma_tau >= 0.0
    assert dist.mu_expose is not None
    assert dist.mu_withhold is not None
    assert dist.n_members == 5


def test_point_prediction_matches_distribution_mean():
    """predict_one().tau_hat must equal predict_distribution().mu_tau."""
    critic = _fit_critic()
    ex = _make_example(task_id="t_new", memory_id="m_new", receiver_id="recv_0")

    pred = critic.predict_one(ex)
    dist = critic.predict_distribution(ex)

    assert pred.tau_hat is not None
    assert dist.mu_tau is not None
    assert abs(pred.tau_hat - dist.mu_tau) < 1e-12
    assert pred.mu_expose == dist.mu_expose
    assert pred.mu_withhold == dist.mu_withhold


def test_cluster_bootstrap_uses_task_ids():
    """Training stats must report bootstrap_cluster_unit == 'task_id'."""
    examples = _make_training_set(n_tasks=5, n_per_task=2)
    critic = _fit_critic(examples, n_bootstrap=3)

    stats = critic._training_stats
    assert stats["bootstrap_cluster_unit"] == "task_id"
    assert stats["n_unique_tasks"] == 5
    assert stats["n_bootstrap"] == 3


def test_self_transfer_prediction_is_invalid():
    """Self-transfer (source == receiver) must produce None predictions."""
    critic = _fit_critic()
    # source_agent_id == receiver_id => self-transfer
    ex = _make_example(
        task_id="t1",
        memory_id="m1",
        receiver_id="agent_a",
        source_agent_id="agent_a",
    )
    dist = critic.predict_distribution(ex)
    assert dist.mu_tau is None
    assert dist.sigma_tau is None
    assert dist.mu_expose is None
    assert dist.mu_withhold is None

    pred = critic.predict_one(ex)
    assert pred.tau_hat is None
    assert not pred.admitted


def test_invalid_training_examples_are_excluded():
    """Examples with None scores must be excluded from training."""
    examples = _make_training_set()
    # Add invalid examples (None scores)
    examples.append(
        _make_example(
            task_id="t_invalid",
            memory_id="m_invalid",
            receiver_id="recv_0",
            expose_score=None,
            withhold_score=0.5,
        )
    )
    examples.append(
        _make_example(
            task_id="t_invalid2",
            memory_id="m_invalid2",
            receiver_id="recv_0",
            expose_score=0.5,
            withhold_score=None,
        )
    )
    critic = _fit_critic(examples)
    stats = critic._training_stats
    assert stats["invalid_excluded"] == 2
    # The valid examples from _make_training_set should still be used
    assert stats["n_examples_used"] == len(_make_training_set())


def test_frozen_critic_cannot_refit():
    """Frozen critic must raise RuntimeError on fit()."""
    critic = _fit_critic()
    critic.freeze()
    assert critic.is_frozen

    with pytest.raises(RuntimeError, match="frozen"):
        critic.fit(_make_training_set())


def test_checkpoint_roundtrip_preserves_distribution():
    """save() + load() must produce identical predictions."""
    critic = _fit_critic(n_bootstrap=7)
    critic.freeze()

    ex = _make_example(task_id="t_test", memory_id="m_test", receiver_id="recv_1")
    dist_before = critic.predict_distribution(ex)

    with tempfile.TemporaryDirectory() as tmp_dir:
        path = str(Path(tmp_dir) / "critic_bootstrap.joblib")
        sha_saved = critic.save(path)
        loaded = BootstrapOfficialScoreTransferCritic.load(path)

    assert loaded.is_frozen
    assert loaded.n_bootstrap == 7
    assert len(loaded.members) == 7

    dist_after = loaded.predict_distribution(ex)

    assert dist_before.mu_tau == pytest.approx(dist_after.mu_tau, abs=1e-10)
    assert dist_before.sigma_tau == pytest.approx(dist_after.sigma_tau, abs=1e-10)
    assert dist_before.mu_expose == pytest.approx(dist_after.mu_expose, abs=1e-10)
    assert dist_before.mu_withhold == pytest.approx(dist_after.mu_withhold, abs=1e-10)
    assert dist_before.n_members == dist_after.n_members

    # Loaded critic must produce a stable, non-empty SHA
    sha = loaded.checkpoint_sha256()
    assert isinstance(sha, str) and len(sha) == 64


def test_same_seed_reproduces_same_predictions():
    """Two critics trained with the same seed must produce identical predictions."""
    examples = _make_training_set()

    critic_a = _fit_critic(examples, n_bootstrap=5, seed=42)
    critic_b = _fit_critic(examples, n_bootstrap=5, seed=42)

    ex = _make_example(task_id="t_repro", memory_id="m_repro", receiver_id="recv_0")
    dist_a = critic_a.predict_distribution(ex)
    dist_b = critic_b.predict_distribution(ex)

    assert dist_a.mu_tau == pytest.approx(dist_b.mu_tau, abs=1e-10)
    assert dist_a.sigma_tau == pytest.approx(dist_b.sigma_tau, abs=1e-10)
    assert dist_a.n_members == dist_b.n_members


def test_n_bootstrap_members_equals_requested_count():
    """Number of trained members must match n_bootstrap parameter."""
    for n in (3, 7, 11):
        critic = _fit_critic(n_bootstrap=n)
        assert len(critic.members) == n
        assert critic._training_stats["n_bootstrap"] == n
