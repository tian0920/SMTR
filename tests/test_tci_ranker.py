"""Tests for TCI ranker and offline evaluator (Tasks 13-15).

Verifies:
  - Pairwise loss computation.
  - Ranker training convergence.
  - Offline metrics (accuracy, margin, regret).
  - Checkpoint save/load.
  - Operator breakdown.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from smtr.router.tci_ranker import (
    TCIRanker,
    TCIRankerCheckpoint,
    TCIRankerConfig,
    score_memory,
    tci_pairwise_loss,
)
from smtr.router.tci_metrics import (
    TCIMetrics,
    compute_regret,
    evaluate_tci_ranker,
    top1_transfer_effect_hit_rate,
)


# ──────────────────────────────────────────────────────────────
# Pairwise loss tests
# ──────────────────────────────────────────────────────────────
class TestPairwiseLoss:
    def test_zero_margin(self) -> None:
        """When scores are equal, loss = log(2)."""
        s_orig = np.array([1.0])
        s_pert = np.array([1.0])
        dirs = np.array([1.0])
        loss = tci_pairwise_loss(s_orig, s_pert, dirs)
        assert loss == pytest.approx(np.log(2), abs=1e-5)

    def test_correct_ordering_low_loss(self) -> None:
        """When direction * (s_orig - s_pert) > 0, loss should be small."""
        s_orig = np.array([5.0])
        s_pert = np.array([0.0])
        dirs = np.array([1.0])
        loss = tci_pairwise_loss(s_orig, s_pert, dirs)
        assert loss < 0.01

    def test_wrong_ordering_high_loss(self) -> None:
        """When direction * (s_orig - s_pert) < 0, loss should be large."""
        s_orig = np.array([0.0])
        s_pert = np.array([5.0])
        dirs = np.array([1.0])
        loss = tci_pairwise_loss(s_orig, s_pert, dirs)
        assert loss > 4.0

    def test_negative_direction(self) -> None:
        """Negative direction reverses the expected ordering."""
        s_orig = np.array([0.0])
        s_pert = np.array([5.0])
        dirs = np.array([-1.0])
        loss = tci_pairwise_loss(s_orig, s_pert, dirs)
        assert loss < 0.01

    def test_batch_loss(self) -> None:
        """Batch loss averages over all pairs."""
        s_orig = np.array([5.0, 0.0])
        s_pert = np.array([0.0, 5.0])
        dirs = np.array([1.0, 1.0])
        loss = tci_pairwise_loss(s_orig, s_pert, dirs)
        assert loss > 0


# ──────────────────────────────────────────────────────────────
# Score memory tests
# ──────────────────────────────────────────────────────────────
class TestScoreMemory:
    def test_linear_score(self) -> None:
        features = np.array([1.0, 0.0, 1.0])
        weights = np.array([0.5, 0.3, 0.2])
        bias = 0.1
        s = score_memory(features, weights, bias)
        assert s == pytest.approx(0.8)

    def test_zero_weights(self) -> None:
        features = np.array([1.0, 2.0])
        weights = np.zeros(2)
        s = score_memory(features, weights, 0.5)
        assert s == pytest.approx(0.5)


# ──────────────────────────────────────────────────────────────
# Ranker training tests
# ──────────────────────────────────────────────────────────────
class TestTCIRanker:
    def test_training_reduces_loss(self) -> None:
        """Training should reduce loss from initial value."""
        rng = np.random.RandomState(42)
        n, d = 50, 16
        feat_orig = rng.randn(n, d)
        feat_pert = feat_orig + rng.randn(n, d) * 0.1
        dirs = np.ones(n)

        config = TCIRankerConfig(
            feature_dim=d, learning_rate=0.1, n_epochs=100, seed=42
        )
        ranker = TCIRanker(config)
        result = ranker.train(feat_orig, feat_pert, dirs)

        assert result["final_loss"] < result["loss_history"][0]

    def test_empty_training(self) -> None:
        """Empty input should return zero loss."""
        config = TCIRankerConfig(feature_dim=4)
        ranker = TCIRanker(config)
        result = ranker.train(
            np.zeros((0, 4)), np.zeros((0, 4)), np.zeros(0)
        )
        assert result["final_loss"] == 0.0
        assert result["n_pairs"] == 0

    def test_checkpoint_save_load(self) -> None:
        """Checkpoint round-trip preserves weights."""
        config = TCIRankerConfig(feature_dim=8, seed=7)
        ranker = TCIRanker(config)
        ranker.weights = np.ones(8) * 0.5
        ranker.bias = 0.3

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "ckpt.joblib"
            ranker.to_checkpoint().save(path)
            loaded = TCIRankerCheckpoint.load(path)

        assert np.allclose(loaded.weights, ranker.weights)
        assert loaded.bias == pytest.approx(ranker.bias)

    def test_from_checkpoint(self) -> None:
        """Load ranker from checkpoint preserves state."""
        config = TCIRankerConfig(feature_dim=4, seed=7)
        ranker = TCIRanker(config)
        ranker.weights = np.array([1.0, 2.0, 3.0, 4.0])
        ranker.bias = 0.5

        ckpt = ranker.to_checkpoint()
        loaded = TCIRanker.from_checkpoint(ckpt)
        assert np.allclose(loaded.weights, ranker.weights)
        assert loaded.bias == pytest.approx(ranker.bias)


# ──────────────────────────────────────────────────────────────
# Offline evaluator tests
# ──────────────────────────────────────────────────────────────
class TestTCIMetrics:
    def test_perfect_accuracy(self) -> None:
        """Correct ordering → accuracy = 1.0."""
        s_orig = np.array([5.0, 6.0, 7.0])
        s_pert = np.array([0.0, 1.0, 2.0])
        dirs = np.array([1.0, 1.0, 1.0])
        metrics = evaluate_tci_ranker(s_orig, s_pert, dirs)
        assert metrics.pairwise_accuracy == pytest.approx(1.0)
        assert metrics.n_pairs == 3

    def test_random_accuracy(self) -> None:
        """Wrong ordering → accuracy < 1.0."""
        s_orig = np.array([0.0])
        s_pert = np.array([5.0])
        dirs = np.array([1.0])
        metrics = evaluate_tci_ranker(s_orig, s_pert, dirs)
        assert metrics.pairwise_accuracy == pytest.approx(0.0)

    def test_operator_breakdown(self) -> None:
        """Operator breakdown counts and accuracy."""
        s_orig = np.array([5.0, 0.0, 5.0])
        s_pert = np.array([0.0, 5.0, 0.0])
        dirs = np.array([1.0, 1.0, 1.0])
        ops = ["precondition", "required_tool", "precondition"]
        metrics = evaluate_tci_ranker(s_orig, s_pert, dirs, ops)

        assert metrics.operator_count["precondition"] == 2
        assert metrics.operator_count["required_tool"] == 1
        assert metrics.operator_accuracy["precondition"] == pytest.approx(1.0)
        assert metrics.operator_accuracy["required_tool"] == pytest.approx(0.0)

    def test_empty_input(self) -> None:
        metrics = evaluate_tci_ranker(
            np.array([]), np.array([]), np.array([])
        )
        assert metrics.n_pairs == 0
        assert metrics.pairwise_accuracy == 0.0

    def test_pairwise_margin(self) -> None:
        """Margin should be mean of d*(s_m - s_m~)."""
        s_orig = np.array([3.0, 1.0])
        s_pert = np.array([1.0, 2.0])
        dirs = np.array([1.0, 1.0])
        metrics = evaluate_tci_ranker(s_orig, s_pert, dirs)
        expected_margin = np.mean([2.0, -1.0])
        assert metrics.pairwise_margin == pytest.approx(expected_margin)


class TestRegret:
    def test_no_regret(self) -> None:
        """Perfect ordering → regret = 0."""
        s_orig = np.array([5.0, 6.0])
        s_pert = np.array([0.0, 1.0])
        dirs = np.array([1.0, 1.0])
        regret = compute_regret(s_orig, s_pert, dirs)
        assert regret == pytest.approx(0.0)

    def test_full_regret(self) -> None:
        """All wrong → regret = 1.0."""
        s_orig = np.array([0.0])
        s_pert = np.array([5.0])
        dirs = np.array([1.0])
        regret = compute_regret(s_orig, s_pert, dirs)
        assert regret == pytest.approx(1.0)


class TestTop1HitRate:
    def test_perfect_hit(self) -> None:
        """Top scored = top effect."""
        scores = np.array([[0.1, 0.9], [0.8, 0.2]])
        effects = np.array([[0.0, 1.0], [1.0, 0.0]])
        rate = top1_transfer_effect_hit_rate(scores, effects)
        assert rate == pytest.approx(1.0)

    def test_zero_hit(self) -> None:
        """Top scored ≠ top effect."""
        scores = np.array([[0.9, 0.1]])
        effects = np.array([[0.0, 1.0]])
        rate = top1_transfer_effect_hit_rate(scores, effects)
        assert rate == pytest.approx(0.0)

    def test_empty(self) -> None:
        rate = top1_transfer_effect_hit_rate(np.array([]), np.array([]))
        assert rate == 0.0
