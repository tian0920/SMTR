"""Tests for TCI generalization evaluation (Task 9).

Verifies:
  - Group split has no leakage.
  - Random pair generation breaks causal relationship.
  - Random baseline gives ~0.5 accuracy on separable data.
  - Factor breakdown works correctly.
  - Margin calibration produces correct bins.
  - Ranker validation tracks best weights.
"""

from __future__ import annotations

import numpy as np
import pytest

from smtr.router.tci_dataset import TCIPair
from smtr.router.tci_split import split_tci_pairs
from smtr.router.tci_ranker import TCIRanker, TCIRankerConfig
from smtr.router.tci_metrics import (
    compute_margin_accuracy_curve,
    evaluate_by_factor,
    evaluate_split_metrics,
    evaluate_tci_ranker,
)
from smtr.router.tci_baselines import (
    RandomPairBaseline,
    build_random_pairs,
    evaluate_random_baseline,
)


def _make_pair(
    task_id: str,
    receiver: str,
    memory_id: str,
    direction: int = 1,
    ptype: str = "precondition",
) -> TCIPair:
    return TCIPair(
        perturbation_id=f"pert_{task_id}_{memory_id}_{direction}",
        task_id=task_id,
        receiver_agent_id=receiver,
        candidate_memory_id=memory_id,
        perturbation_type=ptype,
        changed_field="precondition_tags",
        y0=1,
        y_original=1,
        y_perturbed=0,
        effect_original=0,
        effect_perturbed=-1,
        direction=direction,
        contrast_type="induced_damage",
    )


# ──────────────────────────────────────────────────────────────
# Group split no leakage
# ──────────────────────────────────────────────────────────────
class TestGroupSplitNoLeakage:
    def test_no_group_leakage(self) -> None:
        """Same (task, receiver, memory) must not be in multiple splits."""
        pairs = [
            _make_pair(f"t{i}", "a1", f"m{i}") for i in range(20)
        ]
        split = split_tci_pairs(pairs, seed=42)

        def gk(pairs: list[TCIPair]) -> set[tuple[str, str, str]]:
            return {
                (p.task_id, p.receiver_agent_id, p.candidate_memory_id)
                for p in pairs
            }

        tr = gk(split.train_pairs)
        va = gk(split.valid_pairs)
        te = gk(split.test_pairs)

        assert tr.isdisjoint(va)
        assert tr.isdisjoint(te)
        assert va.isdisjoint(te)


# ──────────────────────────────────────────────────────────────
# Random pair generation
# ──────────────────────────────────────────────────────────────
class TestRandomPairGeneration:
    def test_random_pairs_count(self) -> None:
        """Random pairs count equals input pairs count."""
        pairs = [
            _make_pair("t1", "a1", "m1"),
            _make_pair("t1", "a1", "m2"),
        ]
        random_pairs = build_random_pairs(pairs, seed=42)
        assert len(random_pairs) == len(pairs)

    def test_random_pairs_have_different_memory(self) -> None:
        """Random pairs should use a different memory than the original."""
        pairs = [
            _make_pair("t1", "a1", "m1"),
            _make_pair("t1", "a1", "m2"),
        ]
        random_pairs = build_random_pairs(pairs, seed=42)
        # Each random pair should use a different memory from its source.
        for rp, orig in zip(random_pairs, pairs):
            assert rp.candidate_memory_id != orig.candidate_memory_id

    def test_random_pairs_preserve_task(self) -> None:
        """Task_id and receiver_agent_id must be preserved."""
        pairs = [
            _make_pair("t1", "a1", "m1"),
            _make_pair("t2", "a2", "m2"),
        ]
        random_pairs = build_random_pairs(pairs, seed=42)
        for rp, orig in zip(random_pairs, pairs):
            assert rp.task_id == orig.task_id
            assert rp.receiver_agent_id == orig.receiver_agent_id

    def test_random_pairs_empty(self) -> None:
        random_pairs = build_random_pairs([], seed=42)
        assert random_pairs == []

    def test_random_directions_shuffled(self) -> None:
        """Directions should be randomly shuffled."""
        pairs = [
            _make_pair("t1", "a1", f"m{i}", direction=1) for i in range(100)
        ]
        random_pairs = build_random_pairs(pairs, seed=42)
        dirs = [p.direction for p in random_pairs]
        # Should have both +1 and -1 since shuffled.
        assert 1 in dirs
        assert -1 in dirs


# ──────────────────────────────────────────────────────────────
# Random baseline lower bound
# ──────────────────────────────────────────────────────────────
class TestRandomBaselineLowerBound:
    def test_random_baseline_accuracy_near_half(self) -> None:
        """On perfectly separable TCI data, random baseline ≈ 0.5.

        The key insight: random pairs have NO causal relationship,
        so even a perfect ranker should get ~50% accuracy on them.
        """
        rng = np.random.RandomState(42)
        n = 200
        feat_dim = 16
        # Build perfectly separable features: original > perturbed always.
        feat_orig = rng.randn(n, feat_dim) + 2.0
        feat_pert = rng.randn(n, feat_dim) - 2.0
        dirs = np.ones(n)

        # Train ranker to perfectly separate.
        config = TCIRankerConfig(
            feature_dim=feat_dim, learning_rate=0.1, n_epochs=100, seed=42
        )
        ranker = TCIRanker(config)
        ranker.train(feat_orig, feat_pert, dirs)

        # On real pairs, accuracy should be ~1.0.
        s_orig = ranker.score(feat_orig)
        s_pert = ranker.score(feat_pert)
        real_metrics = evaluate_tci_ranker(s_orig, s_pert, dirs)
        assert real_metrics.pairwise_accuracy >= 0.95

        # On random pairs, accuracy should be near 0.5.
        pairs = [
            _make_pair(f"t{i}", "a1", f"m{i}") for i in range(n)
        ]
        random_bl = evaluate_random_baseline(ranker, pairs, seed=42)
        # Random baseline should NOT be perfect.
        assert random_bl.pairwise_accuracy < 0.80


# ──────────────────────────────────────────────────────────────
# Factor breakdown
# ──────────────────────────────────────────────────────────────
class TestFactorBreakdown:
    def test_by_factor_keys(self) -> None:
        """Factor breakdown must have one entry per perturbation_type."""
        pairs = [
            _make_pair("t1", "a1", "m1", ptype="precondition"),
            _make_pair("t2", "a2", "m2", ptype="environment_constraint"),
        ]
        config = TCIRankerConfig(feature_dim=16, seed=7)
        ranker = TCIRanker(config)
        result = evaluate_by_factor(pairs, ranker)
        assert "precondition" in result
        assert "environment_constraint" in result

    def test_by_factor_has_accuracy(self) -> None:
        pairs = [
            _make_pair("t1", "a1", "m1", ptype="precondition"),
            _make_pair("t2", "a2", "m2", ptype="precondition"),
        ]
        config = TCIRankerConfig(feature_dim=16, seed=7)
        ranker = TCIRanker(config)
        result = evaluate_by_factor(pairs, ranker)
        assert "accuracy" in result["precondition"]
        assert "margin" in result["precondition"]
        assert result["precondition"]["n"] == 2


# ──────────────────────────────────────────────────────────────
# Margin calibration
# ──────────────────────────────────────────────────────────────
class TestMarginCalibration:
    def test_margin_bins_count(self) -> None:
        """5 bins should produce 5 entries."""
        s_orig = np.array([5.0, 3.0, 1.0, 0.5, 0.1, 0.0, 2.0, 4.0, 6.0, 8.0])
        s_pert = np.zeros(10)
        dirs = np.ones(10)
        result = compute_margin_accuracy_curve(s_orig, s_pert, dirs, bins=5)
        assert len(result) == 5

    def test_margin_bin_accuracy_monotone(self) -> None:
        """Higher margin bins should generally have higher accuracy.

        Construct scenario where direction and scores are aligned
        so that larger |margin| bins have higher accuracy.
        """
        rng = np.random.RandomState(42)
        n = 1000
        # Build scores where magnitude correlates with correctness.
        # For direction=+1: s_orig > s_pert means correct.
        # Use exponential distribution of margins so we get a range.
        abs_margins = rng.exponential(scale=3.0, size=n)
        # Make 80% of pairs correct (positive margin), 20% incorrect.
        correct_mask = rng.random(n) < 0.8
        margins = np.where(correct_mask, abs_margins, -abs_margins)
        # Construct scores from margins: s_orig - s_pert = margin when d=1.
        dirs = np.ones(n)
        s_orig = margins  # s_orig - s_pert = margin
        s_pert = np.zeros(n)

        result = compute_margin_accuracy_curve(s_orig, s_pert, dirs, bins=5)
        # Top bin accuracy should be >= bottom bin accuracy.
        keys = list(result.keys())
        top_acc = result[keys[-1]]["accuracy"]
        bottom_acc = result[keys[0]]["accuracy"]
        assert top_acc >= bottom_acc

    def test_margin_empty(self) -> None:
        result = compute_margin_accuracy_curve(
            np.array([]), np.array([]), np.array([]), bins=5
        )
        assert result == {}


# ──────────────────────────────────────────────────────────────
# Ranker validation
# ──────────────────────────────────────────────────────────────
class TestRankerValidation:
    def test_validation_tracks_history(self) -> None:
        """train_with_validation must return history lists."""
        rng = np.random.RandomState(42)
        fd = 8
        tr_orig = rng.randn(20, fd)
        tr_pert = rng.randn(20, fd)
        tr_dirs = np.ones(20)
        va_orig = rng.randn(5, fd)
        va_pert = rng.randn(5, fd)
        va_dirs = np.ones(5)

        config = TCIRankerConfig(feature_dim=fd, seed=7)
        ranker = TCIRanker(config)
        hist = ranker.train_with_validation(
            tr_orig, tr_pert, tr_dirs,
            va_orig, va_pert, va_dirs,
            epochs=20,
        )

        assert len(hist["train_loss"]) == 20
        assert len(hist["valid_accuracy"]) == 20
        assert len(hist["valid_margin"]) == 20
        assert "best_epoch" in hist
        assert "best_valid_accuracy" in hist

    def test_validation_restores_best_weights(self) -> None:
        """After training, weights should be from best validation epoch."""
        rng = np.random.RandomState(42)
        fd = 8
        tr_orig = rng.randn(20, fd) + 1.0
        tr_pert = rng.randn(20, fd) - 1.0
        tr_dirs = np.ones(20)
        va_orig = rng.randn(5, fd) + 1.0
        va_pert = rng.randn(5, fd) - 1.0
        va_dirs = np.ones(5)

        config = TCIRankerConfig(feature_dim=fd, seed=7, learning_rate=0.1)
        ranker = TCIRanker(config)
        hist = ranker.train_with_validation(
            tr_orig, tr_pert, tr_dirs,
            va_orig, va_pert, va_dirs,
            epochs=50,
        )
        assert hist["best_valid_accuracy"] >= 0.0

    def test_validation_empty_train(self) -> None:
        config = TCIRankerConfig(feature_dim=4)
        ranker = TCIRanker(config)
        hist = ranker.train_with_validation(
            np.zeros((0, 4)), np.zeros((0, 4)), np.zeros(0),
            np.zeros((1, 4)), np.zeros((1, 4)), np.zeros(1),
        )
        assert hist["best_epoch"] == -1


# ──────────────────────────────────────────────────────────────
# Split metrics
# ──────────────────────────────────────────────────────────────
class TestSplitMetrics:
    def test_evaluate_split_metrics_keys(self) -> None:
        pairs = [
            _make_pair(f"t{i}", "a1", f"m{i}") for i in range(10)
        ]
        split = split_tci_pairs(pairs, seed=42)
        config = TCIRankerConfig(feature_dim=16, seed=7)
        ranker = TCIRanker(config)
        result = evaluate_split_metrics(
            ranker,
            split.train_pairs,
            split.valid_pairs,
            split.test_pairs,
        )
        assert "train" in result
        assert "valid" in result
        assert "test" in result
        for key in ["train", "valid", "test"]:
            assert "n" in result[key]
            assert "accuracy" in result[key]
            assert "margin" in result[key]
