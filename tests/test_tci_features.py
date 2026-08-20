"""Tests for TCI structural feature encoder (Task 9).

Tests:
  - memory_id invariance (same structure, different id → same features)
  - changed factor sensitivity (env constraint change → feature distance > 0)
  - receiver interaction (same memory, different receivers → different interaction)
  - no hash leakage (metadata contains no id/digest/hash)
  - feature shift (zero_shift_rate < 1 for valid contrasts)
  - build_tci_pairs_with_features end-to-end
"""

from __future__ import annotations

import pytest
import numpy as np

from smtr.router.tci_features import (
    TCIFeatureEncoder,
    TCIFeature,
    interaction_feature,
    validate_no_identity_leakage,
    TCI_FEATURE_DIM,
)
from smtr.router.memory_card_features import MemoryCardFeatureEncoder
from smtr.router.receiver_features import ReceiverFeatureEncoder
from smtr.router.task_features import TaskFeatureEncoder
from smtr.router.tci_metrics import compute_feature_shift


# ── Helpers ──


def _make_card(**overrides):
    """Build a minimal memory card dict."""
    card = {
        "memory_id": "mem_001",
        "goal_summary": "database query procedure",
        "task_tags": ["database"],
        "required_tools": ["sql_execute"],
        "required_capabilities": ["database_query"],
        "execution_role_tags": ["executor"],
        "environment_constraints": ["database_environment"],
        "precondition_tags": [],
        "procedure_type": "observed_actions",
        "procedure_length_bucket": "medium",
        "read_write_scope": "read_write",
    }
    card.update(overrides)
    return card


def _make_receiver(**overrides):
    """Build a minimal receiver context dict."""
    ctx = {
        "receiver_role": "executor",
        "receiver_capabilities": ["database_query"],
        "receiver_tool_names": ["sql_execute"],
        "environment_signature": ["database_environment"],
    }
    ctx.update(overrides)
    return ctx


def _make_task(**overrides):
    """Build a minimal task context dict."""
    ctx = {
        "task_instruction": "execute database query with joins",
        "scenario": "database",
        "task_tags": ["database"],
    }
    ctx.update(overrides)
    return ctx


# ── Tests ──


class TestMemoryIdInvariance:
    """Two memories with different ids but same structure → same features."""

    def test_same_structure_different_id(self):
        enc = TCIFeatureEncoder(feature_dim=128, seed=42)
        card_a = _make_card(memory_id="mem_a")
        card_b = _make_card(memory_id="mem_b")
        recv = _make_receiver()
        task = _make_task()

        fa = enc.encode(card_a, recv, task)
        fb = enc.encode(card_b, recv, task)

        assert fa.vector == fb.vector, (
            "Features must be identical when structure is the same, "
            "regardless of memory_id"
        )

    def test_different_precondition(self):
        enc = TCIFeatureEncoder(feature_dim=128, seed=42)
        card_a = _make_card(precondition_tags=["admin_role_required"])
        card_b = _make_card(precondition_tags=["multi_region_consensus"])
        recv = _make_receiver()
        task = _make_task()

        fa = enc.encode(card_a, recv, task)
        fb = enc.encode(card_b, recv, task)

        dist = np.linalg.norm(np.array(fa.vector) - np.array(fb.vector))
        assert dist > 0, "Features must differ when preconditions differ"


class TestChangedFactorSensitivity:
    """Changing environment_constraint → feature distance > 0."""

    def test_environment_change(self):
        enc = TCIFeatureEncoder(feature_dim=128, seed=42)
        card_base = _make_card(
            environment_constraints=["database_environment"]
        )
        card_perturbed = _make_card(
            environment_constraints=[
                "database_environment",
                "gpu_acceleration_required",
            ]
        )
        recv = _make_receiver()
        task = _make_task()

        f_base = enc.encode(card_base, recv, task)
        f_pert = enc.encode(card_perturbed, recv, task)

        dist = np.linalg.norm(
            np.array(f_base.vector) - np.array(f_pert.vector)
        )
        assert dist > 0, "Feature must change when env constraint added"

    def test_precondition_change(self):
        enc = TCIFeatureEncoder(feature_dim=128, seed=42)
        card_base = _make_card(precondition_tags=[])
        card_perturbed = _make_card(
            precondition_tags=["exclusive_table_lock"]
        )
        recv = _make_receiver()
        task = _make_task()

        f_base = enc.encode(card_base, recv, task)
        f_pert = enc.encode(card_perturbed, recv, task)

        dist = np.linalg.norm(
            np.array(f_base.vector) - np.array(f_pert.vector)
        )
        assert dist > 0, "Feature must change when precondition added"


class TestReceiverInteraction:
    """Same memory, different receivers → different interaction features."""

    def test_different_receivers(self):
        enc = TCIFeatureEncoder(feature_dim=128, seed=42)
        card = _make_card()
        recv_a = _make_receiver(
            receiver_capabilities=["database_query", "planning"]
        )
        recv_b = _make_receiver(
            receiver_capabilities=["communication", "coordination"]
        )
        task = _make_task()

        fa = enc.encode(card, recv_a, task)
        fb = enc.encode(card, recv_b, task)

        dist = np.linalg.norm(np.array(fa.vector) - np.array(fb.vector))
        assert dist > 0, (
            "Interaction features must differ for different receivers"
        )


class TestNoHashLeakage:
    """Feature metadata must not contain id, digest, or hash fields."""

    def test_no_forbidden_keys(self):
        enc = TCIFeatureEncoder(feature_dim=128, seed=42)
        card = _make_card()
        recv = _make_receiver()
        task = _make_task()

        feat = enc.encode(card, recv, task)

        forbidden = {
            "memory_id", "agent_id", "task_id",
            "digest", "hash", "perturbation_id",
        }
        for key in forbidden:
            assert key not in feat.metadata, (
                f"Forbidden key '{key}' leaked into feature metadata"
            )

    def test_validate_no_leakage_clean(self):
        feat = TCIFeature(
            vector=[1.0, 2.0, 3.0],
            feature_names=["a", "b", "c"],
            metadata={"perturbation_type": "precondition"},
        )
        assert validate_no_identity_leakage(feat) is True

    def test_validate_no_leakage_dirty(self):
        feat = TCIFeature(
            vector=[1.0, 2.0, 3.0],
            feature_names=["a", "b", "c"],
            metadata={"memory_id": "mem_001"},
        )
        with pytest.raises(ValueError, match="leakage"):
            validate_no_identity_leakage(feat)


class TestFeatureDimension:
    """Output dimension must be exactly feature_dim."""

    def test_default_dim(self):
        enc = TCIFeatureEncoder(feature_dim=128, seed=42)
        feat = enc.encode(_make_card(), _make_receiver(), _make_task())
        assert len(feat.vector) == 128

    def test_custom_dim(self):
        enc = TCIFeatureEncoder(feature_dim=64, seed=42)
        feat = enc.encode(_make_card(), _make_receiver(), _make_task())
        assert len(feat.vector) == 64


class TestFeatureShift:
    """Feature shift analysis for valid contrasts."""

    def test_shift_with_structural_features(self):
        """Pairs with different preconditions should have non-zero shift."""
        enc = TCIFeatureEncoder(feature_dim=128, seed=42)
        card_orig = _make_card(precondition_tags=[])
        card_pert = _make_card(
            precondition_tags=["admin_role_required"]
        )
        recv = _make_receiver()
        task = _make_task()

        f_orig = enc.encode(card_orig, recv, task)
        f_pert = enc.encode(card_pert, recv, task)

        # Build a mock TCIPair.
        from smtr.router.tci_dataset import TCIPair
        pair = TCIPair(
            perturbation_id="test",
            task_id="1",
            receiver_agent_id="agent1",
            candidate_memory_id="mem1",
            perturbation_type="precondition",
            changed_field="precondition_tags",
            y0=1, y_original=1, y_perturbed=0,
            effect_original=0, effect_perturbed=-1,
            direction=1,
            contrast_type="induced_damage",
            original_features=tuple(f_orig.vector),
            perturbed_features=tuple(f_pert.vector),
        )

        shift = compute_feature_shift([pair])
        assert shift["mean_shift"] > 0
        assert shift["zero_shift_rate"] == 0.0

    def test_shift_no_features(self):
        """Pairs without structural features should have zero shift."""
        from smtr.router.tci_dataset import TCIPair
        pair = TCIPair(
            perturbation_id="test",
            task_id="1",
            receiver_agent_id="agent1",
            candidate_memory_id="mem1",
            perturbation_type="precondition",
            changed_field="precondition_tags",
            y0=1, y_original=1, y_perturbed=0,
            effect_original=0, effect_perturbed=-1,
            direction=1,
            contrast_type="induced_damage",
        )
        shift = compute_feature_shift([pair])
        assert shift["zero_shift_rate"] == 1.0


class TestInteractionFeature:
    """Element-wise product interaction."""

    def test_basic_interaction(self):
        r = np.array([1.0, 2.0, 3.0])
        m = np.array([4.0, 5.0, 6.0])
        result = interaction_feature(r, m)
        np.testing.assert_array_equal(result, [4.0, 10.0, 18.0])

    def test_unequal_dims(self):
        r = np.array([1.0, 2.0])
        m = np.array([3.0, 4.0, 5.0])
        result = interaction_feature(r, m)
        assert len(result) == 3  # padded to memory dim
        np.testing.assert_array_equal(result[:2], [3.0, 8.0])
        assert result[2] == 0.0  # padded


class TestBuildPairsWithFeatures:
    """End-to-end test of build_tci_pairs_with_features."""

    def test_build_with_features(self):
        from smtr.intervention.intervention_contrast import (
            InterventionContrast,
        )
        from smtr.router.tci_dataset import build_tci_pairs_with_features

        contrast = InterventionContrast(
            perturbation_id="pert_001",
            task_id="1",
            receiver_agent_id="agent1",
            candidate_memory_id="mem1",
            perturbation_type="precondition",
            changed_field="precondition_tags",
            y0=1, y_original=1, y_perturbed=0,
            effect_original=0, effect_perturbed=-1,
            contrast_direction=1,
            source_record_digest="edge_001",
            original_memory_digest="abc",
            perturbed_memory_digest="def",
        )

        enc = TCIFeatureEncoder(feature_dim=128, seed=42)
        original_cards = {
            "mem1": _make_card(precondition_tags=[]),
        }
        perturbed_cards = {
            "pert_001": _make_card(
                precondition_tags=["admin_role_required"]
            ),
        }
        receiver_contexts = {"agent1": _make_receiver()}
        task_contexts = {"1": _make_task()}

        pairs = build_tci_pairs_with_features(
            [contrast],
            feature_encoder=enc,
            original_cards=original_cards,
            perturbed_cards=perturbed_cards,
            receiver_contexts=receiver_contexts,
            task_contexts=task_contexts,
        )

        assert len(pairs) == 1
        p = pairs[0]
        assert p.has_structural_features
        assert len(p.original_features) == 128
        assert len(p.perturbed_features) == 128
        # Features should differ (precondition changed).
        assert p.original_features != p.perturbed_features


class TestMemoryCardFeatureEncoder:
    """Unit tests for memory card feature encoder."""

    def test_encode_returns_correct_dim(self):
        enc = MemoryCardFeatureEncoder()
        card = _make_card()
        features = enc.encode(card)
        assert len(features) == enc.feature_dim

    def test_unknown_token(self):
        enc = MemoryCardFeatureEncoder()
        card = _make_card(
            precondition_tags=["unknown_precondition_xyz"]
        )
        features = enc.encode(card)
        # The <unk> slot in precondition block should be 1.
        assert features[len(enc.precondition_vocab)] == 1.0

    def test_multi_hot(self):
        enc = MemoryCardFeatureEncoder()
        card = _make_card(
            precondition_tags=[
                "admin_role_required",
                "multi_region_consensus",
            ]
        )
        features = enc.encode(card)
        # First two precondition slots should be 1.
        assert features[0] == 1.0
        assert features[1] == 1.0
        assert features[2] == 0.0
