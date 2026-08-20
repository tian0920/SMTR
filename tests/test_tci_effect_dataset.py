"""Tests for TCI Effect Dataset and Value Head (Task 11).

Covers:
  - Test 1: Effect computation (beneficial, neutral, harmful)
  - Test 2: TCIEffectExample validation
  - Test 3: TCIEffectBatch properties
  - Test 4: TCIValueHead save/load
  - Test 5: Training mode transitions
  - Test 6: Backward compatibility
  - Test 7: Random effect baseline
  - Test 8: Effect accuracy computation
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from smtr.core.types import (
    AgentProfile,
    CandidateExposureInput,
    MemoryRoutingCard,
    ReceiverState,
)
from smtr.router.random_effect_baseline import (
    build_random_effect_baseline,
)
from smtr.router.tci_effect_builder import (
    build_tci_effect_examples,
    compute_effect_accuracy,
)
from smtr.router.tci_effect_dataset import (
    TCIEffectBatch,
    TCIEffectExample,
)
from smtr.router.transfer_critic import (
    EFFECT_CLASSES,
    FourOutcomeTransferCritic,
    TCIValueHead,
    _VALID_CRITIC_TRAINING_MODES,
)
from smtr.router.transfer_features import HashingTransferFeatureEncoder


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_card(
    *, memory_id: str = "mem-1",
    precondition_tags: tuple[str, ...] = (),
) -> MemoryRoutingCard:
    return MemoryRoutingCard(
        memory_id=memory_id,
        goal_summary="procedure",
        task_tags=("database",),
        required_tools=("sql_execute",),
        required_capabilities=("database_query",),
        execution_role_tags=("executor",),
        environment_constraints=(),
        precondition_tags=precondition_tags,
        procedure_type="single_source",
        procedure_length_bucket="short",
        read_write_scope="read_only",
    )


def _make_receiver(*, agent_id: str = "agent-1") -> AgentProfile:
    return AgentProfile(
        agent_id=agent_id,
        role="executor",
        capabilities=("database_query",),
        tool_names=("sql_execute",),
    )


def _make_input(
    card: MemoryRoutingCard,
    receiver: AgentProfile,
    task_id: str = "task-1",
) -> CandidateExposureInput:
    state = ReceiverState(
        task_id=task_id,
        scenario="database",
        task_instruction="query database",
        receiver=receiver,
    )
    return CandidateExposureInput(
        receiver_state=state, candidate_card=card
    )


# ---------------------------------------------------------------------------
# Test 1: Effect Computation
# ---------------------------------------------------------------------------

class TestEffectComputation:
    def test_beneficial_effect(self) -> None:
        """Y0=0, Ym=1 → effect=+1 (beneficial)."""
        ex = TCIEffectExample(
            memory_features=[0.1, 0.2, 0.3],
            transfer_effect=1,
            effect_source="tci_intervention",
            contrast_type="precondition",
            perturbation_type="precondition_tags",
        )
        assert ex.transfer_effect == 1

    def test_neutral_effect(self) -> None:
        """Y0=1, Ym=1 → effect=0 (neutral)."""
        ex = TCIEffectExample(
            memory_features=[0.1, 0.2, 0.3],
            transfer_effect=0,
            effect_source="tci_intervention",
            contrast_type="precondition",
            perturbation_type="precondition_tags",
        )
        assert ex.transfer_effect == 0

    def test_harmful_effect(self) -> None:
        """Y0=1, Ym=0 → effect=-1 (harmful)."""
        ex = TCIEffectExample(
            memory_features=[0.1, 0.2, 0.3],
            transfer_effect=-1,
            effect_source="tci_intervention",
            contrast_type="precondition",
            perturbation_type="precondition_tags",
        )
        assert ex.transfer_effect == -1

    def test_invalid_effect_rejected(self) -> None:
        """effect=2 should be rejected."""
        with pytest.raises(ValueError, match="must be -1, 0, or"):
            TCIEffectExample(
                memory_features=[0.1],
                transfer_effect=2,
                effect_source="tci_intervention",
                contrast_type="precondition",
                perturbation_type="precondition_tags",
            )


# ---------------------------------------------------------------------------
# Test 2: TCIEffectBatch
# ---------------------------------------------------------------------------

class TestTCIEffectBatch:
    def test_empty_batch(self) -> None:
        batch = TCIEffectBatch(examples=[])
        assert batch.n_examples == 0
        assert batch.features.shape == (0, 0)

    def test_batch_properties(self) -> None:
        examples = [
            TCIEffectExample(
                memory_features=[1.0, 0.0],
                transfer_effect=1,
                effect_source="tci_intervention",
                contrast_type="precondition",
                perturbation_type="precondition_tags",
            ),
            TCIEffectExample(
                memory_features=[0.0, 1.0],
                transfer_effect=-1,
                effect_source="tci_intervention",
                contrast_type="precondition",
                perturbation_type="precondition_tags",
            ),
            TCIEffectExample(
                memory_features=[0.5, 0.5],
                transfer_effect=0,
                effect_source="tci_intervention",
                contrast_type="precondition",
                perturbation_type="precondition_tags",
            ),
        ]
        batch = TCIEffectBatch(examples=examples)
        assert batch.n_examples == 3
        assert batch.features.shape == (3, 2)
        assert list(batch.effects) == [1, -1, 0]

    def test_effect_distribution(self) -> None:
        examples = [
            TCIEffectExample(
                memory_features=[0.0],
                transfer_effect=1,
                effect_source="tci_intervention",
                contrast_type="precondition",
                perturbation_type="precondition_tags",
            ),
            TCIEffectExample(
                memory_features=[0.0],
                transfer_effect=1,
                effect_source="tci_intervention",
                contrast_type="precondition",
                perturbation_type="precondition_tags",
            ),
            TCIEffectExample(
                memory_features=[0.0],
                transfer_effect=0,
                effect_source="tci_intervention",
                contrast_type="precondition",
                perturbation_type="precondition_tags",
            ),
            TCIEffectExample(
                memory_features=[0.0],
                transfer_effect=-1,
                effect_source="tci_intervention",
                contrast_type="precondition",
                perturbation_type="precondition_tags",
            ),
        ]
        batch = TCIEffectBatch(examples=examples)
        dist = batch.effect_distribution()
        assert dist == {-1: 1, 0: 1, 1: 2}


# ---------------------------------------------------------------------------
# Test 3: TCIValueHead
# ---------------------------------------------------------------------------

class TestTCIValueHead:
    def test_value_head_predict(self) -> None:
        """Value head predicts correct shape."""
        X_train = np.array([
            [1.0, 0.0], [0.0, 1.0], [0.5, 0.5],
            [1.0, 0.0], [0.0, 1.0], [0.5, 0.5],
        ])
        y_train = np.array([1, -1, 0, 1, -1, 0])
        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        clf.fit(X_train, y_train)
        head = TCIValueHead(model=clf, n_examples=6)

        X_test = np.array([[1.0, 0.0], [0.0, 1.0]])
        preds = head.predict(X_test)
        assert preds.shape == (2,)
        assert all(p in (-1, 0, 1) for p in preds)

    def test_value_head_proba(self) -> None:
        """Value head predict_proba returns valid probabilities."""
        X_train = np.array([
            [1.0, 0.0], [0.0, 1.0], [0.5, 0.5],
            [1.0, 0.0], [0.0, 1.0], [0.5, 0.5],
        ])
        y_train = np.array([1, -1, 0, 1, -1, 0])
        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        clf.fit(X_train, y_train)
        head = TCIValueHead(model=clf, n_examples=6)

        X_test = np.array([[1.0, 0.0]])
        probs = head.predict_proba(X_test)
        assert probs.shape == (1, 3)
        assert abs(probs.sum() - 1.0) < 0.01


# ---------------------------------------------------------------------------
# Test 4: Checkpoint Save/Load
# ---------------------------------------------------------------------------

class TestValueHeadCheckpoint:
    def test_value_head_checkpoint_save_load(self) -> None:
        """Value head persists through save/load."""
        critic = FourOutcomeTransferCritic(
            n_features=32, n_bootstrap=3, seed=7, critic_mode="flat"
        )
        inputs, labels = [], []
        for lb in ("neutral_failure", "negative_transfer",
                    "positive_transfer", "neutral_success"):
            for i in range(4):
                inputs.append(_make_input(
                    _make_card(memory_id=f"m-{lb}-{i}"),
                    _make_receiver(),
                ))
                labels.append(lb)

        # Build TCI inputs.
        card_orig = _make_card(
            memory_id="tci-orig",
            precondition_tags=("admin_role_required",),
        )
        card_pert = _make_card(memory_id="tci-pert")
        receiver = _make_receiver(agent_id="tci-receiver")
        inp_orig = _make_input(card_orig, receiver, task_id="tci-task")
        inp_pert = _make_input(card_pert, receiver, task_id="tci-task")
        tci_inputs = [
            (inp_orig, inp_pert, 1, "precondition"),
            (inp_orig, inp_pert, -1, "environment_constraint"),
        ]

        # Build effect batch.
        examples = [
            TCIEffectExample(
                memory_features=[0.1] * 32,
                transfer_effect=1,
                effect_source="tci_intervention",
                contrast_type="precondition",
                perturbation_type="precondition_tags",
            ),
            TCIEffectExample(
                memory_features=[0.2] * 32,
                transfer_effect=-1,
                effect_source="tci_intervention",
                contrast_type="environment_constraint",
                perturbation_type="environment_constraints",
            ),
            TCIEffectExample(
                memory_features=[0.3] * 32,
                transfer_effect=0,
                effect_source="tci_intervention",
                contrast_type="precondition",
                perturbation_type="precondition_tags",
            ),
        ]
        effect_batch = TCIEffectBatch(examples=examples)

        critic.fit(
            inputs, labels,
            coverage_mode="pilot",
            tci_inputs=tci_inputs,
            tci_effect_batch=effect_batch,
        )

        assert critic.training_mode == "tci_value_augmented"
        assert critic.tci_value_head is not None
        assert critic.tci_value_examples == 3

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "value_critic.joblib"
            critic.save(path)
            loaded = FourOutcomeTransferCritic.load(path)

        assert loaded.training_mode == "tci_value_augmented"
        assert loaded.tci_value_head is not None
        assert loaded.tci_value_examples == 3
        assert loaded.tci_rank_examples == 4  # 2 pairs → 4 examples.

    def test_old_checkpoint_no_value_head(self) -> None:
        """Old checkpoints without value head load correctly."""
        critic = FourOutcomeTransferCritic(
            n_features=32, n_bootstrap=3, seed=7, critic_mode="flat"
        )
        inputs, labels = [], []
        for lb in ("neutral_failure", "negative_transfer",
                    "positive_transfer", "neutral_success"):
            for i in range(4):
                inputs.append(_make_input(
                    _make_card(memory_id=f"m-{lb}-{i}"),
                    _make_receiver(),
                ))
                labels.append(lb)

        critic.fit(inputs, labels, coverage_mode="pilot")
        assert critic.tci_value_head is None
        assert critic.training_mode == "observational"

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "obs_critic.joblib"
            critic.save(path)
            loaded = FourOutcomeTransferCritic.load(path)

        assert loaded.tci_value_head is None
        assert loaded.tci_value_examples == 0


# ---------------------------------------------------------------------------
# Test 5: Training Mode Transitions
# ---------------------------------------------------------------------------

class TestTrainingModes:
    def test_value_augmented_mode_in_valid_modes(self) -> None:
        assert "tci_value_augmented" in _VALID_CRITIC_TRAINING_MODES

    def test_effect_classes_defined(self) -> None:
        assert EFFECT_CLASSES == (-1, 0, 1)


# ---------------------------------------------------------------------------
# Test 6: Random Effect Baseline
# ---------------------------------------------------------------------------

class TestRandomEffectBaseline:
    def test_random_batch_same_size(self) -> None:
        """Random batch has same number of examples as TCI batch."""
        examples = [
            TCIEffectExample(
                memory_features=[float(i)],
                transfer_effect=int(np.sign(i - 2)),
                effect_source="tci_intervention",
                contrast_type="precondition",
                perturbation_type="precondition_tags",
            )
            for i in range(5)
        ]
        tci_batch = TCIEffectBatch(examples=examples)
        random_batch = build_random_effect_baseline(tci_batch, seed=42)

        assert random_batch.n_examples == tci_batch.n_examples

    def test_random_effects_in_valid_range(self) -> None:
        """Random effects are in {-1, 0, 1}."""
        examples = [
            TCIEffectExample(
                memory_features=[0.0],
                transfer_effect=1,
                effect_source="tci_intervention",
                contrast_type="precondition",
                perturbation_type="precondition_tags",
            )
            for _ in range(20)
        ]
        tci_batch = TCIEffectBatch(examples=examples)
        random_batch = build_random_effect_baseline(tci_batch, seed=42)

        for ex in random_batch.examples:
            assert ex.transfer_effect in (-1, 0, 1)


# ---------------------------------------------------------------------------
# Test 7: Effect Accuracy
# ---------------------------------------------------------------------------

class TestEffectAccuracy:
    def test_perfect_accuracy(self) -> None:
        """Perfect value head gets accuracy=1.0."""
        X = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
        y = np.array([1, -1, 0])
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X, y)
        head = TCIValueHead(model=clf, n_examples=3)

        examples = [
            TCIEffectExample(
                memory_features=[1.0, 0.0],
                transfer_effect=1,
                effect_source="tci_intervention",
                contrast_type="precondition",
                perturbation_type="precondition_tags",
            ),
            TCIEffectExample(
                memory_features=[0.0, 1.0],
                transfer_effect=-1,
                effect_source="tci_intervention",
                contrast_type="precondition",
                perturbation_type="precondition_tags",
            ),
        ]
        batch = TCIEffectBatch(examples=examples)
        result = compute_effect_accuracy(head, batch)
        assert result["accuracy"] == 1.0
        assert result["n_examples"] == 2

    def test_empty_batch_accuracy(self) -> None:
        """Empty batch returns zero accuracy."""
        head = TCIValueHead(
            model=LogisticRegression(), n_examples=0
        )
        batch = TCIEffectBatch(examples=[])
        result = compute_effect_accuracy(head, batch)
        assert result["accuracy"] == 0.0
        assert result["n_examples"] == 0


# ---------------------------------------------------------------------------
# Test 8: Effect Builder
# ---------------------------------------------------------------------------

class TestEffectBuilder:
    def test_build_from_empty_contrasts(self) -> None:
        """Empty contrasts produces empty batch."""
        encoder = HashingTransferFeatureEncoder(n_features=32)
        batch = build_tci_effect_examples(
            contrasts=[],
            feature_encoder=encoder,
            tci_inputs=None,
        )
        assert batch.n_examples == 0
