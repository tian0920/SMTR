"""Tests for TCI routing evaluation (Codex Task 10).

Covers:
  - Test 1: Perfect critic → regret=0, top1_hit=1.0
  - Test 2: Random critic → close to random
  - Test 3: TCI mode checkpoint save/load
  - Routing metrics: PTC, NTE, Regret, Top-1 Hit
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from smtr.core.types import (
    AgentProfile,
    CandidateExposureInput,
    MemoryRoutingCard,
    ReceiverState,
)
from smtr.router.tci_augmentation import (
    SOURCE_TCI_INDUCED_DAMAGE,
    TCIAugmentedBatch,
    TCIAugmentedExample,
    build_tci_augmentation_examples,
)
from smtr.router.tci_metrics import (
    compute_negative_transfer_exposure,
    compute_positive_transfer_capture,
    compute_routing_metrics_summary,
    compute_top1_hit_rate,
    compute_transfer_regret,
)
from smtr.router.tci_routing_eval import (
    RoutingMetrics,
    RoutingSelection,
    evaluate_memory_selection,
    score_candidate,
    select_best_candidate,
)
from smtr.router.transfer_critic import (
    FourOutcomeTransferCritic,
    _VALID_CRITIC_TRAINING_MODES,
    TCI_SCHEMA_VERSION,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_card(
    *,
    memory_id: str = "mem-1",
    precondition_tags: tuple[str, ...] = (),
    environment_constraints: tuple[str, ...] = (),
    required_capabilities: tuple[str, ...] = ("database_query",),
    required_tools: tuple[str, ...] = ("sql_execute",),
    execution_role_tags: tuple[str, ...] = ("executor",),
) -> MemoryRoutingCard:
    return MemoryRoutingCard(
        memory_id=memory_id,
        goal_summary="procedure",
        task_tags=("database",),
        required_tools=required_tools,
        required_capabilities=required_capabilities,
        execution_role_tags=execution_role_tags,
        environment_constraints=environment_constraints,
        precondition_tags=precondition_tags,
        procedure_type="single_source",
        procedure_length_bucket="short",
        read_write_scope="read_only",
    )


def _make_receiver(
    *,
    agent_id: str = "agent-1",
    role: str = "executor",
    capabilities: tuple[str, ...] = ("database_query",),
    tool_names: tuple[str, ...] = ("sql_execute",),
) -> AgentProfile:
    return AgentProfile(
        agent_id=agent_id,
        role=role,  # type: ignore[arg-type]
        capabilities=capabilities,
        tool_names=tool_names,
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


def _make_balanced_inputs_and_labels(n_per_class: int = 4):
    """Generate balanced 4-class training data."""
    inputs, labels = [], []
    for lb in ("neutral_failure", "negative_transfer",
               "positive_transfer", "neutral_success"):
        for i in range(n_per_class):
            inputs.append(_make_input(
                _make_card(memory_id=f"m-{lb}-{i}"),
                _make_receiver(),
            ))
            labels.append(lb)
    return inputs, labels


# ---------------------------------------------------------------------------
# Test 1: Perfect critic → regret=0, top1_hit=1.0
# ---------------------------------------------------------------------------

class TestPerfectCritic:
    """Simulates a perfect critic that assigns high scores to
    memories with positive effects and low scores to those with
    negative effects."""

    def test_perfect_critic_regret_zero(self) -> None:
        """Perfect critic should have zero regret and top1_hit=1."""
        # Create a "perfect" critic by training on data that
        # perfectly separates positive from negative transfer.
        critic = FourOutcomeTransferCritic(
            n_features=32, n_bootstrap=5, seed=7, critic_mode="flat"
        )

        # Create training data where memory features perfectly
        # predict transfer outcome.
        inputs, labels = [], []
        for i in range(8):
            # Positive transfer: strong precondition.
            card_pos = _make_card(
                memory_id=f"pos-{i}",
                precondition_tags=("admin_role_required",),
                environment_constraints=("gpu_acceleration_required",),
            )
            inputs.append(_make_input(card_pos, _make_receiver()))
            labels.append("positive_transfer")

            # Negative transfer: no precondition, different env.
            card_neg = _make_card(
                memory_id=f"neg-{i}",
                precondition_tags=(),
                environment_constraints=("temporary_table_creation_required",),
            )
            inputs.append(_make_input(card_neg, _make_receiver()))
            labels.append("negative_transfer")

        # Add neutral classes for minimum class diversity.
        for i in range(4):
            inputs.append(_make_input(
                _make_card(memory_id=f"ns-{i}",
                           precondition_tags=("multi_region_consensus",)),
                _make_receiver(),
            ))
            labels.append("neutral_success")
            inputs.append(_make_input(
                _make_card(memory_id=f"nf-{i}"),
                _make_receiver(),
            ))
            labels.append("neutral_failure")

        critic.fit(inputs, labels, coverage_mode="pilot")

        # Create test candidate set: 3 memories with known effects.
        # Memory A: positive effect (+1)
        # Memory B: negative effect (-1)
        # Memory C: neutral (0)
        card_a = _make_card(
            memory_id="A",
            precondition_tags=("admin_role_required",),
            environment_constraints=("gpu_acceleration_required",),
        )
        card_b = _make_card(
            memory_id="B",
            precondition_tags=(),
            environment_constraints=("temporary_table_creation_required",),
        )
        card_c = _make_card(memory_id="C")

        candidates = [
            _make_input(card_a, _make_receiver()),
            _make_input(card_b, _make_receiver()),
            _make_input(card_c, _make_receiver()),
        ]
        effects = [1.0, -1.0, 0.0]

        selection = select_best_candidate(critic, candidates, effects)
        # The perfect critic should select A (positive effect).
        assert selection.is_positive_transfer
        assert not selection.is_negative_transfer
        # Regret should be small (ideally 0 if A is correctly ranked highest).
        assert selection.regret < 0.5
        # Top-1 hit: A is the best effect memory.
        if selection.selected_memory_id == "A":
            assert selection.is_top1_hit

    def test_evaluate_memory_selection_perfect(self) -> None:
        """Evaluate multiple candidate sets with a well-trained critic."""
        critic = FourOutcomeTransferCritic(
            n_features=32, n_bootstrap=5, seed=7, critic_mode="flat"
        )
        inputs, labels = _make_balanced_inputs_and_labels(6)
        critic.fit(inputs, labels, coverage_mode="pilot")

        # 2 candidate sets.
        card1 = _make_card(memory_id="m1",
                           precondition_tags=("admin_role_required",))
        card2 = _make_card(memory_id="m2")

        candidates_list = [
            [_make_input(card1, _make_receiver()),
             _make_input(card2, _make_receiver())],
            [_make_input(card2, _make_receiver())],
        ]
        effects_list = [
            [1.0, 0.0],
            [0.5],
        ]
        metrics = evaluate_memory_selection(
            candidates_list, effects_list, critic
        )
        assert metrics.n_selections == 2
        assert 0.0 <= metrics.positive_capture <= 1.0
        assert 0.0 <= metrics.negative_exposure <= 1.0
        assert metrics.transfer_regret >= 0.0
        assert 0.0 <= metrics.top1_hit_rate <= 1.0


# ---------------------------------------------------------------------------
# Test 2: Random critic → close to random
# ---------------------------------------------------------------------------

class TestRandomCritic:
    def test_random_critic_near_random(self) -> None:
        """An untrained/fresh critic should be close to random on
        routing metrics."""
        critic = FourOutcomeTransferCritic(
            n_features=32, n_bootstrap=3, seed=0, critic_mode="flat"
        )
        inputs, labels = _make_balanced_inputs_and_labels(4)
        critic.fit(inputs, labels, coverage_mode="pilot")

        # Many candidate sets to get stable statistics.
        rng = np.random.default_rng(42)
        candidates_list = []
        effects_list = []
        for _ in range(20):
            n_cands = rng.integers(2, 5)
            cands = []
            effs = []
            for j in range(n_cands):
                cands.append(_make_input(
                    _make_card(memory_id=f"rand-{rng.integers(0, 1000)}"),
                    _make_receiver(),
                ))
                effs.append(float(rng.choice([-1, 0, 1])))
            candidates_list.append(cands)
            effects_list.append(effs)

        metrics = evaluate_memory_selection(
            candidates_list, effects_list, critic
        )
        assert metrics.n_selections == 20
        # With random scores, top-1 hit should be well below 1.0.
        assert metrics.top1_hit_rate < 0.8


# ---------------------------------------------------------------------------
# Test 3: TCI mode checkpoint save/load
# ---------------------------------------------------------------------------

class TestTCIModeCheckpoint:
    def test_tci_augmented_mode_checkpoint(self) -> None:
        """TCI augmented training sets training_mode correctly
        and persists to checkpoint."""
        critic = FourOutcomeTransferCritic(
            n_features=32, n_bootstrap=3, seed=7, critic_mode="flat"
        )
        inputs, labels = _make_balanced_inputs_and_labels(4)

        # Build TCI inputs.
        card_orig = _make_card(
            memory_id="tci-orig",
            precondition_tags=("admin_role_required",),
        )
        card_pert = _make_card(
            memory_id="tci-pert", precondition_tags=()
        )
        receiver = _make_receiver(agent_id="tci-receiver")
        inp_orig = _make_input(card_orig, receiver, task_id="tci-task")
        inp_pert = _make_input(card_pert, receiver, task_id="tci-task")
        tci_inputs = [(inp_orig, inp_pert, 1, "precondition")]

        critic.fit(
            inputs, labels,
            coverage_mode="pilot",
            tci_inputs=tci_inputs,
        )
        assert critic.training_mode == "tci_augmented"
        assert critic.n_tci_examples == 2  # 1 pair → 2 examples.
        assert critic.tci_schema_version == TCI_SCHEMA_VERSION

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tci_critic.joblib"
            critic.save(path)
            loaded = FourOutcomeTransferCritic.load(path)
        assert loaded.training_mode == "tci_augmented"
        assert loaded.n_observational_examples == 16
        assert loaded.n_tci_examples == 2
        assert loaded.tci_schema_version == TCI_SCHEMA_VERSION

    def test_observational_mode_checkpoint(self) -> None:
        """Observational-only training sets training_mode correctly."""
        critic = FourOutcomeTransferCritic(
            n_features=32, n_bootstrap=3, seed=7, critic_mode="flat"
        )
        inputs, labels = _make_balanced_inputs_and_labels(4)
        critic.fit(inputs, labels, coverage_mode="pilot")
        assert critic.training_mode == "observational"
        assert critic.n_tci_examples == 0
        assert critic.tci_schema_version is None

    def test_old_checkpoint_defaults(self) -> None:
        """Old checkpoints without training_mode fields load with defaults."""
        import joblib
        import sklearn
        critic = FourOutcomeTransferCritic(
            n_features=32, n_bootstrap=3, seed=7, critic_mode="flat"
        )
        inputs, labels = _make_balanced_inputs_and_labels(4)
        critic.fit(inputs, labels, coverage_mode="pilot")
        # Simulate old checkpoint: remove new fields before save.
        data = {
            "members": critic.members,
            "n_features": critic.n_features,
            "n_bootstrap": critic.n_bootstrap,
            "feature_block": critic.feature_block,
            "seed": critic.seed,
            "encoder": critic.encoder,
            "schema_version": critic.encoder.schema_version,
            "sklearn_version": sklearn.__version__,
            "critic_mode": critic.critic_mode,
            "factorized_members": [],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "old.joblib"
            joblib.dump(data, path)
            loaded = FourOutcomeTransferCritic.load(path)
        assert loaded.training_mode == "observational"
        assert loaded.n_observational_examples == 0
        assert loaded.n_tci_examples == 0
        assert loaded.tci_schema_version is None


# ---------------------------------------------------------------------------
# Routing metrics functions
# ---------------------------------------------------------------------------

class TestRoutingMetricsFunctions:
    def _make_selections(self):
        """Create a fixed set of selections for metric testing."""
        return [
            RoutingSelection(
                selected_memory_id="m1", selected_score=0.5,
                selected_effect=1.0, best_effect=1.0,
                regret=0.0, is_positive_transfer=True,
                is_negative_transfer=False, is_top1_hit=True,
            ),
            RoutingSelection(
                selected_memory_id="m2", selected_score=0.3,
                selected_effect=-1.0, best_effect=1.0,
                regret=2.0, is_positive_transfer=False,
                is_negative_transfer=True, is_top1_hit=False,
            ),
            RoutingSelection(
                selected_memory_id="m3", selected_score=0.1,
                selected_effect=0.0, best_effect=1.0,
                regret=1.0, is_positive_transfer=False,
                is_negative_transfer=False, is_top1_hit=False,
            ),
        ]

    def test_positive_transfer_capture(self) -> None:
        sels = self._make_selections()
        ptc = compute_positive_transfer_capture(sels)
        assert abs(ptc - 1.0 / 3.0) < 1e-9

    def test_negative_transfer_exposure(self) -> None:
        sels = self._make_selections()
        nte = compute_negative_transfer_exposure(sels)
        assert abs(nte - 1.0 / 3.0) < 1e-9

    def test_transfer_regret(self) -> None:
        sels = self._make_selections()
        regret = compute_transfer_regret(sels)
        assert abs(regret - 1.0) < 1e-9  # mean(0, 2, 1)

    def test_top1_hit_rate(self) -> None:
        sels = self._make_selections()
        hit = compute_top1_hit_rate(sels)
        assert abs(hit - 1.0 / 3.0) < 1e-9

    def test_routing_metrics_summary(self) -> None:
        sels = self._make_selections()
        summary = compute_routing_metrics_summary(sels)
        assert summary["n_selections"] == 3
        assert abs(summary["positive_capture"] - 1.0 / 3.0) < 1e-9
        assert abs(summary["negative_exposure"] - 1.0 / 3.0) < 1e-9
        assert abs(summary["transfer_regret"] - 1.0) < 1e-9

    def test_empty_selections(self) -> None:
        summary = compute_routing_metrics_summary([])
        assert summary["n_selections"] == 0
        assert summary["positive_capture"] == 0.0
        assert summary["negative_exposure"] == 0.0


# ---------------------------------------------------------------------------
# TCI augmentation data interface
# ---------------------------------------------------------------------------

class TestTCIAugmentation:
    def test_build_tci_augmentation_examples(self) -> None:
        card_orig = _make_card(memory_id="orig",
                               precondition_tags=("admin_role_required",))
        card_pert = _make_card(memory_id="pert")
        receiver = _make_receiver()
        inp_orig = _make_input(card_orig, receiver)
        inp_pert = _make_input(card_pert, receiver)

        batch = build_tci_augmentation_examples(
            [(inp_orig, inp_pert, 1, "precondition")],
        )
        assert batch.n_examples == 2
        assert batch.labels == ["positive_transfer", "negative_transfer"]
        assert batch.source_types[0] == SOURCE_TCI_INDUCED_DAMAGE
        assert batch.source_types[1] == SOURCE_TCI_INDUCED_DAMAGE

    def test_zero_direction_skipped(self) -> None:
        card = _make_card(memory_id="m1")
        receiver = _make_receiver()
        inp = _make_input(card, receiver)
        batch = build_tci_augmentation_examples(
            [(inp, inp, 0, "precondition")],
        )
        assert batch.n_examples == 0

    def test_negative_direction_flips(self) -> None:
        card_orig = _make_card(memory_id="orig")
        card_pert = _make_card(memory_id="pert")
        receiver = _make_receiver()
        inp_orig = _make_input(card_orig, receiver)
        inp_pert = _make_input(card_pert, receiver)

        batch = build_tci_augmentation_examples(
            [(inp_orig, inp_pert, -1, "environment_constraint")],
        )
        assert batch.n_examples == 2
        # direction < 0: perturbed is better → perturbed → positive.
        assert batch.labels[0] == "positive_transfer"
        assert batch.labels[1] == "negative_transfer"


# ---------------------------------------------------------------------------
# Training mode validation
# ---------------------------------------------------------------------------

class TestTrainingModeConstants:
    def test_valid_training_modes(self) -> None:
        assert "observational" in _VALID_CRITIC_TRAINING_MODES
        assert "tci_augmented" in _VALID_CRITIC_TRAINING_MODES
        assert len(_VALID_CRITIC_TRAINING_MODES) == 2

    def test_tci_schema_version(self) -> None:
        assert TCI_SCHEMA_VERSION == "v1"


# ---------------------------------------------------------------------------
# evaluate_memory_selection edge cases
# ---------------------------------------------------------------------------

class TestEvaluateMemorySelection:
    def test_empty_candidates(self) -> None:
        critic = FourOutcomeTransferCritic(
            n_features=32, n_bootstrap=3, seed=7, critic_mode="flat"
        )
        inputs, labels = _make_balanced_inputs_and_labels(4)
        critic.fit(inputs, labels, coverage_mode="pilot")

        metrics = evaluate_memory_selection([], [], critic)
        assert metrics.n_selections == 0

    def test_single_candidate(self) -> None:
        critic = FourOutcomeTransferCritic(
            n_features=32, n_bootstrap=3, seed=7, critic_mode="flat"
        )
        inputs, labels = _make_balanced_inputs_and_labels(4)
        critic.fit(inputs, labels, coverage_mode="pilot")

        card = _make_card(memory_id="only")
        candidates = [[_make_input(card, _make_receiver())]]
        effects = [[1.0]]

        metrics = evaluate_memory_selection(candidates, effects, critic)
        assert metrics.n_selections == 1
        assert metrics.top1_hit_rate == 1.0  # Only one choice.
        assert metrics.transfer_regret == 0.0

    def test_mismatched_lengths_raises(self) -> None:
        critic = FourOutcomeTransferCritic(
            n_features=32, n_bootstrap=3, seed=7, critic_mode="flat"
        )
        inputs, labels = _make_balanced_inputs_and_labels(4)
        critic.fit(inputs, labels, coverage_mode="pilot")

        with pytest.raises(ValueError):
            evaluate_memory_selection(
                [[_make_input(_make_card(), _make_receiver())]],
                [[1.0, 2.0]],  # Mismatched.
                critic,
            )
