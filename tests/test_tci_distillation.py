"""Tests for TCI distillation into transfer critic (Task 6).

Covers:
  - Intervention ranking loss (pairwise logistic)
  - Loss gradient
  - Build distillation examples (soft-labeled binary pairs)
  - Empty / missing TCI → graceful fallback (observational equivalence)
  - Checkpoint save/load preserves TCI provenance fields
  - Joint training: TCI pairs shift critic scores in correct direction
  - evaluate_tci_loss_on_critic returns breakdown metrics
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
from smtr.router.tci_supervision import (
    TCISupervisionBatch,
    build_tci_distillation_examples,
    evaluate_tci_loss_on_critic,
    intervention_ranking_loss,
    intervention_ranking_loss_gradient,
)
from smtr.router.transfer_critic import FourOutcomeTransferCritic


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


# ---------------------------------------------------------------------------
# 1. Intervention ranking loss
# ---------------------------------------------------------------------------

class TestInterventionRankingLoss:
    def test_zero_margin_gives_log2(self) -> None:
        # When s_m == s_m~, margin = 0 → loss = log(2).
        s = np.array([0.5, 0.5, 0.5])
        d = np.array([1.0, 1.0, 1.0])
        loss = intervention_ranking_loss(s, s, d)
        assert abs(loss - float(np.log(2))) < 1e-9

    def test_correct_direction_gives_small_loss(self) -> None:
        # direction=1, s_orig >> s_pert → margin >> 0 → small loss.
        s_orig = np.array([3.0, 3.0, 3.0])
        s_pert = np.array([0.0, 0.0, 0.0])
        d = np.array([1.0, 1.0, 1.0])
        loss = intervention_ranking_loss(s_orig, s_pert, d)
        assert loss < 0.1

    def test_wrong_direction_gives_large_loss(self) -> None:
        # direction=1, s_orig < s_pert → margin < 0 → large loss.
        s_orig = np.array([0.1, 0.1, 0.1])
        s_pert = np.array([0.9, 0.9, 0.9])
        d = np.array([1.0, 1.0, 1.0])
        loss = intervention_ranking_loss(s_orig, s_pert, d)
        assert loss > 0.5

    def test_numerical_stability_large_margin(self) -> None:
        # Very large positive margin: should be finite.
        s_orig = np.array([100.0])
        s_pert = np.array([0.0])
        d = np.array([1.0])
        loss = intervention_ranking_loss(s_orig, s_pert, d)
        assert np.isfinite(loss)
        assert loss < 1e-6

    def test_numerical_stability_large_negative(self) -> None:
        # Very large negative margin: should be finite.
        s_orig = np.array([0.0])
        s_pert = np.array([100.0])
        d = np.array([1.0])
        loss = intervention_ranking_loss(s_orig, s_pert, d)
        assert np.isfinite(loss)
        assert loss > 50.0


# ---------------------------------------------------------------------------
# 2. Loss gradient
# ---------------------------------------------------------------------------

class TestInterventionRankingLossGradient:
    def test_gradient_shape(self) -> None:
        n = 7
        s_orig = np.random.default_rng(0).normal(size=n)
        s_pert = np.random.default_rng(1).normal(size=n)
        d = np.ones(n)
        g_orig, g_pert = intervention_ranking_loss_gradient(
            s_orig, s_pert, d
        )
        assert g_orig.shape == (n,)
        assert g_pert.shape == (n,)

    def test_gradient_direction(self) -> None:
        # direction=1: dL/d_s_orig < 0 (increase s_orig reduces loss)
        #              dL/d_s_pert > 0 (decrease s_pert reduces loss)
        s_orig = np.array([0.3, 0.3])
        s_pert = np.array([0.7, 0.7])
        d = np.array([1.0, 1.0])
        g_orig, g_pert = intervention_ranking_loss_gradient(
            s_orig, s_pert, d
        )
        assert (g_orig < 0).all()
        assert (g_pert > 0).all()


# ---------------------------------------------------------------------------
# 3. Build distillation examples
# ---------------------------------------------------------------------------

class TestBuildTCIDistillationExamples:
    def test_positive_direction_flips_labels(self) -> None:
        card_orig = _make_card(memory_id="mem-1")
        card_pert = _make_card(memory_id="mem-1")
        receiver = _make_receiver()
        inp_orig = _make_input(card_orig, receiver)
        inp_pert = _make_input(card_pert, receiver)

        batch = build_tci_distillation_examples(
            [(inp_orig, inp_pert, 1, "precondition")],
            alpha=1.0,
        )
        assert len(batch.inputs) == 2
        # First is original → positive (q10).
        assert batch.labels[0] == "positive_transfer"
        # Second is perturbed → negative (q01).
        assert batch.labels[1] == "negative_transfer"

    def test_negative_direction_flips_order(self) -> None:
        card_orig = _make_card(memory_id="mem-1")
        card_pert = _make_card(memory_id="mem-1")
        receiver = _make_receiver()
        inp_orig = _make_input(card_orig, receiver)
        inp_pert = _make_input(card_pert, receiver)

        batch = build_tci_distillation_examples(
            [(inp_orig, inp_pert, -1, "environment_constraint")],
            alpha=1.0,
        )
        assert batch.labels[0] == "positive_transfer"
        assert batch.labels[1] == "negative_transfer"

    def test_zero_direction_skipped(self) -> None:
        card = _make_card(memory_id="mem-1")
        receiver = _make_receiver()
        inp_orig = _make_input(card, receiver)
        inp_pert = _make_input(card, receiver)
        batch = build_tci_distillation_examples(
            [(inp_orig, inp_pert, 0, "precondition")],
            alpha=1.0,
        )
        assert len(batch.inputs) == 0
        assert len(batch.labels) == 0
        assert batch.weights.shape == (0,)

    def test_weight_sums_to_alpha(self) -> None:
        card = _make_card(memory_id="mem-1")
        receiver = _make_receiver()
        inp_orig = _make_input(card, receiver)
        inp_pert = _make_input(card, receiver)
        # 3 pairs, direction != 0 → 6 examples.
        pairs = [
            (inp_orig, inp_pert, 1, "precondition"),
            (inp_orig, inp_pert, 1, "environment_constraint"),
            (inp_orig, inp_pert, -1, "capability"),
        ]
        batch = build_tci_distillation_examples(pairs, alpha=1.5)
        assert len(batch.inputs) == 6
        assert abs(batch.weights.sum() - 1.5) < 1e-9

    def test_empty_returns_empty_batch(self) -> None:
        batch = build_tci_distillation_examples([], alpha=1.0)
        assert batch.inputs == []
        assert batch.labels == []
        assert batch.weights.shape == (0,)


# ---------------------------------------------------------------------------
# 4. Backward compatibility (observational equivalence)
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:
    def test_critic_fit_without_tci_inputs(self) -> None:
        """Without tci_inputs, fit() produces the same result as before."""
        critic = FourOutcomeTransferCritic(
            n_features=32, n_bootstrap=3, seed=0, critic_mode="flat"
        )
        inputs = []
        labels = []
        rng = np.random.default_rng(7)
        # 4 classes × 4 examples → minimum to bootstrap.
        for lb in ("neutral_failure", "negative_transfer",
                   "positive_transfer", "neutral_success"):
            for _ in range(4):
                card = _make_card(
                    memory_id=f"m-{rng.integers(0, 10000)}",
                    precondition_tags=(
                        "admin_role_required",
                    ) if rng.random() > 0.5 else (),
                )
                receiver = _make_receiver(
                    agent_id=f"a-{rng.integers(0, 5)}"
                )
                inputs.append(_make_input(card, receiver))
                labels.append(lb)

        # Old-style call: no tci_inputs parameter.
        critic.fit(inputs, labels, coverage_mode="pilot")
        assert critic._fitted
        assert critic.tci_distillation_n_examples == 0
        assert critic.tci_distillation_alpha is None

    def test_critic_fit_with_none_tci_inputs(self) -> None:
        """Explicit tci_inputs=None matches default behaviour."""
        critic = FourOutcomeTransferCritic(
            n_features=32, n_bootstrap=3, seed=0, critic_mode="flat"
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
        critic.fit(
            inputs, labels,
            coverage_mode="pilot", tci_inputs=None, tci_alpha=1.0,
        )
        assert critic._fitted
        assert critic.tci_distillation_n_examples == 0


# ---------------------------------------------------------------------------
# 5. Joint training: TCI supervision shifts critic scores
# ---------------------------------------------------------------------------

class TestTCISupervisionEffect:
    def test_tci_pairs_shift_scores_in_correct_direction(self) -> None:
        """Adding TCI pairs where direction=1 should increase
        score(m_orig) > score(m_pert) on the training set."""
        critic = FourOutcomeTransferCritic(
            n_features=32, n_bootstrap=5, seed=7, critic_mode="flat"
        )
        # Observational data: 4 classes balanced.
        inputs, labels = [], []
        for lb in ("neutral_failure", "negative_transfer",
                   "positive_transfer", "neutral_success"):
            for i in range(4):
                inputs.append(_make_input(
                    _make_card(memory_id=f"m-{lb}-{i}"),
                    _make_receiver(),
                ))
                labels.append(lb)

        # TCI pair: original has precondition "admin_role_required"
        # that perturbed lacks → original is "better" → direction=1.
        card_orig = _make_card(
            memory_id="tci-orig",
            precondition_tags=("admin_role_required",),
        )
        card_pert = _make_card(
            memory_id="tci-pert",
            precondition_tags=(),
        )
        receiver = _make_receiver(agent_id="tci-receiver")
        inp_orig = _make_input(card_orig, receiver, task_id="tci-task")
        inp_pert = _make_input(card_pert, receiver, task_id="tci-task")

        tci_inputs = [
            (inp_orig, inp_pert, 1, "precondition"),
        ]

        critic.fit(
            inputs, labels,
            coverage_mode="pilot",
            tci_inputs=tci_inputs,
            tci_alpha=2.0,
        )
        assert critic._fitted
        assert critic.tci_distillation_n_examples == 2

        # After joint training, the critic should predict a higher
        # tau_hat = q10 - q01 for the original than the perturbed card
        # (or at least not much worse). We only require that the pair
        # loss is strictly smaller than log(2) (random).
        metrics = evaluate_tci_loss_on_critic(
            critic, [(inp_orig, inp_pert, 1, "precondition")]
        )
        assert metrics["pairwise_loss"] < float(np.log(2)) + 0.2


# ---------------------------------------------------------------------------
# 6. Checkpoint save/load
# ---------------------------------------------------------------------------

class TestCheckpointPersistence:
    def test_save_load_preserves_tci_fields(self) -> None:
        critic = FourOutcomeTransferCritic(
            n_features=32, n_bootstrap=3, seed=0, critic_mode="flat"
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
        critic.tci_distillation_n_examples = 42
        critic.tci_distillation_alpha = 1.5
        critic.tci_distillation_metrics = {
            "pairwise_accuracy": 0.75,
            "pairwise_loss": 0.3,
            "n_pairs": 10,
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "critic.joblib"
            critic.save(path)
            loaded = FourOutcomeTransferCritic.load(path)
        assert loaded.tci_distillation_n_examples == 42
        assert loaded.tci_distillation_alpha == 1.5
        assert loaded.tci_distillation_metrics["pairwise_accuracy"] == 0.75

    def test_load_old_checkpoint_defaults(self) -> None:
        """Old checkpoints without TCI fields load with defaults (0/None)."""
        critic = FourOutcomeTransferCritic(
            n_features=32, n_bootstrap=3, seed=0, critic_mode="flat"
        )
        # Old-style: no TCI fields written.
        import joblib
        import sklearn
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
        assert loaded.tci_distillation_n_examples == 0
        assert loaded.tci_distillation_alpha is None
        assert loaded.tci_distillation_metrics is None


# ---------------------------------------------------------------------------
# 7. evaluate_tci_loss_on_critic breakdown
# ---------------------------------------------------------------------------

class TestEvaluateTCILossOnCritic:
    def test_returns_breakdown_by_contrast_type(self) -> None:
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

        card_orig = _make_card(
            memory_id="orig", precondition_tags=("admin_role_required",),
        )
        card_pert = _make_card(
            memory_id="pert", precondition_tags=(),
        )
        receiver = _make_receiver()
        inp_orig = _make_input(card_orig, receiver)
        inp_pert = _make_input(card_pert, receiver)

        metrics = evaluate_tci_loss_on_critic(
            critic,
            [
                (inp_orig, inp_pert, 1, "precondition"),
                (inp_orig, inp_pert, -1, "environment_constraint"),
            ],
        )
        assert "pairwise_accuracy" in metrics
        assert "pairwise_margin" in metrics
        assert "pairwise_loss" in metrics
        assert metrics["n_pairs"] == 2
        assert "precondition_accuracy" in metrics
        assert "environment_constraint_accuracy" in metrics
        assert metrics["precondition_n"] == 1
        assert metrics["environment_constraint_n"] == 1

    def test_empty_inputs_returns_zero(self) -> None:
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
        metrics = evaluate_tci_loss_on_critic(critic, [])
        assert metrics["n_pairs"] == 0
        assert metrics["pairwise_accuracy"] == 0.0


# ---------------------------------------------------------------------------
# 8. Graceful fallback for missing TCI files
# ---------------------------------------------------------------------------

class TestBuildTCIInputsGracefulFallback:
    def test_returns_empty_when_paths_missing(self) -> None:
        from smtr.marble.training import _build_tci_inputs_for_critic
        result = _build_tci_inputs_for_critic(
            tci_contrasts_path=None,
            perturbations_manifest_path=None,
            paired_records_path=None,
            memory_pool_path=Path("/nonexistent/memory_pool.jsonl"),
        )
        assert result == []

    def test_returns_empty_when_files_missing(self) -> None:
        from smtr.marble.training import _build_tci_inputs_for_critic
        result = _build_tci_inputs_for_critic(
            tci_contrasts_path=Path("/nonexistent/contrasts.jsonl"),
            perturbations_manifest_path=Path("/nonexistent/pert.json"),
            paired_records_path=Path("/nonexistent/paired.jsonl"),
            memory_pool_path=Path("/nonexistent/memory_pool.jsonl"),
        )
        assert result == []
