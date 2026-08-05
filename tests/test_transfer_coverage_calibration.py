"""Tests for label coverage fail-fast, calibration and epsilon selection (清单第七/八章)."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from smtr.core.types import (
    AgentProfile,
    CandidateExposureInput,
    MemoryRoutingCard,
    ReceiverState,
)
from smtr.marble.paired_outcomes import LABEL_TO_OUTCOMES
from smtr.router.transfer_calibration import (
    compute_four_class_metrics,
    compute_probability_metrics,
    select_epsilon,
)
from smtr.router.transfer_coverage import (
    InsufficientTransferCoverageError,
    validate_transfer_label_coverage,
)
from smtr.router.transfer_critic import FourOutcomeTransferCritic

ALL_LABELS = ["neutral_failure", "negative_transfer", "positive_transfer", "neutral_success"]


def _make_inputs_and_labels(labels: list[str]) -> tuple[list[CandidateExposureInput], list[str]]:
    inputs = []
    for i, label in enumerate(labels):
        writer = AgentProfile(agent_id=f"w{i % 2}", role="planner")
        receiver = AgentProfile(agent_id=f"r{i % 3}", role="executor")
        card = MemoryRoutingCard(
            memory_id=f"m{i % 5}",
            goal_summary=f"goal {i}",
            writer=writer,
            source_task_id=f"src{i}",
            source_scenario="database",
        )
        rs = ReceiverState(
            task_id=f"task{i % 4}",
            scenario="database",
            task_instruction=f"fix query {i}",
            receiver=receiver,
        )
        inputs.append(CandidateExposureInput(receiver_state=rs, candidate_card=card))
    return inputs, labels


def _four_class_labels(n: int = 40) -> list[str]:
    return [ALL_LABELS[i % 4] for i in range(n)]


def _records_for(inputs: list[CandidateExposureInput], labels: list[str]) -> list[dict]:
    """Paired records aligned with inputs, one per seed record."""
    records = []
    for i, (inp, label) in enumerate(zip(inputs, labels)):
        y_share, y_withhold = LABEL_TO_OUTCOMES[label]
        records.append(
            {
                "task_id": inp.receiver_state.task_id,
                "receiver_agent_id": inp.receiver_state.receiver.agent_id,
                "candidate_memory_id": inp.candidate_card.memory_id,
                "generation_seed": i,
                "label": label,
                "share": {"team_success": bool(y_share)},
                "withhold": {"team_success": bool(y_withhold)},
            }
        )
    return records


def test_training_fails_without_negative_transfer():
    """Training without negative_transfer must fail fast in every mode."""
    labels = ["positive_transfer", "neutral_success", "neutral_failure"] * 10
    inputs, _ = _make_inputs_and_labels(labels)
    critic = FourOutcomeTransferCritic(n_bootstrap=3, n_features=64, seed=0)
    with pytest.raises(InsufficientTransferCoverageError):
        critic.fit(inputs, labels, coverage_mode="formal")
    with pytest.raises(InsufficientTransferCoverageError):
        critic.fit(inputs, labels, coverage_mode="pilot")


def test_formal_mode_requires_all_four_classes():
    """Formal training must reject data missing any neutral class."""
    labels = ["positive_transfer", "negative_transfer"] * 10
    inputs, _ = _make_inputs_and_labels(labels)
    critic = FourOutcomeTransferCritic(n_bootstrap=3, n_features=64, seed=0)
    with pytest.raises(InsufficientTransferCoverageError):
        critic.fit(inputs, labels, coverage_mode="formal")
    # Pilot allows missing neutrals when positive+negative are both present.
    critic.fit(inputs, labels, coverage_mode="pilot")
    assert critic.coverage_report["label_counts"]["negative_transfer"] == 10
    assert critic.coverage_report["positive_transfer_edges"] > 0
    assert critic.coverage_report["negative_transfer_edges"] > 0
    assert 0 < critic.coverage_report["minority_class_rate"] <= 0.5


def test_validate_coverage_reports_shape():
    report = validate_transfer_label_coverage(_four_class_labels(), mode="formal")
    assert set(report["label_counts"]) == set(ALL_LABELS)
    assert report["minority_class_rate"] == pytest.approx(0.25)


def test_bootstrap_members_preserve_required_classes():
    """Every bootstrap member must see all training classes; no zero-padding."""
    inputs, labels = _make_inputs_and_labels(_four_class_labels(48))
    critic = FourOutcomeTransferCritic(n_bootstrap=7, n_features=64, seed=0)
    critic.fit(inputs, labels)
    assert len(critic.members) == 7
    for member in critic.members:
        assert set(member.classes_) == {0, 1, 2, 3}
        assert member.class_weight == "balanced"


def test_calibration_metrics_shape():
    labels = _four_class_labels(20)
    rng = np.random.default_rng(0)
    probs = rng.dirichlet([1, 1, 1, 1], size=len(labels))
    pred_labels = [ALL_LABELS[int(np.argmax(p))] for p in probs]

    cls_metrics = compute_four_class_metrics(labels, pred_labels)
    for key in (
        "accuracy",
        "macro_f1",
        "per_class_precision",
        "per_class_recall",
        "confusion_matrix",
        "negative_transfer_recall",
    ):
        assert key in cls_metrics

    prob_metrics = compute_probability_metrics(labels, probs)
    for key in (
        "multiclass_log_loss",
        "multiclass_brier_score",
        "expected_calibration_error",
        "negative_transfer_brier_score",
        "q01_calibration_curve",
    ):
        assert key in prob_metrics
    assert prob_metrics["multiclass_brier_score"] >= 0
    assert 0 <= prob_metrics["expected_calibration_error"] <= 1


def test_checkpoint_contains_validation_selected_epsilon(tmp_path):
    """epsilon_star must be selected on validation data and persisted."""
    inputs, labels = _make_inputs_and_labels(_four_class_labels(60))
    critic = FourOutcomeTransferCritic(n_bootstrap=3, n_features=64, seed=0)
    critic.fit(inputs, labels)
    selection = critic.calibrate_q01(
        inputs, labels, _records_for(inputs, labels), delta=0.5
    )

    checkpoint = tmp_path / "critic.joblib"
    critic.save(checkpoint)
    loaded = FourOutcomeTransferCritic.load(checkpoint)
    assert loaded.epsilon_star == selection["epsilon_star"]
    assert loaded.epsilon_star in (0.05, 0.10, 0.20, 0.30)
    assert selection["selected_on"] == "validation"
    assert loaded.q01_calibrator is not None
    pred = loaded.predict(inputs[0])
    assert 0.0 <= loaded.calibrated_q01(pred) <= 1.0


def test_test_split_does_not_select_risk_budget():
    """Epsilon selection must only accept validation-side inputs.

    The selection API has no test-set parameter, and selecting on one array
    never reads any other data source.
    """
    params = set(inspect.signature(select_epsilon).parameters)
    assert params == {
        "tau_hat",
        "q01_calibrated",
        "labels",
        "epsilons",
        "delta",
    }, "select_epsilon must not accept test-split data"

    tau = np.array([0.5, 0.4, -0.2, 0.3, 0.1, -0.1])
    q01 = np.array([0.02, 0.4, 0.1, 0.05, 0.25, 0.6])
    labels = [
        "positive_transfer",
        "negative_transfer",
        "neutral_failure",
        "neutral_success",
        "negative_transfer",
        "positive_transfer",
    ]
    result = select_epsilon(tau, q01, labels, delta=0.10)
    assert result["epsilon_star"] in (0.05, 0.10, 0.20, 0.30)
    # Selection is deterministic in the validation arrays alone.
    again = select_epsilon(tau.copy(), q01.copy(), list(labels), delta=0.10)
    assert again["epsilon_star"] == result["epsilon_star"]
    # Risk constraint respected at the selected budget.
    chosen = result["candidates_evaluated"][str(result["epsilon_star"])]
    assert chosen["negative_transfer_exposure_rate"] <= 0.10
