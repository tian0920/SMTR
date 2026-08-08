"""Tests for Commit 9: bootstrap uncertainty (清单第九章)."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from smtr.core.types import (
    AgentProfile,
    CandidateExposureInput,
    MemoryRoutingCard,
    ReceiverState,
    TransferPrediction,
    TransferPredictionDistribution,
)
from smtr.marble.paired_outcomes import LABEL_TO_OUTCOMES
from smtr.router.transfer_critic import FourOutcomeTransferCritic

ALL_LABELS = ["neutral_failure", "negative_transfer", "positive_transfer", "neutral_success"]


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


def _dist(tau_mean: float, eta_mean: float, tau_lower: float, eta_upper: float):
    q01 = eta_mean
    q10 = tau_mean + eta_mean
    return TransferPredictionDistribution(
        mean=TransferPrediction(
            q00_neutral_failure=max(0.0, 1.0 - q01 - q10),
            q01_negative_transfer=q01,
            q10_positive_transfer=q10,
            q11_neutral_success=0.0,
        ),
        tau_std=0.1,
        eta_std=0.05,
        tau_lower=tau_lower,
        eta_upper=eta_upper,
    )


class TestPredictDistribution:
    def test_distribution_matches_mean_and_quantiles(self):
        labels = [ALL_LABELS[i % 4] for i in range(40)]
        inputs, labels = _make_inputs_and_labels(labels)
        critic = FourOutcomeTransferCritic(n_bootstrap=7, n_features=64, seed=0)
        critic.fit(inputs, labels)

        mean_pred = critic.predict(inputs[0])
        dist = critic.predict_distribution(inputs[0])

        assert dist.mean == mean_pred, "ensemble mean must match point predict"
        assert dist.tau_std >= 0.0
        assert dist.eta_std >= 0.0
        # Quantile bounds must stay within the estimand range
        assert -1.0 <= dist.tau_lower <= 1.0
        assert 0.0 <= dist.eta_upper <= 1.0

    def test_distribution_is_frozen(self):
        dist = _dist(0.5, 0.1, 0.2, 0.2)
        with pytest.raises(FrozenInstanceError):
            dist.tau_lower = 0.0  # type: ignore[misc]

    def test_calibrated_eta_upper_stays_bounded(self):
        labels = [ALL_LABELS[i % 4] for i in range(60)]
        inputs, labels = _make_inputs_and_labels(labels)
        critic = FourOutcomeTransferCritic(n_bootstrap=5, n_features=64, seed=0)
        critic.fit(inputs, labels)
        critic.calibrate_q01(inputs, labels, _records_for(inputs, labels), delta=0.5)

        dist = critic.predict_distribution(inputs[0])
        assert 0.0 <= dist.eta_upper <= 1.0
