"""Tests for Commit 9: bootstrap uncertainty + SMTR-UCB (清单第九章)."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from unittest.mock import MagicMock

import pytest

from smtr.core.types import (
    AgentProfile,
    CandidateExposureInput,
    MemoryRoutingCard,
    ReceiverState,
    TransferPrediction,
    TransferPredictionDistribution,
)
from smtr.router.exposure_router import SMTRExposureRouter, SMTRUCBRouter
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


def _card(memory_id: str) -> MemoryRoutingCard:
    return MemoryRoutingCard(
        memory_id=memory_id,
        goal_summary="goal",
        writer=AgentProfile(agent_id="w1", role="planner"),
        source_task_id="src",
        source_scenario="database",
    )


def _receiver_state() -> ReceiverState:
    return ReceiverState(
        task_id="t1",
        scenario="database",
        task_instruction="do stuff",
        receiver=AgentProfile(agent_id="r1", role="executor"),
    )


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
        critic.calibrate_q01(inputs, labels, delta=0.5)

        dist = critic.predict_distribution(inputs[0])
        assert 0.0 <= dist.eta_upper <= 1.0


class TestSMTRUCBRouter:
    def test_withholds_uncertain_positive_candidate(self):
        """SMTR-UCB must be stricter than the point-estimate SMTR rule."""
        critic = MagicMock()
        critic.epsilon_star = 0.2
        critic.predict.return_value = TransferPrediction(
            q00_neutral_failure=0.5, q01_negative_transfer=0.1,
            q10_positive_transfer=0.5, q11_neutral_success=0.0,
        )
        critic.predict_calibrated.return_value = critic.predict.return_value.model_copy(
            update={"eta_hat_calibrated": 0.1})
        # Mean tau = 0.4 > 0 and eta = 0.1 <= budget: SMTR would share.
        critic.predict_distribution.return_value = _dist(0.4, 0.1, -0.1, 0.2)

        cards = [_card("mem1")]
        rs = _receiver_state()

        smtr = SMTRExposureRouter(critic=critic)
        assert [d.action for d in smtr.decide(rs, cards)] == ["share"]

        ucb = SMTRUCBRouter(critic=critic)
        decisions = ucb.decide(rs, cards)
        assert [d.action for d in decisions] == ["withhold"]
        assert decisions[0].reason == "tau_lower<=0"

    def test_shares_confident_safe_candidate(self):
        critic = MagicMock()
        critic.epsilon_star = 0.2
        critic.predict_distribution.return_value = _dist(0.5, 0.05, 0.2, 0.1)
        ucb = SMTRUCBRouter(critic=critic)
        decisions = ucb.decide(_receiver_state(), [_card("mem1")])
        assert [d.action for d in decisions] == ["share"]
        assert decisions[0].reason == "tau_lower>0 and eta_upper<=epsilon_star"

    def test_withholds_when_eta_upper_exceeds_budget(self):
        critic = MagicMock()
        critic.epsilon_star = 0.2
        critic.predict_distribution.return_value = _dist(0.5, 0.1, 0.3, 0.5)
        ucb = SMTRUCBRouter(critic=critic)
        decisions = ucb.decide(_receiver_state(), [_card("mem1")])
        assert [d.action for d in decisions] == ["withhold"]
        assert decisions[0].reason == "eta_upper>epsilon_star"

    def test_shares_at_most_one_memory(self):
        critic = MagicMock()
        critic.epsilon_star = 0.2
        critic.predict_distribution.side_effect = [
            _dist(0.6, 0.05, 0.3, 0.1),
            _dist(0.4, 0.05, 0.2, 0.1),
        ]
        ucb = SMTRUCBRouter(critic=critic)
        decisions = ucb.decide(_receiver_state(), [_card("mem1"), _card("mem2")])
        shared = {d.memory_id for d in decisions if d.action == "share"}
        assert shared == {"mem1"}, "top ensemble-mean tau wins under v1 limit"
