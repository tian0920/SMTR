"""Commit 3 tests: calibrated eta + checkpoint epsilon_star drive routing (清单第三章)."""

from __future__ import annotations

import pytest

from smtr.core.types import (
    AgentProfile,
    MemoryRoutingCard,
    ReceiverState,
    TransferPrediction,
)
from smtr.router.baselines import SMTRNoRiskRouter
from smtr.router.exposure_router import SMTRExposureRouter


def _card(memory_id: str = "mem1") -> MemoryRoutingCard:
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


class _StubCritic:
    """Critic stub exposing raw and calibrated eta separately."""

    def __init__(
        self,
        *,
        tau_hat: float,
        eta_raw: float,
        eta_calibrated: float,
        epsilon_star: float | None = None,
    ) -> None:
        self.tau_hat = tau_hat
        self.eta_raw = eta_raw
        self.eta_calibrated = eta_calibrated
        self.epsilon_star = epsilon_star
        self.q01_calibrator = None
        self.feature_block = "full"

    def _raw(self) -> TransferPrediction:
        q01 = self.eta_raw
        q10 = self.tau_hat + self.eta_raw
        return TransferPrediction(
            q00_neutral_failure=max(0.0, 1.0 - q01 - q10),
            q01_negative_transfer=q01,
            q10_positive_transfer=q10,
            q11_neutral_success=0.0,
        )

    def predict(self, item) -> TransferPrediction:
        return self._raw()

    def predict_calibrated(self, item) -> TransferPrediction:
        return self._raw().model_copy(
            update={"eta_hat_calibrated": self.eta_calibrated})


class TestCalibratedEtaRouting:
    def test_router_uses_calibrated_eta_not_raw_eta(self):
        # Case 1: raw eta=0.30 (unsafe), calibrated eta=0.08 <= eps*=0.10,
        # tau=0.20 > 0 -> share. Raw eta must not gate the decision.
        critic_share = _StubCritic(
            tau_hat=0.20, eta_raw=0.30, eta_calibrated=0.08, epsilon_star=0.10)
        decisions = SMTRExposureRouter(critic=critic_share).decide(
            _receiver_state(), [_card()])
        assert [d.action for d in decisions] == ["share"]
        assert decisions[0].reason == "tau>0 and eta_calibrated<=epsilon_star"

        # Case 2: raw eta=0.05 (looks safe), calibrated eta=0.20 > eps*=0.10
        # -> withhold. Calibrated eta must gate the decision.
        critic_withhold = _StubCritic(
            tau_hat=0.20, eta_raw=0.05, eta_calibrated=0.20, epsilon_star=0.10)
        decisions = SMTRExposureRouter(critic=critic_withhold).decide(
            _receiver_state(), [_card()])
        assert [d.action for d in decisions] == ["withhold"]
        assert decisions[0].reason == "eta_calibrated>epsilon_star"

    def test_test_router_uses_checkpoint_epsilon_star(self):
        # Budget must come from the checkpoint: eps*=0.05 rejects eta_cal=0.08,
        # eps*=0.10 accepts it. No explicit budget is passed anywhere.
        strict = _StubCritic(
            tau_hat=0.20, eta_raw=0.08, eta_calibrated=0.08, epsilon_star=0.05)
        decisions = SMTRExposureRouter(critic=strict).decide(
            _receiver_state(), [_card()])
        assert [d.action for d in decisions] == ["withhold"]

        relaxed = _StubCritic(
            tau_hat=0.20, eta_raw=0.08, eta_calibrated=0.08, epsilon_star=0.10)
        decisions = SMTRExposureRouter(critic=relaxed).decide(
            _receiver_state(), [_card()])
        assert [d.action for d in decisions] == ["share"]

    def test_missing_epsilon_star_fails_in_formal_mode(self):
        critic = _StubCritic(
            tau_hat=0.20, eta_raw=0.05, eta_calibrated=0.05, epsilon_star=None)
        router = SMTRExposureRouter(critic=critic)
        with pytest.raises(
            ValueError,
            match="Checkpoint does not contain validation-selected epsilon_star.",
        ):
            router.decide(_receiver_state(), [_card()])

    def test_no_risk_router_ignores_eta(self):
        # SMTR-no-risk must ignore eta entirely (not epsilon=1): a huge raw eta
        # with positive tau still shares, and epsilon_star is never required.
        critic = _StubCritic(
            tau_hat=0.20, eta_raw=0.99, eta_calibrated=0.99, epsilon_star=None)
        decisions = SMTRNoRiskRouter(critic=critic).decide(
            _receiver_state(), [_card()])
        assert [d.action for d in decisions] == ["share"]
        assert decisions[0].reason == "tau>0_no_risk_constraint"

        # Negative tau still withholds even with zero risk.
        critic_neg = _StubCritic(
            tau_hat=-0.10, eta_raw=0.0, eta_calibrated=0.0, epsilon_star=None)
        decisions = SMTRNoRiskRouter(critic=critic_neg).decide(
            _receiver_state(), [_card()])
        assert [d.action for d in decisions] == ["withhold"]


class TestRiskBudgetOverrideGuard:
    def test_explicit_budget_forbidden_without_debug_flag(self):
        critic = _StubCritic(
            tau_hat=0.20, eta_raw=0.05, eta_calibrated=0.05, epsilon_star=0.10)
        with pytest.raises(ValueError, match="allow_risk_budget_override"):
            SMTRExposureRouter(critic=critic, negative_risk_budget=0.2)

    def test_explicit_budget_allowed_in_debug_mode(self):
        critic = _StubCritic(
            tau_hat=0.20, eta_raw=0.15, eta_calibrated=0.15, epsilon_star=0.10)
        # eps*=0.10 would withhold (eta_cal=0.15); debug override 0.2 shares.
        router = SMTRExposureRouter(
            critic=critic, negative_risk_budget=0.2,
            allow_risk_budget_override=True)
        decisions = router.decide(_receiver_state(), [_card()])
        assert [d.action for d in decisions] == ["share"]
