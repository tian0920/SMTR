"""SMTR-v1 tau-only routing: eta is diagnostic, never a routing gate.

The old calibrated-eta + epsilon_star routing was removed.  SMTR-v1
decides solely on tau > 0 (net transfer effect).  eta (= q01) is
reported for diagnostics but does not gate share/withhold.
"""

from __future__ import annotations

from smtr.core.types import (
    AgentProfile,
    MemoryRoutingCard,
    ReceiverState,
    TransferPrediction,
)
from smtr.router.exposure_router import SMTRExposureRouter


def _card(memory_id: str = "mem1") -> MemoryRoutingCard:
    return MemoryRoutingCard(
        memory_id=memory_id,
        goal_summary="goal",
    )


def _receiver_state() -> ReceiverState:
    return ReceiverState(
        task_id="t1",
        scenario="database",
        task_instruction="do stuff",
        receiver=AgentProfile(agent_id="r1", role="executor"),
    )


class _StubCritic:
    """Critic stub exposing tau and eta independently."""

    def __init__(
        self,
        *,
        tau_hat: float,
        eta_raw: float = 0.0,
        eta_calibrated: float = 0.0,
    ) -> None:
        self.tau_hat = tau_hat
        self.eta_raw = eta_raw
        self.eta_calibrated = eta_calibrated
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


class TestTauOnlyRouting:
    """SMTR-v1 routes on tau > 0 regardless of eta magnitude."""

    def test_positive_tau_shares_even_with_high_eta(self):
        # tau=0.20 > 0, eta=0.99 (huge risk) -> still share.
        critic = _StubCritic(tau_hat=0.20, eta_raw=0.99, eta_calibrated=0.99)
        decisions = SMTRExposureRouter(critic=critic).decide(
            _receiver_state(), [_card()])
        assert [d.action for d in decisions] == ["share"]
        assert decisions[0].reason == "tau>0"

    def test_negative_tau_withholds_even_with_zero_eta(self):
        # tau=-0.10, eta=0.00 (zero risk) -> withhold.
        critic = _StubCritic(tau_hat=-0.10, eta_raw=0.0, eta_calibrated=0.0)
        decisions = SMTRExposureRouter(critic=critic).decide(
            _receiver_state(), [_card()])
        assert [d.action for d in decisions] == ["withhold"]
        assert decisions[0].reason == "tau<=0"

    def test_zero_tau_withholds(self):
        # tau=0.0 exactly -> withhold (strict > 0).
        critic = _StubCritic(tau_hat=0.0, eta_raw=0.05, eta_calibrated=0.05)
        decisions = SMTRExposureRouter(critic=critic).decide(
            _receiver_state(), [_card()])
        assert [d.action for d in decisions] == ["withhold"]

    def test_no_epsilon_star_required(self):
        # SMTR-v1 never reads epsilon_star; no checkpoint metadata needed.
        critic = _StubCritic(tau_hat=0.30, eta_raw=0.50, eta_calibrated=0.50)
        assert not hasattr(critic, "epsilon_star")
        decisions = SMTRExposureRouter(critic=critic).decide(
            _receiver_state(), [_card()])
        assert [d.action for d in decisions] == ["share"]

    def test_eta_reported_in_trace(self):
        # eta is diagnostic: reported in traces but never gates decisions.
        critic = _StubCritic(tau_hat=0.20, eta_raw=0.30, eta_calibrated=0.08)
        traces = SMTRExposureRouter(critic=critic).trace(
            _receiver_state(), [_card()])
        assert traces[0]["action"] == "share"
        assert traces[0]["eta_calibrated"] == 0.08
        assert traces[0]["eta_raw"] == 0.30
