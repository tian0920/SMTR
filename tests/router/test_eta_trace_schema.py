"""R6 Test 6: formal SMTR traces carry the raw/calibrated eta schema.

Every router trace must include ``eta_raw`` / ``eta_calibrated`` /
``risk_budget`` (清单 P0-7), and the share/withhold gate must compare
``eta_calibrated`` (never the raw value) against the risk budget.
"""

from __future__ import annotations

from smtr.core.types import (
    AgentProfile,
    MemoryRoutingCard,
    ReceiverState,
    TransferPrediction,
)
from smtr.router.exposure_router import SMTRExposureRouter


class _StubCritic:
    def __init__(
        self, *, tau_hat: float, eta_raw: float, eta_calibrated: float,
        epsilon_star: float = 0.10,
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


def _card() -> MemoryRoutingCard:
    return MemoryRoutingCard(
        memory_id="mem1",
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


class TestEtaTraceSchema:
    def test_trace_carries_raw_calibrated_and_budget(self):
        critic = _StubCritic(
            tau_hat=0.20, eta_raw=0.30, eta_calibrated=0.08, epsilon_star=0.10)
        router = SMTRExposureRouter(critic=critic)
        traces = router.trace(_receiver_state(), [_card()])
        assert len(traces) == 1
        trace = traces[0]
        for field in ("eta_raw", "eta_calibrated", "risk_budget"):
            assert field in trace, f"trace must include {field}"
        assert trace["eta_raw"] == 0.30
        assert trace["eta_calibrated"] == 0.08
        assert trace["risk_budget"] == 0.10
        assert trace["tau_hat"] == 0.20

    def test_router_gate_uses_eta_calibrated(self):
        # Raw eta unsafe but calibrated safe -> share: the gate ignores raw.
        share_critic = _StubCritic(
            tau_hat=0.20, eta_raw=0.30, eta_calibrated=0.08, epsilon_star=0.10)
        decisions = SMTRExposureRouter(critic=share_critic).decide(
            _receiver_state(), [_card()])
        assert [d.action for d in decisions] == ["share"]
        assert decisions[0].eta_raw == 0.30
        assert decisions[0].eta_calibrated == 0.08
        assert decisions[0].risk_budget == 0.10

        # Raw eta looks safe but calibrated unsafe -> withhold: the gate
        # must compare eta_calibrated against the budget.
        withhold_critic = _StubCritic(
            tau_hat=0.20, eta_raw=0.02, eta_calibrated=0.25, epsilon_star=0.10)
        decisions = SMTRExposureRouter(critic=withhold_critic).decide(
            _receiver_state(), [_card()])
        assert [d.action for d in decisions] == ["withhold"]
        assert decisions[0].reason == "eta_calibrated>epsilon_star"
