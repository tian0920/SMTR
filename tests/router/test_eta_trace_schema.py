"""R6 Test 6: formal SMTR traces carry the raw/calibrated eta schema.

Every router trace must include ``eta_raw`` / ``eta_calibrated`` (清单 P0-7),
and the share/withhold gate must use ``tau > 0`` as the sole decision
variable.  eta (= q01) is diagnostic only; no epsilon* risk budget.
"""

from __future__ import annotations

from types import SimpleNamespace

from smtr.core.types import (
    AgentProfile,
    MemoryRoutingCard,
    ReceiverState,
)
from smtr.router.exposure_router import SMTRExposureRouter


class _StubCritic:
    def __init__(
        self, *, tau_hat: float, eta_raw: float, eta_calibrated: float,
    ) -> None:
        self.tau_hat = tau_hat
        self.eta_raw = eta_raw
        self.eta_calibrated = eta_calibrated
        self.q01_calibrator = None
        self.feature_block = "full"

    def predict(self, item):
        q01 = self.eta_raw
        q10 = self.tau_hat + self.eta_raw
        return SimpleNamespace(
            tau_hat=self.tau_hat,
            eta_hat=q01,
        )

    def predict_calibrated(self, item):
        q01 = self.eta_raw
        return SimpleNamespace(
            tau_hat=self.tau_hat,
            eta_hat=q01,
            eta_hat_calibrated=self.eta_calibrated,
        )


def _card() -> MemoryRoutingCard:
    return MemoryRoutingCard(
        memory_id="mem1",
        goal_summary="goal",
    )


def _receiver_state() -> ReceiverState:
    return ReceiverState(
        task_id="t1",
        scenario="database",
        task_instruction="do stuff",
        receiver=AgentProfile(agent_id="r1", role="executor"),
    )


class TestEtaTraceSchema:
    def test_trace_carries_raw_and_calibrated(self):
        critic = _StubCritic(
            tau_hat=0.20, eta_raw=0.30, eta_calibrated=0.08)
        router = SMTRExposureRouter(critic=critic)
        traces = router.trace(_receiver_state(), [_card()])
        assert len(traces) == 1
        trace = traces[0]
        for field in ("eta_raw", "eta_calibrated"):
            assert field in trace, f"trace must include {field}"
        assert trace["eta_raw"] == 0.30
        assert trace["eta_calibrated"] == 0.08
        assert trace["tau_hat"] == 0.20

    def test_router_gate_uses_tau_only(self):
        # Positive tau -> share, regardless of eta magnitude.
        share_critic = _StubCritic(
            tau_hat=0.20, eta_raw=0.30, eta_calibrated=0.08)
        decisions = SMTRExposureRouter(critic=share_critic).decide(
            _receiver_state(), [_card()])
        assert [d.action for d in decisions] == ["share"]
        assert decisions[0].reason == "tau>0"

        # Non-positive tau -> withhold, even when eta is small.
        withhold_critic = _StubCritic(
            tau_hat=-0.10, eta_raw=0.02, eta_calibrated=0.01)
        decisions = SMTRExposureRouter(critic=withhold_critic).decide(
            _receiver_state(), [_card()])
        assert [d.action for d in decisions] == ["withhold"]
        assert decisions[0].reason == "tau<=0"
