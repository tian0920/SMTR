"""R6 Test 5: negative-transfer risk is calibrated exactly once.

The calibrator must run once at router prediction time; the risk-utility
curve consumes ``eta_calibrated`` directly and never re-invokes the
calibrator (清单 P0-8/P0-10).
"""

from __future__ import annotations

import numpy as np
import pytest

from smtr.core.types import (
    AgentProfile,
    MemoryRoutingCard,
    ReceiverState,
    TransferPrediction,
)
from smtr.marble.paired_evaluation import _smtr_risk_utility_curve
from smtr.router.exposure_router import SMTRExposureRouter


class CountingCalibrator:
    """Isotonic-like calibrator stub that records every predict() call."""

    def __init__(self) -> None:
        self.calls = 0

    def predict(self, x):
        self.calls += 1
        return np.asarray(x, dtype=float) * 0.5


class _CountingCritic:
    """Critic that applies the counting calibrator once per prediction."""

    def __init__(self, calibrator: CountingCalibrator) -> None:
        self.q01_calibrator = calibrator
        self.epsilon_star = 0.2
        self.feature_block = "full"

    def _raw(self) -> TransferPrediction:
        return TransferPrediction(
            q00_neutral_failure=0.4,
            q01_negative_transfer=0.3,
            q10_positive_transfer=0.6,
            q11_neutral_success=0.0,
        )

    def predict(self, item) -> TransferPrediction:
        return self._raw()

    def predict_calibrated(self, item) -> TransferPrediction:
        raw = self._raw()
        calibrated = float(self.q01_calibrator.predict(np.asarray([raw.eta_hat_raw]))[0])
        return raw.model_copy(update={"eta_hat_calibrated": calibrated})


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


class TestEtaCalibratedOnce:
    def test_calibrator_called_once_at_prediction_not_in_curve(self):
        calibrator = CountingCalibrator()
        critic = _CountingCritic(calibrator)
        router = SMTRExposureRouter(critic=critic)

        # Router prediction phase: exactly one calibrator call per candidate.
        decisions = router.decide(_receiver_state(), [_card()])
        assert calibrator.calls == 1
        assert [d.action for d in decisions] == ["share"]

        dec = decisions[0]
        trace = {
            "task_id": "t1",
            "generation_seed": 7,
            "receiver_agent_id": "r1",
            "candidate_memory_id": dec.memory_id,
            "action": dec.action,
            "tau_hat": dec.tau_hat,
            "eta_raw": dec.eta_raw,
            "eta_calibrated": dec.eta_calibrated,
        }
        outcome = {
            "task_id": "t1",
            "generation_seed": 7,
            "receiver_agent_id": "r1",
            "candidate_memory_id": dec.memory_id,
            "share": {"team_success": True},
            "withhold": {"team_success": False},
        }

        # Risk-utility curve: no additional calibrator call, and the
        # artifact declares a single calibration at prediction time.
        result = _smtr_risk_utility_curve(
            [trace], [outcome], critic, experiment_mode="formal"
        )
        assert calibrator.calls == 1
        assert result["risk_value_type"] == "calibrated_eta"
        assert result["calibration_applied_times"] == 1
        assert result["n_matched_candidates"] == 1

    def test_legacy_trace_rejected_in_formal_but_tolerated_in_pilot(self):
        critic = _CountingCritic(CountingCalibrator())
        legacy_trace = {
            "task_id": "t1",
            "generation_seed": 7,
            "receiver_agent_id": "r1",
            "candidate_memory_id": "mem1",
            "action": "share",
            "tau_hat": 0.3,
            "eta_hat": 0.15,
        }
        outcome = {
            "task_id": "t1",
            "generation_seed": 7,
            "receiver_agent_id": "r1",
            "candidate_memory_id": "mem1",
            "share": {"team_success": True},
            "withhold": {"team_success": False},
        }

        # R6 P0-9: formal mode never guesses raw vs calibrated.
        with pytest.raises(ValueError, match="requires trace fields"):
            _smtr_risk_utility_curve(
                [legacy_trace], [outcome], critic, experiment_mode="formal"
            )

        # Pilot mode falls back to eta_calibrated = trace["eta_hat"] with a
        # warning instead of failing.
        with pytest.warns(UserWarning, match="falls back"):
            result = _smtr_risk_utility_curve(
                [legacy_trace], [outcome], critic, experiment_mode="pilot"
            )
        assert result["n_matched_candidates"] == 1
