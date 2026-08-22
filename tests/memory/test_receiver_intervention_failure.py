"""Tests for MissingCounterfactualOutcomeError in ReceiverInterventionEvaluator.

Covers:
- missing outcome (no source available)
- malformed reward (non-tuple, wrong length)
- nan / inf reward
- valid outcome (happy path, no error)
"""

import math

import pytest

from smtr.memory.receiver_intervention import (
    MissingCounterfactualOutcomeError,
    ReceiverInterventionEvaluator,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _eval() -> ReceiverInterventionEvaluator:
    return ReceiverInterventionEvaluator()


def _eval_with_fn(fn):
    return ReceiverInterventionEvaluator(outcome_fn=fn)


# ---------------------------------------------------------------------------
# Tests: missing outcome
# ---------------------------------------------------------------------------

class TestMissingOutcome:
    def test_no_source_raises(self) -> None:
        """outcome_fn is None AND receiver not in paired_outcomes → raise."""
        ev = _eval()
        with pytest.raises(MissingCounterfactualOutcomeError) as exc_info:
            ev.evaluate(
                memory_id="m1",
                receiver_ids=["agent1"],
                episode_id=42,
            )
        err = exc_info.value
        assert err.memory_id == "m1"
        assert err.receiver_id == "agent1"
        assert err.episode_id == 42
        assert "no outcome source" in str(err)

    def test_receiver_not_in_paired_outcomes_raises(self) -> None:
        """paired_outcomes exists but doesn't contain the requested receiver."""
        ev = _eval()
        with pytest.raises(MissingCounterfactualOutcomeError) as exc_info:
            ev.evaluate(
                memory_id="m1",
                receiver_ids=["agent1", "agent2"],
                paired_outcomes={"agent1": (1.0, 0.0)},
                episode_id=7,
            )
        assert exc_info.value.receiver_id == "agent2"

    def test_outcome_fn_returns_none_raises(self) -> None:
        """outcome_fn returns None → MissingCounterfactualOutcomeError."""
        ev = _eval_with_fn(lambda mid, rid, state: None)
        with pytest.raises(MissingCounterfactualOutcomeError) as exc_info:
            ev.evaluate(
                memory_id="m2",
                receiver_ids=["agent1"],
                episode_id=10,
            )
        assert "no outcome source" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Tests: malformed reward
# ---------------------------------------------------------------------------

class TestMalformedReward:
    def test_none_expose_raises(self) -> None:
        """expose_reward is None → raise."""
        ev = _eval()
        with pytest.raises(MissingCounterfactualOutcomeError) as exc_info:
            ev.evaluate(
                memory_id="m1",
                receiver_ids=["agent1"],
                paired_outcomes={"agent1": (None, 0.0)},
                episode_id=1,
            )
        assert "malformed" in str(exc_info.value)

    def test_none_withhold_raises(self) -> None:
        """withhold_reward is None → raise."""
        ev = _eval()
        with pytest.raises(MissingCounterfactualOutcomeError):
            ev.evaluate(
                memory_id="m1",
                receiver_ids=["agent1"],
                paired_outcomes={"agent1": (1.0, None)},
            )

    def test_string_reward_raises(self) -> None:
        """Non-numeric reward → raise."""
        ev = _eval()
        with pytest.raises(MissingCounterfactualOutcomeError) as exc_info:
            ev.evaluate(
                memory_id="m1",
                receiver_ids=["agent1"],
                paired_outcomes={"agent1": ("bad", 0.0)},
            )
        assert "malformed" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Tests: nan / inf reward
# ---------------------------------------------------------------------------

class TestNonFiniteReward:
    def test_nan_raises(self) -> None:
        ev = _eval()
        with pytest.raises(MissingCounterfactualOutcomeError) as exc_info:
            ev.evaluate(
                memory_id="m1",
                receiver_ids=["agent1"],
                paired_outcomes={"agent1": (float("nan"), 0.0)},
            )
        assert "not finite" in str(exc_info.value)

    def test_inf_raises(self) -> None:
        ev = _eval()
        with pytest.raises(MissingCounterfactualOutcomeError) as exc_info:
            ev.evaluate(
                memory_id="m1",
                receiver_ids=["agent1"],
                paired_outcomes={"agent1": (1.0, float("inf"))},
            )
        assert "not finite" in str(exc_info.value)

    def test_negative_inf_raises(self) -> None:
        ev = _eval()
        with pytest.raises(MissingCounterfactualOutcomeError):
            ev.evaluate(
                memory_id="m1",
                receiver_ids=["agent1"],
                paired_outcomes={"agent1": (float("-inf"), 0.0)},
            )


# ---------------------------------------------------------------------------
# Tests: valid outcome (happy path)
# ---------------------------------------------------------------------------

class TestValidOutcome:
    def test_valid_paired_outcomes(self) -> None:
        """Valid paired outcomes produce correct decisions."""
        ev = _eval()
        result = ev.evaluate(
            memory_id="m1",
            receiver_ids=["agent1", "agent2"],
            paired_outcomes={
                "agent1": (1.0, 0.0),  # delta=1.0 → validated
                "agent2": (0.0, 1.0),  # delta=-1.0 → rejected
            },
            episode_id=42,
        )
        assert result.n_validated == 1
        assert result.n_rejected == 1
        assert result.validated_receivers == ["agent1"]
        assert result.rejected_receivers == ["agent2"]

    def test_valid_outcome_fn(self) -> None:
        """Valid outcome_fn works correctly."""
        ev = _eval_with_fn(lambda mid, rid, state: (0.8, 0.3))
        result = ev.evaluate(
            memory_id="m1",
            receiver_ids=["agent1"],
            episode_id=5,
        )
        assert result.n_validated == 1
        assert result.receiver_outcomes[0].delta == pytest.approx(0.5)

    def test_zero_delta_rejects(self) -> None:
        """Zero delta → rejected (this is a real measurement, not missing)."""
        ev = _eval()
        result = ev.evaluate(
            memory_id="m1",
            receiver_ids=["agent1"],
            paired_outcomes={"agent1": (0.5, 0.5)},
            episode_id=1,
        )
        assert result.n_rejected == 1
        assert result.n_validated == 0

    def test_episode_id_in_error(self) -> None:
        """Error message includes memory_id, receiver_id, episode_id."""
        ev = _eval()
        with pytest.raises(MissingCounterfactualOutcomeError) as exc_info:
            ev.evaluate(
                memory_id="mem_abc",
                receiver_ids=["agent2"],
                episode_id=99,
            )
        msg = str(exc_info.value)
        assert "mem_abc" in msg
        assert "agent2" in msg
        assert "99" in msg
