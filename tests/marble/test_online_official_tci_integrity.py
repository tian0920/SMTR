"""P0-5 Integrity Suite — Cases 1-3: Official Task Score → TCI delta.

Invariants tested:
  1. TCI delta is computed ONLY from official Task Score (not team_success).
  2. team_success does NOT affect TCI decision.
  3. Invalid official outcome → delta=None (NOT delta=0, silent-zero ban).
"""

from __future__ import annotations

import pytest

from smtr.marble.outcome.official_metric_outcome import (
    OfficialMetricOutcomeEvaluator,
    ReceiverTCIOutcome,
)
from smtr.memory.online_receiver_intervention import (
    OnlineValidationRecord,
    VALIDATION_STATUS_INVALID,
    VALIDATION_STATUS_VALIDATED,
    VALIDATION_STATUS_REJECTED,
)


# -----------------------------------------------------------------------
# Case 1: TCI delta only from official Task Score
# -----------------------------------------------------------------------

class TestTCIDeltaFromOfficialTaskScore:
    """TCI delta must be computed exclusively from official metric scores."""

    def test_01_bargaining_delta_matches_official_metric(self):
        """Bargaining scenario: delta = expose_normalized - withhold_normalized."""
        evaluator = OfficialMetricOutcomeEvaluator(scenario="bargaining")

        expose_result = {
            "task_evaluation": {
                "buyer": {"effectiveness": 4, "progress": 3, "interaction": 4},
                "seller": {"effectiveness": 3, "progress": 4, "interaction": 3},
            }
        }
        withhold_result = {
            "task_evaluation": {
                "buyer": {"effectiveness": 2, "progress": 2, "interaction": 2},
                "seller": {"effectiveness": 2, "progress": 2, "interaction": 2},
            }
        }

        outcome = evaluator.compute_delta(
            expose_result=expose_result,
            withhold_result=withhold_result,
            receiver_id="agent1",
        )

        assert isinstance(outcome, ReceiverTCIOutcome)
        assert outcome.raw_metric_name == "avg_negotiation_quality"
        assert outcome.oriented_delta is not None
        # expose avg = (4+3+4+3+4+3)/6 = 3.5 → normalized = (3.5-1)/4 = 0.625
        # withhold avg = (2+2+2+2+2+2)/6 = 2.0 → normalized = (2.0-1)/4 = 0.25
        # delta = 0.625 - 0.25 = 0.375
        assert abs(outcome.oriented_delta - 0.375) < 1e-6

    def test_01_database_delta_from_root_cause_recall(self):
        """Database scenario: delta computed from root_cause_recall."""
        evaluator = OfficialMetricOutcomeEvaluator(scenario="database")

        expose_result = {
            "task_evaluation": {
                "root_cause": ["memory_leak", "null_pointer"],
                "predicted": ["memory_leak", "null_pointer"],
            }
        }
        withhold_result = {
            "task_evaluation": {
                "root_cause": ["memory_leak", "null_pointer"],
                "predicted": ["memory_leak"],
            }
        }

        outcome = evaluator.compute_delta(
            expose_result=expose_result,
            withhold_result=withhold_result,
            receiver_id="agent1",
        )

        assert outcome.oriented_delta is not None
        # expose: recall = 2/2 = 1.0
        # withhold: recall = 1/2 = 0.5
        # delta = 1.0 - 0.5 = 0.5
        assert abs(outcome.oriented_delta - 0.5) < 1e-6


# -----------------------------------------------------------------------
# Case 2: team_success does NOT affect TCI decision
# -----------------------------------------------------------------------

class TestTeamSuccessDoesNotAffectDecision:
    """team_success is diagnostic only; it must not influence delta or decision."""

    def test_02_team_success_ignored_when_official_metric_valid(self):
        """Even if team_success differs, delta comes from official metric."""
        evaluator = OfficialMetricOutcomeEvaluator(scenario="minecraft")

        # expose: block_hit_rate = 4/5 = 0.8, withhold: 2/5 = 0.4
        expose_result = {"task_evaluation": 4.0}
        withhold_result = {"task_evaluation": 2.0}

        outcome = evaluator.compute_delta(
            expose_result=expose_result,
            withhold_result=withhold_result,
            receiver_id="agent1",
        )

        # Delta is purely from official metric
        assert outcome.oriented_delta is not None
        assert abs(outcome.oriented_delta - 0.4) < 1e-6  # 0.8 - 0.4
        assert outcome.has_positive_delta is True

    def test_02_team_success_true_both_branches_no_effect(self):
        """Both branches succeed (team_success=True) but official metric differs."""
        evaluator = OfficialMetricOutcomeEvaluator(scenario="research")

        expose_result = {
            "task_evaluation": {"innovation": 4, "safety": 4, "feasibility": 4},
        }
        withhold_result = {
            "task_evaluation": {"innovation": 2, "safety": 2, "feasibility": 2},
        }

        outcome = evaluator.compute_delta(
            expose_result=expose_result,
            withhold_result=withhold_result,
            receiver_id="agent1",
        )

        # expose avg=4 → norm=(4-1)/4=0.75; withhold avg=2 → norm=(2-1)/4=0.25
        assert outcome.oriented_delta is not None
        assert abs(outcome.oriented_delta - 0.5) < 1e-6


# -----------------------------------------------------------------------
# Case 3: Invalid official outcome → delta=None (silent-zero ban)
# -----------------------------------------------------------------------

class TestInvalidOutcomeDeltaIsNone:
    """Invalid official outcome MUST produce delta=None, never delta=0."""

    def test_03_missing_task_evaluation_produces_none_delta(self):
        """Missing task_evaluation → is_valid=False → oriented_delta=None."""
        evaluator = OfficialMetricOutcomeEvaluator(scenario="bargaining")

        expose_result = {"task_evaluation": None}  # invalid
        withhold_result = {
            "task_evaluation": {
                "buyer": {"effectiveness": 3, "progress": 3, "interaction": 3},
                "seller": {"effectiveness": 3, "progress": 3, "interaction": 3},
            }
        }

        outcome = evaluator.compute_delta(
            expose_result=expose_result,
            withhold_result=withhold_result,
            receiver_id="agent1",
        )

        # CRITICAL: delta must be None, NOT 0.0
        assert outcome.oriented_delta is None
        assert outcome.is_valid is False
        assert outcome.has_positive_delta is False
        assert outcome.has_negative_delta is False

    def test_03_both_branches_invalid_produces_none_delta(self):
        """Both branches invalid → oriented_delta=None."""
        evaluator = OfficialMetricOutcomeEvaluator(scenario="coding")

        expose_result = {}  # no task_evaluation, no code_quality
        withhold_result = {}

        outcome = evaluator.compute_delta(
            expose_result=expose_result,
            withhold_result=withhold_result,
            receiver_id="agent1",
        )

        assert outcome.oriented_delta is None
        assert outcome.is_valid is False

    def test_03b_online_validation_record_invalid_has_none_delta(self):
        """OnlineValidationRecord with invalid decision has delta=None."""
        record = OnlineValidationRecord(
            memory_id="mem_001",
            receiver_id="agent1",
            task_id="task_01",
            scenario="bargaining",
            seed=0,
            delta=None,
            decision=VALIDATION_STATUS_INVALID,
            expose_metric_valid=False,
            withhold_metric_valid=True,
        )

        assert record.delta is None
        assert record.decision == "invalid"
        assert record.decision != VALIDATION_STATUS_VALIDATED
        assert record.decision != VALIDATION_STATUS_REJECTED

    def test_03c_valid_record_has_numeric_delta(self):
        """Valid record must have a numeric delta (not None)."""
        record = OnlineValidationRecord(
            memory_id="mem_002",
            receiver_id="agent1",
            task_id="task_01",
            scenario="bargaining",
            seed=0,
            delta=0.15,
            decision=VALIDATION_STATUS_VALIDATED,
            expose_metric_valid=True,
            withhold_metric_valid=True,
            normalized_expose_score=0.6,
            normalized_withhold_score=0.45,
        )

        assert record.delta is not None
        assert record.delta > 0
        assert record.decision == "validated"
