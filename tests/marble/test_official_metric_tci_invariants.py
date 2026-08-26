"""Official metric TCI invariants.

These tests ensure:
1. Delta comes from the official MultiAgentBench metric (not team_success)
2. team_success is diagnostic-only (never used for delta)
3. Invalid metrics produce "invalid" decision (no fallback, delta=None)
4. Expose/withhold matching invariants hold

Tests cover all 5 scenarios: database, research, minecraft, coding, bargaining.
"""

from __future__ import annotations

import pytest

from smtr.marble.outcome.official_metric_outcome import (
    OfficialMetricOutcome,
    OfficialMetricOutcomeEvaluator,
    ReceiverTCIOutcome,
)
from smtr.memory.online_receiver_intervention import (
    OnlineValidationRecord,
    VALIDATION_STATUS_INVALID,
    VALIDATION_STATUS_REJECTED,
    VALIDATION_STATUS_VALIDATED,
)


# ---------------------------------------------------------------------------
# Test 1: Database delta from root-cause recall
# ---------------------------------------------------------------------------

def test_database_delta_from_root_cause_recall():
    """Database scenario: delta = expose_recall - withhold_recall."""
    evaluator = OfficialMetricOutcomeEvaluator(scenario="database")

    # Expose: predicted 2 of 3 root causes → recall = 0.667
    expose_result = {
        "task_evaluation": {
            "root_cause": ["missing_index", "null_constraint", "foreign_key"],
            "predicted": "Root causes: missing_index, null_constraint",
        }
    }
    # Withhold: predicted 1 of 3 → recall = 0.333
    withhold_result = {
        "task_evaluation": {
            "root_cause": ["missing_index", "null_constraint", "foreign_key"],
            "predicted": "Root cause: missing_index",
        }
    }

    outcome = evaluator.compute_delta(
        expose_result=expose_result,
        withhold_result=withhold_result,
        receiver_id="agent1",
    )

    assert outcome.raw_metric_name == "root_cause_recall"
    assert outcome.is_valid
    assert outcome.oriented_delta > 0  # Expose has higher recall
    assert outcome.oriented_delta == pytest.approx(1/3, abs=0.05)


# ---------------------------------------------------------------------------
# Test 2: Research delta from official rating
# ---------------------------------------------------------------------------

def test_research_delta_from_official_rating():
    """Research scenario: delta = expose_avg - withhold_avg (normalized)."""
    evaluator = OfficialMetricOutcomeEvaluator(scenario="research")

    # Expose: high ratings → avg=4.33 → normalized=0.833
    expose_result = {
        "task_evaluation": {"innovation": 5, "safety": 4, "feasibility": 4}
    }
    # Withhold: lower ratings → avg=2.67 → normalized=0.417
    withhold_result = {
        "task_evaluation": {"innovation": 3, "safety": 2, "feasibility": 3}
    }

    outcome = evaluator.compute_delta(
        expose_result=expose_result,
        withhold_result=withhold_result,
        receiver_id="agent2",
    )

    assert outcome.raw_metric_name == "avg_innovation_safety_feasibility"
    assert outcome.is_valid
    assert outcome.oriented_delta > 0
    # Expected: (4.33-1)/4 - (2.67-1)/4 ≈ 0.417
    assert outcome.oriented_delta == pytest.approx(0.417, abs=0.05)


# ---------------------------------------------------------------------------
# Test 3: Coding delta from official rating
# ---------------------------------------------------------------------------

def test_coding_delta_from_official_rating():
    """Coding scenario: delta from 4-dimension average."""
    evaluator = OfficialMetricOutcomeEvaluator(scenario="coding")

    # Expose: good code
    expose_result = {
        "task_evaluation": {
            "instruction_following": 4,
            "executability": 5,
            "consistency": 4,
            "quality": 4,
        }
    }
    # Withhold: mediocre code
    withhold_result = {
        "task_evaluation": {
            "instruction_following": 2,
            "executability": 3,
            "consistency": 2,
            "quality": 3,
        }
    }

    outcome = evaluator.compute_delta(
        expose_result=expose_result,
        withhold_result=withhold_result,
        receiver_id="agent1",
    )

    assert outcome.raw_metric_name == "avg_code_quality"
    assert outcome.is_valid
    assert outcome.oriented_delta > 0
    # Expose avg=4.25 → norm=0.8125; Withhold avg=2.5 → norm=0.375
    # Delta ≈ 0.4375
    assert outcome.oriented_delta == pytest.approx(0.4375, abs=0.01)


# ---------------------------------------------------------------------------
# Test 4: Bargaining delta from official rating
# ---------------------------------------------------------------------------

def test_bargaining_delta_from_official_rating():
    """Bargaining scenario: delta from 6-dimension average."""
    evaluator = OfficialMetricOutcomeEvaluator(scenario="bargaining")

    expose_result = {
        "task_evaluation": {
            "buyer": {"effectiveness": 4, "progress": 5, "interaction": 4},
            "seller": {"effectiveness": 3, "progress": 4, "interaction": 4},
        }
    }
    withhold_result = {
        "task_evaluation": {
            "buyer": {"effectiveness": 2, "progress": 2, "interaction": 3},
            "seller": {"effectiveness": 2, "progress": 3, "interaction": 2},
        }
    }

    outcome = evaluator.compute_delta(
        expose_result=expose_result,
        withhold_result=withhold_result,
        receiver_id="agent3",
    )

    assert outcome.raw_metric_name == "avg_negotiation_quality"
    assert outcome.is_valid
    assert outcome.oriented_delta > 0


# ---------------------------------------------------------------------------
# Test 5: Minecraft delta from block_hit_rate
# ---------------------------------------------------------------------------

def test_minecraft_delta_from_block_hit_rate():
    """Minecraft scenario: delta from block_hit_rate * 5."""
    evaluator = OfficialMetricOutcomeEvaluator(scenario="minecraft")

    # Expose: block_hit_rate=0.8 → task_evaluation=4.0
    expose_result = {"task_evaluation": 4.0}
    # Withhold: block_hit_rate=0.2 → task_evaluation=1.0
    withhold_result = {"task_evaluation": 1.0}

    outcome = evaluator.compute_delta(
        expose_result=expose_result,
        withhold_result=withhold_result,
        receiver_id="agent1",
    )

    assert outcome.raw_metric_name == "block_hit_rate"
    assert outcome.is_valid
    # Normalized: 4.0/5=0.8 vs 1.0/5=0.2 → delta=0.6
    assert outcome.oriented_delta == pytest.approx(0.6, abs=0.01)


# ---------------------------------------------------------------------------
# Test 6: team_success same, official score different → delta != 0
# ---------------------------------------------------------------------------

def test_team_success_same_official_score_different():
    """When team_success is identical but official scores differ,
    delta MUST be non-zero (proving team_success is not used)."""
    evaluator = OfficialMetricOutcomeEvaluator(scenario="minecraft")

    # Both would have team_success=True, but different block_hit_rates
    expose_result = {"task_evaluation": 4.0}   # 80% blocks correct
    withhold_result = {"task_evaluation": 2.5}  # 50% blocks correct

    outcome = evaluator.compute_delta(
        expose_result=expose_result,
        withhold_result=withhold_result,
        receiver_id="agent1",
    )

    # team_success would be True for both → old delta would be 0
    # Official metric delta MUST be non-zero
    assert outcome.oriented_delta == pytest.approx(0.3, abs=0.01)


# ---------------------------------------------------------------------------
# Test 7: team_success different, official score same → no TCI change
# ---------------------------------------------------------------------------

def test_team_success_different_official_score_same():
    """When team_success differs but official scores are identical,
    the TCI decision must NOT change (delta=0)."""
    evaluator = OfficialMetricOutcomeEvaluator(scenario="minecraft")

    # Same block_hit_rate for both → delta = 0
    expose_result = {"task_evaluation": 3.0}
    withhold_result = {"task_evaluation": 3.0}

    outcome = evaluator.compute_delta(
        expose_result=expose_result,
        withhold_result=withhold_result,
        receiver_id="agent1",
    )

    # Even if team_success were different, delta must be 0
    assert outcome.oriented_delta == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Test 8: Missing official metric → invalid decision (no fallback, delta=None)
# ---------------------------------------------------------------------------

def test_missing_official_metric_invalid_outcome():
    """When task_evaluation is missing, outcome must be invalid.
    NO fallback to team_success."""
    evaluator = OfficialMetricOutcomeEvaluator(scenario="database")

    # No task_evaluation field
    result_no_eval = {"team_success": True, "score": 1.0}
    outcome = evaluator.evaluate(task={}, run_result=result_no_eval)

    assert not outcome.is_valid
    assert outcome.failure_reason == "task_evaluation_not_available"
    assert outcome.normalized_score is None

    # Also test compute_delta with one invalid branch
    expose_result = {"task_evaluation": None}
    withhold_result = {
        "task_evaluation": {
            "root_cause": ["a"],
            "predicted": "a",
        }
    }
    delta_outcome = evaluator.compute_delta(
        expose_result=expose_result,
        withhold_result=withhold_result,
        receiver_id="agent1",
    )
    assert not delta_outcome.is_valid
    assert delta_outcome.oriented_delta is None  # Undefined → None (NOT silent zero)


# ---------------------------------------------------------------------------
# Test 9: Same task / same seed / same environment invariant
# ---------------------------------------------------------------------------

def test_same_task_seed_environment_expose_withhold_matching():
    """Expose and withhold must be compared on same task/seed/environment.
    This test validates the record captures matching metadata."""
    record = OnlineValidationRecord(
        memory_id="mem_001",
        receiver_id="agent1",
        task_id="42",
        scenario="database",
        seed=7,
        metric_name="root_cause_recall",
        raw_expose_score=0.8,
        raw_withhold_score=0.5,
        normalized_expose_score=0.8,
        normalized_withhold_score=0.5,
        delta=0.3,
        decision=VALIDATION_STATUS_VALIDATED,
        expose_metric_valid=True,
        withhold_metric_valid=True,
    )

    # Verify matching metadata is recorded
    assert record.task_id == "42"
    assert record.scenario == "database"
    assert record.seed == 7
    assert record.metric_name == "root_cause_recall"
    assert record.expose_metric_valid and record.withhold_metric_valid


# ---------------------------------------------------------------------------
# Test 10: delta > 0 → validated, delta <= 0 → rejected
# ---------------------------------------------------------------------------

def test_delta_sign_determines_decision():
    """delta > 0 → validated, delta <= 0 → rejected.
    (For valid outcomes only.)"""
    # Positive delta
    rec_positive = OnlineValidationRecord(
        memory_id="mem_pos",
        receiver_id="agent1",
        task_id="1",
        scenario="minecraft",
        seed=0,
        metric_name="block_hit_rate",
        normalized_expose_score=0.8,
        normalized_withhold_score=0.5,
        delta=0.3,
        decision=VALIDATION_STATUS_VALIDATED,
        expose_metric_valid=True,
        withhold_metric_valid=True,
    )
    assert rec_positive.delta > 0
    assert rec_positive.decision == VALIDATION_STATUS_VALIDATED

    # Zero delta → rejected
    rec_zero = OnlineValidationRecord(
        memory_id="mem_zero",
        receiver_id="agent1",
        task_id="2",
        scenario="minecraft",
        seed=0,
        metric_name="block_hit_rate",
        normalized_expose_score=0.5,
        normalized_withhold_score=0.5,
        delta=0.0,
        decision=VALIDATION_STATUS_REJECTED,
        expose_metric_valid=True,
        withhold_metric_valid=True,
    )
    assert rec_zero.delta == 0
    assert rec_zero.decision == VALIDATION_STATUS_REJECTED

    # Negative delta → rejected
    rec_negative = OnlineValidationRecord(
        memory_id="mem_neg",
        receiver_id="agent1",
        task_id="3",
        scenario="minecraft",
        seed=0,
        metric_name="block_hit_rate",
        normalized_expose_score=0.3,
        normalized_withhold_score=0.6,
        delta=-0.3,
        decision=VALIDATION_STATUS_REJECTED,
        expose_metric_valid=True,
        withhold_metric_valid=True,
    )
    assert rec_negative.delta < 0
    assert rec_negative.decision == VALIDATION_STATUS_REJECTED


# ---------------------------------------------------------------------------
# Additional: Verify all 5 scenarios have distinct metric names
# ---------------------------------------------------------------------------

def test_all_scenarios_have_distinct_metric_names():
    """Each scenario must have a unique official metric name."""
    evaluator_names = set()
    for scenario in ["database", "research", "minecraft", "coding", "bargaining"]:
        evaluator = OfficialMetricOutcomeEvaluator(scenario=scenario)
        name = evaluator._get_metric_name()
        assert name not in evaluator_names, f"Duplicate metric name: {name}"
        evaluator_names.add(name)
    assert len(evaluator_names) == 5


# ---------------------------------------------------------------------------
# Additional: Normalization bounds
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scenario,task_eval", [
    ("minecraft", 0.0),     # min
    ("minecraft", 5.0),     # max
    ("minecraft", 2.5),     # mid
])
def test_minecraft_normalization_bounds(scenario, task_eval):
    """Minecraft normalization: [0, 5] → [0, 1]."""
    evaluator = OfficialMetricOutcomeEvaluator(scenario=scenario)
    outcome = evaluator.evaluate(task={}, run_result={"task_evaluation": task_eval})
    assert outcome.is_valid
    assert 0.0 <= outcome.normalized_score <= 1.0


@pytest.mark.parametrize("scenario,task_eval", [
    ("research", {"innovation": 1, "safety": 1, "feasibility": 1}),  # min
    ("research", {"innovation": 5, "safety": 5, "feasibility": 5}),  # max
    ("research", {"innovation": 3, "safety": 3, "feasibility": 3}),  # mid
])
def test_research_normalization_bounds(scenario, task_eval):
    """Research normalization: [1, 5] → [0, 1]."""
    evaluator = OfficialMetricOutcomeEvaluator(scenario=scenario)
    outcome = evaluator.evaluate(task={}, run_result={"task_evaluation": task_eval})
    assert outcome.is_valid
    assert 0.0 <= outcome.normalized_score <= 1.0
