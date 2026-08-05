"""清单 Test 5: candidate decision coverage (P0-12).

Coverage denominator is the set of core-valid candidate-seed records, not
the trace count; missing traces lower coverage and foreign traces count
as unexpected.
"""

from __future__ import annotations

from smtr.evaluation.metrics import compute_candidate_decision_coverage


def _record(memory_id: str, seed: int, withhold_success: bool = False) -> dict:
    return {
        "task_id": "t1",
        "receiver_agent_id": "r1",
        "candidate_memory_id": memory_id,
        "generation_seed": seed,
        "valid": True,
        "label": "positive_transfer",
        "y_share": 1,
        "y_withhold": int(withhold_success),
        "share": {"team_success": True},
        "withhold": {"team_success": withhold_success},
    }


def _trace(memory_id: str, seed: int) -> dict:
    return {
        "trace_type": "candidate_decision",
        "task_id": "t1",
        "receiver_agent_id": "r1",
        "candidate_memory_id": memory_id,
        "generation_seed": seed,
        "action": "share",
    }


def test_full_coverage_when_every_record_has_trace():
    records = [_record("m1", 0), _record("m1", 1), _record("m2", 0)]
    traces = [_trace("m1", 0), _trace("m1", 1), _trace("m2", 0)]
    result = compute_candidate_decision_coverage(
        candidate_decision_traces=traces, paired_records=records
    )
    assert result["candidate_decision_coverage"] == 1.0
    assert result["valid_candidate_seed_count"] == 3
    assert result["matched_candidate_seed_count"] == 3
    assert result["missing_candidate_seed_count"] == 0
    assert result["unexpected_candidate_seed_trace_count"] == 0


def test_missing_trace_lowers_coverage():
    records = [_record("m1", 0), _record("m1", 1), _record("m2", 0)]
    traces = [_trace("m1", 0), _trace("m2", 0)]
    result = compute_candidate_decision_coverage(
        candidate_decision_traces=traces, paired_records=records
    )
    assert result["candidate_decision_coverage"] < 1.0
    assert result["candidate_decision_coverage"] == 2 / 3
    assert result["missing_candidate_seed_count"] == 1
    assert result["unexpected_candidate_seed_trace_count"] == 0


def test_foreign_trace_counts_as_unexpected():
    records = [_record("m1", 0)]
    traces = [_trace("m1", 0), _trace("m1", 7)]  # seed 7 never observed
    result = compute_candidate_decision_coverage(
        candidate_decision_traces=traces, paired_records=records
    )
    assert result["candidate_decision_coverage"] == 1.0
    assert result["unexpected_candidate_seed_trace_count"] == 1
