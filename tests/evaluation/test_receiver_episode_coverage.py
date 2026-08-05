"""清单 Test 6: receiver episode coverage (P0-13).

One receiver/seed carries exactly one final policy trace; the number of
candidates K inspected by the router must never divide the coverage.
"""

from __future__ import annotations

from smtr.evaluation.metrics import compute_receiver_episode_coverage


def _record(memory_id: str, seed: int) -> dict:
    return {
        "task_id": "t1",
        "receiver_agent_id": "r1",
        "candidate_memory_id": memory_id,
        "generation_seed": seed,
        "valid": True,
        "label": "positive_transfer",
        "y_share": 1,
        "y_withhold": 0,
        "share": {"team_success": True},
        "withhold": {"team_success": False},
    }


def _policy_trace(seed: int, selected: str | None) -> dict:
    return {
        "trace_type": "receiver_policy",
        "task_id": "t1",
        "receiver_agent_id": "r1",
        "generation_seed": seed,
        "selected_memory_id": selected,
        "policy_action": "share" if selected else "withhold",
    }


def test_multiple_candidates_do_not_divide_coverage_by_k():
    # K=3 candidates x 2 seeds => 6 records, but only 2 episodes.
    records = [
        _record(mem, seed) for mem in ("m1", "m2", "m3") for seed in (0, 1)
    ]
    traces = [_policy_trace(0, "m1"), _policy_trace(1, None)]
    result = compute_receiver_episode_coverage(
        receiver_policy_traces=traces, paired_records=records
    )
    # One trace per receiver/seed: coverage must be 1.0, not 2/6.
    assert result["receiver_episode_coverage"] == 1.0
    assert result["expected_receiver_seed_count"] == 2
    assert result["matched_receiver_seed_count"] == 2


def test_full_coverage_when_every_episode_evaluable():
    records = [_record("m1", 0), _record("m2", 0), _record("m1", 1)]
    traces = [_policy_trace(0, "m1"), _policy_trace(1, None)]
    result = compute_receiver_episode_coverage(
        receiver_policy_traces=traces, paired_records=records
    )
    assert result["receiver_episode_coverage"] == 1.0
    assert result["missing_receiver_seed_count"] == 0
    assert result["unexpected_receiver_policy_trace_count"] == 0


def test_missing_episode_lowers_coverage():
    records = [_record("m1", 0), _record("m1", 1)]
    traces = [_policy_trace(0, None)]
    result = compute_receiver_episode_coverage(
        receiver_policy_traces=traces, paired_records=records
    )
    assert result["receiver_episode_coverage"] == 0.5
    assert result["missing_receiver_seed_count"] == 1
