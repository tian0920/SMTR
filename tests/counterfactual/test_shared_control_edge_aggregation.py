"""Edge aggregation is schema-version agnostic (清单 Shared-Control 第10章).

v3 shared-control records and legacy v2 records must produce identical
per-edge empirical statistics for the same outcome sequence.
"""

from __future__ import annotations

from smtr.marble.real_pairs import aggregate_edge_records

OUTCOMES = [
    (True, False),
    (True, False),
    (False, True),
    (True, True),
    (False, False),
    (True, False),
    (True, True),
    (False, False),
]


def _record(*, seed: int, share_success: bool, withhold_success: bool) -> dict:
    return {
        "edge_id": "edge_x1",
        "task_id": "t1",
        "receiver_agent_id": "r1",
        "candidate_memory_id": "m1",
        "generation_seed": seed,
        "valid": True,
        "share": {"team_success": share_success},
        "withhold": {"team_success": withhold_success},
    }


def _v2_records() -> list[dict]:
    return [
        _record(seed=seed, share_success=share, withhold_success=withhold)
        for seed, (share, withhold) in enumerate(OUTCOMES)
    ]


def _v4_records() -> list[dict]:
    records = []
    for seed, (share, withhold) in enumerate(OUTCOMES):
        rec = _record(seed=seed, share_success=share, withhold_success=withhold)
        rec.update(
            {
                "schema_version": "marble_candidate_pair_v4",
                "control_group_id": f"ctrl_{seed:016x}",
                "control_family_id": "t1::r1",
                "control_reused": True,
                "control_definition_version": "shared_no_memory_control_v1",
                "control_group_candidate_count": 1,
                "control_execution_position": "control_first",
                "share_execution_rank": 0,
            }
        )
        records.append(rec)
    return records


def test_v4_and_v2_records_aggregate_identically():
    v2 = aggregate_edge_records(_v2_records())
    v4 = aggregate_edge_records(_v4_records())
    assert len(v2) == 1
    assert len(v4) == 1
    for key in (
        "q00_empirical",
        "q01_empirical",
        "q10_empirical",
        "q11_empirical",
        "tau_empirical",
        "eta_empirical",
        "n_replicates",
        "n_attempted",
    ):
        assert v4[0][key] == v2[0][key], key


def test_aggregate_values_match_expected_counts():
    agg = aggregate_edge_records(_v4_records())[0]
    assert agg["n_replicates"] == 8
    assert agg["n_attempted"] == 8
    assert agg["q00_empirical"] == 2 / 8
    assert agg["q01_empirical"] == 1 / 8
    assert agg["q10_empirical"] == 3 / 8
    assert agg["q11_empirical"] == 2 / 8
    assert agg["tau_empirical"] == 2 / 8
    assert agg["eta_empirical"] == 1 / 8


def test_invalid_v4_records_are_excluded_from_probabilities():
    records = _v4_records()
    records.append(
        {
            **_record(seed=99, share_success=True, withhold_success=True),
            "schema_version": "marble_candidate_pair_v4",
            "control_group_id": "ctrl_deadbeef",
            "valid": False,
            "invalid_reason": "shared_control_invalid:mock_control_failure",
        }
    )
    agg = aggregate_edge_records(records)[0]
    assert agg["n_replicates"] == 8
    assert agg["n_attempted"] == 9
