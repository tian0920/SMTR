"""Episode cost accounting (清单 Shared-Control 第11章).

Shared controls turn the legacy 2-episodes-per-(edge, seed) protocol
into one control episode per group plus one share episode per record.
"""

from __future__ import annotations

from smtr.marble.real_pairs import episode_costs
from tests.marble._shared_control_harness import run_generate


def _record(*, control_group_id: str, share_ok: bool = True) -> dict:
    return {
        "valid": share_ok,
        "control_group_id": control_group_id,
        "share": {
            "real_engine_executed": share_ok,
            "environment_valid": share_ok,
            "native_evaluator_executed": share_ok,
            "cleanup_succeeded": share_ok,
        },
    }


def _manual_records() -> list[dict]:
    # 10 control groups, 4 candidate records per group.
    return [
        _record(control_group_id=f"ctrl_{g:016x}")
        for g in range(10)
        for _ in range(4)
    ]


def test_manual_cost_accounting():
    records = _manual_records()
    assert len(records) == 40
    costs = episode_costs(
        records=records,
        share_episode_attempt_count=40,
        control_episode_attempt_count=10,
        control_episode_valid_count=10,
    )
    assert costs["candidate_seed_attempt_count"] == 40
    assert costs["candidate_seed_record_count"] == 40
    assert costs["valid_candidate_seed_record_count"] == 40
    assert costs["share_episode_attempt_count"] == 40
    assert costs["share_episode_valid_count"] == 40
    assert costs["control_episode_attempt_count"] == 10
    assert costs["control_episode_valid_count"] == 10
    assert costs["actual_engine_episode_attempt_count"] == 50
    assert costs["legacy_equivalent_episode_count"] == 80
    assert costs["saved_episode_count"] == 30
    assert costs["episode_saving_fraction"] == 0.375
    assert costs["control_group_count"] == 10
    assert costs["mean_candidates_per_control"] == 4.0
    assert costs["median_candidates_per_control"] == 4.0


def test_invalid_share_blocks_do_not_count_as_valid():
    records = _manual_records()
    records[0] = _record(control_group_id="ctrl_0000000000000000", share_ok=False)
    costs = episode_costs(
        records=records,
        share_episode_attempt_count=40,
        control_episode_attempt_count=10,
        control_episode_valid_count=10,
    )
    assert costs["share_episode_valid_count"] == 39
    assert costs["valid_candidate_seed_record_count"] == 39
    # Attempt counts are caller-reported and unaffected by validity.
    assert costs["share_episode_attempt_count"] == 40
    assert costs["control_group_count"] == 10


def test_empty_records_yield_zero_costs():
    costs = episode_costs(
        records=[],
        share_episode_attempt_count=0,
        control_episode_attempt_count=0,
        control_episode_valid_count=0,
    )
    assert costs["legacy_equivalent_episode_count"] == 0
    assert costs["saved_episode_count"] == 0
    assert costs["episode_saving_fraction"] == 0.0
    assert costs["mean_candidates_per_control"] == 0.0
    assert costs["median_candidates_per_control"] == 0.0


def test_pipeline_result_reports_episode_savings(tmp_path):
    """2 receivers x 4 candidates x 5 seeds -> controls=10, actual=50,
    legacy=80, saved=30, fraction=0.375 (清单 19.19)."""
    out = run_generate(
        tmp_path,
        entries=[
            {
                "task_id": "t1",
                "receiver_agent_id": "r1",
                "memory_ids": ["m1", "m2", "m3", "m4"],
            },
            {
                "task_id": "t1",
                "receiver_agent_id": "r2",
                "memory_ids": ["m1", "m2", "m3", "m4"],
            },
        ],
        seeds=list(range(5)),
        experiment_mode="formal",
    )
    result = out["result"]
    assert len(out["records"]) == 40
    assert result["control_episode_attempt_count"] == 10
    assert result["share_episode_attempt_count"] == 40
    assert result["actual_engine_episode_attempt_count"] == 50
    assert result["legacy_equivalent_episode_count"] == 80
    assert result["saved_episode_count"] == 30
    assert result["episode_saving_fraction"] == 0.375
    assert result["control_group_count"] == 10
    assert result["mean_candidates_per_control"] == 4.0
