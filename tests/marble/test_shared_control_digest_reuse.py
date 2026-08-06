"""Control digest reuse: every record of a group reuses the same control."""

from __future__ import annotations

from tests.marble._shared_control_harness import run_generate


def test_records_within_one_group_reuse_identical_control(tmp_path):
    """Same seed -> same control group id, raw result digest and outcome."""
    out = run_generate(
        tmp_path,
        entries=[{
            "task_id": "t1",
            "receiver_agent_id": "r1",
            "memory_ids": ["m1", "m2", "m3"],
        }],
        seeds=[0, 1, 2],
    )
    records = out["records"]
    assert len(records) == 9

    by_seed: dict[int, list[dict]] = {}
    for rec in records:
        by_seed.setdefault(rec["generation_seed"], []).append(rec)

    for seed, group_records in by_seed.items():
        assert len(group_records) == 3
        group_ids = {rec["control_group_id"] for rec in group_records}
        raw_digests = {
            rec["digests"]["control_raw_result_digest"] for rec in group_records
        }
        withhold_outcomes = {
            rec["withhold"]["team_success"] for rec in group_records
        }
        assert len(group_ids) == 1, seed
        assert len(raw_digests) == 1, seed
        assert len(withhold_outcomes) == 1, seed
        # Control-side digests are identical across the whole group.
        reference = group_records[0]["digests"]
        for rec in group_records[1:]:
            for key in reference:
                if key.startswith("control_"):
                    assert rec["digests"][key] == reference[key], key


def test_different_seeds_never_share_a_control(tmp_path):
    out = run_generate(
        tmp_path,
        entries=[{
            "task_id": "t1",
            "receiver_agent_id": "r1",
            "memory_ids": ["m1", "m2", "m3"],
        }],
        seeds=[0, 1, 2],
    )
    by_seed = {
        rec["generation_seed"]: rec["control_group_id"] for rec in out["records"]
    }
    assert set(by_seed) == {0, 1, 2}
    assert len(set(by_seed.values())) == 3
