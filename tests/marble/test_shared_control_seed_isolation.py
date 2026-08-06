"""Shared-control seed isolation: one distinct control per seed."""

from __future__ import annotations

from tests.marble._shared_control_harness import run_generate


def test_five_seeds_yield_five_distinct_controls(tmp_path):
    """Formal mode: 5 seeds -> 5 controls with distinct group ids."""
    out = run_generate(
        tmp_path,
        entries=[{
            "task_id": "t1",
            "receiver_agent_id": "r1",
            "memory_ids": ["m1", "m2"],
        }],
        seeds=[0, 1, 2, 3, 4],
        experiment_mode="formal",
    )
    runner = out["runner"]
    assert len(runner.control_calls) == 5
    group_ids = [call["control_group_id"] for call in runner.control_calls]
    assert len(set(group_ids)) == 5
    # 2 candidates x 5 seeds = 10 paired records.
    assert len(out["records"]) == 10
    assert len(runner.share_calls) == 2 * 5


def test_records_pair_each_seed_with_its_own_control(tmp_path):
    """Each seed's records reference only that seed's control group."""
    out = run_generate(
        tmp_path,
        entries=[{
            "task_id": "t1",
            "receiver_agent_id": "r1",
            "memory_ids": ["m1", "m2"],
        }],
        seeds=[0, 1, 2, 3, 4],
        experiment_mode="formal",
    )
    by_seed: dict[int, set[str]] = {}
    for rec in out["records"]:
        by_seed.setdefault(rec["generation_seed"], set()).add(
            rec["control_group_id"]
        )
    assert set(by_seed) == {0, 1, 2, 3, 4}
    per_seed_groups = {seed: groups.pop() for seed, groups in by_seed.items()}
    # Every seed has exactly one control and they never overlap.
    assert len(set(per_seed_groups.values())) == 5
    for rec in out["records"]:
        assert rec["control_group_id"] == per_seed_groups[rec["generation_seed"]]
