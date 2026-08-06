"""Control failure propagation (清单 Shared-Control 第6章).

One invalid shared control must invalidate every paired record of its
group; no automatic withhold re-run may rescue it.
"""

from __future__ import annotations

from smtr.marble.real_pairs import compute_control_group_id
from tests.marble._shared_control_harness import (
    FakeSharedControlRunner,
    run_generate,
)


def test_invalid_control_invalidates_its_whole_group(tmp_path):
    broken_group = compute_control_group_id(
        split_name="validation",
        scenario="database",
        task_id="t1",
        receiver_agent_id="r1",
        generation_seed=0,
    )
    out = run_generate(
        tmp_path,
        entries=[{
            "task_id": "t1",
            "receiver_agent_id": "r1",
            "memory_ids": ["m1", "m2"],
        }],
        seeds=[0, 1, 2],
        runner=FakeSharedControlRunner(
            invalid_control_groups={broken_group},
        ),
    )
    records = out["records"]
    assert len(records) == 6

    for rec in records:
        if rec["generation_seed"] == 0:
            assert rec["valid"] is False, rec
            assert str(rec["invalid_reason"]).startswith(
                "shared_control_invalid:"
            ), rec
        else:
            assert rec["valid"] is True, rec
            assert rec["invalid_reason"] in (None, "")

    # No re-run of the broken control: exactly one call per seed.
    assert len(out["runner"].control_calls) == 3


def test_valid_groups_keep_all_records(tmp_path):
    out = run_generate(
        tmp_path,
        entries=[{
            "task_id": "t1",
            "receiver_agent_id": "r1",
            "memory_ids": ["m1", "m2"],
        }],
        seeds=[0, 1, 2],
    )
    assert all(rec["valid"] for rec in out["records"])
