"""Share failure isolation (清单 Shared-Control 第6章).

A failed share branch invalidates only its own record; the shared
control and every sibling share stay valid.
"""

from __future__ import annotations

from smtr.marble.real_pairs import compute_edge_id
from tests.marble._shared_control_harness import (
    FakeSharedControlRunner,
    run_generate,
)


def test_failed_share_invalidates_only_its_edge(tmp_path):
    broken_edge = compute_edge_id("t1", "r1", "m2")
    out = run_generate(
        tmp_path,
        entries=[{
            "task_id": "t1",
            "receiver_agent_id": "r1",
            "memory_ids": ["m1", "m2", "m3"],
        }],
        seeds=[0, 1, 2],
        runner=FakeSharedControlRunner(invalid_share_edges={broken_edge}),
    )
    records = out["records"]
    assert len(records) == 9

    for rec in records:
        if rec["candidate_memory_id"] == "m2":
            assert rec["valid"] is False, rec
            assert rec["invalid_reason"] == "share_engine_not_executed", rec
        else:
            assert rec["valid"] is True, rec

    # The shared control is unaffected: still one per seed, all reused.
    assert len(out["runner"].control_calls) == 3
    assert all(rec["control_reused"] for rec in records)
