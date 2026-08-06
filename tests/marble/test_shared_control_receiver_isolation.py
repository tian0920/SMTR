"""Shared-control receiver isolation: controls never span receivers."""

from __future__ import annotations

from tests.marble._shared_control_harness import run_generate


def test_two_receivers_get_separate_controls(tmp_path):
    """Same task, two receivers: one control group per receiver per seed."""
    out = run_generate(
        tmp_path,
        entries=[
            {
                "task_id": "t1",
                "receiver_agent_id": "r1",
                "memory_ids": ["m1", "m2"],
            },
            {
                "task_id": "t1",
                "receiver_agent_id": "r2",
                "memory_ids": ["m3", "m4"],
            },
        ],
        seeds=[0, 1, 2],
    )
    runner = out["runner"]
    # 2 receivers x 3 seeds = 6 shared controls.
    assert len(runner.control_calls) == 6
    assert len(runner.share_calls) == 4 * 3

    r1_groups = {
        call["control_group_id"]
        for call in runner.control_calls
        if call["agent_config"]["target_receiver_agent_id"] == "r1"
    }
    r2_groups = {
        call["control_group_id"]
        for call in runner.control_calls
        if call["agent_config"]["target_receiver_agent_id"] == "r2"
    }
    assert len(r1_groups) == 3
    assert len(r2_groups) == 3
    # Control identities are disjoint across receivers.
    assert r1_groups.isdisjoint(r2_groups)


def test_records_never_share_controls_across_receivers(tmp_path):
    """A record's control family equals its own (task, receiver)."""
    out = run_generate(
        tmp_path,
        entries=[
            {
                "task_id": "t1",
                "receiver_agent_id": "r1",
                "memory_ids": ["m1", "m2"],
            },
            {
                "task_id": "t1",
                "receiver_agent_id": "r2",
                "memory_ids": ["m3", "m4"],
            },
        ],
        seeds=[0, 1, 2],
    )
    for rec in out["records"]:
        assert rec["control_family_id"] == (
            f"{rec['task_id']}::{rec['receiver_agent_id']}"
        )
    r1_controls = {
        rec["control_group_id"]
        for rec in out["records"]
        if rec["receiver_agent_id"] == "r1"
    }
    r2_controls = {
        rec["control_group_id"]
        for rec in out["records"]
        if rec["receiver_agent_id"] == "r2"
    }
    assert r1_controls.isdisjoint(r2_controls)
