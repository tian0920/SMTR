"""Shared-control grouping: one control per (task, receiver, seed)."""

from __future__ import annotations

from tests.marble._shared_control_harness import run_generate


def test_single_group_runs_one_control_and_four_shares_per_seed(tmp_path):
    """4 candidates of one group: 1 shared control + 4 shares per seed."""
    out = run_generate(
        tmp_path,
        entries=[{
            "task_id": "t1",
            "receiver_agent_id": "r1",
            "memory_ids": ["m1", "m2", "m3", "m4"],
        }],
        seeds=[0, 1, 2],
    )
    runner = out["runner"]

    # One shared no-memory control per seed of the single group.
    assert len(runner.control_calls) == 3
    for call in runner.control_calls:
        # The forbidden set covers every group candidate, fixed upfront.
        assert call["forbidden_memory_ids"] == ("m1", "m2", "m3", "m4")

    # One candidate-specific share per edge per seed.
    assert len(runner.share_calls) == 4 * 3
    shared_edge_ids = {call["edge_id"] for call in runner.share_calls}
    assert len(shared_edge_ids) == 4

    result = out["result"]
    assert result["attempted"] == 4 * 3
    assert result["share_episode_attempt_count"] == 12
    assert result["control_episode_attempt_count"] == 3
    # Actual episode cost is 15, not the legacy 2 * 12 = 24.
    assert result["actual_engine_episode_attempt_count"] == 15
    assert result["legacy_equivalent_episode_count"] == 24
    assert result["saved_episode_count"] == 9
    assert result["control_group_count"] == 3
    assert result["mean_candidates_per_control"] == 4.0


def test_control_group_id_matches_group_definition(tmp_path):
    """Records carry the group control id, never a candidate-derived id."""
    from smtr.marble.real_pairs import compute_control_group_id

    out = run_generate(
        tmp_path,
        entries=[{
            "task_id": "t1",
            "receiver_agent_id": "r1",
            "memory_ids": ["m1", "m2"],
        }],
        seeds=[0, 1, 2],
    )
    expected = {
        seed: compute_control_group_id(
            split_name="validation",
            scenario="database",
            task_id="t1",
            receiver_agent_id="r1",
            generation_seed=seed,
        )
        for seed in (0, 1, 2)
    }
    for rec in out["records"]:
        assert rec["control_group_id"] == expected[rec["generation_seed"]]
        assert rec["control_family_id"] == "t1::r1"
        assert rec["control_reused"] is True
        assert rec["control_group_candidate_count"] == 2
