"""Core-validity filter tests (清单第十二章 / Commit 9).

Invalid paired records (incomplete branches, mismatched seeds or core-config
digests) must never reach critic training or evaluation; formal experiments
fail fast when filtered records lack four-label coverage or multi-seed edges.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from smtr.marble.core_validity import (
    InsufficientCoreValidityError,
    core_validity_exclusion_reasons,
    filter_core_paired_records,
    is_valid_core_paired_record,
    require_core_formal_validity,
)


def _valid_record(
    *,
    edge_id: str = "edge1",
    seed: int = 0,
    y_share: bool = True,
    y_withhold: bool = False,
    digests: dict | None = None,
) -> dict:
    rec = {
        "task_id": "t1",
        "receiver_agent_id": "r1",
        "candidate_memory_id": "m1",
        "edge_id": edge_id,
        "generation_seed": seed,
        "share": {
            "team_success": y_share,
            "environment_valid": True,
            "real_engine_executed": True,
        },
        "withhold": {
            "team_success": y_withhold,
            "environment_valid": True,
            "real_engine_executed": True,
        },
        "valid": True,
    }
    if digests is not None:
        rec["digests"] = digests
    return rec


def test_valid_pair_passes_core_validity():
    assert is_valid_core_paired_record(_valid_record())


def test_incomplete_share_branch_is_rejected():
    rec = _valid_record()
    rec["share"]["environment_valid"] = False
    reasons = core_validity_exclusion_reasons(rec)
    assert "share_environment_invalid" in reasons
    assert not is_valid_core_paired_record(rec)


def test_missing_team_success_is_rejected():
    rec = _valid_record()
    rec["withhold"].pop("team_success")
    reasons = core_validity_exclusion_reasons(rec)
    # Without a canonical outcome the record fails at the accessor level;
    # with a None outcome it fails the branch-completeness check. Both are
    # valid rejection paths.
    assert reasons
    assert not is_valid_core_paired_record(rec)

    rec_none = _valid_record()
    rec_none["withhold"]["team_success"] = None
    rec_none["withhold"].pop("environment_valid")
    rec_none["withhold"].pop("real_engine_executed")
    rec_none["share"]["team_success"] = True
    assert "withhold_missing_team_success" in core_validity_exclusion_reasons(rec_none)


def test_mismatched_seed_pair_is_rejected():
    rec = _valid_record(seed=3)
    rec["share_generation_seed"] = 3
    rec["withhold_generation_seed"] = 4
    assert "mismatched_generation_seed" in core_validity_exclusion_reasons(rec)
    assert not is_valid_core_paired_record(rec)


def test_mismatched_initial_state_pair_is_rejected():
    rec = _valid_record(digests={
        "share_initial_digest": "aaa",
        "withhold_initial_digest": "bbb",
        "share_task_digest": "t",
        "withhold_task_digest": "t",
    })
    reasons = core_validity_exclusion_reasons(rec)
    assert "mismatched_initial_state_digest" in reasons
    assert not is_valid_core_paired_record(rec)


def test_mismatched_task_digest_pair_is_rejected():
    rec = _valid_record(digests={
        "share_task_digest": "task-a",
        "withhold_task_digest": "task-b",
    })
    assert "mismatched_task_digest" in core_validity_exclusion_reasons(rec)


def test_upstream_invalid_record_is_rejected():
    rec = _valid_record()
    rec["valid"] = False
    rec["invalid_reason"] = "branch_crashed"
    assert any(r.startswith("upstream_invalid") for r in core_validity_exclusion_reasons(rec))


def test_missing_identity_fields_are_rejected():
    rec = _valid_record()
    del rec["receiver_agent_id"]
    del rec["edge_id"]
    reasons = core_validity_exclusion_reasons(rec)
    assert "missing_receiver_agent_id" in reasons
    assert "missing_edge_id" in reasons


def test_filter_reports_counts_and_reasons():
    records = [
        _valid_record(),
        _valid_record(edge_id="edge2", seed=1),
        {**_valid_record(edge_id="edge3"), "valid": False, "invalid_reason": "x"},
    ]
    summary = filter_core_paired_records(records)
    assert summary["total_paired_records"] == 3
    assert summary["valid_paired_records"] == 2
    assert summary["excluded_paired_records"] == 1
    assert any(k.startswith("upstream_invalid") for k in summary["exclusion_reasons"])
    assert len(summary["valid_records"]) == 2


def test_invalid_pair_is_excluded_from_training(tmp_path: Path):
    from smtr.router.transfer_features import load_paired_records_for_training

    pool = tmp_path / "pool.jsonl"
    pool.write_text(json.dumps({
        "memory_id": "m1",
        "routing_card": {
            "writer": {"agent_id": "w1", "role": "executor"},
            "goal_summary": "diagnose",
            "task_tags": ["database"],
        },
    }) + "\n", encoding="utf-8")

    good = _valid_record()
    bad = _valid_record(edge_id="edge2", seed=1)
    bad["share"]["real_engine_executed"] = False  # incomplete branch

    records = tmp_path / "records.jsonl"
    records.write_text(
        "\n".join(json.dumps(r) for r in (good, bad)) + "\n", encoding="utf-8"
    )
    result = load_paired_records_for_training(records, pool)
    assert len(result) == 1, "invalid paired record must not reach critic training"


def test_valid_pair_reaches_training_and_evaluation(tmp_path: Path):
    from smtr.router.transfer_features import load_paired_records_for_training

    pool = tmp_path / "pool.jsonl"
    pool.write_text(json.dumps({
        "memory_id": "m1",
        "routing_card": {
            "writer": {"agent_id": "w1", "role": "executor"},
            "goal_summary": "diagnose",
            "task_tags": ["database"],
        },
    }) + "\n", encoding="utf-8")
    records = tmp_path / "records.jsonl"
    records.write_text(json.dumps(_valid_record()) + "\n", encoding="utf-8")

    result = load_paired_records_for_training(records, pool)
    assert len(result) == 1
    _, label = result[0]
    assert label == "positive_transfer"


def _four_label_records() -> list[dict]:
    outcomes = [(True, False), (False, True), (True, True), (False, False)]
    records = []
    for idx, (ys, yw) in enumerate(outcomes):
        for seed in range(5):
            records.append(_valid_record(edge_id=f"edge{idx}", seed=seed, y_share=ys, y_withhold=yw))
    return records


def test_formal_gate_passes_with_coverage_and_seeds():
    require_core_formal_validity(_four_label_records(), experiment_mode="formal")


def test_formal_gate_fails_without_label_coverage():
    records = [r for r in _four_label_records() if not (
        r["share"]["team_success"] is False and r["withhold"]["team_success"] is True
    )]
    with pytest.raises(InsufficientCoreValidityError, match="labels"):
        require_core_formal_validity(records, experiment_mode="formal")


def test_formal_gate_fails_with_too_few_seeds():
    records = _four_label_records()
    starved = [r for r in records if not (r["edge_id"] == "edge0" and r["generation_seed"] >= 3)]
    with pytest.raises(InsufficientCoreValidityError, match="seeds"):
        require_core_formal_validity(starved, experiment_mode="formal")


def test_pilot_gate_requires_three_seeds():
    records = _four_label_records()
    reduced = [r for r in records if r["generation_seed"] < 3]
    require_core_formal_validity(reduced, experiment_mode="pilot")
    reduced2 = [r for r in records if r["generation_seed"] < 2]
    with pytest.raises(InsufficientCoreValidityError):
        require_core_formal_validity(reduced2, experiment_mode="pilot")


def test_record_persists_split_integrity_metadata():
    """Records must persist split_name and source/target provenance (ch.13)."""
    from unittest.mock import MagicMock

    from smtr.marble.real_pairs import paired_result_to_record

    pair_result = MagicMock()
    pair_result.task_id = "205"
    pair_result.scenario = "database"
    pair_result.candidate_memory_id = "m1"
    pair_result.paired_label = "positive_transfer"
    pair_result.paired_record_valid = True
    pair_result.invalid_reason = None
    pair_result.branch_execution_order = "share_then_withhold"
    for branch in (pair_result.share, pair_result.withhold):
        branch.outcome.success = True
        branch.outcome.environment_valid = True
        branch.outcome.native_evaluator_executed = True
        branch.real_engine_executed = True
        branch.runtime_visibility_verified = True
        branch.cleanup_succeeded = True
        branch.initial_digest = "d"
        branch.initial_logical_fingerprint = {"combined_digest": "ld"}
        branch.agent_config_digest = "a"
        branch.task_digest = "t"
        branch.tool_config_digest = "tc"

    edge = {
        "edge_id": "edge-x",
        "receiver_agent_id": "r1",
        "receiver_role": "critic",
        "receiver_capabilities": [],
        "writer_agent_id": "w1",
        "writer_role": "executor",
        "writer_capabilities": [],
        "candidate_rank": 1,
        "candidate_score": 0.5,
        "source_task_id": "src-task-9",
        "source_trajectory_id": "traj-99",
        "target_task_group": "group-latency",
    }
    record = paired_result_to_record(
        pair_result=pair_result, edge=edge, seed=3, split_name="validation",
    )
    assert record["split_name"] == "validation"
    assert record["source_task_id"] == "src-task-9"
    assert record["source_trajectory_id"] == "traj-99"
    assert record["target_task_id"] == "205"
    assert record["target_task_group"] == "group-latency"
    assert record["edge_id"] == "edge-x"
