"""Candidate manifest split provenance gate (清单 P0-1 2.4).

A test-split formal audit must reject candidate manifests that do not
target the test split or whose memories do not come from the train split.
"""

from __future__ import annotations

import json

from smtr.evaluation.split_audit import audit_split_files


def _rec(task_id: str, memory_id: str) -> dict:
    return {
        "record_type": "marble_candidate_level_pair",
        "task_id": task_id,
        "receiver_agent_id": "r1",
        "candidate_memory_id": memory_id,
        "edge_id": f"{task_id}|r1|{memory_id}",
        "generation_seed": 0,
        "target_trajectory_id": f"traj_target_{task_id}",
        "memory_source_task_id": "train_source_task",
        "memory_source_trajectory_id": "traj_train_source",
        "memory_source_split": "train",
        "share": {"team_success": 1},
        "withhold": {"team_success": 0},
        "valid": True,
        "label": "positive_transfer",
    }


def _write_jsonl(path, records) -> None:
    path.write_text(
        "".join(json.dumps(rec) + "\n" for rec in records), encoding="utf-8"
    )


def _split_files(tmp_path):
    train = tmp_path / "train.jsonl"
    val = tmp_path / "validation.jsonl"
    test = tmp_path / "test.jsonl"
    _write_jsonl(train, [_rec("t_train", "m_train")])
    _write_jsonl(val, [_rec("t_val", "m_val")])
    _write_jsonl(test, [_rec("t_test", "m_test")])
    pool = tmp_path / "memories.jsonl"
    _write_jsonl(pool, [{"memory_id": "m_test", "payload": {"procedure": "x"}}])
    return train, val, test, pool


def _manifest(target_split: str, memory_source_split: str) -> dict:
    return {
        "target_split": target_split,
        "memory_source_split": memory_source_split,
        "candidates": [
            {
                "task_id": "t_test",
                "receiver_agent_id": "r1",
                "candidate_records": [{"candidate_memory_id": "m_test"}],
            }
        ],
    }


def _audit(tmp_path, manifest: dict):
    train, val, test, pool = _split_files(tmp_path)
    manifest_path = tmp_path / "candidates.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return audit_split_files(
        train_records_path=train,
        validation_records_path=val,
        test_records_path=test,
        memory_pool_path=pool,
        test_candidate_manifest_path=manifest_path,
        methods=["b0_no_memory"],
        experiment_mode="formal",
    )


def test_matching_manifest_passes(tmp_path):
    summary = _audit(tmp_path, _manifest("test", "train"))
    assert summary["split_integrity_passed"] is True
    assert summary["candidate_manifest_errors"] == []
    assert summary["candidate_manifest"]["target_split"] == "test"


def test_validation_targeted_manifest_fails_test_audit(tmp_path):
    summary = _audit(tmp_path, _manifest("validation", "train"))
    assert summary["split_integrity_passed"] is False
    assert any(
        "target_split mismatch" in err
        for err in summary["candidate_manifest_errors"]
    )


def test_non_train_memory_source_manifest_fails(tmp_path):
    summary = _audit(tmp_path, _manifest("test", "validation"))
    assert summary["split_integrity_passed"] is False
    assert any(
        "memory_source_split must be 'train'" in err
        for err in summary["candidate_manifest_errors"]
    )


def test_formal_audit_without_candidate_manifest_fails(tmp_path):
    train, val, test, pool = _split_files(tmp_path)
    summary = audit_split_files(
        train_records_path=train,
        validation_records_path=val,
        test_records_path=test,
        memory_pool_path=pool,
        test_candidate_manifest_path=None,
        methods=["b0_no_memory"],
        experiment_mode="formal",
    )
    assert summary["split_integrity_passed"] is False
    assert summary["candidate_manifest_errors"] == [
        "formal split audit requires a test candidate manifest"
    ]
