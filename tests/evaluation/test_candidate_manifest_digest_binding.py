"""Candidate manifest digest binding into the split-audit artifact (清单 P0-1).

The audit must bind the exact candidate manifest file it audited: an
evaluation consuming a different manifest (different bytes) must fail the
formal gate, while an identical manifest at a different path stays valid.
"""

from __future__ import annotations

import json

import pytest

from smtr.evaluation.split_audit import audit_split_files
from smtr.evaluation.split_audit_validation import validate_split_audit_artifact
from smtr.marble.runtime_visibility_audit import file_digest


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


def _candidate_manifest(task_id: str, memory_id: str, marker: str) -> dict:
    return {
        "target_split": "test",
        "memory_source_split": "train",
        "marker": marker,
        "candidates": [
            {
                "task_id": task_id,
                "receiver_agent_id": "r1",
                "candidate_records": [{"candidate_memory_id": memory_id}],
            }
        ],
    }


def _setup(tmp_path):
    train = tmp_path / "train.jsonl"
    val = tmp_path / "validation.jsonl"
    test = tmp_path / "test.jsonl"
    _write_jsonl(train, [_rec("t_train", "m_train")])
    _write_jsonl(val, [_rec("t_val", "m_val")])
    _write_jsonl(test, [_rec("t_test", "m_test")])

    pool = tmp_path / "memories.jsonl"
    _write_jsonl(pool, [{"memory_id": "m_test", "payload": {"procedure": "x"}}])

    manifest_a = tmp_path / "candidates_a.json"
    manifest_b = tmp_path / "candidates_b.json"
    manifest_a.write_text(
        json.dumps(_candidate_manifest("t_test", "m_test", "a")),
        encoding="utf-8",
    )
    manifest_b.write_text(
        json.dumps(_candidate_manifest("t_test", "m_test", "b")),
        encoding="utf-8",
    )

    dataset = tmp_path / "dataset.json"
    dataset.write_text('{"tasks": []}', encoding="utf-8")
    splits = tmp_path / "splits.json"
    splits.write_text('{"records": []}', encoding="utf-8")

    summary = audit_split_files(
        train_records_path=train,
        validation_records_path=val,
        test_records_path=test,
        memory_pool_path=pool,
        test_candidate_manifest_path=manifest_a,
        dataset_manifest_path=dataset,
        split_manifest_path=splits,
        methods=["b0_no_memory"],
        experiment_mode="pilot",
    )
    assert summary["split_integrity_passed"] is True

    audit_path = tmp_path / "split_audit.json"
    audit_path.write_text(json.dumps(summary), encoding="utf-8")
    return {
        "audit_path": audit_path,
        "dataset": dataset,
        "splits": splits,
        "pool": pool,
        "manifest_a": manifest_a,
        "manifest_b": manifest_b,
    }


def _validate(files, *, candidate_manifest):
    return validate_split_audit_artifact(
        split_audit_path=files["audit_path"],
        dataset_manifest_path=files["dataset"],
        split_manifest_path=files["splits"],
        memory_pool_path=files["pool"],
        candidate_manifest_path=candidate_manifest,
        checkpoint_paths={},
        enabled_methods=["b0_no_memory"],
    )


def test_audit_binds_candidate_manifest_digest(tmp_path):
    files = _setup(tmp_path)
    audit = json.loads(files["audit_path"].read_text(encoding="utf-8"))
    assert audit["test_candidate_manifest_digest"] == file_digest(
        files["manifest_a"]
    )


def test_swapped_candidate_manifest_fails_validation(tmp_path):
    files = _setup(tmp_path)
    with pytest.raises(
        ValueError, match="split audit does not match current candidate manifest"
    ):
        _validate(files, candidate_manifest=files["manifest_b"])


def test_identical_content_at_other_path_still_validates(tmp_path):
    files = _setup(tmp_path)
    manifest_copy = tmp_path / "candidates_a_copy.json"
    manifest_copy.write_bytes(files["manifest_a"].read_bytes())
    audit = _validate(files, candidate_manifest=manifest_copy)
    assert audit["split_integrity_passed"] is True
