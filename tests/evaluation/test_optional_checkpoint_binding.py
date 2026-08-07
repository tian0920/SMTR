"""Baseline methods need no checkpoint binding (清单 P0-2 3.4).

Methods without a critic checkpoint (b0_no_memory, semantic_top1,
role_aware_top1) must audit cleanly with an empty checkpoint digest map
and validate without any checkpoint files.
"""

from __future__ import annotations

import json

from smtr.evaluation.split_audit import audit_split_files
from smtr.evaluation.split_audit_validation import validate_split_audit_artifact

_BASELINE_METHODS = ["b0_no_memory", "semantic_top1", "role_aware_top1"]


def _rec(task_id: str, memory_id: str) -> dict:
    return {
        "record_type": "marble_candidate_level_pair",
        "task_id": task_id,
        "receiver_agent_id": "r1",
        "candidate_memory_id": memory_id,
        "edge_id": f"{task_id}|r1|{memory_id}",
        "generation_seed": 0,
        "target_trajectory_id": f"traj_target_{task_id}",
        "memory_source_agent_id": "w1",
        "memory_source_task_id": "train_source_task",
        "memory_source_trajectory_id": "traj_train_source",
        "memory_source_split": "train",
        "control_group_id": f"ctrl_{task_id}",
        "share": {"team_success": 1},
        "withhold": {"team_success": 0},
        "valid": True,
        "label": "positive_transfer",
    }


def _write_jsonl(path, records) -> None:
    path.write_text(
        "".join(json.dumps(rec) + "\n" for rec in records), encoding="utf-8"
    )


def _setup(tmp_path):
    train = tmp_path / "train.jsonl"
    val = tmp_path / "validation.jsonl"
    test = tmp_path / "test.jsonl"
    _write_jsonl(train, [_rec("t_train", "m_train")])
    _write_jsonl(val, [_rec("t_val", "m_val")])
    _write_jsonl(test, [_rec("t_test", "m_test")])

    pool = tmp_path / "memories.jsonl"
    _write_jsonl(pool, [{"memory_id": "m_test", "payload": {"procedure": "x"}}])

    manifest = tmp_path / "candidates.json"
    manifest.write_text(
        json.dumps(
            {
                "target_split": "test",
                "memory_source_split": "train",
                "candidates": [
                    {
                        "task_id": "t_test",
                        "receiver_agent_id": "r1",
                        "candidate_records": [
                            {"candidate_memory_id": "m_test"}
                        ],
                    }
                ],
            }
        ),
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
        test_candidate_manifest_path=manifest,
        dataset_manifest_path=dataset,
        split_manifest_path=splits,
        checkpoint_paths={},
        methods=_BASELINE_METHODS,
        experiment_mode="formal",
    )

    audit_path = tmp_path / "split_audit.json"
    audit_path.write_text(json.dumps(summary), encoding="utf-8")
    return {
        "summary": summary,
        "audit_path": audit_path,
        "dataset": dataset,
        "splits": splits,
        "pool": pool,
        "manifest": manifest,
    }


def test_baseline_formal_audit_needs_no_checkpoint(tmp_path):
    files = _setup(tmp_path)
    summary = files["summary"]
    assert summary["split_integrity_passed"] is True
    assert summary["checkpoint_digests"] == {}
    assert summary["checkpoint_binding_errors"] == []


def test_baseline_validation_without_checkpoints(tmp_path):
    files = _setup(tmp_path)
    audit = validate_split_audit_artifact(
        split_audit_path=files["audit_path"],
        dataset_manifest_path=files["dataset"],
        split_manifest_path=files["splits"],
        memory_pool_path=files["pool"],
        candidate_manifest_path=files["manifest"],
        checkpoint_paths={},
        enabled_methods=_BASELINE_METHODS,
    )
    assert audit["split_integrity_passed"] is True
