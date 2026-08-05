"""R6 Test 2: target trajectory overlap across splits must fail the audit.

Target execution trajectories are disjoint across train/validation/test.
A ``target_trajectory_id`` observed in more than one split is fatal and
``split_integrity_passed`` must be ``False``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from smtr.evaluation.split_audit import audit_split_files, audit_split_leakage


def _rec(task_id: str, *, target_trajectory_id: str, memory: str) -> dict:
    return {
        "task_id": task_id,
        "receiver_agent_id": "receiver_1",
        "candidate_memory_id": memory,
        "edge_id": f"edge_{task_id}_{memory}",
        "target_trajectory_id": target_trajectory_id,
        "memory_source_task_id": "train_task_0",
        "memory_source_trajectory_id": f"src_{task_id}",
        "memory_source_split": "train",
    }


def test_target_trajectory_overlap_across_validation_and_test_fails():
    splits = {
        "train": [
            _rec("t_train_1", target_trajectory_id="traj_target_1", memory="M"),
        ],
        "validation": [
            _rec("t_val_1", target_trajectory_id="traj_shared", memory="M"),
        ],
        "test": [
            _rec("t_test_1", target_trajectory_id="traj_shared", memory="M"),
        ],
    }
    with pytest.raises(ValueError, match="target_trajectory_id leakage"):
        audit_split_leakage(splits)


def test_audit_split_files_reports_failed_integrity(tmp_path: Path):
    splits = {
        "train": [
            _rec("t_train_1", target_trajectory_id="traj_target_1", memory="M"),
        ],
        "validation": [
            _rec("t_val_1", target_trajectory_id="traj_shared", memory="M"),
        ],
        "test": [
            _rec("t_test_1", target_trajectory_id="traj_shared", memory="M"),
        ],
    }
    paths = {}
    for name, records in splits.items():
        path = tmp_path / f"{name}.jsonl"
        path.write_text(
            "".join(json.dumps(r) + "\n" for r in records), encoding="utf-8"
        )
        paths[name] = path

    summary = audit_split_files(
        train_records_path=paths["train"],
        validation_records_path=paths["validation"],
        test_records_path=paths["test"],
    )
    assert summary["split_integrity_passed"] is False
    assert "target_trajectory_id leakage" in summary["error"]
