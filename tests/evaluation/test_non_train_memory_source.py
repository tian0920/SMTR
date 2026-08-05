"""R6 Test 3: memories sourced outside the train split must fail the audit.

Procedural memories are extracted exclusively from train trajectories, so
any candidate whose ``memory_source_split`` is not ``train`` is a fatal
split-integrity violation.
"""

from __future__ import annotations

import pytest

from smtr.evaluation.split_audit import audit_split_leakage


def _rec(task_id: str, *, memory: str, memory_source_split: str) -> dict:
    return {
        "task_id": task_id,
        "receiver_agent_id": "receiver_1",
        "candidate_memory_id": memory,
        "edge_id": f"edge_{task_id}_{memory}",
        "target_trajectory_id": f"traj_{task_id}",
        "memory_source_task_id": f"src_task_{memory}",
        "memory_source_trajectory_id": f"src_traj_{memory}",
        "memory_source_split": memory_source_split,
    }


def test_non_train_memory_source_fails_audit():
    splits = {
        "train": [
            _rec("t_train_1", memory="M_train", memory_source_split="train"),
        ],
        "validation": [
            _rec("t_val_1", memory="M_bad", memory_source_split="test"),
        ],
        "test": [
            _rec("t_test_1", memory="M_ok", memory_source_split="train"),
        ],
    }
    with pytest.raises(ValueError, match="outside the train split"):
        audit_split_leakage(splits)


def test_train_only_sources_pass():
    splits = {
        "train": [
            _rec("t_train_1", memory="M1", memory_source_split="train"),
        ],
        "validation": [
            _rec("t_val_1", memory="M1", memory_source_split="train"),
        ],
        "test": [
            _rec("t_test_1", memory="M1", memory_source_split="train"),
        ],
    }
    audit = audit_split_leakage(splits)
    assert audit["non_train_memory_sources"] == []
    assert audit["split_integrity_passed"] is True
