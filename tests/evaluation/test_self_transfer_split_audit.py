"""R6 Test 4: self-transfer edges must fail the split audit.

A candidate target task must never equal the task its memory was extracted
from (``memory_source_task_id == task_id``); that is self-transfer and is a
fatal split-integrity violation.
"""

from __future__ import annotations

import pytest

from smtr.evaluation.split_audit import audit_split_leakage


def _rec(task_id: str, *, memory: str, memory_source_task_id: str) -> dict:
    return {
        "task_id": task_id,
        "receiver_agent_id": "receiver_1",
        "candidate_memory_id": memory,
        "edge_id": f"edge_{task_id}_{memory}",
        "target_trajectory_id": f"traj_{task_id}",
        "memory_source_task_id": memory_source_task_id,
        "memory_source_trajectory_id": f"src_traj_{memory}",
        "memory_source_split": "train",
    }


def test_self_transfer_edge_fails_audit():
    splits = {
        "train": [
            _rec("t_train_1", memory="M1", memory_source_task_id="t_train_0"),
        ],
        "validation": [
            # The validation target task equals its memory's source task.
            _rec("t_val_1", memory="M2", memory_source_task_id="t_val_1"),
        ],
        "test": [
            _rec("t_test_1", memory="M1", memory_source_task_id="t_train_0"),
        ],
    }
    with pytest.raises(ValueError, match="self-transfer"):
        audit_split_leakage(splits)


def test_legacy_source_task_id_field_still_detected():
    # Legacy artifacts persist ``source_task_id`` instead of
    # ``memory_source_task_id``; the audit must still catch self-transfer.
    legacy = _rec("t_val_1", memory="M2", memory_source_task_id="t_other")
    legacy.pop("memory_source_task_id")
    legacy["source_task_id"] = "t_val_1"
    splits = {
        "train": [
            _rec("t_train_1", memory="M1", memory_source_task_id="t_train_0"),
        ],
        "validation": [legacy],
        "test": [
            _rec("t_test_1", memory="M1", memory_source_task_id="t_train_0"),
        ],
    }
    with pytest.raises(ValueError, match="self-transfer"):
        audit_split_leakage(splits)
