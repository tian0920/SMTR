"""Missing provenance fields fail formal audits closed (清单 P1-1 4.3).

Dropping any required provenance field from a paired record must abort a
formal audit before any overlap check, with a per-field error report.
"""

from __future__ import annotations

import json

import pytest

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


def _audit(tmp_path, *, dropped_field: str | None) -> dict:
    records = {
        "train": _rec("t_train", "m_train"),
        "validation": _rec("t_val", "m_val"),
        "test": _rec("t_test", "m_test"),
    }
    if dropped_field is not None:
        for record in records.values():
            record[dropped_field] = None

    train = tmp_path / "train.jsonl"
    val = tmp_path / "validation.jsonl"
    test = tmp_path / "test.jsonl"
    _write_jsonl(train, [records["train"]])
    _write_jsonl(val, [records["validation"]])
    _write_jsonl(test, [records["test"]])
    pool = tmp_path / "memories.jsonl"
    _write_jsonl(pool, [{
        "memory_id": "m_test",
        "payload": {
            "procedure": "x",
            "provenance": {
                "source_agent_id": "w1",
                "source_task_id": "train_source",
                "source_trajectory_id": "traj_train",
                "source_split": "train",
            },
        },
    }])

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

    return audit_split_files(
        train_records_path=train,
        validation_records_path=val,
        test_records_path=test,
        memory_pool_path=pool,
        test_candidate_manifest_path=manifest,
        methods=["b0_no_memory"],
        experiment_mode="formal",
    )


def test_complete_provenance_passes(tmp_path):
    summary = _audit(tmp_path, dropped_field=None)
    assert summary["split_integrity_passed"] is True


@pytest.mark.parametrize(
    "dropped_field",
    [
        "target_trajectory_id",
        "memory_source_trajectory_id",
        "memory_source_task_id",
        "memory_source_split",
    ],
)
def test_missing_provenance_field_fails_formal_audit(tmp_path, dropped_field):
    summary = _audit(tmp_path, dropped_field=dropped_field)
    assert summary["split_integrity_passed"] is False
    assert any(
        f"missing required provenance field '{dropped_field}'" in err
        for err in summary["provenance_errors"]
    )
    assert summary["missing_provenance"][dropped_field]
    # Fail closed before the leakage checks even run.
    assert "error" not in summary
