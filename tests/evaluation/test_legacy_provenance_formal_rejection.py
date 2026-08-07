"""Legacy provenance schema handling (清单 P1-4).

Records that only carry the legacy ``source_*`` fields are rejected by
formal audits through the provenance schema validation. The legacy
fallback path has been removed; formal paired records must carry
``memory_source_*`` provenance fields.
"""

from __future__ import annotations

import json

from smtr.evaluation.split_audit import audit_split_files


def _legacy_rec(task_id: str, memory_id: str) -> dict:
    return {
        "record_type": "marble_candidate_level_pair",
        "task_id": task_id,
        "receiver_agent_id": "r1",
        "candidate_memory_id": memory_id,
        "edge_id": f"{task_id}|r1|{memory_id}",
        "generation_seed": 0,
        "target_trajectory_id": f"traj_target_{task_id}",
        # Legacy schema: only the old source_* fields are present.
        "source_task_id": "train_source_task",
        "source_trajectory_id": "traj_train_source",
        "share": {"team_success": 1},
        "withhold": {"team_success": 0},
        "valid": True,
        "label": "positive_transfer",
    }


def _write_jsonl(path, records) -> None:
    path.write_text(
        "".join(json.dumps(rec) + "\n" for rec in records), encoding="utf-8"
    )


def _setup(tmp_path, *, mode: str):
    train = tmp_path / "train.jsonl"
    val = tmp_path / "validation.jsonl"
    test = tmp_path / "test.jsonl"
    _write_jsonl(train, [_legacy_rec("t_train", "m_train")])
    _write_jsonl(val, [_legacy_rec("t_val", "m_val")])
    _write_jsonl(test, [_legacy_rec("t_test", "m_test")])

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
        methods=["b0_no_memory"],
        experiment_mode=mode,
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


def test_pilot_accepts_legacy_records(tmp_path):
    files = _setup(tmp_path, mode="pilot")
    summary = files["summary"]
    assert summary["split_integrity_passed"] is True


def test_formal_rejects_legacy_records(tmp_path):
    files = _setup(tmp_path, mode="formal")
    summary = files["summary"]
    assert summary["split_integrity_passed"] is False
    assert summary["provenance_errors"]


def test_formal_provenance_rejects_missing_fields(tmp_path):
    """Formal audit rejects records missing required provenance fields."""
    files = _setup(tmp_path, mode="formal")
    summary = files["summary"]
    assert summary["split_integrity_passed"] is False
    # The legacy records use source_* instead of memory_source_* fields,
    # so the formal provenance validator rejects them.
    assert any(
        "missing required provenance field" in err
        for err in summary["provenance_errors"]
    )
