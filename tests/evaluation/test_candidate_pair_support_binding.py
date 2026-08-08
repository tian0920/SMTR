"""Candidate edge support binding between manifest and test records (清单 P0-1 2.5).

Every test paired-record edge must appear in the candidate manifest and
vice versa: unsupported candidate edges fail a strict audit and missing
manifest coverage is always fatal; non-strict pilots only report them.
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


def _audit(tmp_path, candidate_entries, *, strict: bool, mode: str):
    train = tmp_path / "train.jsonl"
    val = tmp_path / "validation.jsonl"
    test = tmp_path / "test.jsonl"
    _write_jsonl(train, [_rec("t_train", "m_train")])
    _write_jsonl(val, [_rec("t_val", "m_val")])
    _write_jsonl(test, [_rec("t_test", "m_test")])
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

    manifest = {
        "target_split": "test",
        "memory_source_split": "train",
        "candidates": candidate_entries,
    }
    manifest_path = tmp_path / "candidates.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    budget_manifest_path = tmp_path / "budget_candidates.json"
    budget_manifest_path.write_text(
        json.dumps(
            {
                "target_split": "train",
                "memory_source_split": "train",
                "candidates": [],
            }
        ),
        encoding="utf-8",
    )

    return audit_split_files(
        train_records_path=train,
        validation_records_path=val,
        test_records_path=test,
        memory_pool_path=pool,
        test_candidate_manifest_path=manifest_path,
        train_budget_candidate_manifest_path=budget_manifest_path,
        methods=["b0_no_memory"],
        strict_candidate_support=strict,
        experiment_mode=mode,
    )


def _entry(task_id: str, memory_id: str) -> dict:
    return {
        "task_id": task_id,
        "receiver_agent_id": "r1",
        "candidate_records": [{"candidate_memory_id": memory_id}],
    }


def test_fully_supported_edges_pass(tmp_path):
    summary = _audit(tmp_path, [_entry("t_test", "m_test")],
                     strict=True, mode="formal")
    assert summary["split_integrity_passed"] is True
    assert summary["unsupported_candidate_edges"] == []
    assert summary["test_edges_missing_from_manifest"] == []


def test_unsupported_edge_fails_strict_audit(tmp_path):
    entries = [_entry("t_test", "m_test"), _entry("t_extra", "m_extra")]
    summary = _audit(tmp_path, entries, strict=True, mode="formal")
    assert summary["split_integrity_passed"] is False
    assert summary["unsupported_candidate_edges"] == [
        ("t_extra", "r1", "m_extra")
    ]


def test_unsupported_edge_reported_but_not_fatal_when_not_strict(tmp_path):
    entries = [_entry("t_test", "m_test"), _entry("t_extra", "m_extra")]
    summary = _audit(tmp_path, entries, strict=False, mode="pilot")
    assert summary["split_integrity_passed"] is True
    assert summary["strict_candidate_support"] is False
    assert summary["unsupported_candidate_edges"] == [
        ("t_extra", "r1", "m_extra")
    ]


def test_test_edge_missing_from_manifest_always_fatal(tmp_path):
    # The manifest covers a different edge, so the real test edge
    # (t_test, r1, m_test) lacks manifest support.
    summary = _audit(tmp_path, [_entry("t_other", "m_other")],
                     strict=False, mode="pilot")
    assert summary["split_integrity_passed"] is False
    assert summary["test_edges_missing_from_manifest"] == [
        ("t_test", "r1", "m_test")
    ]
