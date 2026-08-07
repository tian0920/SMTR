"""Tests for integrity audit."""

from __future__ import annotations

import json

from smtr.marble.integrity import run_integrity_audit


def _write_candidate_manifest(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "candidates": [{
            "task_id": "t1",
            "receiver_agent_id": "r1",
            "receiver_role": "executor",
            "candidate_records": [{
                "memory_id": "m1",
                "rank": 1,
                "score": 0.5,
            }],
        }],
    }), encoding="utf-8")


def _write_paired_records(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    records = [{
        "record_type": "marble_candidate_level_pair",
        "task_id": "t1",
        "generation_seed": 0,
        "candidate_memory_id": "m1",
        "receiver_agent_id": "r1",
        "receiver_role": "executor",
        "label": "positive_transfer",
        "valid": True,
        "invalid_reason": None,
        "share": {"team_success": True, "runtime_visibility_verified": True,
                  "native_evaluator_executed": True, "cleanup_succeeded": True},
        "withhold": {"team_success": False, "runtime_visibility_verified": True,
                     "native_evaluator_executed": True, "cleanup_succeeded": True},
        "digests": {
            "share_initial_digest": "abc",
            "withhold_initial_digest": "abc",
            "share_task_digest": "td",
            "withhold_task_digest": "td",
            "share_tool_config_digest": "tc",
            "withhold_tool_config_digest": "tc",
            "share_agent_config_digest": "ac",
            "withhold_agent_config_digest": "ac",
        },
    }]
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")


def _write_memory_pool(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"memory_id": "m1", "payload": {"procedure": "test"}, "routing_card": {}}) + "\n", encoding="utf-8")


def test_missing_paired_records_fails(tmp_path):
    """Missing paired records must cause audit failure."""
    cm = tmp_path / "candidates.json"
    mp = tmp_path / "memories.jsonl"
    _write_candidate_manifest(cm)
    _write_memory_pool(mp)
    result = run_integrity_audit(
        candidate_manifest_path=cm,
        paired_records_path=tmp_path / "nonexistent.jsonl",
        memory_pool_path=mp,
    )
    assert result["audit_passed"] is False
    assert "paired_records" in result["missing_artifacts"]


def test_payload_leakage_fails(tmp_path):
    """Payload leakage in paired records must fail audit."""
    cm = tmp_path / "candidates.json"
    pr = tmp_path / "paired.jsonl"
    mp = tmp_path / "memories.jsonl"
    _write_candidate_manifest(cm)
    _write_memory_pool(mp)
    # Write a record with payload leakage
    rec = {"record_type": "marble_candidate_level_pair", "task_id": "t1",
           "candidate_memory_id": "m1", "receiver_agent_id": "r1", "receiver_role": "executor",
           "label": "positive_transfer", "valid": True, "invalid_reason": None,
           "procedure": "LEAKED PROCEDURE",
           "share": {"team_success": True, "runtime_visibility_verified": True,
                     "native_evaluator_executed": True, "cleanup_succeeded": True},
           "withhold": {"team_success": False, "runtime_visibility_verified": True,
                        "native_evaluator_executed": True, "cleanup_succeeded": True},
           "digests": {"share_initial_digest": "a", "withhold_initial_digest": "a",
                       "share_task_digest": "t", "withhold_task_digest": "t",
                       "share_tool_config_digest": "tc", "withhold_tool_config_digest": "tc",
                       "share_agent_config_digest": "ac", "withhold_agent_config_digest": "ac"}}
    pr.write_text(json.dumps(rec) + "\n", encoding="utf-8")
    result = run_integrity_audit(candidate_manifest_path=cm, paired_records_path=pr, memory_pool_path=mp)
    assert result["audit_passed"] is False
    assert result["payload_leakage"] is True


def test_digest_mismatch_fails(tmp_path):
    """Digest mismatch must fail audit."""
    cm = tmp_path / "candidates.json"
    pr = tmp_path / "paired.jsonl"
    mp = tmp_path / "memories.jsonl"
    _write_candidate_manifest(cm)
    _write_memory_pool(mp)
    rec = {"record_type": "marble_candidate_level_pair", "task_id": "t1",
           "candidate_memory_id": "m1", "receiver_agent_id": "r1", "receiver_role": "executor",
           "label": "positive_transfer", "valid": True, "invalid_reason": None,
           "share": {"team_success": True, "runtime_visibility_verified": True,
                     "native_evaluator_executed": True, "cleanup_succeeded": True},
           "withhold": {"team_success": False, "runtime_visibility_verified": True,
                        "native_evaluator_executed": True, "cleanup_succeeded": True},
           "digests": {"share_initial_digest": "aaa", "withhold_initial_digest": "bbb",
                       "share_task_digest": "t", "withhold_task_digest": "t",
                       "share_tool_config_digest": "tc", "withhold_tool_config_digest": "tc",
                       "share_agent_config_digest": "ac", "withhold_agent_config_digest": "ac"}}
    pr.write_text(json.dumps(rec) + "\n", encoding="utf-8")
    result = run_integrity_audit(candidate_manifest_path=cm, paired_records_path=pr, memory_pool_path=mp)
    assert result["audit_passed"] is False
    assert result["branch_isolation_passed"] is False


def _write_split_records(path, task_id, memory_id):
    path.parent.mkdir(parents=True, exist_ok=True)
    rec = {
        "record_type": "marble_candidate_level_pair",
        "task_id": task_id,
        "generation_seed": 0,
        "candidate_memory_id": memory_id,
        "receiver_agent_id": "r1",
        "valid": True,
        "label": "positive_transfer",
    }
    path.write_text(json.dumps(rec) + "\n", encoding="utf-8")


def test_all_pass(tmp_path):
    """All checks passing must yield audit_passed=true."""
    cm = tmp_path / "candidates.json"
    pr = tmp_path / "paired.jsonl"
    mp = tmp_path / "memories.jsonl"
    _write_candidate_manifest(cm)
    _write_paired_records(pr)
    _write_memory_pool(mp)
    # 清单 P0-18: the real split audit requires three isolated split files.
    train = tmp_path / "train.jsonl"
    val = tmp_path / "validation.jsonl"
    test = tmp_path / "test.jsonl"
    _write_split_records(train, "t_train", "m_train")
    _write_split_records(val, "t_val", "m_val")
    _write_split_records(test, "t_test", "m_test")
    result = run_integrity_audit(
        candidate_manifest_path=cm,
        paired_records_path=pr,
        memory_pool_path=mp,
        train_paired_records_path=train,
        validation_paired_records_path=val,
        test_paired_records_path=test,
    )
    assert result["audit_passed"] is True
    assert result["errors"] == []
    assert result["split_integrity_passed"] is True
    assert result["split_audit"]["target_task_overlap"] == []
    assert result["split_audit"]["treatment_edge_overlap"] == []


def test_split_audit_missing_inputs_fails_closed(tmp_path):
    """Without split files the split audit fails closed (清单 P0-18)."""
    cm = tmp_path / "candidates.json"
    pr = tmp_path / "paired.jsonl"
    mp = tmp_path / "memories.jsonl"
    _write_candidate_manifest(cm)
    _write_paired_records(pr)
    _write_memory_pool(mp)
    result = run_integrity_audit(candidate_manifest_path=cm, paired_records_path=pr, memory_pool_path=mp)
    assert result["audit_passed"] is False
    assert result["split_integrity_passed"] is False
    assert any("split audit inputs missing" in e for e in result["errors"])


def test_split_audit_cross_split_edge_fails(tmp_path):
    """A treatment edge crossing splits must fail the integrity audit."""
    cm = tmp_path / "candidates.json"
    pr = tmp_path / "paired.jsonl"
    mp = tmp_path / "memories.jsonl"
    _write_candidate_manifest(cm)
    _write_paired_records(pr)
    _write_memory_pool(mp)
    train = tmp_path / "train.jsonl"
    val = tmp_path / "validation.jsonl"
    test = tmp_path / "test.jsonl"
    # Same treatment edge (t_train, r1, m_shared) in train and test.
    _write_split_records(train, "t_train", "m_shared")
    _write_split_records(val, "t_val", "m_val")
    _write_split_records(test, "t_train", "m_shared")
    result = run_integrity_audit(
        candidate_manifest_path=cm,
        paired_records_path=pr,
        memory_pool_path=mp,
        train_paired_records_path=train,
        validation_paired_records_path=val,
        test_paired_records_path=test,
    )
    assert result["audit_passed"] is False
    assert result["split_integrity_passed"] is False
