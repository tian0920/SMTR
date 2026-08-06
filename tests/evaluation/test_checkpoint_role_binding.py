"""Checkpoint role / feature-block binding errors (清单 P0-2 3.3, 3.5).

A checkpoint whose feature block does not match its declared role fails
the audit, and a method whose required role has no checkpoint path is
reported as a binding error instead of silently proceeding.
"""

from __future__ import annotations

import json

from smtr.evaluation.split_audit import audit_split_files
from smtr.router.transfer_critic import FourOutcomeTransferCritic


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


def _audit(tmp_path, *, methods, checkpoint_paths):
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

    return audit_split_files(
        train_records_path=train,
        validation_records_path=val,
        test_records_path=test,
        memory_pool_path=pool,
        test_candidate_manifest_path=manifest,
        checkpoint_paths=checkpoint_paths,
        methods=methods,
        experiment_mode="formal",
    )


def test_feature_block_mismatch_fails_audit(tmp_path):
    # A full-block checkpoint misdeclared as the global_transfer role.
    critic = FourOutcomeTransferCritic(feature_block="full")
    critic.calibration_split = "validation"
    critic.epsilon_selection_split = "validation"
    critic.training_split = "train"
    ckpt = tmp_path / "misdeclared.joblib"
    critic.save(ckpt)

    summary = _audit(
        tmp_path,
        methods=["global_transfer_critic"],
        checkpoint_paths={"global_transfer": ckpt},
    )
    assert summary["split_integrity_passed"] is False
    assert any(
        "checkpoint feature block mismatch" in err
        and "role='global_transfer'" in err
        and "actual='full'" in err
        for err in summary["checkpoint_binding_errors"]
    )


def test_missing_required_role_fails_audit(tmp_path):
    summary = _audit(
        tmp_path,
        methods=["global_transfer_critic"],
        checkpoint_paths={},
    )
    assert summary["split_integrity_passed"] is False
    assert (
        "method 'global_transfer_critic' requires checkpoint role "
        "'global_transfer'" in summary["checkpoint_binding_errors"]
    )
