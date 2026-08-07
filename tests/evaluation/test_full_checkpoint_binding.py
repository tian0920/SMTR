"""Full SMTR checkpoint digest binding (清单 P0-2 3.8).

The split audit binds the full-role checkpoint by digest; evaluating with a
different full checkpoint file must abort before any episode runs.
"""

from __future__ import annotations

import json

import pytest

from smtr.evaluation.split_audit import audit_split_files
from smtr.evaluation.split_audit_validation import validate_split_audit_artifact
from smtr.marble.runtime_visibility_audit import file_digest
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


def _save_checkpoint(path, *, feature_block, train_path, val_path, pool_path):
    critic = FourOutcomeTransferCritic(feature_block=feature_block)
    critic.calibration_split = "validation"
    critic.epsilon_selection_split = "validation"
    critic.training_split = "train"
    critic.train_record_digest = file_digest(train_path)
    critic.validation_record_digest = file_digest(val_path)
    critic.memory_pool_digest = file_digest(pool_path)
    critic.epsilon_star = 0.2
    critic.method_schema_metadata = {
        "method_schema": "memory_receiver_v1",
        "routing_conditioning": "memory_receiver",
        "writer_features_used": False,
        "provenance_features_used": False,
        "outcome_level": "team_success",
        "treatment_edge_unit": "task_receiver_memory",
    }
    critic.save(path)
    return path


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

    ckpt_a = _save_checkpoint(
        tmp_path / "full_a.joblib",
        feature_block="full",
        train_path=train,
        val_path=val,
        pool_path=pool,
    )
    # Different bytes: a checkpoint trained against a bogus train digest.
    critic_b = FourOutcomeTransferCritic(feature_block="full")
    critic_b.calibration_split = "validation"
    critic_b.epsilon_selection_split = "validation"
    critic_b.training_split = "train"
    critic_b.epsilon_star = 0.3
    ckpt_b = tmp_path / "full_b.joblib"
    critic_b.save(ckpt_b)

    summary = audit_split_files(
        train_records_path=train,
        validation_records_path=val,
        test_records_path=test,
        memory_pool_path=pool,
        test_candidate_manifest_path=manifest,
        dataset_manifest_path=dataset,
        split_manifest_path=splits,
        checkpoint_paths={"full": ckpt_a},
        methods=["smtr"],
        experiment_mode="formal",
    )
    assert summary["split_integrity_passed"] is True
    assert summary["checkpoint_digests"]["full"] == file_digest(ckpt_a)

    audit_path = tmp_path / "split_audit.json"
    audit_path.write_text(json.dumps(summary), encoding="utf-8")
    return {
        "audit_path": audit_path,
        "dataset": dataset,
        "splits": splits,
        "pool": pool,
        "manifest": manifest,
        "ckpt_a": ckpt_a,
        "ckpt_b": ckpt_b,
    }


def _validate(files, *, checkpoint_full):
    return validate_split_audit_artifact(
        split_audit_path=files["audit_path"],
        dataset_manifest_path=files["dataset"],
        split_manifest_path=files["splits"],
        memory_pool_path=files["pool"],
        candidate_manifest_path=files["manifest"],
        checkpoint_paths={"full": checkpoint_full},
        enabled_methods=["smtr"],
    )


def test_bound_checkpoint_validates(tmp_path):
    files = _setup(tmp_path)
    audit = _validate(files, checkpoint_full=files["ckpt_a"])
    assert audit["split_integrity_passed"] is True


def test_swapped_full_checkpoint_fails(tmp_path):
    files = _setup(tmp_path)
    with pytest.raises(
        ValueError, match="checkpoint digest mismatch: role='full'"
    ):
        _validate(files, checkpoint_full=files["ckpt_b"])
