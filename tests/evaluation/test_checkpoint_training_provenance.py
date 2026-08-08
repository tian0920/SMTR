"""Checkpoint training provenance digest checks (清单 P0-2 3.6).

Formal audits fail closed when a checkpoint's persisted training
provenance (training split, train/validation/pool digests) does not
match the files the audit itself consumed.
"""

from __future__ import annotations

import json

import pytest

from smtr.evaluation.split_audit import audit_split_files
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


def _audit(tmp_path, *, mutate) -> dict:
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

    budget_manifest = tmp_path / "budget_candidates.json"
    budget_manifest.write_text(
        json.dumps(
            {
                "target_split": "train",
                "memory_source_split": "train",
                "candidates": [],
            }
        ),
        encoding="utf-8",
    )

    critic = FourOutcomeTransferCritic(feature_block="full")
    critic.calibration_split = "validation"
    critic.epsilon_selection_split = "validation"
    critic.training_split = "train"
    critic.train_record_digest = file_digest(train)
    critic.validation_record_digest = file_digest(val)
    critic.memory_pool_digest = file_digest(pool)
    from smtr.marble.artifact_digests import (
        candidate_manifest_digest_from_path,
    )
    critic.budget_train_candidate_manifest_digest = (
        candidate_manifest_digest_from_path(budget_manifest)
    )
    # Budget manifest has empty candidates, so effective records are empty.
    from smtr.evaluation.training_support import canonical_effective_record_digest
    critic.effective_train_record_digest = canonical_effective_record_digest([])
    critic.effective_train_edge_count = 0
    critic.epsilon_star = 0.2
    critic.method_schema_metadata = {
        "method_schema": "memory_receiver_v1",
        "routing_conditioning": "memory_receiver",
        "writer_features_used": False,
        "provenance_features_used": False,
        "outcome_level": "team_success",
        "treatment_edge_unit": "task_receiver_memory",
    }
    mutate(critic)
    ckpt = tmp_path / "full.joblib"
    critic.save(ckpt)

    return audit_split_files(
        train_records_path=train,
        validation_records_path=val,
        test_records_path=test,
        memory_pool_path=pool,
        test_candidate_manifest_path=manifest,
        train_budget_candidate_manifest_path=budget_manifest,
        checkpoint_paths={"full": ckpt},
        methods=["smtr"],
        experiment_mode="formal",
    )


def _noop(critic) -> None:
    pass


def test_matching_provenance_passes(tmp_path):
    summary = _audit(tmp_path, mutate=_noop)
    assert summary["split_integrity_passed"] is True
    assert summary["checkpoint_binding_errors"] == []


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (
            lambda critic: setattr(critic, "train_record_digest", "bogus"),
            "checkpoint role 'full' train record digest mismatch",
        ),
        (
            lambda critic: setattr(
                critic, "validation_record_digest", "bogus"
            ),
            "checkpoint role 'full' validation record digest mismatch",
        ),
        (
            lambda critic: setattr(critic, "memory_pool_digest", "bogus"),
            "checkpoint role 'full' memory pool digest mismatch",
        ),
        (
            lambda critic: setattr(critic, "training_split", "validation"),
            "checkpoint role 'full' training_split must be 'train', "
            "got 'validation'",
        ),
    ],
)
def test_bad_training_provenance_fails_formal_audit(tmp_path, mutate, expected):
    summary = _audit(tmp_path, mutate=mutate)
    assert summary["split_integrity_passed"] is False
    assert expected in summary["checkpoint_binding_errors"]
