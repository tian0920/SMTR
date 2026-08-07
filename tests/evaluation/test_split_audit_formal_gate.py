"""Test 8 (清单 P0-15~18): split audit formal gate.

The file-level audit wrapper must fail (``split_integrity_passed=False``)
when a treatment edge crosses train/test, when a target task crosses
splits, or when a checkpoint was calibrated on test records — and pass
when all three splits are fully isolated.
"""

from __future__ import annotations

import json

from smtr.evaluation.split_audit import audit_split_files
from smtr.router.transfer_critic import FourOutcomeTransferCritic


def _write_split_records(path, *, task_id, memory_id, seeds=(0,)):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for seed in seeds:
        lines.append(json.dumps({
            "record_type": "marble_candidate_level_pair",
            "task_id": task_id,
            "generation_seed": seed,
            "candidate_memory_id": memory_id,
            "receiver_agent_id": "r1",
            "valid": True,
            "label": "positive_transfer",
        }))
    path.write_text("".join(line + "\n" for line in lines), encoding="utf-8")


def _write_memory_pool(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({
            "memory_id": "m1",
            "payload": {
                "procedure": "x",
                "provenance": {
                    "source_agent_id": "w1",
                    "source_task_id": "train_source",
                    "source_trajectory_id": "traj_train",
                    "source_split": "train",
                },
            },
            "routing_card": {},
        }) + "\n",
        encoding="utf-8",
    )


def _isolated_split_files(tmp_path):
    train = tmp_path / "train.jsonl"
    val = tmp_path / "validation.jsonl"
    test = tmp_path / "test.jsonl"
    _write_split_records(train, task_id="t_train", memory_id="m_train")
    _write_split_records(val, task_id="t_val", memory_id="m_val")
    _write_split_records(test, task_id="t_test", memory_id="m_test")
    return train, val, test


def test_edge_crossing_train_and_test_fails(tmp_path):
    """Same treatment edge seeded in train and test must fail the gate."""
    train = tmp_path / "train.jsonl"
    val = tmp_path / "validation.jsonl"
    test = tmp_path / "test.jsonl"
    # Edge (t_shared, r1, m_shared) appears in both train and test.
    _write_split_records(train, task_id="t_shared", memory_id="m_shared", seeds=(0,))
    _write_split_records(val, task_id="t_val", memory_id="m_val")
    _write_split_records(test, task_id="t_shared", memory_id="m_shared", seeds=(1,))
    pool = tmp_path / "memories.jsonl"
    _write_memory_pool(pool)

    summary = audit_split_files(
        train_records_path=train,
        validation_records_path=val,
        test_records_path=test,
        memory_pool_path=pool,
    )
    assert summary["split_integrity_passed"] is False
    assert "treatment edge" in summary["error"]


def test_target_task_crossing_splits_fails(tmp_path):
    """Same target task in train and validation must fail the gate."""
    train = tmp_path / "train.jsonl"
    val = tmp_path / "validation.jsonl"
    test = tmp_path / "test.jsonl"
    # Distinct edges, but the same target task t_shared in train/validation.
    _write_split_records(train, task_id="t_shared", memory_id="m_a")
    _write_split_records(val, task_id="t_shared", memory_id="m_b")
    _write_split_records(test, task_id="t_test", memory_id="m_c")
    pool = tmp_path / "memories.jsonl"
    _write_memory_pool(pool)

    summary = audit_split_files(
        train_records_path=train,
        validation_records_path=val,
        test_records_path=test,
        memory_pool_path=pool,
    )
    assert summary["split_integrity_passed"] is False
    assert "target_task_id leakage" in summary["error"]


def test_test_calibrated_checkpoint_fails(tmp_path):
    """A checkpoint calibrated on the test split must fail the gate."""
    train, val, test = _isolated_split_files(tmp_path)
    pool = tmp_path / "memories.jsonl"
    _write_memory_pool(pool)

    critic = FourOutcomeTransferCritic()
    critic.calibration_split = "test"
    critic.epsilon_selection_split = "test"
    checkpoint = tmp_path / "critic.joblib"
    critic.save(checkpoint)

    summary = audit_split_files(
        train_records_path=train,
        validation_records_path=val,
        test_records_path=test,
        memory_pool_path=pool,
        checkpoint_paths={"full": checkpoint},
    )
    assert summary["split_integrity_passed"] is False
    assert "test records" in summary["error"]


def test_fully_isolated_splits_pass(tmp_path):
    """Fully isolated splits with a validation checkpoint must pass."""
    train, val, test = _isolated_split_files(tmp_path)
    pool = tmp_path / "memories.jsonl"
    _write_memory_pool(pool)

    critic = FourOutcomeTransferCritic()
    critic.calibration_split = "validation"
    critic.epsilon_selection_split = "validation"
    checkpoint = tmp_path / "critic.joblib"
    critic.save(checkpoint)

    summary = audit_split_files(
        train_records_path=train,
        validation_records_path=val,
        test_records_path=test,
        memory_pool_path=pool,
        checkpoint_paths={"full": checkpoint},
    )
    assert summary["split_integrity_passed"] is True
    assert summary["target_task_overlap"] == []
    assert summary["treatment_edge_overlap"] == []
    assert summary["test_used_for_calibration"] is False
