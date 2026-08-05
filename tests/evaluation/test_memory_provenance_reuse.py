"""R6 Test 1: train-derived memory reuse across validation/test is legal.

Procedural memories are extracted exclusively from train trajectories. The
same train-derived memory may be reused as a candidate for both validation
and test target tasks; this must be reported as provenance reuse, never as
a fatal target-trajectory overlap.
"""

from __future__ import annotations

from smtr.evaluation.split_audit import audit_split_leakage


def _rec(
    task_id: str,
    *,
    target_trajectory_id: str,
    candidate_memory_id: str,
    memory_source_trajectory_id: str = "traj_src_train",
) -> dict:
    return {
        "task_id": task_id,
        "receiver_agent_id": "receiver_1",
        "candidate_memory_id": candidate_memory_id,
        "edge_id": f"edge_{task_id}_{candidate_memory_id}",
        "target_trajectory_id": target_trajectory_id,
        "memory_source_task_id": "train_task_0",
        "memory_source_trajectory_id": memory_source_trajectory_id,
        "memory_source_split": "train",
    }


def test_train_memory_reused_by_validation_and_test_passes_audit():
    # Memory M is extracted from one train trajectory and serves candidates
    # in all three splits (including train's own pool-derived candidates).
    splits = {
        "train": [
            _rec("t_train_1", target_trajectory_id="traj_target_1",
                 candidate_memory_id="M"),
        ],
        "validation": [
            _rec("t_val_1", target_trajectory_id="traj_target_2",
                 candidate_memory_id="M"),
        ],
        "test": [
            _rec("t_test_1", target_trajectory_id="traj_target_3",
                 candidate_memory_id="M"),
        ],
    }

    audit = audit_split_leakage(splits)

    assert audit["split_integrity_passed"] is True
    # Reuse is reported as provenance statistics, not as leakage.
    assert audit["target_trajectory_overlap"] == []
    assert audit["target_task_overlap"] == []
    assert audit["treatment_edge_overlap"] == []
    assert audit["shared_train_memory_provenance_count"] == 1
    reuse = audit["memory_source_trajectory_reuse"]
    assert len(reuse) == 1
    assert reuse[0]["memory_source_trajectory_id"] == "traj_src_train"
    assert reuse[0]["observed_target_splits"] == ["train", "validation", "test"]


def test_memory_reuse_does_not_count_as_target_trajectory_overlap():
    # Even when the same memory serves multiple target tasks in the same
    # split, target trajectories stay disjoint and the audit passes.
    splits = {
        "train": [
            _rec("t_train_1", target_trajectory_id="traj_target_1",
                 candidate_memory_id="M"),
        ],
        "validation": [
            _rec("t_val_1", target_trajectory_id="traj_target_2",
                 candidate_memory_id="M"),
            _rec("t_val_2", target_trajectory_id="traj_target_3",
                 candidate_memory_id="M"),
        ],
        "test": [
            _rec("t_test_1", target_trajectory_id="traj_target_4",
                 candidate_memory_id="M"),
        ],
    }

    audit = audit_split_leakage(splits)
    assert audit["split_integrity_passed"] is True
    assert audit["target_trajectory_overlap"] == []
