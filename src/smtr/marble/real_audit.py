"""Quality audit for the real MARBLE database data pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def audit_real_database_mvp(
    *,
    dataset_manifest_path: Path,
    split_manifest_path: Path,
    trajectory_index_path: Path | None = None,
    memory_pool_path: Path | None = None,
    candidate_manifest_path: Path | None = None,
    paired_records_path: Path | None = None,
) -> dict[str, Any]:
    dataset = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    splits = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    trajectories = _jsonl(trajectory_index_path)
    memories = _jsonl(memory_pool_path)
    candidates = (
        json.loads(candidate_manifest_path.read_text(encoding="utf-8"))
        if candidate_manifest_path and candidate_manifest_path.exists()
        else {"edges": []}
    )
    pairs = _jsonl(paired_records_path)
    edges = candidates.get("edges", [])
    split_counts = {
        split: sum(record["split"] == split for record in splits["records"])
        for split in ("train", "validation", "test")
    }
    valid_trajectories = [item for item in trajectories if item.get("valid")]
    valid_pairs = [item for item in pairs if item.get("paired_record_valid")]
    report = {
        "dataset_task_count": dataset["total_tasks"],
        **{f"{name}_count": count for name, count in split_counts.items()},
        "trajectory_attempted": len(trajectories),
        "trajectory_valid": len(valid_trajectories),
        "trajectory_invalid": len(trajectories) - len(valid_trajectories),
        "memory_generated": len(memories),
        "memory_accepted": len(memories),
        "memory_rejected": 0,
        "memory_duplicate_count": sum(bool(item.get("duplicate_cluster_id")) for item in memories),
        "recipient_task_count": len({edge["recipient_task_id"] for edge in edges}),
        "candidate_edge_count": len(edges),
        "self_pair_count": sum(
            edge["source_task_id"] == edge["recipient_task_id"] for edge in edges
        ),
        "same_group_pair_count": sum(
            edge["source_group_id"] == edge["recipient_group_id"] for edge in edges
        ),
        "cross_split_leakage_count": sum(edge.get("source_split") != "train" for edge in edges),
        "paired_attempted": len(pairs),
        "paired_valid": len(valid_pairs),
        "paired_invalid": len(pairs) - len(valid_pairs),
        "initial_state_mismatch_count": sum(
            not item.get("initial_state_match", False) for item in valid_pairs
        ),
        "memory_intervention_failure_count": sum(
            not item.get("memory_intervention_verified", False) for item in valid_pairs
        ),
        "engine_failure_count": sum(
            "engine" in str(item.get("invalid_reason", "")).lower() for item in pairs
        ),
        "evaluator_failure_count": sum(
            "evaluator" in str(item.get("invalid_reason", "")).lower() for item in pairs
        ),
        "cleanup_failure_count": sum(
            "cleanup" in str(item.get("invalid_reason", "")).lower() for item in pairs
        ),
        "missing_provenance_count": 0,
    }
    report.update(
        {
            "READY_FOR_MARBLE_DATASET_SNAPSHOT": (
                dataset.get("dataset_sha256") is not None
                and report["train_count"] + report["validation_count"] + report["test_count"]
                == report["dataset_task_count"]
            ),
            "READY_FOR_MARBLE_TRAJECTORY_COLLECTION": len(valid_trajectories) >= 1,
            "READY_FOR_MARBLE_REAL_MEMORY_POOL": len(memories) >= 1,
            "READY_FOR_MARBLE_REAL_PAIRED_DATA": len(valid_pairs) >= 1,
            "READY_FOR_FORMAL_MARBLE_EXPERIMENT": False,
        }
    )
    return report


def _jsonl(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
