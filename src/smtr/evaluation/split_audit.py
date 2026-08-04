"""Split leakage audit for paired records (清单第十三章).

Paired records must never be split at the individual-record level; splits
are group-based (target task group). This module audits the per-split paired
record files and fails fast when a required identifier crosses the
train/validation/test boundary.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SPLIT_NAMES = ("train", "validation", "test")


def audit_split_leakage(
    paired_records_by_split: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Audit identifier overlap across train/validation/test paired records.

    Hard requirements (fail fast when non-empty):
      * target_task_overlap   (task_id)
      * source_trajectory_overlap (source_trajectory_id)
      * edge_overlap          (edge_id)

    Advisory (reported, not fatal):
      * candidate_memory_overlap — the memory pool is built from train
        trajectories only, so memory ids are expected to recur in
        validation/test candidates; the overlap is reported so reviewers can
        decide whether further isolation is needed.
    """
    missing = [name for name in SPLIT_NAMES if name not in paired_records_by_split]
    if missing:
        raise ValueError(f"split audit requires all splits, missing: {missing}")

    target_tasks = {
        name: {
            str(rec.get("task_id", ""))
            for rec in paired_records_by_split[name]
            if rec.get("task_id") is not None
        }
        for name in SPLIT_NAMES
    }
    target_task_overlap = _cross_split_overlap(target_tasks)
    if target_task_overlap:
        raise ValueError(
            f"target_task_id leakage across splits: {sorted(target_task_overlap)}"
        )

    source_trajectory_overlap = _cross_split_overlap(
        {
            name: _collect(paired_records_by_split[name], "source_trajectory_id")
            for name in SPLIT_NAMES
        }
    )
    if source_trajectory_overlap:
        raise ValueError(
            "source_trajectory_id leakage across splits: "
            f"{sorted(source_trajectory_overlap)}"
        )

    edge_overlap = _cross_split_overlap(
        {name: _collect(paired_records_by_split[name], "edge_id") for name in SPLIT_NAMES}
    )
    if edge_overlap:
        raise ValueError(f"edge_id leakage across splits: {sorted(edge_overlap)}")

    candidate_memory_overlap = _cross_split_overlap(
        {
            name: _collect(paired_records_by_split[name], "candidate_memory_id")
            for name in SPLIT_NAMES
        }
    )

    return {
        "train_target_tasks": sorted(target_tasks["train"]),
        "validation_target_tasks": sorted(target_tasks["validation"]),
        "test_target_tasks": sorted(target_tasks["test"]),
        "target_task_overlap": sorted(target_task_overlap),
        "source_trajectory_overlap": sorted(source_trajectory_overlap),
        "edge_overlap": sorted(edge_overlap),
        "candidate_memory_overlap": sorted(candidate_memory_overlap),
    }


def write_split_audit(
    audit: dict[str, Any],
    output_path: Path,
) -> Path:
    """Write the split audit JSON (all set-valued fields serialized sorted)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output_path


def _collect(records: list[dict[str, Any]], field: str) -> set[str]:
    return {str(rec[field]) for rec in records if rec.get(field) is not None}


def _cross_split_overlap(sets: dict[str, set[str]]) -> set[str]:
    """Union of pairwise intersections across the three splits."""
    overlap: set[str] = set()
    names = list(sets)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            overlap |= sets[names[i]] & sets[names[j]]
    return overlap
