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

from smtr.counterfactual.edge_keys import TreatmentEdgeKey, treatment_edge_key

SPLIT_NAMES = ("train", "validation", "test")


def audit_split_leakage(
    paired_records_by_split: dict[str, list[dict[str, Any]]],
    *,
    calibration_split: str = "validation",
    epsilon_selection_split: str = "validation",
) -> dict[str, Any]:
    """Audit identifier overlap across train/validation/test paired records.

    Hard requirements (fail fast when non-empty):
      * target_task_overlap   (task_id)
      * source_trajectory_overlap (source_trajectory_id)
      * edge_overlap          (edge_id)
      * treatment edge split consistency (清单 P0-4): every treatment edge
        ``(task_id, receiver_agent_id, candidate_memory_id)`` must appear in
        exactly one split across all of its seed records, i.e.
        ``len(edge_observed_splits[edge]) == 1`` for every edge.
      * non_train_memory_sources (清单 P1-1): memory source trajectories
        must come from the train split only.
      * self_transfer_edges (清单 P1-2): a candidate target task must not
        equal the memory's source task.
      * calibration / epsilon selection must not use test records
        (清单 P0-8 / P1-2): ``test_used_for_calibration`` is computed from
        the recorded split provenance, never assumed.

    Advisory (reported, not fatal):
      * candidate_memory_overlap — the memory pool is built from train
        trajectories only, so memory ids are expected to recur in
        validation/test candidates; the overlap is reported so reviewers can
        decide whether further isolation is needed.

    ``split_integrity_passed`` is computed from the real check results
    (清单 P1-2); it is never initialized to ``True``.
    """
    missing = [name for name in SPLIT_NAMES if name not in paired_records_by_split]
    if missing:
        raise ValueError(f"split audit requires all splits, missing: {missing}")

    # Treatment-edge consistency (清单 P0-4) is checked first: it is the
    # strictest unit, since an edge crossing splits always implies a task
    # crossing splits as well.
    treatment_edges = _treatment_edge_overlap(paired_records_by_split)
    if treatment_edges["treatment_edge_overlap"]:
        raise ValueError(
            "treatment edge seeds split across splits (清单 P0-4 requires "
            f"len(edge_observed_splits[edge]) == 1): "
            f"{treatment_edges['treatment_edge_overlap']}"
        )

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

    # 清单 P1-1/P1-2: memory source trajectories must come from train only,
    # and no candidate may target the task that produced its memory.
    non_train_memory_sources = _non_train_memory_sources(paired_records_by_split)
    self_transfer_edges = _self_transfer_edges(paired_records_by_split)
    test_used_for_calibration = "test" in {
        calibration_split,
        epsilon_selection_split,
    }

    # Computed from the real check results (清单 P1-2); never assumed.
    split_integrity_passed = bool(
        not target_task_overlap
        and not treatment_edges["treatment_edge_overlap"]
        and not source_trajectory_overlap
        and not edge_overlap
        and not non_train_memory_sources
        and not self_transfer_edges
        and not test_used_for_calibration
    )
    if non_train_memory_sources:
        raise ValueError(
            "memory sources outside the train split (清单 P1-1): "
            f"{sorted(non_train_memory_sources)}"
        )
    if self_transfer_edges:
        raise ValueError(
            f"self-transfer edges (target task == memory source task): "
            f"{sorted(self_transfer_edges)}"
        )
    if test_used_for_calibration:
        raise ValueError(
            "calibration/epsilon selection used test records "
            f"(calibration_split={calibration_split!r}, "
            f"epsilon_selection_split={epsilon_selection_split!r}; 清单 P0-8)."
        )

    return {
        "train_target_tasks": sorted(target_tasks["train"]),
        "validation_target_tasks": sorted(target_tasks["validation"]),
        "test_target_tasks": sorted(target_tasks["test"]),
        "target_task_overlap": sorted(target_task_overlap),
        "source_trajectory_overlap": sorted(source_trajectory_overlap),
        "edge_overlap": sorted(edge_overlap),
        "candidate_memory_overlap": sorted(candidate_memory_overlap),
        "treatment_edge_overlap": treatment_edges["treatment_edge_overlap"],
        "split_inconsistent_edges": treatment_edges["split_inconsistent_edges"],
        "treatment_edge_count_by_split": treatment_edges["edge_count_by_split"],
        "non_train_memory_sources": sorted(non_train_memory_sources),
        "self_transfer_edges": sorted(self_transfer_edges),
        "test_used_for_calibration": test_used_for_calibration,
        "calibration_split": calibration_split,
        "epsilon_selection_split": epsilon_selection_split,
        "split_integrity_passed": split_integrity_passed,
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


def _treatment_edge_overlap(
    paired_records_by_split: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Per-edge split membership for treatment edges (清单 P0-4).

    Returns the set of edges observed in more than one split
    (``treatment_edge_overlap``) plus the subset whose seeds were placed in
    multiple splits (``split_inconsistent_edges``). Edges are identified by
    the canonical ``(task_id, receiver_agent_id, candidate_memory_id)`` key.
    """
    edge_observed_splits: dict[TreatmentEdgeKey, set[str]] = {}
    edge_count_by_split: dict[str, int] = {}
    for name in SPLIT_NAMES:
        split_edges = {
            treatment_edge_key(rec)
            for rec in paired_records_by_split[name]
            if rec.get("task_id") is not None
            and rec.get("receiver_agent_id") is not None
            and rec.get("candidate_memory_id") is not None
        }
        edge_count_by_split[name] = len(split_edges)
        for edge in split_edges:
            edge_observed_splits.setdefault(edge, set()).add(name)

    overlap_edges = sorted(
        edge for edge, splits in edge_observed_splits.items() if len(splits) > 1
    )
    return {
        "treatment_edge_overlap": overlap_edges,
        "split_inconsistent_edges": overlap_edges,
        "edge_count_by_split": edge_count_by_split,
        "edge_observed_splits": edge_observed_splits,
    }


def _collect(records: list[dict[str, Any]], field: str) -> set[str]:
    return {str(rec[field]) for rec in records if rec.get(field) is not None}


def _non_train_memory_sources(
    paired_records_by_split: dict[str, list[dict[str, Any]]],
) -> set[str]:
    """Memory ids whose recorded source split is not train (清单 P1-1).

    Only records that persist ``memory_source_split`` are checked; the
    memory pool construction itself is train-only by design, so this is a
    provenance re-check against the persisted records.
    """
    offenders: set[str] = set()
    for name in SPLIT_NAMES:
        for rec in paired_records_by_split[name]:
            source_split = rec.get("memory_source_split")
            if source_split is not None and source_split != "train":
                memory_id = rec.get("candidate_memory_id")
                if memory_id is not None:
                    offenders.add(str(memory_id))
    return offenders


def _self_transfer_edges(
    paired_records_by_split: dict[str, list[dict[str, Any]]],
) -> set[TreatmentEdgeKey]:
    """Edges whose target task equals the memory's source task (清单 P1-2)."""
    offenders: set[TreatmentEdgeKey] = set()
    for name in SPLIT_NAMES:
        for rec in paired_records_by_split[name]:
            source_task = rec.get("source_task_id")
            if source_task in (None, ""):
                continue
            target_task = rec.get("task_id")
            if target_task is not None and str(source_task) == str(target_task):
                if (
                    rec.get("receiver_agent_id") is not None
                    and rec.get("candidate_memory_id") is not None
                ):
                    offenders.add(treatment_edge_key(rec))
    return offenders


def _cross_split_overlap(sets: dict[str, set[str]]) -> set[str]:
    """Union of pairwise intersections across the three splits."""
    overlap: set[str] = set()
    names = list(sets)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            overlap |= sets[names[i]] & sets[names[j]]
    return overlap
