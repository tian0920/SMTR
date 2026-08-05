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
from smtr.marble.runtime_visibility_audit import file_digest

SPLIT_NAMES = ("train", "validation", "test")

# Artifact schema carrying per-file digests so formal evaluations can
# re-verify that the audited files are exactly the ones evaluated (R6 P1-5).
SPLIT_AUDIT_SCHEMA_VERSION = "smtr_split_audit_v2"


def audit_split_leakage(
    paired_records_by_split: dict[str, list[dict[str, Any]]],
    *,
    calibration_split: str = "validation",
    epsilon_selection_split: str = "validation",
) -> dict[str, Any]:
    """Audit identifier overlap across train/validation/test paired records.

    Target identity (task / execution trajectory / treatment edge) must be
    disjoint across splits, while memory provenance may legitimately recur:
    memories are extracted exclusively from train trajectories, so the same
    train-derived memory (and its source trajectory) may serve candidates
    in both validation and test (R6 清单 P0-1/P0-2/P0-3).

    Hard requirements (fail fast when non-empty):
      * target_task_overlap          (task_id)
      * target_trajectory_overlap    (target_trajectory_id)
      * treatment_edge_overlap       ((task_id, receiver_agent_id,
        candidate_memory_id)): every treatment edge must appear in exactly
        one split across all of its seed records.
      * edge_overlap                 (edge_id)
      * non_train_memory_sources: memory source trajectories must come from
        the train split only.
      * self_transfer_edges: a candidate target task must not equal the
        memory's source task.
      * calibration / epsilon selection must not use test records:
        ``test_used_for_calibration`` is computed from the recorded split
        provenance, never assumed.

    Statistics (reported, never fatal; R6 清单 P0-3):
      * shared_train_memory_provenance_count / memory_source_trajectory_reuse
        — train source trajectories observed in more than one split are
        legal memory reuse, not target leakage.
      * candidate_memory_overlap — the memory pool is built from train
        trajectories only, so memory ids are expected to recur in
        validation/test candidates.

    ``split_integrity_passed`` is computed from the real check results;
    it is never initialized to ``True``.
    """
    missing = [name for name in SPLIT_NAMES if name not in paired_records_by_split]
    if missing:
        raise ValueError(f"split audit requires all splits, missing: {missing}")

    # Treatment-edge consistency is checked first: it is the strictest
    # unit, since an edge crossing splits always implies a task crossing
    # splits as well.
    treatment_edges = _treatment_edge_overlap(paired_records_by_split)
    if treatment_edges["treatment_edge_overlap"]:
        raise ValueError(
            "treatment edge seeds split across splits "
            f"(each edge must live in exactly one split): "
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

    target_trajectory_overlap = _cross_split_overlap(
        {
            name: _collect(paired_records_by_split[name], "target_trajectory_id")
            for name in SPLIT_NAMES
        }
    )
    if target_trajectory_overlap:
        raise ValueError(
            "target_trajectory_id leakage across splits: "
            f"{sorted(target_trajectory_overlap)}"
        )

    # Memory source trajectories are provenance, not target identity: reuse
    # of a train-derived memory across splits is legal and only reported
    # (R6 清单 P0-3).
    memory_source_reuse = _memory_source_trajectory_reuse(paired_records_by_split)

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

    non_train_memory_sources = _non_train_memory_sources(paired_records_by_split)
    self_transfer_edges = _self_transfer_edges(paired_records_by_split)
    test_used_for_calibration = "test" in {
        calibration_split,
        epsilon_selection_split,
    }

    # Computed from the real check results; never assumed.
    split_integrity_passed = bool(
        not target_task_overlap
        and not target_trajectory_overlap
        and not treatment_edges["treatment_edge_overlap"]
        and not edge_overlap
        and not non_train_memory_sources
        and not self_transfer_edges
        and not test_used_for_calibration
    )
    if non_train_memory_sources:
        raise ValueError(
            "memory sources outside the train split: "
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
            f"epsilon_selection_split={epsilon_selection_split!r})."
        )

    return {
        "train_target_tasks": sorted(target_tasks["train"]),
        "validation_target_tasks": sorted(target_tasks["validation"]),
        "test_target_tasks": sorted(target_tasks["test"]),
        "target_task_overlap": sorted(target_task_overlap),
        "target_trajectory_overlap": sorted(target_trajectory_overlap),
        "edge_overlap": sorted(edge_overlap),
        "candidate_memory_overlap": sorted(candidate_memory_overlap),
        "treatment_edge_overlap": treatment_edges["treatment_edge_overlap"],
        "split_inconsistent_edges": treatment_edges["split_inconsistent_edges"],
        "treatment_edge_count_by_split": treatment_edges["edge_count_by_split"],
        "non_train_memory_sources": sorted(non_train_memory_sources),
        "self_transfer_edges": sorted(self_transfer_edges),
        "test_used_for_calibration": test_used_for_calibration,
        "shared_train_memory_provenance_count": len(memory_source_reuse),
        "memory_source_trajectory_reuse": memory_source_reuse,
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


def load_paired_records_file(path: Path) -> list[dict[str, Any]]:
    """Load one JSONL paired-records file."""
    records: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def audit_split_files(
    *,
    train_records_path: Path,
    validation_records_path: Path,
    test_records_path: Path,
    memory_pool_path: Path | None = None,
    checkpoint_path: Path | None = None,
    dataset_manifest_path: Path | None = None,
    split_manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Audit persisted split files end to end (清单 P0-15).

    Calibration / epsilon-selection provenance is read from the checkpoint
    when one is supplied; violations never raise out of this wrapper — the
    summary reports ``split_integrity_passed=False`` plus the error so the
    caller decides how to fail.

    The returned summary always carries the v2 artifact metadata (R6 清单
    P1-5): a schema version plus the SHA-256 digest of every audited file,
    so a formal end-to-end evaluation can later prove it evaluated exactly
    the artifacts this audit inspected. Digests use the project's canonical
    file-digest helper, never Python's built-in ``hash()``.
    """
    artifact_metadata = {
        "schema_version": SPLIT_AUDIT_SCHEMA_VERSION,
        "dataset_manifest_digest": _artifact_digest(dataset_manifest_path),
        "split_manifest_digest": _artifact_digest(split_manifest_path),
        "memory_pool_digest": _artifact_digest(memory_pool_path),
        "train_paired_records_digest": _artifact_digest(train_records_path),
        "validation_paired_records_digest": _artifact_digest(
            validation_records_path
        ),
        "test_paired_records_digest": _artifact_digest(test_records_path),
        "checkpoint_digest": _artifact_digest(checkpoint_path),
    }
    splits = {
        "train": load_paired_records_file(train_records_path),
        "validation": load_paired_records_file(validation_records_path),
        "test": load_paired_records_file(test_records_path),
    }
    calibration_split = "validation"
    epsilon_selection_split = "validation"
    if checkpoint_path is not None:
        from smtr.router.transfer_critic import FourOutcomeTransferCritic

        critic = FourOutcomeTransferCritic.load(Path(checkpoint_path))
        calibration_split = getattr(critic, "calibration_split", None) or "unknown"
        epsilon_selection_split = (
            getattr(critic, "epsilon_selection_split", None) or "unknown"
        )

    try:
        summary = audit_split_leakage(
            splits,
            calibration_split=calibration_split,
            epsilon_selection_split=epsilon_selection_split,
        )
    except ValueError as exc:
        return {
            **artifact_metadata,
            "split_integrity_passed": False,
            "error": str(exc),
            "calibration_split": calibration_split,
            "epsilon_selection_split": epsilon_selection_split,
        }

    non_train_pool_sources = sorted(_non_train_memory_pool_sources(memory_pool_path))
    if non_train_pool_sources:
        summary = dict(summary)
        summary["non_train_memory_pool_sources"] = non_train_pool_sources
        summary["split_integrity_passed"] = False
    summary.update(artifact_metadata)
    return summary


def _artifact_digest(path: Path | None) -> str | None:
    """SHA-256 digest of an audited file, or None when not supplied."""
    if path is None:
        return None
    return file_digest(Path(path))


def _non_train_memory_pool_sources(memory_pool_path: Path | None) -> set[str]:
    """Pool entries whose recorded source split is not train."""
    if memory_pool_path is None:
        return set()
    offenders: set[str] = set()
    for line in Path(memory_pool_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        source_split = entry.get("memory_source_split", entry.get("source_split"))
        if source_split is not None and source_split != "train":
            memory_id = entry.get("memory_id")
            if memory_id is not None:
                offenders.add(str(memory_id))
    return offenders


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


def _memory_source_trajectory_reuse(
    paired_records_by_split: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Train source trajectories observed in more than one split (R6 P0-3).

    Reuse of a train-derived memory across validation/test is legal; this
    is a statistic, never a fatal condition. Legacy records that only
    persist ``source_trajectory_id`` are included via fallback.
    """
    observed: dict[str, set[str]] = {}
    for name in SPLIT_NAMES:
        for rec in paired_records_by_split[name]:
            trajectory = rec.get(
                "memory_source_trajectory_id", rec.get("source_trajectory_id")
            )
            if trajectory in (None, ""):
                continue
            observed.setdefault(str(trajectory), set()).add(name)
    return [
        {
            "memory_source_trajectory_id": trajectory,
            "observed_target_splits": [s for s in SPLIT_NAMES if s in splits],
        }
        for trajectory, splits in sorted(observed.items())
        if len(splits) > 1
    ]


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
    """Edges whose target task equals the memory's source task.

    Reads ``memory_source_task_id`` with a ``source_task_id`` fallback for
    legacy artifacts (R6 清单 P0-1).
    """
    offenders: set[TreatmentEdgeKey] = set()
    for name in SPLIT_NAMES:
        for rec in paired_records_by_split[name]:
            source_task = rec.get(
                "memory_source_task_id", rec.get("source_task_id")
            )
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
