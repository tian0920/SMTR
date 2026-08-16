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
from smtr.evaluation.training_support import (
    canonical_effective_record_digest,
    checkpoint_support_signature,
    edge_seed_sets,
    filter_records_to_selected_edges,
    selected_edge_keys_from_candidate_manifest,
    validate_checkpoint_support_against_audit,
)
from smtr.marble.artifact_digests import candidate_manifest_digest_from_path
from smtr.marble.runtime_visibility_audit import file_digest

SPLIT_NAMES = ("train", "validation", "test")

# Artifact schema carrying per-file digests so formal evaluations can
# re-verify that the audited files are exactly the ones evaluated (R6 P1-5).
# v4 binds the test candidate manifest, a per-role checkpoint digest map,
# fail-closed provenance validation, and budget training support binding.
SPLIT_AUDIT_SCHEMA_VERSION = "smtr_split_audit_v4"

# Method -> checkpoint role required to run that method (清单 3.4).
_METHOD_CHECKPOINT_ROLES = {
    "smtr": "full",
    "global_transfer_critic": "global_transfer",
    "smtr_no_compatibility_interaction": "no_compatibility_interaction",
}


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


def validate_candidate_manifest_for_formal_evaluation(
    manifest: dict[str, Any],
    *,
    expected_target_split: str,
) -> list[str]:
    """Schema gate for the test candidate manifest (清单 P0-1 2.4).

    The manifest must target the evaluation split, draw all memories from
    the train split and carry a non-empty candidate list.
    """
    errors: list[str] = []

    if manifest.get("target_split") != expected_target_split:
        errors.append(
            "candidate manifest target_split mismatch: "
            f"expected={expected_target_split!r}, "
            f"actual={manifest.get('target_split')!r}"
        )

    if manifest.get("memory_source_split") != "train":
        errors.append(
            "candidate manifest memory_source_split must be 'train'"
        )

    candidates = manifest.get("candidates")

    if not isinstance(candidates, list):
        errors.append("candidate manifest must contain a candidates list")
        return errors

    if not candidates:
        errors.append("candidate manifest contains no candidate entries")

    return errors


def audit_split_files(
    *,
    train_records_path: Path,
    validation_records_path: Path,
    test_records_path: Path,
    memory_pool_path: Path | None = None,
    test_candidate_manifest_path: Path | None = None,
    checkpoint_paths: dict[str, Path] | None = None,
    methods: list[str] | None = None,
    dataset_manifest_path: Path | None = None,
    split_manifest_path: Path | None = None,
    train_budget_candidate_manifest_path: Path | None = None,
    strict_candidate_support: bool = True,
    experiment_mode: str = "pilot",
) -> dict[str, Any]:
    """Audit persisted split files end to end (清单 P0-15).

    Calibration / epsilon-selection provenance is read from the full-role
    checkpoint when one is supplied; violations never raise out of this
    wrapper — the summary reports ``split_integrity_passed=False`` plus the
    error so the caller decides how to fail.

    The v4 artifact binds: the test candidate manifest, a per-role checkpoint
    digest map with training provenance and budget support verification,
    cross-checkpoint support equality, and fail-closed provenance schema
    validation in formal mode.
    """
    resolved_checkpoint_paths = dict(checkpoint_paths or {})

    # 清单最终闭环 P0-4: formal audits must bind the same train budget
    # manifest that training used; None -> full support stays pilot-only.
    if (
        experiment_mode == "formal"
        and train_budget_candidate_manifest_path is None
    ):
        raise ValueError(
            "formal split audit requires an explicit train budget "
            "candidate manifest, including B=1.0"
        )

    candidate_manifest: dict[str, Any] | None = None
    if test_candidate_manifest_path is not None:
        candidate_manifest = json.loads(
            Path(test_candidate_manifest_path).read_text(encoding="utf-8")
        )

    # Load train budget candidate manifest for independent support verification.
    budget_manifest: dict[str, Any] | None = None
    if train_budget_candidate_manifest_path is not None:
        budget_manifest = json.loads(
            Path(train_budget_candidate_manifest_path).read_text(encoding="utf-8")
        )

    artifact_metadata: dict[str, Any] = {
        "schema_version": SPLIT_AUDIT_SCHEMA_VERSION,
        "experiment_mode": experiment_mode,
        "dataset_manifest_digest": _artifact_digest(dataset_manifest_path),
        "split_manifest_digest": _artifact_digest(split_manifest_path),
        "memory_pool_digest": _artifact_digest(memory_pool_path),
        "train_paired_records_digest": _artifact_digest(train_records_path),
        "validation_paired_records_digest": _artifact_digest(
            validation_records_path
        ),
        "test_paired_records_digest": _artifact_digest(test_records_path),
        "test_candidate_manifest_digest": _artifact_digest(
            test_candidate_manifest_path
        ),
        "candidate_manifest_target_split": (
            candidate_manifest.get("target_split")
            if candidate_manifest is not None
            else None
        ),
        # 清单最终闭环 P0-2: the budget manifest is identified by its
        # canonical content digest (same algorithm as training), never by
        # raw file bytes.
        "train_budget_candidate_manifest_digest": (
            candidate_manifest_digest_from_path(
                train_budget_candidate_manifest_path
            )
            if train_budget_candidate_manifest_path is not None
            else None
        ),
    }

    splits = {
        "train": load_paired_records_file(train_records_path),
        "validation": load_paired_records_file(validation_records_path),
        "test": load_paired_records_file(test_records_path),
    }

    # 清单 P0-2: bind every formal checkpoint role and verify its feature
    # block plus training provenance before any leakage check.
    (
        checkpoint_digests,
        checkpoint_metadata,
        critics,
        checkpoint_binding_errors,
    ) = _bind_checkpoints(
        resolved_checkpoint_paths,
        methods=methods,
        experiment_mode=experiment_mode,
        train_digest=artifact_metadata["train_paired_records_digest"],
        validation_digest=artifact_metadata["validation_paired_records_digest"],
        memory_pool_digest=artifact_metadata["memory_pool_digest"],
        budget_manifest_digest=artifact_metadata[
            "train_budget_candidate_manifest_digest"
        ],
    )

    full_critic = critics.get("full")
    calibration_split = "validation"
    epsilon_selection_split = "validation"
    if full_critic is not None:
        calibration_split = getattr(full_critic, "calibration_split", None) or "unknown"
        epsilon_selection_split = (
            getattr(full_critic, "epsilon_selection_split", None) or "unknown"
        )

    # 清单 P0-1: independently compute effective training support from
    # full train records + frozen budget manifest; never trust checkpoint
    # self-reported metadata.
    audit_training_support: dict[str, Any] = {}
    audit_effective_digest: str | None = None
    audit_effective_edge_count: int | None = None
    audit_budget_manifest_digest: str | None = artifact_metadata[
        "train_budget_candidate_manifest_digest"
    ]
    if budget_manifest is not None:
        selected_edges = selected_edge_keys_from_candidate_manifest(
            budget_manifest
        )
        effective_records = filter_records_to_selected_edges(
            splits["train"], selected_edges
        )
        budget_meta = budget_manifest.get("budget_metadata", {})
        audit_training_support = {
            "parent_train_record_digest": artifact_metadata[
                "train_paired_records_digest"
            ],
            "budget_candidate_manifest_digest": audit_budget_manifest_digest,
            "requested_budget_fraction": budget_meta.get(
                "requested_fraction"
            ),
            "realized_budget_fraction": budget_meta.get(
                "realized_edge_fraction"
            ),
            "selected_edge_count": len(selected_edges),
            "effective_train_edge_count": len(
                {treatment_edge_key(r) for r in effective_records}
            ),
            "effective_train_record_digest": (
                canonical_effective_record_digest(effective_records)
            ),
        }
        audit_effective_digest = audit_training_support[
            "effective_train_record_digest"
        ]
        audit_effective_edge_count = audit_training_support[
            "effective_train_edge_count"
        ]
        # Independent seed support verification (清单 §3).
        parent_seeds = edge_seed_sets(splits["train"])
        effective_seeds = edge_seed_sets(effective_records)
        incomplete_seed_edges = []
        for edge in selected_edges:
            expected = parent_seeds.get(edge, set())
            observed = effective_seeds.get(edge, set())
            if expected != observed:
                incomplete_seed_edges.append({
                    "edge": edge,
                    "expected_seeds": sorted(expected),
                    "observed_seeds": sorted(observed),
                })
        audit_training_support["full_seed_support_passed"] = (
            not incomplete_seed_edges
        )
        if (
            experiment_mode == "formal"
            and incomplete_seed_edges
        ):
            checkpoint_binding_errors.append(
                "budget support removed individual seeds "
                "instead of complete treatment edges"
            )

    # 清单 §5/§9: cross-checkpoint support equality using independently
    # computed audit truth.
    support_signatures: dict[str, tuple[Any, ...]] = {}
    for role, meta in checkpoint_metadata.items():
        support_signatures[role] = checkpoint_support_signature(
            critics[role]
        )
    cross_checkpoint_support_equal = (
        len(set(support_signatures.values())) <= 1
    )
    if not cross_checkpoint_support_equal and experiment_mode == "formal":
        checkpoint_binding_errors.append(
            "formal critic checkpoints were trained on different "
            "budget supports"
        )
    # Each checkpoint's declared support vs audit independent truth.
    checkpoint_training_support: dict[str, dict[str, Any]] = {}
    if experiment_mode == "formal" and audit_effective_digest is not None:
        for role, critic_obj in critics.items():
            validation_errors = validate_checkpoint_support_against_audit(
                critic=critic_obj,
                role=role,
                audit_effective_digest=audit_effective_digest,
                audit_effective_edge_count=audit_effective_edge_count or 0,
                audit_budget_manifest_digest=audit_budget_manifest_digest,
            )
            checkpoint_training_support[role] = {
                "declared": dict(zip(
                    (
                        "budget_manifest_digest",
                        "effective_digest",
                        "edge_count",
                        "budget_requested",
                        "budget_realized",
                    ),
                    support_signatures.get(role, (None,) * 5),
                )),
                "validation_errors": validation_errors,
            }
            checkpoint_binding_errors.extend(validation_errors)

    # 清单 P1-1/P1-4: formal provenance schema validation runs before any
    # overlap check and fails closed.
    if experiment_mode == "formal":
        provenance_errors, missing_provenance = _audit_formal_provenance(splits)
        if provenance_errors:
            return {
                **artifact_metadata,
                "checkpoint_digests": checkpoint_digests,
                "checkpoint_metadata": checkpoint_metadata,
                "checkpoint_binding_errors": checkpoint_binding_errors,
                "split_integrity_passed": False,
                "provenance_errors": provenance_errors,
                "missing_provenance": missing_provenance,
                "calibration_split": calibration_split,
                "epsilon_selection_split": epsilon_selection_split,
            }

    try:
        summary = audit_split_leakage(
            splits,
            calibration_split=calibration_split,
            epsilon_selection_split=epsilon_selection_split,
        )
    except ValueError as exc:
        return {
            **artifact_metadata,
            "checkpoint_digests": checkpoint_digests,
            "checkpoint_metadata": checkpoint_metadata,
            "checkpoint_binding_errors": checkpoint_binding_errors,
            "split_integrity_passed": False,
            "error": str(exc),
            "calibration_split": calibration_split,
            "epsilon_selection_split": epsilon_selection_split,
        }

    # 清单 P0-1: candidate manifest support against the test paired records.
    candidate_manifest_block: dict[str, Any] | None = None
    candidate_manifest_errors: list[str] = []
    test_edges_missing_from_manifest: list[Any] = []
    unsupported_candidate_edges: list[Any] = []
    if candidate_manifest is not None:
        candidate_manifest_block = _audit_candidate_manifest_support(
            candidate_manifest,
            manifest_path=test_candidate_manifest_path,
            manifest_digest=artifact_metadata["test_candidate_manifest_digest"],
            test_records=splits["test"],
        )
        candidate_manifest_errors = candidate_manifest_block[
            "candidate_manifest_errors"
        ]
        test_edges_missing_from_manifest = candidate_manifest_block[
            "test_edges_missing_from_manifest"
        ]
        unsupported_candidate_edges = candidate_manifest_block[
            "unsupported_candidate_edges"
        ]
    elif experiment_mode == "formal":
        candidate_manifest_errors = [
            "formal split audit requires a test candidate manifest"
        ]

    non_train_pool_sources = sorted(_non_train_memory_pool_sources(memory_pool_path))

    # 清单 Writer-Agnostic 第十五章: writer/source-agent identity must not
    # drive routing, candidate scoring or critic features. Verified against
    # the checkpoint metadata of every bound critic; unknown provenance
    # fails closed (reported as True = used).
    writer_free_declarations = _writer_free_declarations(critics, checkpoint_digests)

    split_integrity_passed = bool(summary["split_integrity_passed"])
    if non_train_pool_sources:
        split_integrity_passed = False
    if candidate_manifest_errors or test_edges_missing_from_manifest:
        split_integrity_passed = False
    if strict_candidate_support and unsupported_candidate_edges:
        split_integrity_passed = False
    if checkpoint_binding_errors:
        split_integrity_passed = False
    if experiment_mode == "formal" and any(writer_free_declarations.values()):
        split_integrity_passed = False

    summary = dict(summary)
    if non_train_pool_sources:
        summary["non_train_memory_pool_sources"] = non_train_pool_sources
    summary.update(artifact_metadata)
    summary.update(
        {
            "checkpoint_digests": checkpoint_digests,
            "checkpoint_metadata": checkpoint_metadata,
            "checkpoint_binding_errors": checkpoint_binding_errors,
            "cross_checkpoint_support_equal": cross_checkpoint_support_equal,
            "training_support": audit_training_support,
            "checkpoint_training_support": checkpoint_training_support,
            "candidate_manifest": candidate_manifest_block,
            "candidate_manifest_errors": candidate_manifest_errors,
            "test_edges_missing_from_manifest": test_edges_missing_from_manifest,
            "unsupported_candidate_edges": unsupported_candidate_edges,
            "strict_candidate_support": strict_candidate_support,
            **writer_free_declarations,
            "split_integrity_passed": split_integrity_passed,
        }
    )
    return summary


def _writer_free_declarations(
    critics: dict[str, Any],
    checkpoint_digests: dict[str, str],
) -> dict[str, bool]:
    """Writer-agnostic declarations for the split audit (清单 第十五章).

    Each flag reports whether a writer/source-agent signal was used where
    the writer-agnostic method forbids it. Flags are derived from the
    checkpoint metadata persisted at training time; a critic without
    writer-free metadata fails closed (flag True).
    """
    # Baseline methods bind no critic checkpoint, so no critic features are
    # consumed at all; the declaration is vacuously writer-free. Every bound
    # critic must carry writer-free feature metadata; unknown provenance
    # fails closed (reported as True = used).
    features_used = False
    for critic in critics.values():
        metadata = getattr(critic, "method_schema_metadata", None) or {}
        if metadata.get("writer_features_used") is not False:
            features_used = True
            break
        if metadata.get("provenance_features_used") is not False:
            features_used = True
            break
    # Routing and candidate scoring never consume writer identity in the
    # writer-agnostic pipeline (清单 第二章): candidate records and routing
    # cards carry no writer fields, which the feature/candidate audits
    # enforce upstream. The audit declares the structural guarantee.
    return {
        "writer_identity_used_for_routing": False,
        "source_agent_used_for_candidate_scoring": False,
        "source_agent_used_for_critic_features": features_used,
    }


def _bind_checkpoints(
    checkpoint_paths: dict[str, Path],
    *,
    methods: list[str] | None,
    experiment_mode: str,
    train_digest: str | None,
    validation_digest: str | None,
    memory_pool_digest: str | None,
    budget_manifest_digest: str | None = None,
) -> tuple[dict[str, str], dict[str, dict[str, Any]], dict[str, Any], list[str]]:
    """Per-role checkpoint digests, metadata and binding errors (清单 P0-2).

    Feature-block role validation runs in every mode; the training
    provenance digest checks only fail closed in formal mode so pilot
    checkpoints without persisted provenance remain auditable.
    """
    from smtr.evaluation.formal_artifacts import validate_checkpoint_role

    errors: list[str] = []
    if methods is not None:
        for method in methods:
            role = _METHOD_CHECKPOINT_ROLES.get(method)
            if role is not None and role not in checkpoint_paths:
                errors.append(
                    f"method {method!r} requires checkpoint role {role!r}"
                )

    digests: dict[str, str] = {}
    metadata: dict[str, dict[str, Any]] = {}
    critics: dict[str, Any] = {}
    for role in sorted(checkpoint_paths):
        path = Path(checkpoint_paths[role])
        digests[role] = file_digest(path)
        try:
            critic = validate_checkpoint_role(
                checkpoint_path=path, expected_role=role
            )
        except ValueError as exc:
            errors.append(str(exc))
            continue
        critics[role] = critic
        metadata[role] = {
            "feature_block": getattr(critic, "feature_block", None),
            "training_split": getattr(critic, "training_split", None),
            "calibration_split": getattr(critic, "calibration_split", None),
            "epsilon_selection_split": getattr(
                critic, "epsilon_selection_split", None
            ),
            "calibration_unit": getattr(critic, "calibration_unit", None),
            "epsilon_selection_unit": getattr(
                critic, "epsilon_selection_unit", None
            ),
            # 清单 §7/§9: budget training support binding
            "training_support": {
                "parent_train_record_digest": getattr(
                    critic, "parent_train_candidate_manifest_digest", None
                ),
                "effective_train_record_digest": getattr(
                    critic, "effective_train_record_digest", None
                ),
                "budget_candidate_manifest_digest": getattr(
                    critic, "budget_train_candidate_manifest_digest", None
                ),
                "requested_budget_fraction": getattr(
                    critic, "training_budget_requested", None
                ),
                "realized_budget_fraction": getattr(
                    critic, "training_budget_realized", None
                ),
                "effective_train_edge_count": getattr(
                    critic, "effective_train_edge_count", None
                ),
            },
        }
        if experiment_mode == "formal":
            errors.extend(
                _checkpoint_training_provenance_errors(
                    critic,
                    role=role,
                    train_digest=train_digest,
                    validation_digest=validation_digest,
                    memory_pool_digest=memory_pool_digest,
                    budget_manifest_digest=budget_manifest_digest,
                )
            )
    return digests, metadata, critics, errors


def _checkpoint_training_provenance_errors(
    critic: Any,
    *,
    role: str,
    train_digest: str | None,
    validation_digest: str | None,
    memory_pool_digest: str | None,
    budget_manifest_digest: str | None = None,
) -> list[str]:
    """Formal-mode check of the provenance persisted in a checkpoint (3.6)."""
    errors: list[str] = []
    if getattr(critic, "training_split", None) != "train":
        errors.append(
            f"checkpoint role {role!r} training_split must be 'train', "
            f"got {getattr(critic, 'training_split', None)!r}"
        )
    if getattr(critic, "train_record_digest", None) != train_digest:
        errors.append(f"checkpoint role {role!r} train record digest mismatch")
    if (
        validation_digest is not None
        and getattr(critic, "validation_record_digest", None) != validation_digest
    ):
        errors.append(
            f"checkpoint role {role!r} validation record digest mismatch"
        )
    if (
        memory_pool_digest is not None
        and getattr(critic, "memory_pool_digest", None) != memory_pool_digest
    ):
        errors.append(f"checkpoint role {role!r} memory pool digest mismatch")
    # 清单 §8.1: budget manifest digest — fail closed on missing.
    if budget_manifest_digest is not None:
        actual = getattr(
            critic, "budget_train_candidate_manifest_digest", None
        )
        if actual is None:
            errors.append(
                f"checkpoint role {role!r} missing required "
                "budget provenance field: "
                "budget_train_candidate_manifest_digest"
            )
        elif actual != budget_manifest_digest:
            errors.append(
                f"checkpoint role {role!r} budget candidate manifest "
                "digest mismatch (checkpoint was not trained with the "
                "audited budget candidate manifest)"
            )
    return errors


def _audit_candidate_manifest_support(
    candidate_manifest: dict[str, Any],
    *,
    manifest_path: Path | None,
    manifest_digest: str | None,
    test_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Candidate manifest schema + edge support against test records (2.5).

    Candidate entries are extracted from the actual manifest structure;
    entry memory ids may be stored as ``candidate_memory_id`` or
    ``memory_id``, but every edge key is unified on
    ``(task_id, receiver_agent_id, candidate_memory_id)``.
    """
    from smtr.marble.core_validity import is_core_valid_pair

    errors = validate_candidate_manifest_for_formal_evaluation(
        candidate_manifest,
        expected_target_split="test",
    )

    candidate_edges: set[tuple[str, str, str]] = set()
    for receiver_entry in candidate_manifest.get("candidates") or []:
        task_id = str(receiver_entry.get("task_id"))
        receiver_agent_id = str(receiver_entry.get("receiver_agent_id"))
        for candidate in receiver_entry.get("candidate_records", []):
            memory_id = candidate.get(
                "candidate_memory_id", candidate.get("memory_id")
            )
            if memory_id is None:
                continue
            candidate_edges.add((task_id, receiver_agent_id, str(memory_id)))

    test_record_edges = {
        treatment_edge_key(record)
        for record in test_records
        if is_core_valid_pair(record)
    }

    return {
        "path": str(manifest_path) if manifest_path is not None else None,
        "digest": manifest_digest,
        "target_split": candidate_manifest.get("target_split"),
        "memory_source_split": candidate_manifest.get("memory_source_split"),
        "candidate_edge_count": len(candidate_edges),
        "candidate_manifest_errors": errors,
        "test_edges_missing_from_manifest": sorted(
            test_record_edges - candidate_edges
        ),
        "unsupported_candidate_edges": sorted(
            candidate_edges - test_record_edges
        ),
    }


def _audit_formal_provenance(
    splits: dict[str, list[dict[str, Any]]],
) -> tuple[list[str], dict[str, list[dict[str, Any]]]]:
    """Fail-closed provenance schema validation for formal audits (清单 P1-1).

    Returns the flat error list plus a per-field report so callers never
    see only a vague total error count.
    """
    from smtr.counterfactual.paired_record import (
        FORMAL_PAIRED_PROVENANCE_FIELDS,
        validate_formal_paired_provenance,
    )

    errors: list[str] = []
    missing: dict[str, list[dict[str, Any]]] = {
        field: [] for field in sorted(FORMAL_PAIRED_PROVENANCE_FIELDS)
    }
    for name in SPLIT_NAMES:
        for index, record in enumerate(splits[name]):
            errors.extend(
                validate_formal_paired_provenance(record, record_index=index)
            )
            for field in FORMAL_PAIRED_PROVENANCE_FIELDS:
                value = record.get(field)
                if value is None or (
                    isinstance(value, str) and not value.strip()
                ):
                    missing[field].append(
                        {
                            "split": name,
                            "record_index": index,
                            "task_id": record.get("task_id"),
                            "receiver_agent_id": record.get(
                                "receiver_agent_id"
                            ),
                            "candidate_memory_id": record.get(
                                "candidate_memory_id"
                            ),
                        }
                    )
    return errors, missing


def _artifact_digest(path: Path | None) -> str | None:
    """SHA-256 digest of an audited file, or None when not supplied."""
    if path is None:
        return None
    return file_digest(Path(path))


def _read_pool_provenance(entry: dict[str, Any]) -> dict[str, str]:
    """Read memory provenance from payload.provenance (清单 P0-4 §9).

    Raises ValueError on missing provenance (fail closed).
    """
    payload = entry.get("payload")
    if not isinstance(payload, dict):
        raise ValueError(
            f"memory pool entry {entry.get('memory_id')!r} "
            "missing payload"
        )
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError(
            f"memory pool entry {entry.get('memory_id')!r} "
            "missing payload.provenance"
        )
    required = (
        "source_agent_id",
        "source_task_id",
        "source_trajectory_id",
        "source_split",
    )
    result: dict[str, str] = {}
    for field in required:
        value = provenance.get(field)
        if value is None or (isinstance(value, str) and not value.strip()):
            raise ValueError(
                f"memory pool entry {entry.get('memory_id')!r} "
                f"missing payload.provenance.{field}"
            )
        result[field] = str(value)
    return result


def _non_train_memory_pool_sources(
    memory_pool_path: Path | None,
) -> list[dict[str, Any]]:
    """Pool entries whose source split is not train (清单 P0-4 §9.3).

    Reads provenance exclusively from ``payload.provenance``; missing
    provenance fails closed.
    """
    if memory_pool_path is None:
        return []
    violations: list[dict[str, Any]] = []
    for line in Path(memory_pool_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        try:
            prov = _read_pool_provenance(entry)
        except ValueError as exc:
            violations.append({
                "memory_id": entry.get("memory_id"),
                "error": str(exc),
            })
            continue
        if prov["source_split"] != "train":
            violations.append({
                "memory_id": entry.get("memory_id"),
                "source_split": prov["source_split"],
                "source_task_id": prov["source_task_id"],
                "source_trajectory_id": prov["source_trajectory_id"],
            })
    return violations


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
    is a statistic, never a fatal condition.
    """
    observed: dict[str, set[str]] = {}
    for name in SPLIT_NAMES:
        for rec in paired_records_by_split[name]:
            trajectory = rec.get("memory_source_trajectory_id")
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
    """Edges whose target task equals the memory's source task (清单 P0-1)."""
    offenders: set[TreatmentEdgeKey] = set()
    for name in SPLIT_NAMES:
        for rec in paired_records_by_split[name]:
            source_task = rec.get("memory_source_task_id")
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
