"""Independent training-support verification for formal audits (清单 P0-1).

The split audit must never trust checkpoint self-reported metadata. This
module provides the deterministic functions that both training and audit
call to compute effective training subsets and digests. The audit
independently reconstructs the budget-filtered records from the full train
paired records and the frozen budget candidate manifest, then verifies
that every formal checkpoint's declared support matches.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from smtr.counterfactual.edge_keys import (
    TreatmentEdgeKey,
    candidate_record_edge_key,
    treatment_edge_key,
)
from smtr.counterfactual.paired_record import (
    canonical_paired_record_digest,
    stable_json_digest,
)

# 清单 §4.3: formal checkpoint must carry all these training-support
# fields; any missing field fails the audit closed.
FORMAL_TRAINING_SUPPORT_FIELDS = frozenset({
    "budget_train_candidate_manifest_digest",
    "effective_train_record_digest",
    "effective_train_edge_count",
    "training_budget_requested",
    "training_budget_realized",
})


def selected_edge_keys_from_candidate_manifest(
    manifest: dict[str, Any],
) -> set[TreatmentEdgeKey]:
    """Extract selected treatment-edge keys from a budget candidate manifest.

    Uses the same deterministic key construction as the training pipeline
    so manifest edges and paired-record edges are always comparable.
    """
    selected: set[TreatmentEdgeKey] = set()
    for entry in manifest.get("candidates") or []:
        task_id = str(entry.get("task_id", ""))
        receiver_agent_id = str(entry.get("receiver_agent_id", ""))
        for record in entry.get("candidate_records", []):
            memory_id = record.get(
                "candidate_memory_id", record.get("memory_id")
            )
            if memory_id is not None:
                key = candidate_record_edge_key(
                    task_id=task_id,
                    receiver_agent_id=receiver_agent_id,
                    candidate_memory_id=str(memory_id),
                )
                if key in selected:
                    raise ValueError(
                        f"duplicate treatment edge in manifest: {key}"
                    )
                selected.add(key)
    return selected


def filter_records_to_selected_edges(
    records: list[dict[str, Any]],
    selected_edge_keys: set[TreatmentEdgeKey],
) -> list[dict[str, Any]]:
    """Keep seed records of selected edges, drop the rest.

    Budget filtering removes whole treatment edges; individual seeds of
    a selected edge are never dropped (清单 §2.2).
    """
    return [
        record
        for record in records
        if treatment_edge_key(record) in selected_edge_keys
    ]


def edge_seed_sets(
    records: list[dict[str, Any]],
) -> dict[TreatmentEdgeKey, set[int]]:
    """Map each treatment edge to the generation seeds it covers."""
    result: dict[TreatmentEdgeKey, set[int]] = defaultdict(set)
    for record in records:
        result[treatment_edge_key(record)].add(
            int(record["generation_seed"])
        )
    return dict(result)


def canonical_effective_record_digest(
    records: list[dict[str, Any]],
) -> str:
    """Order-insensitive digest of an effective training record set.

    This is the audit's independent recomputation of the effective
    training digest. Must match the digest persisted in every formal
    checkpoint; otherwise the checkpoint was trained on a different
    support than the audited budget manifest.
    """
    return canonical_paired_record_digest(records)


def checkpoint_support_signature(
    critic: Any,
) -> tuple[Any, ...]:
    """Training-support signature declared by one checkpoint.

    All three formal checkpoints (full, global_transfer,
    no_compatibility_interaction) must report identical signatures for
    the same budget experiment.
    """
    return (
        getattr(critic, "budget_train_candidate_manifest_digest", None),
        getattr(critic, "effective_train_record_digest", None),
        getattr(critic, "effective_train_edge_count", None),
        getattr(critic, "training_budget_requested", None),
        getattr(critic, "training_budget_realized", None),
    )


def validate_checkpoint_support_against_audit(
    *,
    critic: Any,
    role: str,
    audit_effective_digest: str,
    audit_effective_edge_count: int,
    audit_budget_manifest_digest: str | None,
) -> list[str]:
    """Fail-closed comparison of checkpoint metadata vs audit truth.

    Missing fields fail closed (清单 P0-2 §4.2): ``actual is None`` is
    not silently bypassed.
    """
    errors: list[str] = []

    checks: list[tuple[str, Any, Any]] = [
        (
            "effective_train_record_digest",
            getattr(critic, "effective_train_record_digest", None),
            audit_effective_digest,
        ),
        (
            "effective_train_edge_count",
            getattr(critic, "effective_train_edge_count", None),
            audit_effective_edge_count,
        ),
        (
            "budget_train_candidate_manifest_digest",
            getattr(
                critic, "budget_train_candidate_manifest_digest", None
            ),
            audit_budget_manifest_digest,
        ),
    ]

    for field_name, actual, expected in checks:
        if actual is None:
            errors.append(
                f"checkpoint role {role!r} missing required "
                f"budget provenance field: {field_name}"
            )
        elif expected is not None and actual != expected:
            errors.append(
                f"checkpoint role {role!r} budget provenance "
                f"mismatch: {field_name}"
            )

    return errors
