"""Fixed-budget stratified nested train subset sampling.

清单 Shared-Control 第12-15章: the intervention budget ``B`` is an
analysis axis, never a runtime parameter. For each fixed fraction in
``ANALYSIS_BUDGET_FRACTIONS`` a nested, deterministically ordered subset
of *train* candidate treatment edges is selected before any outcome is
observed. Anchor groups are atomic selection units; selection never
reads outcomes, critic predictions, or adaptive signals.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from typing import Any

from smtr.counterfactual.edge_keys import TreatmentEdgeKey, treatment_edge_key
from smtr.marble.real_data import (
    CandidateBudgetMetadata,
    CandidateEntry,
    CandidateRecord,
    DatabaseCandidateManifest,
)
from smtr.marble.real_pairs import stable_hash

ANALYSIS_BUDGET_FRACTIONS: tuple[float, ...] = (0.25, 0.50, 0.75, 1.00)

BUDGET_POLICY_VERSION = "fixed_stratified_nested_v1"

BUDGET_MANIFEST_SCHEMA_VERSION = "marble_candidates_budget_v1"

# Cross-receiver anchor groups form one indivisible selection unit
# spanning all receiver roles (清单 Shared-Control 第12.5/12.6节).
ANCHOR_STRATUM_KEY: tuple[str, str] = (
    "cross_receiver_anchor",
    "all_receiver_roles",
)

_SelectionUnitId = tuple[str, ...]


def manifest_canonical_digest(manifest: DatabaseCandidateManifest) -> str:
    """SHA-256 over the canonical JSON dump of a candidate manifest."""
    payload = json.dumps(
        manifest.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _edge_key(entry: CandidateEntry, record: CandidateRecord) -> TreatmentEdgeKey:
    return (str(entry.task_id), str(entry.receiver_agent_id), str(record.memory_id))


def _unit_priority(unit_id: _SelectionUnitId) -> int:
    """Stable ordering shared by every budget fraction (清单第12.7节)."""
    if unit_id[0] == "anchor_group":
        return stable_hash(BUDGET_POLICY_VERSION, "anchor_group", unit_id[1])
    return stable_hash(BUDGET_POLICY_VERSION, *unit_id)


def _selection_units(
    parent_manifest: DatabaseCandidateManifest,
) -> dict[
    tuple[str, str],
    dict[_SelectionUnitId, list[TreatmentEdgeKey]],
]:
    """Partition parent edges into atomic selection units per stratum.

    Regular candidates: one unit per treatment edge with stratum
    ``(candidate_source, receiver_role)``. Anchor candidates sharing an
    ``anchor_group_id`` form a single indivisible unit in the anchor
    stratum so the group is never split across budgets.
    """
    strata: dict[tuple[str, str], dict[_SelectionUnitId, list[TreatmentEdgeKey]]] = {}
    for entry in parent_manifest.candidates:
        for record in entry.candidate_records:
            edge = _edge_key(entry, record)
            if record.anchor_group_id:
                stratum = ANCHOR_STRATUM_KEY
                unit_id: _SelectionUnitId = ("anchor_group", record.anchor_group_id)
            else:
                stratum = (str(record.candidate_source), str(entry.receiver_role))
                unit_id = ("edge", *edge)
            strata.setdefault(stratum, {}).setdefault(unit_id, []).append(edge)
    return strata


def _selected_units(
    strata: dict[tuple[str, str], dict[_SelectionUnitId, list[TreatmentEdgeKey]]],
    budget_fraction: float,
) -> set[_SelectionUnitId]:
    """Nested stratified selection: top ``k_s(B)`` units per stratum."""
    selected: set[_SelectionUnitId] = set()
    for units in strata.values():
        ordered = sorted(units, key=_unit_priority)
        n_s = len(ordered)
        k_s = min(n_s, max(1, math.ceil(budget_fraction * n_s))) if n_s else 0
        selected.update(ordered[:k_s])
    return selected


def _manifest_edge_set(manifest: DatabaseCandidateManifest) -> set[TreatmentEdgeKey]:
    return {
        _edge_key(entry, record)
        for entry in manifest.candidates
        for record in entry.candidate_records
    }


def _cohort_counts(manifest: DatabaseCandidateManifest) -> dict[str, int]:
    counts = Counter(
        str(record.candidate_source)
        for entry in manifest.candidates
        for record in entry.candidate_records
    )
    return dict(sorted(counts.items()))


def build_budgeted_candidate_manifest(
    *,
    parent_manifest: DatabaseCandidateManifest,
    budget_fraction: float,
) -> DatabaseCandidateManifest:
    """Build a fixed stratified nested train subset manifest (清单第12章).

    Selection is deterministic and nested: the same stable ordering is
    reused for every fraction, so ``E_25 ⊆ E_50 ⊆ E_75 ⊆ E_100``.
    Anchor groups are kept atomic; the realized edge fraction may exceed
    the requested fraction and is reported in ``budget_metadata``.
    """
    if parent_manifest.target_split != "train":
        raise ValueError(
            "budget subsampling is restricted to train candidate manifests"
        )
    if budget_fraction not in ANALYSIS_BUDGET_FRACTIONS:
        raise ValueError(
            "analysis budget must be one of {0.25, 0.50, 0.75, 1.00}"
        )

    strata = _selection_units(parent_manifest)
    parent_edge_count = sum(
        len(edges) for units in strata.values() for edges in units.values()
    )
    parent_unit_count = sum(len(units) for units in strata.values())
    parent_digest = manifest_canonical_digest(parent_manifest)

    if budget_fraction == 1.0:
        # Identity budget: normalized copy of the parent, never resampled
        # (清单第12.10节).
        selected = {
            unit_id for units in strata.values() for unit_id in units
        }
    else:
        selected = _selected_units(strata, budget_fraction)

    selected_edges: set[TreatmentEdgeKey] = set()
    for units in strata.values():
        for unit_id, edges in units.items():
            if unit_id in selected:
                selected_edges.update(edges)

    entries: list[CandidateEntry] = []
    for entry in parent_manifest.candidates:
        kept = [
            record
            for record in entry.candidate_records
            if _edge_key(entry, record) in selected_edges
        ]
        if kept:
            entries.append(entry.model_copy(update={"candidate_records": kept}))

    selected_edge_count = len(selected_edges)
    metadata = CandidateBudgetMetadata(
        policy_version=BUDGET_POLICY_VERSION,
        requested_fraction=float(budget_fraction),
        realized_edge_fraction=(
            selected_edge_count / parent_edge_count if parent_edge_count else 0.0
        ),
        realized_unit_fraction=(
            len(selected) / parent_unit_count if parent_unit_count else 0.0
        ),
        parent_manifest_digest=parent_digest,
        parent_edge_count=parent_edge_count,
        selected_edge_count=selected_edge_count,
        parent_selection_unit_count=parent_unit_count,
        selected_selection_unit_count=len(selected),
        cohort_counts_before=_cohort_counts(parent_manifest),
        cohort_counts_after=_cohort_counts(
            parent_manifest.model_copy(update={"candidates": entries})
        ),
    )
    return parent_manifest.model_copy(
        update={
            "schema_version": BUDGET_MANIFEST_SCHEMA_VERSION,
            "candidates": entries,
            "budget_metadata": metadata,
        }
    )


def filter_paired_records_by_manifest(
    *,
    paired_records: list[dict[str, Any]],
    budget_manifest: DatabaseCandidateManifest,
) -> list[dict[str, Any]]:
    """Filter B=100 paired records down to a budget train subset (清单第15章).

    The budget unit is the treatment edge, so every generation seed of a
    selected edge is retained and no seed rows are dropped individually.
    Control artifacts are referenced, never copied.
    """
    selected_edges = _manifest_edge_set(budget_manifest)
    return [
        record
        for record in paired_records
        if treatment_edge_key(record) in selected_edges
    ]


def audit_budget_manifests(
    *,
    parent_manifest: DatabaseCandidateManifest,
    budget_manifests: dict[float, DatabaseCandidateManifest],
) -> list[str]:
    """Budget manifest audit violations (清单 Shared-Control 第17.2节)."""
    violations: list[str] = []
    parent_edges = _manifest_edge_set(parent_manifest)
    parent_digest = manifest_canonical_digest(parent_manifest)

    edge_sets: dict[float, set[TreatmentEdgeKey]] = {}
    for fraction in sorted(budget_manifests):
        manifest = budget_manifests[fraction]
        tag = f"budget={fraction:.2f}"
        meta = manifest.budget_metadata
        if manifest.target_split != "train":
            violations.append(f"{tag}: target_split is not train")
        if meta is None:
            violations.append(f"{tag}: missing budget_metadata")
            edge_sets[fraction] = _manifest_edge_set(manifest)
            continue
        if meta.parent_manifest_digest != parent_digest:
            violations.append(f"{tag}: parent manifest digest mismatch")
        if meta.outcome_fields_used:
            violations.append(f"{tag}: budget selection used outcome fields")
        if meta.critic_predictions_used:
            violations.append(f"{tag}: budget selection used critic predictions")
        if meta.adaptive_sampling_used:
            violations.append(f"{tag}: budget selection was adaptive")

        edges = _manifest_edge_set(manifest)
        edge_sets[fraction] = edges
        if not edges.issubset(parent_edges):
            violations.append(f"{tag}: selected edges are not a parent subset")

        # Anchor groups must stay atomic inside the subset.
        anchor_edges: dict[str, set[TreatmentEdgeKey]] = {}
        for entry in parent_manifest.candidates:
            for record in entry.candidate_records:
                if record.anchor_group_id:
                    anchor_edges.setdefault(
                        record.anchor_group_id, set()
                    ).add(_edge_key(entry, record))
        for group_id, group_edges in sorted(anchor_edges.items()):
            kept = group_edges & edges
            if kept and kept != group_edges:
                violations.append(
                    f"{tag}: anchor group {group_id} split across budgets"
                )

        if fraction == 1.0 and edges != parent_edges:
            violations.append(
                f"{tag}: full-budget edge set differs from parent manifest"
            )

    ordered = sorted(edge_sets)
    for lower, upper in zip(ordered, ordered[1:], strict=False):
        if not edge_sets[lower].issubset(edge_sets[upper]):
            violations.append(
                f"budget subsets are not nested: "
                f"{lower:.2f} not subset of {upper:.2f}"
            )
    return violations
