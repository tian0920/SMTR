"""Identity budget B=1.0 (清单 Shared-Control 第12.10节).

The full budget is a normalized copy of the parent: nothing is
resampled, nothing is dropped, anchors included.
"""

from __future__ import annotations

from smtr.marble.budget_sampling import (
    BUDGET_MANIFEST_SCHEMA_VERSION,
    audit_budget_manifests,
    build_budgeted_candidate_manifest,
)
from smtr.marble.real_data import (
    CandidateEntry,
    CandidateRecord,
    DatabaseCandidateManifest,
)


def _record(
    memory_id: str, rank: int, *, anchor_group_id: str | None = None
) -> CandidateRecord:
    return CandidateRecord(
        memory_id=memory_id,
        writer_agent_id=f"w_{memory_id}",
        writer_role="planner",
        receiver_role="executor",
        match_type="matched_writer_receiver",
        candidate_source=(
            "cross_receiver_anchor" if anchor_group_id else "semantic_top"
        ),
        anchor_group_id=anchor_group_id,
        rank=rank,
        score=0.9,
    )


def _parent() -> DatabaseCandidateManifest:
    return DatabaseCandidateManifest(
        target_split="train",
        candidates=[
            CandidateEntry(
                task_id="t1",
                receiver_agent_id="r1",
                receiver_role="executor",
                candidate_records=[
                    _record("a_shared", rank=1, anchor_group_id="a1"),
                    _record("m2", rank=2),
                    _record("m3", rank=3),
                ],
            ),
            CandidateEntry(
                task_id="t1",
                receiver_agent_id="r2",
                receiver_role="verifier",
                candidate_records=[
                    _record("a_shared", rank=1, anchor_group_id="a1"),
                    _record("m4", rank=2),
                ],
            ),
        ],
    )


def _edge_set(manifest: DatabaseCandidateManifest) -> set[tuple[str, str, str]]:
    return {
        (entry.task_id, entry.receiver_agent_id, record.memory_id)
        for entry in manifest.candidates
        for record in entry.candidate_records
    }


def test_full_budget_is_identity():
    parent = _parent()
    manifest = build_budgeted_candidate_manifest(
        parent_manifest=parent, budget_fraction=1.0
    )
    assert _edge_set(manifest) == _edge_set(parent)
    assert manifest.schema_version == BUDGET_MANIFEST_SCHEMA_VERSION
    assert manifest.target_split == "train"


def test_full_budget_metadata_reports_identity():
    parent = _parent()
    manifest = build_budgeted_candidate_manifest(
        parent_manifest=parent, budget_fraction=1.0
    )
    meta = manifest.budget_metadata
    assert meta is not None
    assert meta.requested_fraction == 1.0
    assert meta.realized_edge_fraction == 1.0
    assert meta.realized_unit_fraction == 1.0
    assert meta.selected_edge_count == meta.parent_edge_count
    assert meta.selected_selection_unit_count == meta.parent_selection_unit_count


def test_anchor_edges_survive_the_identity_budget():
    parent = _parent()
    manifest = build_budgeted_candidate_manifest(
        parent_manifest=parent, budget_fraction=1.0
    )
    anchor_edges = {
        (entry.task_id, entry.receiver_agent_id, record.memory_id)
        for entry in parent.candidates
        for record in entry.candidate_records
        if record.anchor_group_id
    }
    assert anchor_edges == {
        ("t1", "r1", "a_shared"),
        ("t1", "r2", "a_shared"),
    }
    assert anchor_edges <= _edge_set(manifest)


def test_full_budget_audit_is_clean():
    parent = _parent()
    manifest = build_budgeted_candidate_manifest(
        parent_manifest=parent, budget_fraction=1.0
    )
    violations = audit_budget_manifests(
        parent_manifest=parent, budget_manifests={1.0: manifest}
    )
    assert violations == []
