"""Anchor group atomicity (清单 Shared-Control 第12.5/12.6节).

A cross-receiver anchor group is one indivisible selection unit: under
every budget its edges are either all present or all absent.
"""

from __future__ import annotations

from smtr.marble.budget_sampling import (
    ANALYSIS_BUDGET_FRACTIONS,
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
        match_type="compatible",
        candidate_source=(
            "cross_receiver_anchor" if anchor_group_id else "semantic_top"
        ),
        anchor_group_id=anchor_group_id,
        rank=rank,
        score=0.9,
    )


def _parent() -> DatabaseCandidateManifest:
    # Two anchor groups spanning two receivers each, plus regular edges.
    return DatabaseCandidateManifest(
        target_split="train",
        candidates=[
            CandidateEntry(
                task_id="t1",
                receiver_agent_id="r1",
                receiver_role="executor",
                candidate_records=[
                    _record("a1_mem", rank=1, anchor_group_id="a1"),
                    _record("a2_mem", rank=2, anchor_group_id="a2"),
                    _record("m1", rank=3),
                    _record("m2", rank=4),
                ],
            ),
            CandidateEntry(
                task_id="t1",
                receiver_agent_id="r2",
                receiver_role="verifier",
                candidate_records=[
                    _record("a1_mem", rank=1, anchor_group_id="a1"),
                    _record("a2_mem", rank=2, anchor_group_id="a2"),
                    _record("m3", rank=3),
                    _record("m4", rank=4),
                ],
            ),
        ],
    )


def _anchor_edges(
    manifest: DatabaseCandidateManifest, group_id: str
) -> set[tuple[str, str, str]]:
    return {
        (entry.task_id, entry.receiver_agent_id, record.memory_id)
        for entry in manifest.candidates
        for record in entry.candidate_records
        if record.anchor_group_id == group_id
    }


def test_anchor_groups_are_all_in_or_all_out_at_every_budget():
    parent = _parent()
    for group_id in ("a1", "a2"):
        full = _anchor_edges(parent, group_id)
        assert len(full) == 2  # spans both receivers
        for fraction in ANALYSIS_BUDGET_FRACTIONS:
            manifest = build_budgeted_candidate_manifest(
                parent_manifest=parent, budget_fraction=fraction
            )
            kept = _anchor_edges(manifest, group_id)
            assert kept == set() or kept == full, (group_id, fraction, kept)


def test_audit_never_reports_split_anchor_groups():
    parent = _parent()
    manifests = {
        fraction: build_budgeted_candidate_manifest(
            parent_manifest=parent, budget_fraction=fraction
        )
        for fraction in ANALYSIS_BUDGET_FRACTIONS
    }
    violations = audit_budget_manifests(
        parent_manifest=parent, budget_manifests=manifests
    )
    assert violations == []
    assert not any("split across budgets" in v for v in violations)


def test_audit_detects_a_manually_split_anchor():
    parent = _parent()
    manifest = build_budgeted_candidate_manifest(
        parent_manifest=parent, budget_fraction=1.0
    )
    assert manifest.budget_metadata is not None
    # Drop one receiver's anchor edge to fake a split.
    broken_entries = []
    for entry in manifest.candidates:
        if entry.receiver_agent_id == "r2":
            kept = [
                record
                for record in entry.candidate_records
                if record.anchor_group_id != "a1"
            ]
            broken_entries.append(
                entry.model_copy(update={"candidate_records": kept})
            )
        else:
            broken_entries.append(entry)
    broken = manifest.model_copy(update={"candidates": broken_entries})
    violations = audit_budget_manifests(
        parent_manifest=parent, budget_manifests={1.0: broken}
    )
    assert any("split across budgets" in v for v in violations)
