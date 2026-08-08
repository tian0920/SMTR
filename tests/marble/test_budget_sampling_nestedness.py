"""Budget subsets are nested (清单 Shared-Control 第12.7节).

The same stable ordering is reused for every fraction, so
E_25 ⊆ E_50 ⊆ E_75 ⊆ E_100 edge set always holds.
"""

from __future__ import annotations

from smtr.marble.budget_sampling import (
    ANALYSIS_BUDGET_FRACTIONS,
    build_budgeted_candidate_manifest,
)
from smtr.marble.real_data import (
    CandidateEntry,
    CandidateRecord,
    DatabaseCandidateManifest,
)


def _record(memory_id: str, rank: int) -> CandidateRecord:
    return CandidateRecord(
        memory_id=memory_id,
        writer_agent_id=f"w_{memory_id}",
        writer_role="planner",
        receiver_role="executor",
        match_type="compatible",
        rank=rank,
        score=0.9,
    )


def _parent() -> DatabaseCandidateManifest:
    return DatabaseCandidateManifest(
        target_split="train",
        candidates=[
            CandidateEntry(
                task_id="t1",
                receiver_agent_id=f"r{receiver_idx}",
                receiver_role="executor",
                candidate_records=[
                    _record(f"m{receiver_idx}_{i}", rank=i + 1)
                    for i in range(8)
                ],
            )
            for receiver_idx in range(4)
        ],
    )


def test_budget_subsets_are_nested():
    parent = _parent()
    edge_sets = {}
    for fraction in ANALYSIS_BUDGET_FRACTIONS:
        manifest = build_budgeted_candidate_manifest(
            parent_manifest=parent, budget_fraction=fraction
        )
        edge_sets[fraction] = {
            (entry.task_id, entry.receiver_agent_id, record.memory_id)
            for entry in manifest.candidates
            for record in entry.candidate_records
        }

    fractions = sorted(edge_sets)
    for lower, upper in zip(fractions, fractions[1:], strict=False):
        assert edge_sets[lower] <= edge_sets[upper], (lower, upper)
    # Strict growth until the identity budget.
    for lower, upper in zip(fractions, fractions[1:], strict=False):
        assert edge_sets[lower] < edge_sets[upper], (lower, upper)


def test_selection_is_deterministic():
    parent = _parent()
    first = build_budgeted_candidate_manifest(
        parent_manifest=parent, budget_fraction=0.5
    )
    second = build_budgeted_candidate_manifest(
        parent_manifest=parent, budget_fraction=0.5
    )
    assert first.model_dump(mode="json") == second.model_dump(mode="json")
