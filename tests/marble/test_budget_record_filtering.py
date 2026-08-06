"""Edge-level record filtering (清单 Shared-Control 第15章).

The budget unit is the treatment edge: every generation seed of a
selected edge is retained and no seed row is dropped individually.
"""

from __future__ import annotations

from collections import defaultdict

from smtr.counterfactual.edge_keys import treatment_edge_key
from smtr.marble.budget_sampling import (
    build_budgeted_candidate_manifest,
    filter_paired_records_by_manifest,
)
from smtr.marble.real_data import (
    CandidateEntry,
    CandidateRecord,
    DatabaseCandidateManifest,
)

SEEDS = [0, 1, 2, 3, 4]


def _record(memory_id: str, rank: int) -> CandidateRecord:
    return CandidateRecord(
        memory_id=memory_id,
        writer_agent_id=f"w_{memory_id}",
        writer_role="planner",
        receiver_role="executor",
        match_type="matched_writer_receiver",
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
                    _record(f"m{i}", rank=i + 1) for i in range(8)
                ],
            )
        ],
    )


def _paired_records() -> list[dict]:
    return [
        {
            "task_id": "t1",
            "receiver_agent_id": "r1",
            "candidate_memory_id": f"m{i}",
            "generation_seed": seed,
            "valid": True,
            "schema_version": "marble_candidate_pair_v3",
            "control_group_id": f"ctrl_{seed:016x}",
        }
        for i in range(8)
        for seed in SEEDS
    ]


def test_selected_edges_keep_every_seed():
    parent = _parent()
    records = _paired_records()
    assert len(records) == 8 * len(SEEDS)

    manifest = build_budgeted_candidate_manifest(
        parent_manifest=parent, budget_fraction=0.5
    )
    filtered = filter_paired_records_by_manifest(
        paired_records=records, budget_manifest=manifest
    )

    selected_edges = {
        (entry.task_id, entry.receiver_agent_id, record.memory_id)
        for entry in manifest.candidates
        for record in entry.candidate_records
    }
    assert selected_edges  # budget never empties a stratum

    by_edge = defaultdict(set)
    for rec in filtered:
        key = treatment_edge_key(rec)
        assert key in selected_edges
        by_edge[key].add(rec["generation_seed"])

    # All-in or all-out: a selected edge keeps every seed.
    assert set(by_edge) == selected_edges
    for key, seeds in by_edge.items():
        assert seeds == set(SEEDS), key


def test_unselected_edges_are_fully_removed():
    parent = _parent()
    records = _paired_records()
    manifest = build_budgeted_candidate_manifest(
        parent_manifest=parent, budget_fraction=0.5
    )
    filtered = filter_paired_records_by_manifest(
        paired_records=records, budget_manifest=manifest
    )

    selected_edges = {
        (entry.task_id, entry.receiver_agent_id, record.memory_id)
        for entry in manifest.candidates
        for record in entry.candidate_records
    }
    removed = [
        rec
        for rec in records
        if treatment_edge_key(rec) not in selected_edges
    ]
    assert removed  # B=0.5 actually drops edges on 8 regular units
    assert all(rec not in filtered for rec in removed)
    assert len(filtered) == len(selected_edges) * len(SEEDS)


def test_full_budget_keeps_everything():
    parent = _parent()
    records = _paired_records()
    manifest = build_budgeted_candidate_manifest(
        parent_manifest=parent, budget_fraction=1.0
    )
    filtered = filter_paired_records_by_manifest(
        paired_records=records, budget_manifest=manifest
    )
    assert filtered == records
