"""Test 12 (清单 Fixed-Budget 第16章): anchor-group atomicity seen through
the training-chain selection API. Every anchor_group_id is kept whole or
dropped whole at every budget fraction.
"""

from __future__ import annotations

from smtr.marble.budget_sampling import (
    ANALYSIS_BUDGET_FRACTIONS,
    build_budgeted_candidate_manifest,
    selected_treatment_edges_from_manifest,
)
from smtr.marble.real_data import (
    CandidateEntry,
    CandidateRecord,
    DatabaseCandidateManifest,
)
from tests.marble._budget_training_harness import candidate_record


def _anchor_record(memory_id: str, rank: int, group: str) -> CandidateRecord:
    return CandidateRecord(
        memory_id=memory_id,
        writer_agent_id=f"w_{memory_id}",
        writer_role="planner",
        receiver_role="executor",
        match_type="matched_writer_receiver",
        candidate_source="cross_receiver_anchor",
        anchor_group_id=group,
        rank=rank,
        score=0.9,
    )


def _parent() -> DatabaseCandidateManifest:
    # One anchor group spanning two receivers plus regular single-receiver
    # edges, so atomicity can be observed across receivers.
    return DatabaseCandidateManifest(
        target_split="train",
        candidates=[
            CandidateEntry(
                task_id="t1",
                receiver_agent_id="r1",
                receiver_role="executor",
                candidate_records=[
                    _anchor_record("a1_mem", rank=1, group="a1"),
                    candidate_record("m1", rank=2),
                    candidate_record("m2", rank=3),
                    candidate_record("m3", rank=4),
                ],
            ),
            CandidateEntry(
                task_id="t1",
                receiver_agent_id="r2",
                receiver_role="verifier",
                candidate_records=[
                    _anchor_record("a1_mem", rank=1, group="a1"),
                    candidate_record("m4", rank=2),
                    candidate_record("m5", rank=3),
                    candidate_record("m6", rank=4),
                ],
            ),
        ],
    )


def test_anchor_group_is_atomic_in_training_selection():
    parent = _parent()
    anchor_edges = {("t1", "r1", "a1_mem"), ("t1", "r2", "a1_mem")}

    for fraction in ANALYSIS_BUDGET_FRACTIONS:
        manifest = build_budgeted_candidate_manifest(
            parent_manifest=parent, budget_fraction=fraction
        )
        selected = selected_treatment_edges_from_manifest(manifest)
        kept_anchor = selected & anchor_edges
        assert kept_anchor == set() or kept_anchor == anchor_edges, (
            fraction,
            kept_anchor,
        )
