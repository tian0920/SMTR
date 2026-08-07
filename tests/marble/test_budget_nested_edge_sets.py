"""Test 5 (清单 Fixed-Budget 第16章): budget edge sets are nested.

B25 ⊆ B50 ⊆ B75 ⊆ B100, so raising the budget only ever adds treatment
edges and never swaps them.
"""

from __future__ import annotations

from smtr.marble.budget_sampling import (
    selected_treatment_edges_from_manifest,
)
from smtr.marble.real_data import CandidateEntry, DatabaseCandidateManifest
from tests.marble._budget_training_harness import (
    build_budget_manifest,
    candidate_record,
)


def _wide_parent(n_edges: int = 16) -> DatabaseCandidateManifest:
    return DatabaseCandidateManifest(
        target_split="train",
        candidates=[
            CandidateEntry(
                task_id="t1",
                receiver_agent_id="r1",
                receiver_role="executor",
                candidate_records=[
                    candidate_record(f"m{i}", rank=i + 1)
                    for i in range(n_edges)
                ],
            )
        ],
    )


def test_budget_edge_sets_are_nested():
    parent = _wide_parent()
    edge_sets = {
        fraction: selected_treatment_edges_from_manifest(
            build_budget_manifest(parent, budget_fraction=fraction)
        )
        for fraction in (0.25, 0.50, 0.75, 1.00)
    }

    assert edge_sets[0.25] <= edge_sets[0.50]
    assert edge_sets[0.50] <= edge_sets[0.75]
    assert edge_sets[0.75] <= edge_sets[1.00]
    # Counts grow monotonically and B=1.0 covers the whole parent.
    assert len(edge_sets[0.25]) <= len(edge_sets[0.50])
    assert len(edge_sets[0.50]) <= len(edge_sets[0.75])
    assert len(edge_sets[1.00]) == 16
