"""Budget sampling restrictions (清单 Shared-Control 第12.1/12.2节).

Budget subsampling only applies to train candidate manifests and only
to the fixed fraction grid {0.25, 0.50, 0.75, 1.00}.
"""

from __future__ import annotations

import pytest

from smtr.marble.budget_sampling import build_budgeted_candidate_manifest
from smtr.marble.real_data import (
    CandidateEntry,
    CandidateRecord,
    DatabaseCandidateManifest,
)


def _manifest(target_split: str) -> DatabaseCandidateManifest:
    return DatabaseCandidateManifest(
        target_split=target_split,
        candidates=[
            CandidateEntry(
                task_id="t1",
                receiver_agent_id="r1",
                receiver_role="executor",
                candidate_records=[
                    CandidateRecord(
                        memory_id="m1",
                        receiver_role="executor",
                        memory_receiver_match_type="compatible",
                        rank=1,
                        score=0.9,
                    )
                ],
            )
        ],
    )


@pytest.mark.parametrize("split", ["validation", "test", ""])
def test_non_train_manifests_are_rejected(split: str):
    with pytest.raises(
        ValueError, match="restricted to train candidate manifests"
    ):
        build_budgeted_candidate_manifest(
            parent_manifest=_manifest(split), budget_fraction=0.5
        )


@pytest.mark.parametrize("fraction", [0.3, 0.1, 0.9, 0.99, 1.5, 0.0])
def test_off_grid_fractions_are_rejected(fraction: float):
    with pytest.raises(
        ValueError, match="analysis budget must be one of"
    ):
        build_budgeted_candidate_manifest(
            parent_manifest=_manifest("train"), budget_fraction=fraction
        )


def test_train_manifest_is_accepted():
    manifest = build_budgeted_candidate_manifest(
        parent_manifest=_manifest("train"), budget_fraction=0.25
    )
    assert manifest.budget_metadata is not None
