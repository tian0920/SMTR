"""Test 7 (清单 Fixed-Budget 第16章): dropping one seed of a selected edge
is rejected in formal mode — budgeting may only remove whole edges.
"""

from __future__ import annotations

import pytest

from smtr.marble.training import prepare_effective_training_records
from tests.marble._budget_training_harness import (
    build_budget_manifest,
    full_paired_records,
    parent_manifest,
    selected_memory_ids,
    write_budget_manifest,
    write_records,
)


def test_partial_seed_support_fails_formal(tmp_path):
    parent = parent_manifest()
    manifest = build_budget_manifest(parent, budget_fraction=0.50)
    selected = sorted(selected_memory_ids(manifest))
    victim = selected[0]

    # Materialized budgeted records (mode B) with seed 4 deleted on one
    # selected edge: a whole-edge budget can never produce this file.
    budgeted = [
        rec
        for rec in full_paired_records(
            memory_ids=selected,
        )
        if not (rec["candidate_memory_id"] == victim and rec["generation_seed"] == 4)
    ]
    train_path = write_records(tmp_path, budgeted, "train_b050.jsonl")
    manifest_path = write_budget_manifest(tmp_path, manifest)

    with pytest.raises(ValueError, match="seed"):
        prepare_effective_training_records(
            train_records_path=train_path,
            budget_candidate_manifest_path=manifest_path,
            experiment_mode="formal",
            train_records_already_budgeted=True,
        )
