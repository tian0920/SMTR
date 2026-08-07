"""Test 9 (清单 Fixed-Budget 第16章): pre-budgeted train records (mode B)
contain an edge the budget manifest did not select. Formal mode must fail.
"""

from __future__ import annotations

import pytest

from smtr.marble.training import prepare_effective_training_records
from tests.marble._budget_training_harness import (
    build_budget_manifest,
    full_paired_records,
    parent_manifest,
    write_budget_manifest,
    write_records,
)


def test_unexpected_edge_in_budgeted_records_fails_formal(tmp_path):
    parent = parent_manifest()
    manifest = build_budget_manifest(parent, budget_fraction=0.50)

    # Claim the file is already budgeted, but it still carries all eight
    # edges while the manifest selected only four.
    train_path = write_records(tmp_path, full_paired_records(), "train.jsonl")
    manifest_path = write_budget_manifest(tmp_path, manifest)

    with pytest.raises(ValueError, match="outside the budget manifest"):
        prepare_effective_training_records(
            train_records_path=train_path,
            budget_candidate_manifest_path=manifest_path,
            experiment_mode="formal",
            train_records_already_budgeted=True,
        )
