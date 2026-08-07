"""Test 8 (清单 Fixed-Budget 第16章): the budget manifest selects an edge
that the full train records do not contain. Formal mode must fail.
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


def test_manifest_edge_without_paired_records_fails_formal(tmp_path):
    # Parent manifest has nine edges; B=1.0 selects all of them, but the
    # train records file only covers the first eight.
    parent = parent_manifest(memory_ids=[f"m{i}" for i in range(9)])
    manifest = build_budget_manifest(parent, budget_fraction=1.0)

    train_path = write_records(
        tmp_path,
        full_paired_records(memory_ids=[f"m{i}" for i in range(8)]),
        "train.jsonl",
    )
    manifest_path = write_budget_manifest(tmp_path, manifest)

    with pytest.raises(ValueError, match="without paired training records"):
        prepare_effective_training_records(
            train_records_path=train_path,
            budget_candidate_manifest_path=manifest_path,
            experiment_mode="formal",
        )
