"""Test 4 (清单 Fixed-Budget 第16章): B=1.0 is the identity condition.

The full-budget manifest keeps every parent edge and record, so the
effective subset is byte-for-byte the parent training set.
"""

from __future__ import annotations

from smtr.counterfactual.edge_keys import treatment_edge_key
from smtr.marble.training import prepare_effective_training_records
from tests.marble._budget_training_harness import (
    build_budget_manifest,
    full_paired_records,
    parent_manifest,
    write_budget_manifest,
    write_records,
)


def test_b100_is_identity(tmp_path):
    parent = parent_manifest()
    manifest = build_budget_manifest(parent, budget_fraction=1.00)
    records = full_paired_records()

    train_path = write_records(tmp_path, records, "train.jsonl")
    manifest_path = write_budget_manifest(tmp_path, manifest)

    prepared = prepare_effective_training_records(
        train_records_path=train_path,
        budget_candidate_manifest_path=manifest_path,
        experiment_mode="formal",
    )

    parent_edge_keys = {treatment_edge_key(rec) for rec in records}
    assert set(prepared.selected_edge_keys) == parent_edge_keys
    assert prepared.effective_edge_count == prepared.parent_edge_count == 8
    assert prepared.effective_record_count == prepared.parent_record_count
    assert prepared.records == records
    assert prepared.requested_budget_fraction == 1.0
    # Effective digest is stable and equals the parent digest at B=1.0.
    assert (
        prepared.effective_train_record_digest
        == prepared.parent_train_record_digest
    )
    again = prepare_effective_training_records(
        train_records_path=train_path,
        budget_candidate_manifest_path=manifest_path,
        experiment_mode="formal",
    )
    assert (
        again.effective_train_record_digest
        == prepared.effective_train_record_digest
    )
