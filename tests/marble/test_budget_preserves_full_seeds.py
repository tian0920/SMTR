"""Test 6 (清单 Fixed-Budget 第16章): budget filtering preserves the full
seed support of every selected edge.
"""

from __future__ import annotations

from collections import defaultdict

from smtr.counterfactual.edge_keys import treatment_edge_key
from smtr.marble.training import prepare_effective_training_records
from tests.marble._budget_training_harness import (
    SEEDS,
    build_budget_manifest,
    full_paired_records,
    parent_manifest,
    write_budget_manifest,
    write_records,
)


def test_selected_edges_keep_all_five_seeds(tmp_path):
    parent = parent_manifest()
    manifest = build_budget_manifest(parent, budget_fraction=0.50)

    train_path = write_records(tmp_path, full_paired_records(), "train.jsonl")
    manifest_path = write_budget_manifest(tmp_path, manifest)

    prepared = prepare_effective_training_records(
        train_records_path=train_path,
        budget_candidate_manifest_path=manifest_path,
        experiment_mode="formal",
    )

    seeds_by_edge: dict[tuple[str, str, str], set[int]] = defaultdict(set)
    for rec in prepared.records:
        seeds_by_edge[treatment_edge_key(rec)].add(rec["generation_seed"])

    assert set(seeds_by_edge) == set(prepared.selected_edge_keys)
    for edge_key, seeds in seeds_by_edge.items():
        assert seeds == set(SEEDS), edge_key
    assert prepared.all_selected_edges_have_full_seed_support
    assert prepared.incomplete_seed_support_edge_count == 0
