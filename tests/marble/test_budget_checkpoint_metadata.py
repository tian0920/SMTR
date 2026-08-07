"""Test 11 (清单 Fixed-Budget 第16章): checkpoint metadata reports the
effective train edge count and it equals what critic.fit actually saw.
"""

from __future__ import annotations

from smtr.marble.training import train_critic
from tests.marble._budget_training_harness import (
    CapturingCritic,
    build_budget_manifest,
    full_paired_records,
    input_edge_key,
    parent_manifest,
    selected_memory_ids,
    write_budget_manifest,
    write_memory_pool,
    write_records,
)


def test_training_support_metadata_matches_fit(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "smtr.marble.training.FourOutcomeTransferCritic", CapturingCritic
    )
    parent = parent_manifest()
    manifest = build_budget_manifest(parent, budget_fraction=0.50)
    selected = selected_memory_ids(manifest)
    assert len(selected) == 4

    train_path = write_records(tmp_path, full_paired_records(), "train.jsonl")
    manifest_path = write_budget_manifest(tmp_path, manifest)
    pool_path = write_memory_pool(tmp_path)

    metrics = train_critic(
        train_records_path=train_path,
        memory_pool_path=pool_path,
        output_path=tmp_path / "critic.json",
        n_bootstrap=3,
        n_features=64,
        coverage_mode="pilot",
        budget_candidate_manifest_path=manifest_path,
    )

    support = metrics["training_support"]
    critic = CapturingCritic.last
    assert critic is not None and critic.fit_inputs is not None
    fit_edges = {input_edge_key(item) for item in critic.fit_inputs}

    assert support["effective_train_edge_count"] == len(fit_edges) == 4
    assert support["parent_train_edge_count"] == 8
    assert support["effective_train_record_count"] == 4 * 5
    assert support["parent_train_record_count"] == 8 * 5
    assert support["selected_edge_count_from_manifest"] == 4
    assert support["all_selected_edges_found"]
    assert support["unexpected_training_edge_count"] == 0
    assert support["all_selected_edges_have_full_seed_support"]
