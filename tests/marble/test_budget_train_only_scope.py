"""Test 14 (清单 Fixed-Budget 第16章): budgeting scopes train treatment
edges only. The validation split keeps its full record and edge support
while train support shrinks to the budget selection.
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


def test_validation_support_stays_full_under_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "smtr.marble.training.FourOutcomeTransferCritic", CapturingCritic
    )
    parent = parent_manifest()
    manifest = build_budget_manifest(parent, budget_fraction=0.50)
    selected = selected_memory_ids(manifest)
    assert len(selected) == 4

    train_path = write_records(tmp_path, full_paired_records(), "train.jsonl")
    # Validation carries all eight edges, unfiltered.
    validation_path = write_records(
        tmp_path, full_paired_records(), "validation.jsonl"
    )
    manifest_path = write_budget_manifest(tmp_path, manifest)
    pool_path = write_memory_pool(tmp_path)

    metrics = train_critic(
        train_records_path=train_path,
        memory_pool_path=pool_path,
        validation_records_path=validation_path,
        output_path=tmp_path / "critic.json",
        n_bootstrap=3,
        n_features=64,
        coverage_mode="pilot",
        budget_candidate_manifest_path=manifest_path,
    )

    # Train is budget-filtered...
    critic = CapturingCritic.last
    assert critic is not None and critic.fit_inputs is not None
    fit_edges = {input_edge_key(item) for item in critic.fit_inputs}
    assert len(fit_edges) == 4
    assert metrics["training_support"]["effective_train_edge_count"] == 4

    # ...while validation keeps full support.
    assert metrics["validation_records"] == 8 * 5
    assert metrics["validation_edges"] == 8

    policy = metrics["budget_policy"]
    assert policy["budget_scope"] == "train_treatment_edges_only"
    assert policy["validation_support"] == "full"
    assert policy["test_support"] == "full"
