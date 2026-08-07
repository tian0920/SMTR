"""Test 2 (清单 Fixed-Budget 第16章): features and labels are built from the
budget-filtered subset only. Unselected edges carry a recognizable label
(negative_transfer) that must never appear in the fitted label vector.
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


def test_features_and_labels_come_from_filtered_subset(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "smtr.marble.training.FourOutcomeTransferCritic", CapturingCritic
    )
    parent = parent_manifest()
    manifest = build_budget_manifest(parent, budget_fraction=0.50)
    selected = selected_memory_ids(manifest)
    all_ids = [f"m{i}" for i in range(8)]
    unselected = set(all_ids) - selected

    # Recognizable outcomes on the edges the budget must drop.
    outcomes = {mid: (0, 1) for mid in unselected}  # negative_transfer
    records = full_paired_records(outcomes=outcomes)
    train_path = write_records(tmp_path, records, "train.jsonl")
    manifest_path = write_budget_manifest(tmp_path, manifest)
    pool_path = write_memory_pool(tmp_path)

    train_critic(
        train_records_path=train_path,
        memory_pool_path=pool_path,
        output_path=tmp_path / "critic.json",
        n_bootstrap=3,
        n_features=64,
        coverage_mode="pilot",
        budget_candidate_manifest_path=manifest_path,
    )

    critic = CapturingCritic.last
    assert critic is not None and critic.fit_inputs is not None
    # No unselected edge's feature vector reaches X.
    fit_memory_ids = {
        input_edge_key(item)[2] for item in critic.fit_inputs
    }
    assert fit_memory_ids.isdisjoint(unselected)
    # No unselected edge's label reaches y.
    assert "negative_transfer" not in critic.fit_labels
    assert set(critic.fit_labels) == {"neutral_success"}
