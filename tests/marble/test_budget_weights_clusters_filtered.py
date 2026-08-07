"""Test 3 (清单 Fixed-Budget 第16章): edge-equal sample weights and
control-family bootstrap clusters cover only the budget-filtered subset.
"""

from __future__ import annotations

from smtr.marble.training import train_critic
from tests.marble._budget_training_harness import (
    SEEDS,
    CapturingCritic,
    build_budget_manifest,
    full_paired_records,
    parent_manifest,
    selected_memory_ids,
    write_budget_manifest,
    write_memory_pool,
    write_records,
)


def test_weights_and_clusters_cover_only_selected_edges(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "smtr.marble.training.FourOutcomeTransferCritic", CapturingCritic
    )
    parent = parent_manifest()
    manifest = build_budget_manifest(parent, budget_fraction=0.50)
    selected = selected_memory_ids(manifest)

    train_path = write_records(tmp_path, full_paired_records(), "train.jsonl")
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
    assert critic is not None and critic.fit_kwargs is not None
    sample_weights = critic.fit_kwargs["sample_weights"]
    bootstrap_clusters = critic.fit_kwargs["bootstrap_clusters"]

    # Weights cover exactly the effective records: one per kept seed row,
    # edge-equal 1/5 so each selected edge contributes total weight 1.
    assert len(sample_weights) == len(selected) * len(SEEDS)
    assert set(sample_weights.tolist()) == {1.0 / len(SEEDS)}

    # Cluster row indices never reference a dropped edge's records.
    assert bootstrap_clusters is not None
    total_cluster_rows = sum(len(rows) for rows in bootstrap_clusters.values())
    assert total_cluster_rows == len(selected) * len(SEEDS)
