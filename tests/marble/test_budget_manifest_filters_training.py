"""Test 1 (清单 Fixed-Budget 第16章): the budget manifest filters training
before ``critic.fit``. Full records carry eight treatment edges; a B=0.50
manifest selects four, and only those four reach the critic.
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


def test_critic_fit_only_sees_selected_edges(tmp_path, monkeypatch):
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
    fit_edges = {input_edge_key(item) for item in critic.fit_inputs}
    assert {key[2] for key in fit_edges} == selected
    assert len(fit_edges) == 4
    # Every seed of every selected edge reaches fit; nothing else does.
    assert len(critic.fit_inputs) == len(selected) * 5
    unselected = {f"m{i}" for i in range(8)} - selected
    assert all(key[2] not in unselected for key in fit_edges)
