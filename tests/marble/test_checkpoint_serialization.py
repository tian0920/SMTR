"""清单最终闭环 §36: checkpoint serialization round-trip.

Train a minimal critic through the real training path (with a frozen
budget manifest), save the checkpoint and reload it; every training
provenance field the split audit later depends on must survive the
round-trip unchanged.
"""

from __future__ import annotations

from smtr.marble.training import train_critic
from smtr.router.transfer_critic import FourOutcomeTransferCritic
from tests.marble._budget_training_harness import (
    build_budget_manifest,
    full_paired_records,
    parent_manifest,
    write_budget_manifest,
    write_memory_pool,
    write_records,
)


# Mixed outcomes across edges so the real critic sees at least two of the
# four transfer label classes.
_MIXED_OUTCOMES: dict[str, tuple[int, int]] = {
    "m0": (1, 1),
    "m1": (1, 1),
    "m2": (1, 0),
    "m3": (1, 0),
    "m4": (0, 1),
    "m5": (0, 1),
    "m6": (0, 0),
    "m7": (0, 0),
}


def _train_with_budget(tmp_path):
    parent = parent_manifest()
    manifest = build_budget_manifest(parent, budget_fraction=0.50)
    train_path = write_records(
        tmp_path, full_paired_records(outcomes=_MIXED_OUTCOMES), "train.jsonl"
    )
    manifest_path = write_budget_manifest(tmp_path, manifest)
    pool_path = write_memory_pool(tmp_path)
    checkpoint_path = tmp_path / "critic.joblib"

    metrics = train_critic(
        train_records_path=train_path,
        memory_pool_path=pool_path,
        output_path=checkpoint_path,
        n_bootstrap=3,
        n_features=64,
        coverage_mode="pilot",
        budget_candidate_manifest_path=manifest_path,
    )
    return metrics, checkpoint_path


def test_checkpoint_roundtrip_preserves_training_provenance(tmp_path):
    metrics, checkpoint_path = _train_with_budget(tmp_path)
    loaded = FourOutcomeTransferCritic.load(checkpoint_path)

    digests = metrics["artifact_digests"]
    support = metrics["training_support"]
    policy = metrics["budget_policy"]

    # Every field the formal split audit recomputes against must survive
    # save/load exactly.
    assert loaded.effective_train_record_digest == digests["effective_train_records"]
    assert loaded.effective_train_edge_count == support["effective_train_edge_count"]
    assert (
        loaded.budget_train_candidate_manifest_digest
        == digests["budget_candidate_manifest"]
    )
    assert loaded.training_budget_requested == policy["requested_fraction"]
    assert loaded.training_budget_realized == policy["realized_fraction"]
    assert loaded.feature_block == metrics["feature_block"] == "full"

    # None of these fields may silently degrade to None on load.
    assert loaded.effective_train_record_digest is not None
    assert loaded.effective_train_edge_count is not None
    assert loaded.budget_train_candidate_manifest_digest is not None
    assert loaded.training_budget_requested is not None
    assert loaded.training_budget_realized is not None


def test_checkpoint_roundtrip_without_budget_keeps_none_fields(tmp_path):
    train_path = write_records(
        tmp_path, full_paired_records(outcomes=_MIXED_OUTCOMES), "train.jsonl"
    )
    pool_path = write_memory_pool(tmp_path)
    checkpoint_path = tmp_path / "pilot_critic.joblib"

    train_critic(
        train_records_path=train_path,
        memory_pool_path=pool_path,
        output_path=checkpoint_path,
        n_bootstrap=3,
        n_features=64,
        coverage_mode="pilot",
    )
    loaded = FourOutcomeTransferCritic.load(checkpoint_path)

    # Pilot training without a manifest means full support: no budget
    # manifest digest is invented on load, while the implicit budget
    # fractions are 1.0 (all edges trained).
    assert loaded.budget_train_candidate_manifest_digest is None
    assert loaded.training_budget_requested == 1.0
    assert loaded.training_budget_realized == 1.0
    # Effective support metadata is still recorded for the full data.
    assert loaded.effective_train_record_digest is not None
    assert loaded.effective_train_edge_count is not None
