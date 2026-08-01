#!/usr/bin/env python3
"""Train the full-feature M0 critic for gate ablation."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from smtr.router.transfer_critic import (  # noqa: E402
    CriticCheckpointMetadata,
    FourOutcomeTransferCritic,
)
from smtr.router.transfer_evaluation import group_split  # noqa: E402
from smtr.router.transfer_features import (  # noqa: E402
    HashingTransferFeatureEncoder,
    load_paired_records_for_training,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    records_path = Path("data/paired_records_pi3_v22.jsonl")
    output_path = Path("checkpoints/critic_full_gate_ablation_v1.joblib")
    split_manifest_path = Path("data/manifests/gate_ablation_split_v1.json")
    training_manifest_path = Path("data/manifests/gate_ablation_training_v1.json")
    seed = 0
    n_bootstrap = 31
    test_fraction = 0.2
    n_features = 512

    if not records_path.exists():
        raise SystemExit(f"missing training records: {records_path}")

    records = load_paired_records_for_training(records_path)
    fingerprints = {record.continuation_policy_fingerprint for record in records}
    if len(fingerprints) != 1:
        raise SystemExit("training records contain mixed continuation policy fingerprints")
    continuation_policy_fingerprint = next(iter(fingerprints))

    train, test = group_split(records, seed=seed, test_fraction=test_fraction)
    train_episode_ids = sorted({record.episode_id for record in train})
    test_episode_ids = sorted({record.episode_id for record in test})

    split_manifest = {
        "manifest_version": "2.0",
        "splitter": "smtr.router.transfer_evaluation.group_split",
        "records_path": str(records_path),
        "records_sha256": sha256(records_path),
        "seed": seed,
        "test_fraction": test_fraction,
        "group_key": "episode_id",
        "train_episode_ids": train_episode_ids,
        "test_episode_ids": test_episode_ids,
        "train_record_count": len(train),
        "test_record_count": len(test),
    }
    split_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    split_manifest_path.write_text(
        json.dumps(split_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    encoder = HashingTransferFeatureEncoder(
        n_features=n_features,
        feature_block="full",
    )
    critic = FourOutcomeTransferCritic(encoder=encoder).fit(
        train,
        seed=seed,
        n_bootstrap=n_bootstrap,
    )
    metadata = CriticCheckpointMetadata(
        feature_block="full",
        uses_selected_set=True,
        uses_pairwise_interactions=True,
        training_records_sha256=sha256(records_path),
        split_manifest_sha256=sha256(split_manifest_path),
        continuation_policy_fingerprint=continuation_policy_fingerprint,
        ensemble_size=len(critic.models),
        hashing_dimension=encoder.n_features,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    critic.save(output_path, metadata=metadata)

    training_manifest = {
        "manifest_version": "2.0",
        "records_path": str(records_path),
        "records_sha256": sha256(records_path),
        "split_manifest_path": str(split_manifest_path),
        "split_manifest_sha256": sha256(split_manifest_path),
        "continuation_policy_fingerprint": continuation_policy_fingerprint,
        "feature_schema_version": encoder.schema_version,
        "feature_block": "full",
        "uses_selected_set": True,
        "uses_pairwise_interactions": True,
        "bootstrap_group_key": "episode_id",
        "training_seed": seed,
        "ensemble_size": len(critic.models),
        "hashing_dimension": encoder.n_features,
        "checkpoint_path": str(output_path),
        "checkpoint_sha256": sha256(output_path),
        "checkpoint_metadata_path": str(output_path.with_suffix(".metadata.json")),
        "checkpoint_metadata_sha256": sha256(output_path.with_suffix(".metadata.json")),
    }
    training_manifest_path.write_text(
        json.dumps(training_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    loaded = FourOutcomeTransferCritic.load(output_path, require_metadata=True)
    if loaded.encoder.feature_block != "full":
        raise SystemExit("checkpoint self-check failed: feature_block mismatch")
    print(f"trained={output_path}")
    print(f"metadata={output_path.with_suffix('.metadata.json')}")
    print(f"training_manifest={training_manifest_path}")


if __name__ == "__main__":
    main()
