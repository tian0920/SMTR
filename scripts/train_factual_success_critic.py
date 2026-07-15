#!/usr/bin/env python3
"""Train the FactualSuccess-SMTR binary critic."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from smtr.router.factual_success_critic import (  # noqa: E402
    FactualSuccessCheckpointMetadata,
    FactualSuccessCritic,
    choose_threshold_for_exposure,
)
from smtr.router.transfer_evaluation import group_split  # noqa: E402
from smtr.router.transfer_features import (  # noqa: E402
    HashingTransferFeatureEncoder,
    load_paired_records_for_training,
    prediction_input_from_record,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _target_mean_exposure(path: Path) -> float:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "mean_exposure_per_invocation" in payload:
        return float(payload["mean_exposure_per_invocation"])
    distribution = payload.get("count_distribution")
    if distribution:
        return sum(int(count) * float(prob) for count, prob in distribution.items())
    raise ValueError("SMTR validation exposure manifest lacks exposure target")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--smtr-validation-exposure-manifest", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--n-features", type=int, default=512)
    args = parser.parse_args()

    records_path = Path(args.records)
    output_path = Path(args.output)
    split_manifest_path = Path(args.split_manifest)
    exposure_manifest_path = Path(args.smtr_validation_exposure_manifest)

    records = load_paired_records_for_training(records_path)
    train, validation = group_split(
        records,
        seed=args.seed,
        test_fraction=args.test_fraction,
    )
    encoder = HashingTransferFeatureEncoder(
        n_features=args.n_features,
        feature_block="full",
    )
    critic = FactualSuccessCritic(encoder=encoder).fit(train, seed=args.seed)
    validation_probabilities = [
        critic.predict_factual_success(prediction_input_from_record(record))
        for record in validation
    ]
    validation_invocations = {
        (record.episode_id, record.graph_node, record.receiver_agent_id, record.task_stage)
        for record in validation
    }
    threshold = choose_threshold_for_exposure(
        probabilities=validation_probabilities,
        target_mean_exposure=_target_mean_exposure(exposure_manifest_path),
        invocation_count=len(validation_invocations),
    )

    split_manifest = {
        "manifest_version": "1.0",
        "splitter": "smtr.router.transfer_evaluation.group_split",
        "records_path": str(records_path),
        "records_sha256": sha256(records_path),
        "seed": args.seed,
        "test_fraction": args.test_fraction,
        "group_key": "episode_id",
        "train_episode_ids": sorted({record.episode_id for record in train}),
        "validation_episode_ids": sorted({record.episode_id for record in validation}),
        "target": "share_success",
        "label_field": "y_share",
        "threshold_source_split": "validation",
        "threshold_selection_objective": "match_smtr_validation_mean_exposure",
        "smtr_validation_exposure_manifest": str(exposure_manifest_path),
        "smtr_validation_mean_exposure": _target_mean_exposure(exposure_manifest_path),
        "threshold": threshold,
        "validation_invocation_count": len(validation_invocations),
    }
    split_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    split_manifest_path.write_text(
        json.dumps(split_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    metadata = FactualSuccessCheckpointMetadata(
        training_records_sha256=sha256(records_path),
        split_manifest_sha256=sha256(split_manifest_path),
        training_seed=args.seed,
        hashing_dimension=encoder.n_features,
        threshold=threshold,
    )
    critic.save(output_path, metadata=metadata)
    FactualSuccessCritic.load(output_path, require_metadata=True)
    print(f"trained={output_path}")
    print(f"threshold={threshold:.6f}")


if __name__ == "__main__":
    main()
