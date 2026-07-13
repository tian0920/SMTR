#!/usr/bin/env python3
"""Train A1-NoSet critic: no-selected-set ablation.

Uses the same training records, split, and hyperparameters as M0 (critic_pi3_v22),
but with feature_block="context_plus_candidate" which excludes all selected-set
and pairwise interaction features.

Usage:
    python scripts/train_a1_critic.py
"""

import json
import sys
from pathlib import Path

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from smtr.counterfactual.schemas import transfer_class_from_outcomes
from smtr.router.transfer_evaluation import group_split
from smtr.router.transfer_critic import FourOutcomeTransferCritic, CLASS_ORDER, LABEL_TO_CLASS
from smtr.router.transfer_features import (
    HashingTransferFeatureEncoder,
    load_paired_records_for_training,
    prediction_input_from_record,
)


def file_sha256(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def distribution(values):
    from collections import Counter
    counts = Counter(values)
    total = sum(counts.values())
    return {k: v / total for k, v in sorted(counts.items())}


def main():
    input_path = Path("data/paired_records_pi3_v22.jsonl")
    output_path = Path("checkpoints/critic_no_selected_set_v1.joblib")

    # Same hyperparameters as M0 training
    seed = 7
    n_bootstrap = 31
    test_fraction = 0.2
    n_features = 512
    feature_block = "context_plus_candidate"

    print(f"Loading records from {input_path}...")
    records = load_paired_records_for_training(input_path)
    print(f"  {len(records)} records loaded")

    # Verify same split as M0
    train, test = group_split(records, seed=seed, test_fraction=test_fraction)
    print(f"  train={len(train)}, test={len(test)}")

    # Create encoder with context_plus_candidate block
    encoder = HashingTransferFeatureEncoder(
        n_features=n_features,
        feature_block=feature_block,
    )

    # Verify no selected-set or interaction tokens
    sample_items = [prediction_input_from_record(r) for r in train[:5]]
    for item in sample_items:
        tokens = encoder.tokens(item)
        selected_tokens = [t for t in tokens if t.startswith("selected_") and not t.startswith("selected_count:")]
        interaction_tokens = [t for t in tokens if t.startswith("interaction_")]
        assert not selected_tokens, f"A1 must not have selected-set tokens: {selected_tokens}"
        assert not interaction_tokens, f"A1 must not have interaction tokens: {interaction_tokens}"
    print("  Feature audit passed: no selected-set or interaction tokens")

    # Encode training data
    train_items = [prediction_input_from_record(r) for r in train]
    x = encoder.transform(train_items)
    y = np.array(
        [CLASS_ORDER.index(LABEL_TO_CLASS[r.transfer_class]) for r in train]
    )

    # Create and fit critic using the SAME procedure as FourOutcomeTransferCritic.fit()
    from sklearn.linear_model import LogisticRegression
    from smtr.router.transfer_critic import SmoothedClassPriorModel

    critic = FourOutcomeTransferCritic(encoder=encoder)
    rng = np.random.default_rng(seed)
    critic.models = []
    critic.bootstrap_seeds = []

    for _ in range(n_bootstrap):
        bootstrap_seed = int(rng.integers(0, 2**31 - 1))
        critic.bootstrap_seeds.append(bootstrap_seed)
        sample_rng = np.random.default_rng(bootstrap_seed)
        indices = sample_rng.integers(0, len(train), size=len(train))
        sample_y = y[indices]
        if len(set(sample_y.tolist())) < 2:
            critic.models.append(SmoothedClassPriorModel(sample_y.tolist()))
            continue
        model = LogisticRegression(max_iter=2000, solver="lbfgs")
        model.fit(x[indices], sample_y)
        critic.models.append(model)

    critic._fit_support(x)

    # Save checkpoint
    output_path.parent.mkdir(parents=True, exist_ok=True)
    critic.save(output_path)
    checkpoint_sha = file_sha256(output_path)

    # Compute metadata
    metadata = {
        "critic_version": critic.critic_version,
        "feature_block": feature_block,
        "selected_set_features_enabled": False,
        "pairwise_features_enabled": False,
        "encoder_schema_version": encoder.schema_version,
        "n_features": encoder.n_features,
        "n_bootstrap": n_bootstrap,
        "bootstrap_seeds": critic.bootstrap_seeds,
        "seed": seed,
        "test_fraction": test_fraction,
        "train_record_count": len(train),
        "test_record_count": len(test),
        "class_distribution_train": dict(
            zip(
                ["positive", "negative", "neutral_success", "neutral_failure"],
                [sum(1 for r in train if r.transfer_class == c) for c in ["positive", "negative", "neutral_success", "neutral_failure"]],
            )
        ),
        "class_distribution_test": dict(
            zip(
                ["positive", "negative", "neutral_success", "neutral_failure"],
                [sum(1 for r in test if r.transfer_class == c) for c in ["positive", "negative", "neutral_success", "neutral_failure"]],
            )
        ),
        "support_threshold": critic.support_threshold,
        "training_records_digest": input_path.name,
        "checkpoint_sha256": checkpoint_sha,
    }

    metadata_path = output_path.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    print(f"\nA1-NoSet critic saved to {output_path}")
    print(f"  feature_block={feature_block}")
    print(f"  selected_set_features_enabled=False")
    print(f"  pairwise_features_enabled=False")
    print(f"  train={len(train)}, test={len(test)}")
    print(f"  checkpoint_sha256={checkpoint_sha[:16]}...")


if __name__ == "__main__":
    main()
