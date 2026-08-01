"""MARBLE critic training pipeline.

Consumes candidate-level paired records and memory pool,
constructs CandidateExposureInput features, and fits FourOutcomeTransferCritic.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from smtr.core.types import CandidateExposureInput
from smtr.router.transfer_critic import FourOutcomeTransferCritic
from smtr.router.transfer_features import load_paired_records_for_training

_DEFAULT_SEED = 7
_DEFAULT_N_BOOTSTRAP = 31
_DEFAULT_N_FEATURES = 512
_DEFAULT_FEATURE_BLOCK = "full"


def train_critic(
    *,
    train_records_path: Path,
    memory_pool_path: Path,
    validation_records_path: Path | None = None,
    output_path: Path,
    seed: int = _DEFAULT_SEED,
    n_bootstrap: int = _DEFAULT_N_BOOTSTRAP,
    n_features: int = _DEFAULT_N_FEATURES,
    feature_block: str = _DEFAULT_FEATURE_BLOCK,
) -> dict[str, Any]:
    """Train four-outcome transfer critic from paired records."""
    # Load training data
    train_data = load_paired_records_for_training(train_records_path, memory_pool_path)
    if not train_data:
        raise ValueError(f"no valid training records in {train_records_path}")

    inputs = [item for item, _ in train_data]
    labels = [label for _, label in train_data]

    label_counts = Counter(labels)

    # Fit critic
    critic = FourOutcomeTransferCritic(
        n_features=n_features,
        n_bootstrap=n_bootstrap,
        feature_block=feature_block,
        seed=seed,
    )
    critic.fit(inputs, labels)

    # Save checkpoint
    critic.save(output_path)

    # Validation metrics
    metrics: dict[str, Any] = {
        "train_records": len(train_data),
        "label_distribution": dict(label_counts),
        "n_features": n_features,
        "n_bootstrap": n_bootstrap,
        "feature_block": feature_block,
        "seed": seed,
        "checkpoint": str(output_path),
    }

    if validation_records_path and validation_records_path.exists():
        val_data = load_paired_records_for_training(validation_records_path, memory_pool_path)
        if val_data:
            val_inputs = [item for item, _ in val_data]
            val_labels = [label for _, label in val_data]
            preds = critic.predict_batch(val_inputs)
            correct = sum(
                1
                for pred, lb in zip(preds, val_labels)
                if _predicted_label(pred) == lb
            )
            metrics["validation_records"] = len(val_data)
            metrics["validation_accuracy"] = correct / len(val_data)

    # Write metrics alongside checkpoint
    metrics_path = output_path.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return metrics


def _predicted_label(pred) -> str:
    """Get the most likely label from a TransferPrediction."""
    probs = [
        pred.q00_neutral_failure,
        pred.q01_negative_transfer,
        pred.q10_positive_transfer,
        pred.q11_neutral_success,
    ]
    labels = ["neutral_failure", "negative_transfer", "positive_transfer", "neutral_success"]
    return labels[int(np.argmax(probs))]
