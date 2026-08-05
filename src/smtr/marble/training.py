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
from smtr.counterfactual.edge_keys import (
    edge_equal_sample_weights,
    group_records_by_edge,
)
from smtr.router.transfer_calibration import (
    compute_four_class_metrics,
    compute_probability_metrics,
    predicted_label,
)
from smtr.router.transfer_critic import FourOutcomeTransferCritic
from smtr.router.transfer_features import (
    load_paired_records_for_training,
    load_paired_records_with_metadata,
)

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
    coverage_mode: str = "formal",
    risk_delta: float = 0.10,
) -> dict[str, Any]:
    """Train four-outcome transfer critic from paired records."""
    # Load training data with the underlying records so multi-seed treatment
    # edges can be grouped (清单 P0-3): edge-equal sample weights and
    # edge-cluster bootstrap treat seeds as repeated trials of one edge.
    train_data = load_paired_records_with_metadata(train_records_path, memory_pool_path)
    if not train_data:
        raise ValueError(f"no valid training records in {train_records_path}")

    inputs = [item for item, _, _ in train_data]
    labels = [label for _, label, _ in train_data]
    train_records = [rec for _, _, rec in train_data]
    edge_clusters = group_records_by_edge(train_records)
    sample_weights = edge_equal_sample_weights(train_records)

    label_counts = Counter(labels)

    # Fit critic
    critic = FourOutcomeTransferCritic(
        n_features=n_features,
        n_bootstrap=n_bootstrap,
        feature_block=feature_block,
        seed=seed,
    )
    critic.fit(
        inputs,
        labels,
        coverage_mode=coverage_mode,
        sample_weights=sample_weights,
        edge_clusters=edge_clusters,
    )

    # Write feature audit
    feature_audit = _build_feature_audit(
        critic=critic,
        inputs=inputs,
        feature_block=feature_block,
    )
    audit_path = output_path.with_suffix(".feature_audit.json")
    audit_path.write_text(json.dumps(feature_audit, indent=2), encoding="utf-8")

    # Validation metrics + q01 calibration + validation-selected epsilon_star.
    # The risk budget is chosen here on validation data only; the test split
    # must only read epsilon_star from the checkpoint.
    metrics: dict[str, Any] = {
        "train_records": len(train_data),
        "train_edges": len(edge_clusters),
        "label_distribution": dict(label_counts),
        "coverage_mode": coverage_mode,
        "coverage_report": critic.coverage_report,
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
            pred_labels = [predicted_label(_pred_vector(pred)) for pred in preds]
            metrics["validation_records"] = len(val_data)
            metrics["validation_accuracy"] = sum(
                1 for p, t in zip(pred_labels, val_labels) if p == t
            ) / len(val_data)
            metrics["validation_classification"] = compute_four_class_metrics(
                val_labels, pred_labels
            )
            metrics["validation_probability"] = compute_probability_metrics(
                val_labels, np.array([_pred_vector(pred) for pred in preds])
            )
            selection = critic.calibrate_q01(val_inputs, val_labels, delta=risk_delta)
            metrics["epsilon_star"] = selection["epsilon_star"]
            metrics["risk_delta"] = risk_delta
            metrics["epsilon_selected_on"] = "validation"

    # Save checkpoint after calibration so epsilon_star is persisted.
    critic.save(output_path)

    # Write metrics alongside checkpoint
    metrics_path = output_path.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return metrics


def _pred_vector(pred) -> np.ndarray:
    """Probability vector in LABELS order from a TransferPrediction."""
    return np.array([
        pred.q00_neutral_failure,
        pred.q01_negative_transfer,
        pred.q10_positive_transfer,
        pred.q11_neutral_success,
    ])


def _predicted_label(pred) -> str:
    """Get the most likely label from a TransferPrediction."""
    labels = ["neutral_failure", "negative_transfer", "positive_transfer", "neutral_success"]
    return labels[int(np.argmax(_pred_vector(pred)))]


def _build_feature_audit(
    *,
    critic: FourOutcomeTransferCritic,
    inputs: list[CandidateExposureInput],
    feature_block: str,
) -> dict[str, Any]:
    """Build feature audit JSON for checkpoint."""
    from smtr.router.transfer_features import FORBIDDEN_FEATURE_TOKENS

    # Check a sample of tokens
    sample = inputs[:min(100, len(inputs))]
    all_tokens: list[str] = []
    for item in sample:
        all_tokens.extend(critic.encoder.tokens(item))

    # Check writer-receiver features present
    wr_present = any(t.startswith("wr_pair:") for t in all_tokens)

    # Check forbidden leakage
    forbidden_found = False
    observed_prefixes: set[str] = set()
    for token in all_tokens:
        prefix = token.lower().split(":", 1)[0]
        observed_prefixes.add(prefix)
        if prefix in FORBIDDEN_FEATURE_TOKENS:
            forbidden_found = True

    return {
        "schema_version": "2.0",
        "feature_block": feature_block,
        "sample_count": len(sample),
        "writer_receiver_features_present": wr_present,
        "forbidden_feature_leakage": forbidden_found,
        "observed_prefixes": sorted(observed_prefixes),
    }
