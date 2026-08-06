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
    group_records_by_control_family,
    group_records_by_edge,
)
from smtr.marble.runtime_visibility_audit import file_digest
from smtr.router.transfer_calibration import (
    compute_four_class_metrics,
    compute_probability_metrics,
    predicted_label,
)
from smtr.router.transfer_critic import FourOutcomeTransferCritic
from smtr.router.transfer_features import load_paired_records_with_metadata

_DEFAULT_SEED = 7
_DEFAULT_N_BOOTSTRAP = 31
_DEFAULT_N_FEATURES = 512
_DEFAULT_FEATURE_BLOCK = "full"


def train_critic(
    *,
    train_records_path: Path,
    memory_pool_path: Path,
    validation_records_path: Path | None = None,
    test_records_path: Path | None = None,
    output_path: Path,
    seed: int = _DEFAULT_SEED,
    n_bootstrap: int = _DEFAULT_N_BOOTSTRAP,
    n_features: int = _DEFAULT_N_FEATURES,
    feature_block: str = _DEFAULT_FEATURE_BLOCK,
    coverage_mode: str = "formal",
    risk_delta: float = 0.10,
    budget_candidate_manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Train four-outcome transfer critic from paired records."""
    # Load training data with the underlying records so multi-seed treatment
    # edges can be grouped (清单 P0-3): edge-equal sample weights keep loss
    # balanced per treatment edge, while bootstrap clusters are
    # task-receiver control families (清单 Shared-Control 第10章) so rows
    # sharing one no-memory control resample together.
    train_data = load_paired_records_with_metadata(train_records_path, memory_pool_path)
    if not train_data:
        raise ValueError(f"no valid training records in {train_records_path}")

    inputs = [item for item, _, _ in train_data]
    labels = [label for _, label, _ in train_data]
    train_records = [rec for _, _, rec in train_data]
    edge_groups = group_records_by_edge(train_records)
    bootstrap_clusters = group_records_by_control_family(train_records)
    sample_weights = edge_equal_sample_weights(train_records)

    label_counts = Counter(labels)

    # 清单 P0-16: formal training must pass the split audit before the
    # critic is fitted; any leakage aborts training immediately.
    split_audit_summary = None
    if coverage_mode == "formal":
        split_audit_summary = _run_training_split_audit(
            train_records=train_records,
            validation_records_path=validation_records_path,
            test_records_path=test_records_path,
        )

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
        bootstrap_clusters=bootstrap_clusters,
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
        "train_edges": len(edge_groups),
        "train_control_families": len(bootstrap_clusters),
        "loss_weighting_unit": "treatment_edge",
        "bootstrap_cluster_unit": "task_receiver_control_family",
        "label_distribution": dict(label_counts),
        "coverage_mode": coverage_mode,
        "coverage_report": critic.coverage_report,
        "n_features": n_features,
        "n_bootstrap": n_bootstrap,
        "feature_block": feature_block,
        "seed": seed,
        "checkpoint": str(output_path),
    }
    if split_audit_summary is not None:
        metrics["split_audit"] = split_audit_summary

    if validation_records_path and validation_records_path.exists():
        val_data = load_paired_records_with_metadata(
            validation_records_path, memory_pool_path
        )
        if val_data:
            val_inputs = [item for item, _, _ in val_data]
            val_labels = [label for _, label, _ in val_data]
            val_records = [rec for _, _, rec in val_data]
            preds = critic.predict_batch(val_inputs)
            pred_labels = [predicted_label(_pred_vector(pred)) for pred in preds]
            metrics["validation_records"] = len(val_data)
            metrics["validation_edges"] = len(group_records_by_edge(val_records))
            metrics["validation_accuracy"] = sum(
                1 for p, t in zip(pred_labels, val_labels) if p == t
            ) / len(val_data)
            metrics["validation_classification"] = compute_four_class_metrics(
                val_labels, pred_labels
            )
            metrics["validation_probability"] = compute_probability_metrics(
                val_labels, np.array([_pred_vector(pred) for pred in preds])
            )
            # 清单 P0-7/P0-8: edge-level q01 calibration and epsilon
            # selection happen on validation edges only.
            selection = critic.calibrate_q01(
                val_inputs,
                val_labels,
                val_records,
                split_name="validation",
                delta=risk_delta,
            )
            metrics["epsilon_star"] = selection["epsilon_star"]
            metrics["risk_delta"] = risk_delta
            metrics["epsilon_selected_on"] = "validation"
            metrics["calibration_split"] = "validation"
            metrics["epsilon_selection_split"] = "validation"
            metrics["validation_edge_count"] = selection["validation_edge_count"]
            # 清单 P0-8: calibration / epsilon-selection provenance.
            metrics["calibration_unit"] = selection.get(
                "selection_unit", "treatment_edge"
            )
            metrics["calibration_method"] = (
                critic.q01_calibrator.method
                if critic.q01_calibrator is not None
                else "unfitted"
            )
            metrics["calibration_status"] = (
                critic.q01_calibrator.calibration_status
                if critic.q01_calibrator is not None
                else "unfitted"
            )
            metrics["calibration_edge_count"] = selection["validation_edge_count"]
            metrics["epsilon_selection_unit"] = selection.get(
                "selection_unit", "treatment_edge"
            )
            metrics["epsilon_validation_edge_count"] = selection[
                "validation_edge_count"
            ]

    # 清单 P0-2: bind the training provenance into the checkpoint so a later
    # split audit can verify the exact artifacts this critic was fitted on,
    # not just the checkpoint file digest.
    critic.training_split = "train"
    critic.train_record_digest = file_digest(Path(train_records_path))
    critic.validation_record_digest = (
        file_digest(Path(validation_records_path))
        if validation_records_path and Path(validation_records_path).exists()
        else None
    )
    critic.memory_pool_digest = file_digest(Path(memory_pool_path))

    # 清单 Shared-Control 第16.1节: shared-control and budget provenance are
    # bound into every checkpoint; budget fields come from the budgeted
    # train candidate manifest when this checkpoint trains a B subset.
    from smtr.counterfactual.paired_record import (
        SHARED_CONTROL_DEFINITION_VERSION,
    )

    critic.shared_control_definition_version = SHARED_CONTROL_DEFINITION_VERSION
    critic.loss_weighting_unit = "treatment_edge"
    critic.bootstrap_cluster_unit = "task_receiver_control_family"
    critic.adaptive_sampling_used = False
    critic.adaptive_stopping_used = False
    if budget_candidate_manifest_path is not None:
        from smtr.marble.budget_sampling import manifest_canonical_digest
        from smtr.marble.real_data import DatabaseCandidateManifest

        budget_manifest = DatabaseCandidateManifest.model_validate_json(
            Path(budget_candidate_manifest_path).read_text(encoding="utf-8")
        )
        budget_meta = budget_manifest.budget_metadata
        if budget_meta is None:
            raise ValueError(
                "budget candidate manifest lacks budget_metadata: "
                f"{budget_candidate_manifest_path}"
            )
        critic.training_budget_policy = budget_meta.policy_version
        critic.training_budget_requested = budget_meta.requested_fraction
        critic.training_budget_realized = budget_meta.realized_edge_fraction
        critic.parent_train_candidate_manifest_digest = (
            budget_meta.parent_manifest_digest
        )
        critic.budget_train_candidate_manifest_digest = (
            manifest_canonical_digest(budget_manifest)
        )
        metrics["training_budget_policy"] = budget_meta.policy_version
        metrics["training_budget_requested"] = budget_meta.requested_fraction
        metrics["training_budget_realized"] = budget_meta.realized_edge_fraction

    # Save checkpoint after calibration so epsilon_star is persisted.
    critic.save(output_path)

    # Write metrics alongside checkpoint
    metrics_path = output_path.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return metrics


def _run_training_split_audit(
    *,
    train_records: list[dict[str, Any]],
    validation_records_path: Path | None,
    test_records_path: Path | None,
) -> dict[str, Any]:
    """Split audit gate for formal critic training (清单 P0-16).

    Without a test file the audit still checks train/validation task and
    treatment-edge isolation plus memory-source provenance; test isolation
    is re-checked before the formal evaluation.
    """
    from smtr.evaluation.split_audit import audit_split_leakage, load_paired_records_file

    splits: dict[str, list[dict[str, Any]]] = {"train": list(train_records)}
    for name, path in (
        ("validation", validation_records_path),
        ("test", test_records_path),
    ):
        if path is not None and Path(path).exists():
            splits[name] = load_paired_records_file(Path(path))
        else:
            splits[name] = []

    try:
        summary = audit_split_leakage(
            splits,
            calibration_split="validation",
            epsilon_selection_split="validation",
        )
    except ValueError as exc:
        raise ValueError(
            f"formal critic training aborted: split audit failed: {exc}"
        ) from exc
    if not summary["split_integrity_passed"]:
        raise ValueError("formal critic training aborted: split audit failed")
    return summary


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
