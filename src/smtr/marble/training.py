"""MARBLE critic training pipeline.

Consumes candidate-level paired records and memory pool,
constructs CandidateExposureInput features, and fits FourOutcomeTransferCritic.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from smtr.core.types import CandidateExposureInput
from smtr.counterfactual.edge_keys import (
    TreatmentEdgeKey,
    edge_equal_sample_weights,
    group_records_by_control_family,
    group_records_by_edge,
    treatment_edge_key,
)
from smtr.counterfactual.paired_record import (
    canonical_paired_record_digest,
    edge_to_seed_set,
)
from smtr.marble.runtime_visibility_audit import file_digest
from smtr.router.transfer_calibration import (
    compute_four_class_metrics,
    compute_probability_metrics,
    predicted_label,
)
from smtr.router.transfer_critic import FourOutcomeTransferCritic
from smtr.router.transfer_features import (
    build_training_data_from_records,
    load_paired_records_with_metadata,
)

_DEFAULT_SEED = 7
_DEFAULT_N_BOOTSTRAP = 31
_DEFAULT_N_FEATURES = 512
_DEFAULT_FEATURE_BLOCK = "full"


@dataclass(frozen=True)
class EffectiveTrainingRecords:
    """Budget-filtered training records with provenance (清单 Fixed-Budget 3.3).

    ``records`` is the exact record list that may reach feature
    construction and critic.fit: complete treatment edges only, never
    individual generation seeds.
    """

    records: list[dict[str, Any]]
    parent_record_count: int
    effective_record_count: int
    parent_edge_count: int
    effective_edge_count: int
    requested_budget_fraction: float
    realized_budget_fraction: float
    parent_train_record_digest: str
    effective_train_record_digest: str
    budget_manifest_digest: str | None
    selected_edge_keys: tuple[TreatmentEdgeKey, ...]
    all_selected_edges_have_full_seed_support: bool
    all_selected_edges_found: bool
    unexpected_training_edge_count: int
    incomplete_seed_support_edge_count: int


def prepare_effective_training_records(
    *,
    train_records_path: Path,
    budget_candidate_manifest_path: Path | None,
    experiment_mode: str,
    train_records_already_budgeted: bool = False,
) -> EffectiveTrainingRecords:
    """Load train records and apply the budget manifest before features.

    清单 Fixed-Budget 第3-6章: budgeting removes complete treatment
    edges and never individual generation seeds. The returned records are
    the only records allowed to reach feature construction and
    ``critic.fit``; validation and test splits are never touched here.
    """
    from smtr.evaluation.split_audit import load_paired_records_file

    raw_train_records = load_paired_records_file(Path(train_records_path))
    if not raw_train_records:
        raise ValueError(
            f"no paired training records in {train_records_path}"
        )

    parent_edge_keys = {
        treatment_edge_key(rec) for rec in raw_train_records
    }
    parent_edge_seeds = edge_to_seed_set(raw_train_records)

    budget_meta = None
    if budget_candidate_manifest_path is None:
        selected_edge_keys = set(parent_edge_keys)
        requested_fraction = 1.0
        realized_fraction = 1.0
        budget_manifest_digest: str | None = None
    else:
        from smtr.marble.artifact_digests import (
            candidate_manifest_digest,
        )
        from smtr.marble.budget_sampling import (
            filter_paired_records_by_edge_keys,
            selected_treatment_edges_from_manifest,
        )
        from smtr.marble.real_data import DatabaseCandidateManifest

        manifest = DatabaseCandidateManifest.model_validate_json(
            Path(budget_candidate_manifest_path).read_text(encoding="utf-8")
        )
        if manifest.target_split != "train":
            raise ValueError(
                "budget candidate manifest must target the train split: "
                f"{budget_candidate_manifest_path}"
            )
        budget_meta = manifest.budget_metadata
        if budget_meta is None:
            raise ValueError(
                "budget candidate manifest lacks budget_metadata: "
                f"{budget_candidate_manifest_path}"
            )
        selected_edge_keys = selected_treatment_edges_from_manifest(manifest)
        requested_fraction = budget_meta.requested_fraction
        realized_fraction = budget_meta.realized_edge_fraction
        budget_manifest_digest = candidate_manifest_digest(manifest)

    if train_records_already_budgeted:
        # 清单 Fixed-Budget 第12章 mode B: the records file was already
        # materialized by materialize-budgeted-records; its edge set is
        # validated against the manifest instead of re-filtered.
        effective_records = list(raw_train_records)
    elif budget_candidate_manifest_path is None:
        effective_records = list(raw_train_records)
    else:
        effective_records = filter_paired_records_by_edge_keys(
            records=raw_train_records,
            selected_edge_keys=selected_edge_keys,
        )

    if not effective_records:
        raise ValueError(
            "budget filtering produced an empty training record set"
        )

    observed_effective_edge_keys = {
        treatment_edge_key(rec) for rec in effective_records
    }
    if not observed_effective_edge_keys:
        raise ValueError(
            "budget filtering produced no treatment edges"
        )

    missing_selected_edges = sorted(
        selected_edge_keys - observed_effective_edge_keys
    )
    unexpected_training_edges = sorted(
        observed_effective_edge_keys - selected_edge_keys
    )
    if experiment_mode == "formal":
        if missing_selected_edges:
            raise ValueError(
                "budget manifest contains treatment edges without "
                "paired training records"
            )
        if unexpected_training_edges:
            raise ValueError(
                "effective training records contain edges outside the "
                "budget manifest"
            )

    if (
        budget_meta is not None
        and requested_fraction == 1.0
        and not train_records_already_budgeted
    ):
        if selected_edge_keys != parent_edge_keys:
            raise ValueError(
                "B=1.0 budget manifest must preserve the complete "
                "parent treatment-edge set"
            )
        if len(effective_records) != len(raw_train_records):
            raise ValueError(
                "B=1.0 filtering changed the number of paired records"
            )

    # 清单 Fixed-Budget 第6章: selected edges keep their full seed set.
    effective_edge_seeds = edge_to_seed_set(effective_records)
    incomplete_seed_support_edge_count = 0
    for edge_key in selected_edge_keys:
        expected = parent_edge_seeds.get(edge_key, set())
        observed = effective_edge_seeds.get(edge_key, set())
        if observed != expected:
            incomplete_seed_support_edge_count += 1
    if (
        experiment_mode == "formal"
        and incomplete_seed_support_edge_count > 0
    ):
        raise ValueError(
            "budget filtering must remove whole edges, not individual "
            "generation seeds"
        )

    required_seed_count = 5 if experiment_mode == "formal" else 3
    wrong_seed_count_edges = [
        edge_key
        for edge_key, seeds in effective_edge_seeds.items()
        if len(seeds) < required_seed_count
    ]
    if experiment_mode == "formal" and wrong_seed_count_edges:
        raise ValueError(
            "budget training records have incomplete seed support"
        )

    # 清单 Fixed-Budget 第10.1节: strong assertion before any fitting.
    if budget_meta is not None and len(
        observed_effective_edge_keys
    ) != budget_meta.selected_edge_count:
        raise ValueError(
            "effective training edge count does not match budget "
            "manifest metadata"
        )

    return EffectiveTrainingRecords(
        records=effective_records,
        parent_record_count=len(raw_train_records),
        effective_record_count=len(effective_records),
        parent_edge_count=len(parent_edge_keys),
        effective_edge_count=len(observed_effective_edge_keys),
        requested_budget_fraction=requested_fraction,
        realized_budget_fraction=realized_fraction,
        parent_train_record_digest=canonical_paired_record_digest(
            raw_train_records
        ),
        effective_train_record_digest=canonical_paired_record_digest(
            effective_records
        ),
        budget_manifest_digest=budget_manifest_digest,
        selected_edge_keys=tuple(sorted(selected_edge_keys)),
        all_selected_edges_have_full_seed_support=(
            incomplete_seed_support_edge_count == 0
        ),
        all_selected_edges_found=not missing_selected_edges,
        unexpected_training_edge_count=len(unexpected_training_edges),
        incomplete_seed_support_edge_count=(
            incomplete_seed_support_edge_count
        ),
    )


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
    train_records_already_budgeted: bool = False,
    experiment_mode: str | None = None,
) -> dict[str, Any]:
    """Train four-outcome transfer critic from paired records."""
    # 清单 Fixed-Budget 第3章: the budget manifest is applied before any
    # feature construction or critic fitting. Budgeting removes complete
    # treatment edges and never individual generation seeds; validation
    # and test records are never filtered here.
    mode = experiment_mode if experiment_mode is not None else coverage_mode
    # 清单最终闭环 P0-4: formal critic training always requires an explicit
    # budget candidate manifest — including B=1.0. Pilot/debug may keep
    # None -> full support.
    if mode == "formal" and budget_candidate_manifest_path is None:
        raise ValueError(
            "formal critic training requires an explicit budget candidate "
            "manifest, including B=1.0"
        )
    prepared = prepare_effective_training_records(
        train_records_path=train_records_path,
        budget_candidate_manifest_path=budget_candidate_manifest_path,
        experiment_mode=mode,
        train_records_already_budgeted=train_records_already_budgeted,
    )
    if not prepared.all_selected_edges_have_full_seed_support:
        raise ValueError(
            "budget training records have incomplete seed support"
        )

    # Build features/labels from the budget-filtered records only, keeping
    # the raw record beside each example so multi-seed treatment edges can
    # be grouped (清单 P0-3): edge-equal sample weights keep loss balanced
    # per treatment edge, while bootstrap clusters are task-receiver
    # control families (清单 Shared-Control 第10章) so rows sharing one
    # no-memory control resample together.
    train_data = build_training_data_from_records(
        prepared.records, memory_pool_path
    )
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
            train_records_path=train_records_path,
            prepared=prepared,
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
    # 清单 Fixed-Budget 第9章: the effective (budget-filtered) subset, not
    # just the parent file, is the ground truth for what this critic saw.
    critic.effective_train_record_digest = (
        prepared.effective_train_record_digest
    )
    # 清单最终闭环 P0-1: the effective train edge count is a top-level
    # authoritative checkpoint field, not only a nested metadata copy.
    critic.effective_train_edge_count = prepared.effective_edge_count

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
    # 清单最终闭环 P0-1: budget fractions are top-level authoritative
    # checkpoint fields sourced from the prepared effective records.
    critic.training_budget_requested = prepared.requested_budget_fraction
    critic.training_budget_realized = prepared.realized_budget_fraction
    budget_meta = None
    if budget_candidate_manifest_path is not None:
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
        critic.parent_train_candidate_manifest_digest = (
            budget_meta.parent_manifest_digest
        )
        critic.budget_train_candidate_manifest_digest = (
            prepared.budget_manifest_digest
        )
        metrics["training_budget_policy"] = budget_meta.policy_version
        metrics["training_budget_requested"] = budget_meta.requested_fraction
        metrics["training_budget_realized"] = budget_meta.realized_edge_fraction

    # 清单 Fixed-Budget 第10章: structured budget provenance blocks. 第14章:
    # budgeting scopes train treatment edges only; validation/test stay full.
    budget_policy_block: dict[str, Any] = {
        "name": budget_meta.policy_version if budget_meta else None,
        "requested_fraction": prepared.requested_budget_fraction,
        "realized_fraction": prepared.realized_budget_fraction,
        "adaptive_sampling": (
            budget_meta.adaptive_sampling_used if budget_meta else False
        ),
        "outcome_fields_used": (
            budget_meta.outcome_fields_used if budget_meta else False
        ),
        "critic_predictions_used": (
            budget_meta.critic_predictions_used if budget_meta else False
        ),
        "budget_scope": "train_treatment_edges_only",
        "validation_support": "full",
        "test_support": "full",
    }
    training_support_block: dict[str, Any] = {
        "parent_train_record_count": prepared.parent_record_count,
        "effective_train_record_count": prepared.effective_record_count,
        "parent_train_edge_count": prepared.parent_edge_count,
        "effective_train_edge_count": prepared.effective_edge_count,
        "selected_edge_count_from_manifest": (
            budget_meta.selected_edge_count
            if budget_meta is not None
            else prepared.effective_edge_count
        ),
        "all_selected_edges_found": prepared.all_selected_edges_found,
        "unexpected_training_edge_count": (
            prepared.unexpected_training_edge_count
        ),
        "incomplete_seed_support_edge_count": (
            prepared.incomplete_seed_support_edge_count
        ),
        "all_selected_edges_have_full_seed_support": (
            prepared.all_selected_edges_have_full_seed_support
        ),
    }
    artifact_digests_block: dict[str, Any] = {
        "parent_train_records": prepared.parent_train_record_digest,
        "effective_train_records": prepared.effective_train_record_digest,
        "budget_candidate_manifest": prepared.budget_manifest_digest,
    }
    critic.budget_policy_metadata = budget_policy_block
    critic.training_support_metadata = training_support_block
    critic.training_artifact_digests = artifact_digests_block
    metrics["budget_policy"] = budget_policy_block
    metrics["training_support"] = training_support_block
    metrics["artifact_digests"] = artifact_digests_block

    # 清单 Writer-Agnostic 第十章: bind the writer-agnostic method-schema
    # metadata into the checkpoint so formal evaluation can reject legacy
    # writer-conditioned checkpoints.
    from smtr.marble.formal_protocol import (
        REQUIRED_FORMAL_CHECKPOINT_METADATA,
    )

    critic.method_schema_metadata = dict(
        REQUIRED_FORMAL_CHECKPOINT_METADATA
    )

    # Save checkpoint after calibration so epsilon_star is persisted.
    critic.save(output_path)

    # Write metrics alongside checkpoint
    metrics_path = output_path.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return metrics


def _run_training_split_audit(
    *,
    train_records: list[dict[str, Any]],
    train_records_path: Path,
    prepared: EffectiveTrainingRecords,
    validation_records_path: Path | None,
    test_records_path: Path | None,
) -> dict[str, Any]:
    """Split audit gate for formal critic training (清单 P0-16).

    Without a test file the audit still checks train/validation task and
    treatment-edge isolation plus memory-source provenance; test isolation
    is re-checked before the formal evaluation. 清单 Fixed-Budget 第13章:
    the audit also verifies the effective training-record digest and
    persists train-record provenance in its summary.
    """
    from smtr.evaluation.split_audit import audit_split_leakage, load_paired_records_file

    if canonical_paired_record_digest(
        train_records
    ) != prepared.effective_train_record_digest:
        raise ValueError(
            "checkpoint effective training-record digest mismatch"
        )

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
    summary["train_records_provenance"] = {
        "parent_file_digest": file_digest(Path(train_records_path)),
        "effective_record_digest": prepared.effective_train_record_digest,
        "budget_manifest_digest": prepared.budget_manifest_digest,
        "requested_budget_fraction": prepared.requested_budget_fraction,
        "effective_edge_count": prepared.effective_edge_count,
    }
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
    """Build feature audit JSON for checkpoint (清单 Writer-Agnostic 7.2).

    Reports whether writer/provenance, receiver and memory-receiver
    interaction features are present. Formal full checkpoints must have
    writer_features_present=False, provenance_features_present=False,
    receiver_features_present=True and
    memory_receiver_interactions_present=True.
    """
    from smtr.router.transfer_features import (
        FORBIDDEN_FEATURE_TOKENS,
        FORBIDDEN_PROVENANCE_FEATURE_PREFIXES,
    )

    # Check a sample of tokens
    sample = inputs[:min(100, len(inputs))]
    all_tokens: list[str] = []
    for item in sample:
        all_tokens.extend(critic.encoder.tokens(item))

    # Writer/provenance presence check (清单 7.1): any token whose prefix
    # matches a forbidden provenance name fails the audit immediately.
    provenance_found = False
    writer_found = False
    receiver_found = False
    interaction_found = False

    # Check forbidden leakage
    forbidden_found = False
    observed_prefixes: set[str] = set()
    for token in all_tokens:
        prefix = token.lower().split(":", 1)[0]
        observed_prefixes.add(prefix)
        if prefix in FORBIDDEN_FEATURE_TOKENS:
            forbidden_found = True
        if any(prefix.startswith(banned) for banned in FORBIDDEN_PROVENANCE_FEATURE_PREFIXES):
            provenance_found = True
            if prefix.startswith("writer") or prefix.startswith("wr_"):
                writer_found = True
        if prefix in {"receiver_role", "receiver_cap", "receiver_tool"}:
            receiver_found = True
        if prefix.startswith("mr_"):
            interaction_found = True

    return {
        "schema_version": "3.0",
        "feature_block": feature_block,
        "sample_count": len(sample),
        "routing_conditioning": "memory_receiver",
        "writer_features_present": writer_found,
        "provenance_features_present": provenance_found,
        "receiver_features_present": receiver_found,
        "memory_receiver_interactions_present": interaction_found,
        "forbidden_feature_leakage": forbidden_found or provenance_found,
        "observed_prefixes": sorted(observed_prefixes),
    }
