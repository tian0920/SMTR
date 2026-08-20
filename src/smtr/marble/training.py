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
    build_routing_card_from_pool_entry,
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


def _build_tci_inputs_for_critic(
    *,
    tci_contrasts_path: Path | None,
    perturbations_manifest_path: Path | None,
    paired_records_path: Path | None,
    memory_pool_path: Path,
    marble_source_path: Path | None = None,
) -> list[tuple[CandidateExposureInput, CandidateExposureInput, int, str]]:
    """Build TCI supervision tuples for critic.fit.

    Returns list of ``(input_original, input_perturbed, direction,
    contrast_type)``. Empty list when any path is missing (graceful
    fallback to observational-only training).

    The receiver/task context is taken from the paired records file
    matched by (task_id, receiver_agent_id) — the same context that
    observational training uses. The original card comes from the
    memory pool; the perturbed card from the perturbations manifest.
    Both cards share one ReceiverState so the critic sees a true
    memory-level contrast in its own feature space.
    """
    from smtr.core.types import (
        AgentProfile,
        CandidateExposureInput,
        MemoryRoutingCard,
        ReceiverState,
    )

    if (
        tci_contrasts_path is None
        or perturbations_manifest_path is None
        or paired_records_path is None
        or not tci_contrasts_path.exists()
        or not perturbations_manifest_path.exists()
        or not paired_records_path.exists()
    ):
        return []

    # ---- Load memory pool (dict by memory_id) ----
    pool: dict[str, dict] = {}
    for line in memory_pool_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            mem = json.loads(line)
            pool[mem["memory_id"]] = mem

    # ---- Load perturbations manifest ----
    manifest = json.loads(
        perturbations_manifest_path.read_text(encoding="utf-8")
    )
    perturbations_by_id: dict[str, dict] = {}
    for entry in manifest.get("perturbations", []):
        spec = entry.get("spec", {})
        pid = spec.get("perturbation_id")
        if pid:
            perturbations_by_id[pid] = entry

    # ---- Build receiver context lookup from paired records ----
    # Keyed by (task_id, receiver_agent_id); first match wins.
    def _lookup_context(rec: dict) -> tuple[str, str] | None:
        tid = str(rec.get("task_id", ""))
        rid = rec.get("receiver_agent_id", "")
        return (tid, rid) if tid and rid else None

    context_records: dict[tuple[str, str], dict] = {}
    for line in paired_records_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        key = _lookup_context(rec)
        if key is not None and key not in context_records:
            context_records[key] = rec

    if not context_records:
        return []

    # ---- Load contrasts ----
    contrasts: list[dict] = []
    for line in tci_contrasts_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            contrasts.append(json.loads(line))

    tci_inputs: list[
        tuple[CandidateExposureInput, CandidateExposureInput, int, str]
    ] = []
    for c in contrasts:
        pid = c.get("perturbation_id")
        direction = c.get("contrast_direction", 0)
        if pid is None or direction == 0:
            continue
        entry = perturbations_by_id.get(pid)
        if entry is None:
            continue
        spec = entry["spec"]
        perturbed_card_raw = entry.get("perturbed_card")
        if perturbed_card_raw is None:
            continue

        candidate_memory_id = spec.get("candidate_memory_id")
        if candidate_memory_id not in pool:
            continue

        # Original card from pool.
        original_card = build_routing_card_from_pool_entry(
            pool[candidate_memory_id]
        )
        # Perturbed card from manifest.
        perturbed_card = MemoryRoutingCard(**perturbed_card_raw)

        # Receiver context lookup by (task_id, receiver_agent_id).
        task_id = str(spec.get("task_id", c.get("task_id", "")))
        receiver_agent_id = spec.get(
            "receiver_agent_id", c.get("receiver_agent_id", "")
        )
        rec = context_records.get((task_id, receiver_agent_id))
        if rec is None:
            continue

        receiver = AgentProfile(
            agent_id=rec.get("receiver_agent_id", ""),
            role=rec.get("receiver_role", "unknown"),
            capabilities=tuple(rec.get("receiver_capabilities", [])),
            model_name=rec.get("receiver_model_name"),
            tool_names=tuple(rec.get("receiver_tool_names", [])),
        )
        receiver_state = ReceiverState(
            task_id=task_id,
            scenario=rec.get("scenario", "database"),
            task_instruction=rec.get("task_instruction", ""),
            receiver=receiver,
            subtask=rec.get("subtask"),
            environment_signature=tuple(
                rec.get("environment_signature", [])
            ),
            local_context_summary=rec.get("local_context_summary", ""),
            team_context_summary=rec.get("team_context_summary", ""),
        )

        input_orig = CandidateExposureInput(
            receiver_state=receiver_state,
            candidate_card=original_card,
        )
        input_pert = CandidateExposureInput(
            receiver_state=receiver_state,
            candidate_card=perturbed_card,
        )

        # contrast_type derived from the perturbation operator
        # (precondition, environment_constraint, capability, etc.)
        contrast_type = spec.get(
            "perturbation_type", spec.get("changed_field", "unknown")
        )
        tci_inputs.append(
            (input_orig, input_pert, int(direction), contrast_type)
        )

    return tci_inputs


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
    marble_source_path: Path | None = None,
    critic_mode: str = "flat",
    tci_contrasts_path: Path | None = None,
    tci_perturbations_manifest_path: Path | None = None,
    tci_paired_records_path: Path | None = None,
    tci_alpha: float = 1.0,
) -> dict[str, Any]:
    """Train four-outcome transfer critic from paired records.

    Optional TCI distillation (observational+tci training mode):
    when ``tci_contrasts_path`` is provided along with the perturbation
    manifest and paired records, TCI intervention pairs are encoded in
    the critic's feature space and appended to the observational training
    data with total weight ``tci_alpha``. When not provided, behaviour
    is identical to the original observational training.
    """
    # 清单 Formal Protocol §3: experiment_mode and coverage_mode must agree;
    # all downstream protocol checks use the unified ``mode``.
    from smtr.evaluation.experiment_protocol import validate_mode_consistency

    mode = validate_mode_consistency(
        experiment_mode=experiment_mode,
        coverage_mode=coverage_mode,
    )
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
        prepared.records, memory_pool_path,
        marble_source_path=marble_source_path,
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
    if mode == "formal":
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
        critic_mode=critic_mode,
    )
    critic.fit(
        inputs,
        labels,
        records=train_records,
        coverage_mode=coverage_mode,
        sample_weights=sample_weights,
        bootstrap_clusters=bootstrap_clusters,
        tci_inputs=_build_tci_inputs_for_critic(
            tci_contrasts_path=tci_contrasts_path,
            perturbations_manifest_path=tci_perturbations_manifest_path,
            paired_records_path=tci_paired_records_path,
            memory_pool_path=memory_pool_path,
            marble_source_path=marble_source_path,
        ),
        tci_alpha=tci_alpha,
    )
    # TCI distillation provenance (Task 6).
    if tci_contrasts_path is not None:
        critic.tci_distillation_alpha = tci_alpha
        tci_eval_inputs = _build_tci_inputs_for_critic(
            tci_contrasts_path=tci_contrasts_path,
            perturbations_manifest_path=tci_perturbations_manifest_path,
            paired_records_path=tci_paired_records_path,
            memory_pool_path=memory_pool_path,
            marble_source_path=marble_source_path,
        )
        if tci_eval_inputs:
            from smtr.router.tci_supervision import (
                evaluate_tci_loss_on_critic,
            )
            critic.tci_distillation_metrics = (
                evaluate_tci_loss_on_critic(critic, tci_eval_inputs)
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
        "critic_mode": critic_mode,
        # TCI Distillation provenance (Task 6).
        "tci_distillation_n_examples": critic.tci_distillation_n_examples,
        "tci_distillation_alpha": critic.tci_distillation_alpha,
        "tci_distillation_metrics": critic.tci_distillation_metrics,
        "tci_training_mode": (
            "observational+tci"
            if tci_contrasts_path is not None
            else "observational"
        ),
    }
    if critic_mode == "opportunity_factorized":
        metrics["factorization_version"] = "counterfactual_opportunity_v1"
        metrics["head_support"] = critic.head_support_report
    if split_audit_summary is not None:
        metrics["split_audit"] = split_audit_summary

    if validation_records_path and validation_records_path.exists():
        val_data = load_paired_records_with_metadata(
            validation_records_path, memory_pool_path,
            marble_source_path=marble_source_path,
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

    # 清单 Formal Protocol §2: seed protocol metadata bound into every
    # checkpoint so downstream stages can verify the same protocol.
    from smtr.evaluation.experiment_protocol import build_seed_protocol_block

    observed_seeds = sorted({
        int(rec.get("generation_seed", -1))
        for rec in prepared.records
    })
    critic.seed_protocol_metadata = build_seed_protocol_block(
        mode=mode, seeds=observed_seeds
    )
    metrics["seed_protocol"] = critic.seed_protocol_metadata

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
        "schema_version": "3.1",
        "feature_block": feature_block,
        "sample_count": len(sample),
        "routing_conditioning": "memory_receiver",
        "writer_features_present": writer_found,
        "provenance_features_present": provenance_found,
        "receiver_features_present": receiver_found,
        "memory_receiver_interactions_present": interaction_found,
        "task_memory_interaction_present": any(
            p.startswith("tm_") for p in observed_prefixes
        ),
        "procedure_signature_present": any(
            p.startswith("psi_") for p in observed_prefixes
        ),
        "forbidden_feature_leakage": forbidden_found or provenance_found,
        "observed_prefixes": sorted(observed_prefixes),
    }
