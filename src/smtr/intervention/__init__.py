"""SMTR transfer-critical counterfactual intervention (P2) package."""

from smtr.intervention.perturbation_schema import (
    SCHEMA_VERSION,
    INTERVENTION_CONTRAST_SCHEMA_VERSION,
    PERTURBATION_TYPES,
    PerturbationSpec,
    PerturbationOutcomeRecord,
)
from smtr.intervention.transfer_perturbation import (
    PerturbedMemory,
    TransferPerturbationOperator,
    RequiredToolPerturbation,
    RequiredCapabilityPerturbation,
    PreconditionPerturbation,
    EnvironmentConstraintPerturbation,
    ProcedureDependencyPerturbation,
    OPERATOR_PRIORITY,
    get_all_operators,
    validate_single_factor_change,
)
from smtr.intervention.perturbation_selector import (
    edge_has_transfer_event,
    select_balanced_operator,
    select_perturbation_edges,
)
from smtr.intervention.perturbation_runner import (
    run_perturbed_exposure_branch,
)
from smtr.intervention.perturbation_analysis import (
    compute_perturbation_metrics,
    compute_operator_level_metrics,
    compute_baseline_conditioned_flips,
    compute_support_gain,
    compute_triple_counts,
    compute_operator_distribution,
    compute_pilot_gate,
    compute_contrast_summary,
    compute_operator_contrast,
    validate_real_execution_records,
    TripleCounts,
    PilotGate,
    ContrastSummary,
    OperatorContrastStats,
)
from smtr.intervention.intervention_contrast import (
    InterventionContrast,
    compute_transfer_effect,
    compute_contrast_direction,
    is_valid_contrast,
)
from smtr.intervention.contrast_types import (
    ContrastType,
    classify_contrast,
)
from smtr.intervention.contrast_builder import (
    build_intervention_contrasts,
)

__all__ = [
    "SCHEMA_VERSION",
    "INTERVENTION_CONTRAST_SCHEMA_VERSION",
    "PERTURBATION_TYPES",
    "PerturbationSpec",
    "PerturbationOutcomeRecord",
    "PerturbedMemory",
    "TransferPerturbationOperator",
    "RequiredToolPerturbation",
    "RequiredCapabilityPerturbation",
    "PreconditionPerturbation",
    "EnvironmentConstraintPerturbation",
    "ProcedureDependencyPerturbation",
    "OPERATOR_PRIORITY",
    "get_all_operators",
    "validate_single_factor_change",
    "edge_has_transfer_event",
    "select_balanced_operator",
    "select_perturbation_edges",
    "run_perturbed_exposure_branch",
    "compute_perturbation_metrics",
    "compute_operator_level_metrics",
    "compute_baseline_conditioned_flips",
    "compute_support_gain",
    "compute_triple_counts",
    "compute_operator_distribution",
    "compute_pilot_gate",
    "compute_contrast_summary",
    "compute_operator_contrast",
    "validate_real_execution_records",
    "TripleCounts",
    "PilotGate",
    "ContrastSummary",
    "OperatorContrastStats",
    "InterventionContrast",
    "compute_transfer_effect",
    "compute_contrast_direction",
    "is_valid_contrast",
    "ContrastType",
    "classify_contrast",
    "build_intervention_contrasts",
]
