"""P2 perturbation artifact schema (清单 §4).

Defines frozen dataclass records for transfer-critical counterfactual
interventions. Each perturbation changes exactly ONE transfer-critical
factor while keeping all other memory fields byte-equivalent.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

SCHEMA_VERSION = "smtr_transfer_perturb_v1"

INTERVENTION_CONTRAST_SCHEMA_VERSION = "smtr_intervention_contrast_v1"

PERTURBATION_TYPES: frozenset[str] = frozenset(
    {
        "required_tool",
        "required_capability",
        "precondition",
        "environment_constraint",
        "procedure_dependency",
    }
)


@dataclass(frozen=True)
class PerturbationSpec:
    """Specification of a single transfer-critical perturbation.

    Invariants (清单 §4.1, §4.3):
      - ``perturbation_type`` must be in ``PERTURBATION_TYPES``.
      - ``changed_field`` describes exactly which routing-card field was
        modified (e.g. "required_tools", "precondition_tags").
      - ``original_memory_digest`` / ``perturbed_memory_digest`` allow
        cryptographic provenance verification.
    """

    perturbation_id: str

    task_id: str
    receiver_agent_id: str
    candidate_memory_id: str

    perturbation_type: str
    changed_field: str

    original_value: Any
    perturbed_value: Any

    source_record_id: str
    control_group_key: str

    generation_seed: int

    original_memory_digest: str
    perturbed_memory_digest: str

    def __post_init__(self) -> None:
        if self.perturbation_type not in PERTURBATION_TYPES:
            raise ValueError(
                f"unknown perturbation_type: {self.perturbation_type!r}; "
                f"valid: {sorted(PERTURBATION_TYPES)}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialise to JSON-safe dict."""
        d = asdict(self)
        d["schema_version"] = SCHEMA_VERSION
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PerturbationSpec:
        """Reconstruct from dict (ignore schema_version)."""
        keys = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in keys})


@dataclass(frozen=True)
class PerturbationOutcomeRecord:
    """Outcome record pairing original, withhold and perturbed branches.

    Provenance invariants (清单 §4.3):
      - ``y0`` reuses existing shared no-memory control (never re-executed).
      - ``y_original`` reuses existing share branch (never re-executed).
      - Only ``y_perturbed`` requires a new MARBLE execution.
      - ``task_id``, ``receiver_agent_id``, ``generation_seed`` must match
        the original paired record.
    """

    schema_version: str

    spec: PerturbationSpec

    y0: bool
    y_original: bool
    y_perturbed: bool

    original_branch_id: str
    perturbed_branch_id: str

    task_id: str
    receiver_agent_id: str
    candidate_memory_id: str

    generation_seed: int

    runtime_metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Serialise to JSON-safe dict."""
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PerturbationOutcomeRecord:
        """Reconstruct from dict."""
        spec = PerturbationSpec.from_dict(data["spec"])
        keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in keys and k != "spec"}
        return cls(spec=spec, **filtered)


def compute_memory_digest(card_dict: dict[str, Any]) -> str:
    """Deterministic SHA256 digest of a routing-card dict."""
    raw = json.dumps(card_dict, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def compute_perturbation_id(spec: PerturbationSpec) -> str:
    """Deterministic ID for a perturbation spec."""
    key = (
        spec.task_id,
        spec.receiver_agent_id,
        spec.candidate_memory_id,
        spec.perturbation_type,
        str(spec.generation_seed),
    )
    raw = "|".join(key)
    return "pert_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
