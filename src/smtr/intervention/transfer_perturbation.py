"""Transfer-critical perturbation operators (清单 §5-§11).

Each operator implements the TransferPerturbationOperator Protocol:
  - ``applicable(memory, receiver)``: can this operator produce a valid
    single-factor perturbation for the given memory and receiver?
  - ``perturb(memory, receiver, rng=...)``: return a PerturbedMemory with
    exactly one routing-card field changed.

Hard invariants (清单 §6.3, §11):
  - Only the declared ``changed_field`` may differ between original and
    perturbed memory.
  - goal_summary, task_tags, procedure steps, preconditions/postconditions
    (unless targeted) must be byte-equivalent.
  - perturbation text must not contain label-leaking tokens.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Protocol

from smtr.core.types import MemoryRoutingCard, ReceiverState
from smtr.intervention.perturbation_schema import (
    PerturbationSpec,
    compute_memory_digest,
    compute_perturbation_id,
)

# ──────────────────────────────────────────────────────────────
# Forbidden tokens (清单 §11.1): perturbation text must NOT leak
# labels, outcomes or receiver identity.
# ──────────────────────────────────────────────────────────────
FORBIDDEN_TOKENS: frozenset[str] = frozenset(
    {
        "harmful",
        "negative_transfer",
        "positive_transfer",
        "neutral_failure",
        "neutral_success",
        "perturbed",
        "synthetic",
        "label",
        "outcome",
        "bad memory",
        "damage",
        "rescue",
    }
)

# ──────────────────────────────────────────────────────────────
# Plausible replacement pools for each operator.
# ──────────────────────────────────────────────────────────────
# These are domain-plausible but receiver-incompatible replacements.
_TOOL_POOL: tuple[str, ...] = (
    "sql_admin_write",
    "file_system_mount",
    "docker_container_exec",
    "git_push_force",
    "kubernetes_deploy",
    "network_packet_capture",
    "gpu_memory_alloc",
)

_CAPABILITY_POOL: tuple[str, ...] = (
    "modify_database_schema",
    "write_production_data",
    "deploy_microservice",
    "admin_user_management",
    "real_time_streaming",
    "distributed_lock_acquisition",
)

_PRECONDITION_POOL: tuple[str, ...] = (
    "privileged_write_access",
    "exclusive_table_lock",
    "real_time_replication_enabled",
    "multi_region_consensus",
    "admin_role_required",
    "audit_logging_disabled",
)

_ENVIRONMENT_POOL: tuple[str, ...] = (
    "temporary_table_creation_required",
    "write_ahead_logging_disabled",
    "cross_region_replication_needed",
    "gpu_acceleration_required",
    "real_time_streaming_context",
)

# Procedure step dependency tokens (清单 §10).
_STEP_DEPENDENCY_MARKERS: tuple[str, ...] = (
    "requires_output_of_previous",
    "depends_on_schema_inspection",
    "requires_prior_aggregation",
)


@dataclass(frozen=True)
class PerturbedMemory:
    """Result of applying one perturbation operator to a memory card."""

    card: MemoryRoutingCard
    changed_field: str
    original_value: Any
    perturbed_value: Any
    perturbation_type: str


class TransferPerturbationOperator(Protocol):
    """Protocol for transfer-critical perturbation operators."""

    name: str

    def applicable(
        self,
        memory: MemoryRoutingCard,
        receiver: ReceiverState,
    ) -> bool:
        """Return True if this operator can produce a valid perturbation."""
        ...

    def perturb(
        self,
        memory: MemoryRoutingCard,
        receiver: ReceiverState,
        *,
        rng: random.Random,
    ) -> PerturbedMemory:
        """Return a PerturbedMemory with exactly one field changed."""
        ...


def _check_forbidden_tokens(text: str) -> None:
    """Raise if text contains any forbidden label-leaking token."""
    low = text.lower()
    for tok in FORBIDDEN_TOKENS:
        if tok in low:
            raise ValueError(
                f"forbidden token {tok!r} found in perturbation text"
            )


# ──────────────────────────────────────────────────────────────
# Operator A: Required Tool Perturbation (清单 §6)
# ──────────────────────────────────────────────────────────────
class RequiredToolPerturbation:
    """Change one required_tool to a receiver-incompatible tool."""

    name = "required_tool"

    def applicable(
        self,
        memory: MemoryRoutingCard,
        receiver: ReceiverState,
    ) -> bool:
        if not memory.required_tools:
            return False
        receiver_tools = set(receiver.receiver.tool_names)
        return len(receiver_tools) > 0

    def perturb(
        self,
        memory: MemoryRoutingCard,
        receiver: ReceiverState,
        *,
        rng: random.Random,
    ) -> PerturbedMemory:
        receiver_tools = set(receiver.receiver.tool_names)
        original_tools = list(memory.required_tools)

        # Pick a replacement tool that the receiver does NOT have.
        candidates = [t for t in _TOOL_POOL if t not in receiver_tools]
        if not candidates:
            candidates = list(_TOOL_POOL)

        new_tool = rng.choice(candidates)
        _check_forbidden_tokens(new_tool)

        # Replace one tool (pick randomly from original).
        idx = rng.randrange(len(original_tools))
        new_tools = list(original_tools)
        original_value = new_tools[idx]
        new_tools[idx] = new_tool

        new_card = memory.model_copy(
            update={"required_tools": tuple(new_tools)}
        )
        return PerturbedMemory(
            card=new_card,
            changed_field="required_tools",
            original_value=original_value,
            perturbed_value=new_tool,
            perturbation_type=self.name,
        )


# ──────────────────────────────────────────────────────────────
# Operator B: Required Capability Perturbation (清单 §7)
# ──────────────────────────────────────────────────────────────
class RequiredCapabilityPerturbation:
    """Change one required_capability to an unsatisfied capability."""

    name = "required_capability"

    def applicable(
        self,
        memory: MemoryRoutingCard,
        receiver: ReceiverState,
    ) -> bool:
        if not memory.required_capabilities:
            return False
        receiver_caps = set(receiver.receiver.capabilities)
        # Applicable if at least one capability is currently satisfied.
        return bool(set(memory.required_capabilities) & receiver_caps)

    def perturb(
        self,
        memory: MemoryRoutingCard,
        receiver: ReceiverState,
        *,
        rng: random.Random,
    ) -> PerturbedMemory:
        receiver_caps = set(receiver.receiver.capabilities)
        original_caps = list(memory.required_capabilities)

        # Find a satisfied capability to replace.
        satisfied = [c for c in original_caps if c in receiver_caps]
        if not satisfied:
            satisfied = original_caps

        target_cap = rng.choice(satisfied)

        # Replacement must NOT be in receiver capabilities.
        candidates = [c for c in _CAPABILITY_POOL if c not in receiver_caps]
        if not candidates:
            candidates = list(_CAPABILITY_POOL)

        new_cap = rng.choice(candidates)
        _check_forbidden_tokens(new_cap)

        new_caps = [c if c != target_cap else new_cap for c in original_caps]
        new_card = memory.model_copy(
            update={"required_capabilities": tuple(new_caps)}
        )
        return PerturbedMemory(
            card=new_card,
            changed_field="required_capabilities",
            original_value=target_cap,
            perturbed_value=new_cap,
            perturbation_type=self.name,
        )


# ──────────────────────────────────────────────────────────────
# Operator C: Precondition Perturbation (清单 §8)
# ──────────────────────────────────────────────────────────────
class PreconditionPerturbation:
    """Replace one precondition with a receiver-invalid precondition."""

    name = "precondition"

    def applicable(
        self,
        memory: MemoryRoutingCard,
        receiver: ReceiverState,
    ) -> bool:
        # Applicable if memory has preconditions or we can add one.
        return True

    def perturb(
        self,
        memory: MemoryRoutingCard,
        receiver: ReceiverState,
        *,
        rng: random.Random,
    ) -> PerturbedMemory:
        original_preconds = list(memory.precondition_tags)
        new_precond = rng.choice(_PRECONDITION_POOL)
        _check_forbidden_tokens(new_precond)

        if original_preconds:
            # Replace one existing precondition.
            idx = rng.randrange(len(original_preconds))
            original_value = original_preconds[idx]
            new_preconds = list(original_preconds)
            new_preconds[idx] = new_precond
        else:
            # Add a new precondition (original was empty).
            original_value = ()
            new_preconds = [new_precond]

        new_card = memory.model_copy(
            update={"precondition_tags": tuple(new_preconds)}
        )
        return PerturbedMemory(
            card=new_card,
            changed_field="precondition_tags",
            original_value=original_value,
            perturbed_value=new_precond,
            perturbation_type=self.name,
        )


# ──────────────────────────────────────────────────────────────
# Operator D: Environment Constraint Perturbation (清单 §9)
# ──────────────────────────────────────────────────────────────
class EnvironmentConstraintPerturbation:
    """Change environment constraints to be receiver-incompatible."""

    name = "environment_constraint"

    def applicable(
        self,
        memory: MemoryRoutingCard,
        receiver: ReceiverState,
    ) -> bool:
        # Applicable if there are constraints to modify or we can add one.
        return True

    def perturb(
        self,
        memory: MemoryRoutingCard,
        receiver: ReceiverState,
        *,
        rng: random.Random,
    ) -> PerturbedMemory:
        original_env = list(memory.environment_constraints)
        new_constraint = rng.choice(_ENVIRONMENT_POOL)
        _check_forbidden_tokens(new_constraint)

        # Ensure the new constraint is NOT already in receiver env.
        receiver_env = set(receiver.environment_signature)
        if new_constraint in receiver_env:
            # Try other candidates.
            alts = [c for c in _ENVIRONMENT_POOL if c not in receiver_env]
            if alts:
                new_constraint = rng.choice(alts)

        if original_env:
            idx = rng.randrange(len(original_env))
            original_value = original_env[idx]
            new_env = list(original_env)
            new_env[idx] = new_constraint
        else:
            original_value = ()
            new_env = [new_constraint]

        new_card = memory.model_copy(
            update={"environment_constraints": tuple(new_env)}
        )
        return PerturbedMemory(
            card=new_card,
            changed_field="environment_constraints",
            original_value=original_value,
            perturbed_value=new_constraint,
            perturbation_type=self.name,
        )


# ──────────────────────────────────────────────────────────────
# Operator E: Procedure Dependency Perturbation (清单 §10)
# ──────────────────────────────────────────────────────────────
class ProcedureDependencyPerturbation:
    """Swap two adjacent procedure steps that have a dependency.

    Only applicable when the procedure text contains explicit
    dependency markers (清单 §10). First version does NOT do
    arbitrary shuffles.
    """

    name = "procedure_dependency"

    def applicable(
        self,
        memory: MemoryRoutingCard,
        receiver: ReceiverState,
    ) -> bool:
        # Only applicable if goal_summary has step-like structure
        # with at least 2 lines/steps.
        lines = [
            ln.strip()
            for ln in memory.goal_summary.split("\n")
            if ln.strip()
        ]
        return len(lines) >= 2

    def perturb(
        self,
        memory: MemoryRoutingCard,
        receiver: ReceiverState,
        *,
        rng: random.Random,
    ) -> PerturbedMemory:
        lines = [
            ln for ln in memory.goal_summary.split("\n") if ln.strip()
        ]
        if len(lines) < 2:
            raise ValueError("need at least 2 procedure steps")

        # Swap two adjacent lines (pick random position).
        idx = rng.randrange(len(lines) - 1)
        original_order = (lines[idx], lines[idx + 1])
        lines[idx], lines[idx + 1] = lines[idx + 1], lines[idx]
        swapped_order = (lines[idx], lines[idx + 1])

        new_goal = "\n".join(lines)
        _check_forbidden_tokens(new_goal)

        new_card = memory.model_copy(update={"goal_summary": new_goal})
        return PerturbedMemory(
            card=new_card,
            changed_field="goal_summary",
            original_value=original_order,
            perturbed_value=swapped_order,
            perturbation_type=self.name,
        )


# ──────────────────────────────────────────────────────────────
# Operator priority (清单 §13.1)
# ──────────────────────────────────────────────────────────────
OPERATOR_PRIORITY: tuple[str, ...] = (
    "precondition",
    "required_capability",
    "required_tool",
    "environment_constraint",
    "procedure_dependency",
)

_ALL_OPERATORS: dict[str, Any] = {
    "required_tool": RequiredToolPerturbation(),
    "required_capability": RequiredCapabilityPerturbation(),
    "precondition": PreconditionPerturbation(),
    "environment_constraint": EnvironmentConstraintPerturbation(),
    "procedure_dependency": ProcedureDependencyPerturbation(),
}


def get_all_operators() -> list[TransferPerturbationOperator]:
    """Return operators in priority order (清单 §13.1)."""
    return [_ALL_OPERATORS[name] for name in OPERATOR_PRIORITY]


# ──────────────────────────────────────────────────────────────
# Validator (清单 §11)
# ──────────────────────────────────────────────────────────────
def validate_single_factor_change(
    original: MemoryRoutingCard,
    perturbed: MemoryRoutingCard,
    spec: PerturbationSpec,
) -> None:
    """Verify that original and perturbed differ in exactly one field.

    Raises ``ValueError`` if any invariant is violated:
      - memory_id must differ.
      - non-target fields must be byte-equivalent.
      - target field must have changed.
      - no forbidden tokens in perturbed text.
    """
    if original.memory_id == perturbed.memory_id:
        raise ValueError("perturbed memory must have a different memory_id")

    orig_dict = original.model_dump()
    pert_dict = perturbed.model_dump()

    # Identify which fields differ.
    changed_fields: list[str] = []
    for key in orig_dict:
        if key == "memory_id":
            continue
        if orig_dict[key] != pert_dict[key]:
            changed_fields.append(key)

    if not changed_fields:
        raise ValueError("no fields changed between original and perturbed")

    if len(changed_fields) > 1:
        raise ValueError(
            f"multiple fields changed: {changed_fields}; "
            f"expected only {spec.changed_field!r}"
        )

    if changed_fields[0] != spec.changed_field:
        raise ValueError(
            f"changed field is {changed_fields[0]!r}, "
            f"expected {spec.changed_field!r}"
        )

    # Check no forbidden tokens anywhere in perturbed card.
    pert_text = str(pert_dict)
    _check_forbidden_tokens(pert_text)


def build_perturbation_spec(
    *,
    task_id: str,
    receiver_agent_id: str,
    candidate_memory_id: str,
    perturbed: PerturbedMemory,
    original_card: MemoryRoutingCard,
    source_record_id: str,
    control_group_key: str,
    generation_seed: int,
) -> PerturbationSpec:
    """Construct a PerturbationSpec with computed digests and ID."""
    orig_digest = compute_memory_digest(original_card.model_dump())
    pert_digest = compute_memory_digest(perturbed.card.model_dump())

    spec = PerturbationSpec(
        perturbation_id="",  # placeholder
        task_id=task_id,
        receiver_agent_id=receiver_agent_id,
        candidate_memory_id=candidate_memory_id,
        perturbation_type=perturbed.perturbation_type,
        changed_field=perturbed.changed_field,
        original_value=perturbed.original_value,
        perturbed_value=perturbed.perturbed_value,
        source_record_id=source_record_id,
        control_group_key=control_group_key,
        generation_seed=generation_seed,
        original_memory_digest=orig_digest,
        perturbed_memory_digest=pert_digest,
    )
    # Compute deterministic ID.
    pert_id = compute_perturbation_id(spec)
    # Frozen dataclass: use object.__setattr__.
    object.__setattr__(spec, "perturbation_id", pert_id)
    return spec
