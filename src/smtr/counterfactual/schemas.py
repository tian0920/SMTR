from datetime import datetime
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

from smtr.memory.schemas import ContextFingerprint, utc_now
from smtr.memory.snapshot import MemoryStoreSnapshot
from smtr.router.candidate_proposer import CandidateProposal


class DecisionPoint(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: str = "1.0"
    episode_id: str
    task_id: str
    graph_node: str
    receiver_agent_id: str
    receiver_role: str
    task_stage: str
    graph_state_snapshot: dict[str, Any]
    environment_snapshot: dict[str, Any]
    candidate_proposal: CandidateProposal
    memory_store_snapshot: MemoryStoreSnapshot
    run_seed: int
    capture_index: int
    snapshot_digest: str


class RuntimeSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True)

    graph_state: dict[str, Any]
    environment_snapshot: dict[str, Any]
    graph_node: str
    receiver_agent_id: str
    run_seed: int
    random_state: dict[str, Any]
    memory_store_revision: int
    snapshot_digest: str


class CandidateTraversalPlan(BaseModel):
    model_config = ConfigDict(frozen=True)

    candidate_order: list[str]
    target_index: int
    target_memory_id: str
    selected_before: list[str] = Field(default_factory=list)
    selected_before_positions: list[int] = Field(default_factory=list)
    traversal_seed: int
    target_selection_seed: int = 0
    prefix_sampling_seed: int = 0
    target_selection_policy_name: str = "legacy"
    target_selection_policy_version: str = "1"
    prefix_sampling_policy_name: str = "legacy_empty"
    prefix_sampling_policy_version: str = "1"
    target_selection_probability: float | None = None
    prefix_sampling_probability: float | None = None

    @model_validator(mode="after")
    def validate_plan(self) -> "CandidateTraversalPlan":
        if not self.selected_before_positions and self.selected_before:
            unknown = set(self.selected_before) - set(self.candidate_order)
            if unknown:
                raise ValueError(
                    f"selected_before contains non-candidate IDs: {sorted(unknown)}"
                )
            object.__setattr__(
                self,
                "selected_before_positions",
                [self.candidate_order.index(memory_id) for memory_id in self.selected_before],
            )
        validate_selection_prefix(self)
        if self.candidate_order[self.target_index] != self.target_memory_id:
            raise ValueError("target_memory_id must match candidate_order[target_index]")
        return self


def validate_selection_prefix(plan: CandidateTraversalPlan) -> None:
    if len(set(plan.candidate_order)) != len(plan.candidate_order):
        raise ValueError("candidate_order must not contain duplicate IDs")
    if not 0 <= plan.target_index < len(plan.candidate_order):
        raise ValueError("target_index is outside candidate_order")
    if len(set(plan.selected_before)) != len(plan.selected_before):
        raise ValueError("selected_before must not contain duplicate IDs")
    if len(plan.selected_before_positions) != len(plan.selected_before):
        raise ValueError("selected_before_positions must align with selected_before")
    if plan.target_memory_id in plan.selected_before:
        raise ValueError("selected_before must not contain target memory")
    candidates = set(plan.candidate_order)
    unknown = set(plan.selected_before) - candidates
    if unknown:
        raise ValueError(f"selected_before contains non-candidate IDs: {sorted(unknown)}")
    allowed_prefix = set(plan.candidate_order[: plan.target_index])
    after_target = set(plan.selected_before) - allowed_prefix
    if after_target:
        raise ValueError(
            "selected_before must only contain memories before target_index: "
            f"{sorted(after_target)}"
        )
    expected_positions = [
        plan.candidate_order.index(memory_id) for memory_id in plan.selected_before
    ]
    if plan.selected_before_positions != expected_positions:
        raise ValueError("selected_before_positions must match candidate traversal order")
    if any(position >= plan.target_index for position in plan.selected_before_positions):
        raise ValueError("selected_before_positions must be before target_index")


class RoutingFeatureSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True)

    memory_id: str
    active_payload_version: int
    goal_summary: str
    task_tags: list[str] = Field(default_factory=list)
    precondition_summary: str = ""
    postcondition_summary: str = ""
    required_environment_facts: dict[str, str | bool | int | float] = Field(default_factory=dict)
    forbidden_environment_facts: dict[str, str | bool | int | float] = Field(default_factory=dict)
    compatible_receiver_roles: list[str] = Field(default_factory=list)
    compatible_receiver_capabilities: list[str] = Field(default_factory=list)
    execution_success_alpha: float = 1.0
    execution_success_beta: float = 1.0
    execution_success_count: int = 0
    execution_failure_count: int = 0
    paired_positive_transfer_count: int = 0
    paired_negative_transfer_count: int = 0
    paired_neutral_transfer_count: int = 0
    card_schema_version: str = "1.0"


class BranchOutcome(BaseModel):
    model_config = ConfigDict(frozen=True)

    team_success: bool
    team_reward: float
    team_summary: str
    final_environment_observation: dict[str, Any]
    selected_memory_ids_by_agent: dict[str, list[str]]
    router_trace: list[dict[str, Any]]
    target_memory_visible_to_receiver: bool
    selected_final_at_target_node: list[str]


class EvaluationGroupMetadata(BaseModel):
    model_config = ConfigDict(frozen=True)

    scenario_family: str = "unknown"
    environment_regime: str = "unknown"
    target_memory_family: str = "unknown"
    prefix_structure_family: str = "unknown"
    factor_combination_id: str = "unknown"
    surface_variant_id: str = "default"
    mechanism_group_id: str = "unknown"


class ContinuationBehaviorMetadata(BaseModel):
    model_config = ConfigDict(frozen=True)

    policy_fingerprint: str = ""
    policy_kind: str = "frozen_no_share"
    node_seed: int = 0
    invocation_index: int = 0
    max_total_shares_per_invocation: int = 0
    max_exploratory_shares_per_invocation: int = 0
    exploration_round_probability: float = 0.0
    exploration_eligible_count: int = 0
    exploration_target_memory_id: str | None = None
    forced_intervention: bool = False


class PairedInterventionRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: str = "1.3"
    record_id: str = Field(default_factory=lambda: str(uuid4()))
    episode_id: str
    task_id: str
    graph_node: str
    receiver_agent_id: str
    receiver_role: str
    task_stage: str
    collection_round_id: str | None = None
    continuation_policy_fingerprint: str | None = None
    base_memory_store_revision: int | None = None
    base_memory_snapshot_digest: str | None = None
    evaluation_group_metadata: EvaluationGroupMetadata = Field(
        default_factory=EvaluationGroupMetadata
    )
    continuation_behavior_metadata: list[ContinuationBehaviorMetadata] = Field(
        default_factory=list
    )
    migrated_from_schema: str | None = None
    candidate_memory_id: str
    candidate_payload_version: int
    candidate_card_snapshot: RoutingFeatureSnapshot | None = None
    selected_before_card_snapshots: list[RoutingFeatureSnapshot] = Field(default_factory=list)
    selected_before_payload_versions: dict[str, int] = Field(default_factory=dict)
    candidate_order: list[str]
    target_index: int
    selected_before: list[str]
    prefix_size: int = 0
    target_selection_policy_name: str = "legacy"
    target_selection_policy_version: str = "1"
    prefix_sampling_policy_name: str = "legacy_empty"
    prefix_sampling_policy_version: str = "1"
    target_selection_probability: float | None = None
    prefix_sampling_probability: float | None = None
    decision_context: ContextFingerprint
    memory_store_revision: int
    memory_snapshot_digest: str
    runtime_snapshot_digest: str
    continuation_policy_name: str
    continuation_policy_version: str
    common_seed: int
    share_outcome: BranchOutcome
    withhold_outcome: BranchOutcome
    y_share: int
    y_withhold: int
    transfer_class: Literal[
        "positive",
        "negative",
        "neutral_success",
        "neutral_failure",
    ]
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_transfer_class(self) -> "PairedInterventionRecord":
        expected = transfer_class_from_outcomes(self.y_share, self.y_withhold)
        if self.transfer_class != expected:
            raise ValueError(
                f"transfer_class must be {expected} for ({self.y_share}, {self.y_withhold})"
            )
        if self.candidate_card_snapshot is not None:
            if self.candidate_card_snapshot.memory_id != self.candidate_memory_id:
                raise ValueError("candidate_card_snapshot does not match candidate_memory_id")
            if (
                self.candidate_card_snapshot.active_payload_version
                != self.candidate_payload_version
            ):
                raise ValueError("candidate_card_snapshot version mismatch")
        snapshot_ids = [snapshot.memory_id for snapshot in self.selected_before_card_snapshots]
        if snapshot_ids != self.selected_before:
            raise ValueError("selected_before_card_snapshots must align with selected_before")
        return self

    @computed_field
    @property
    def marginal_effect(self) -> int:
        return self.y_share - self.y_withhold

    @computed_field
    @property
    def negative_transfer_indicator(self) -> int:
        return int(self.y_share == 0 and self.y_withhold == 1)


def transfer_class_from_outcomes(
    y_share: int,
    y_withhold: int,
) -> Literal["positive", "negative", "neutral_success", "neutral_failure"]:
    if y_share == 1 and y_withhold == 0:
        return "positive"
    if y_share == 0 and y_withhold == 1:
        return "negative"
    if y_share == 1 and y_withhold == 1:
        return "neutral_success"
    return "neutral_failure"


def routing_feature_snapshot_from_card(card) -> RoutingFeatureSnapshot:
    return RoutingFeatureSnapshot(
        memory_id=card.memory_id,
        active_payload_version=card.active_payload_version,
        goal_summary=card.goal_summary,
        task_tags=list(card.task_tags),
        precondition_summary=card.precondition_summary,
        postcondition_summary=card.postcondition_summary,
        required_environment_facts=dict(card.required_environment_facts),
        forbidden_environment_facts=dict(card.forbidden_environment_facts),
        compatible_receiver_roles=list(card.compatible_receiver_roles),
        compatible_receiver_capabilities=list(card.compatible_receiver_capabilities),
        execution_success_alpha=card.execution_success_alpha,
        execution_success_beta=card.execution_success_beta,
        execution_success_count=card.execution_success_count,
        execution_failure_count=card.execution_failure_count,
        paired_positive_transfer_count=card.paired_positive_transfer_count,
        paired_negative_transfer_count=card.paired_negative_transfer_count,
        paired_neutral_transfer_count=card.paired_neutral_transfer_count,
    )
