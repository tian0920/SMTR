"""Schemas for SMTR ablation experiment runner."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Valid method IDs for the ablation experiment
VALID_METHOD_IDS = frozenset({
    "B0",
    "B1-Top1",
    "B1-AllCandidates",
    "B1-Matched",
    "SMTR",
    "EffectOnly-SMTR",
    "Static-SMTR",
    "FactualSuccess-SMTR",
})

# Mapping from method_id to internal registry key
METHOD_ID_TO_REGISTRY = {
    "B0": "b0_no_memory",
    "B1-Top1": "b1_top1",
    "B1-AllCandidates": "b1_all_candidates",
    "B1-Matched": "b1_matched",
    "SMTR": "smtr",
    "EffectOnly-SMTR": "effect_only_smtr",
    "Static-SMTR": "static_smtr",
    "FactualSuccess-SMTR": "factual_success_smtr",
}


class ExperimentConfig(BaseModel):
    """Configuration for a comparison experiment."""

    model_config = ConfigDict(frozen=True)

    db_path: str
    critic_checkpoint: str | None = None
    task_seeds: list[int] = Field(default_factory=lambda: [0])
    generation_seeds: list[int] = Field(default_factory=lambda: [0])
    traversal_seeds: list[int] = Field(default_factory=lambda: [0, 1, 2])
    scenario_replicates: int = 1
    top_k: int = 4
    max_shares_per_invocation: int | None = 3
    negative_risk_budget: float = 0.2
    output_dir: str
    overwrite: bool = False
    fail_fast: bool = True
    bootstrap_seed: int = 42
    bootstrap_n: int = 2000
    # Counterfactual scenario name (None = default toy task)
    scenario: str | None = None
    # Methods to run (None = default B0/B1/M0 for backward compat)
    methods: list[str] | None = None
    enable_ablation_methods: bool = False
    # A1 critic checkpoint (if different from M0)
    a1_critic_checkpoint: str | None = None
    factual_success_checkpoint: str | None = None
    factual_success_threshold: float | None = None
    # B1-Matched budget manifest path
    budget_manifest_path: str | None = None
    explicit_permutation: list[int] | None = None
    permutation_id: str | None = None
    permutation_application_policy: str | None = None

    @model_validator(mode="after")
    def validate_config(self) -> "ExperimentConfig":
        if self.scenario_replicates < 1:
            raise ValueError("scenario_replicates must be >= 1")
        if not self.task_seeds:
            raise ValueError("task_seeds must not be empty")
        if not self.generation_seeds:
            raise ValueError("generation_seeds must not be empty")
        if not self.traversal_seeds:
            raise ValueError("traversal_seeds must not be empty")
        if not 0.0 <= self.negative_risk_budget <= 1.0:
            raise ValueError("negative_risk_budget must be in [0, 1]")
        return self


class DecisionRecord(BaseModel):
    """A single router decision within one invocation."""

    model_config = ConfigDict(frozen=True)

    decision_index: int
    memory_id: str
    action: Literal["share", "withhold"]
    reason: str
    proposal_rank: int | None = None
    proposal_score: float | None = None
    traversal_position: int
    selected_before_memory_ids: list[str] = Field(default_factory=list)
    selected_before_digest: str
    selected_before_actual: list[str] = Field(default_factory=list)
    selected_before_critic: list[str] = Field(default_factory=list)
    selected_before_actual_digest: str | None = None
    selected_before_critic_digest: str | None = None
    tau_mean: float | None = None
    tau_lcb: float | None = None
    tau_ucb: float | None = None
    negative_risk_mean: float | None = None
    negative_risk_lcb: float | None = None
    negative_risk_ucb: float | None = None
    robust_diagnostics: dict[str, float] | None = None
    support_distance: float | None = None
    gate_name: str | None = None
    conditioning_policy_name: str | None = None
    effect_condition_passed: bool | None = None
    risk_condition_passed: bool | None = None
    effect_condition_status: str | None = None
    risk_condition_status: str | None = None
    paired_record_id: str | None = None
    true_transfer_class: str | None = None
    true_tau: float | None = None


class RoutingInvocationRecord(BaseModel):
    """A complete routing invocation at one graph node."""

    model_config = ConfigDict(frozen=True)

    invocation_id: str
    graph_node: str
    receiver_agent_id: str
    receiver_role: str
    context_fingerprint_digest: str
    candidate_request_digest: str
    candidate_memory_ids: list[str] = Field(default_factory=list)
    candidate_scores: list[float] = Field(default_factory=list)
    proposal_order: list[str] = Field(default_factory=list)
    traversal_order: list[str] = Field(default_factory=list)
    traversal_policy_name: str | None = None
    traversal_seed: int | None = None
    permutation_indices: list[int] = Field(default_factory=list)
    decisions: list[DecisionRecord] = Field(default_factory=list)
    selected_memory_ids: list[str] = Field(default_factory=list)
    visible_payload_memory_ids: list[str] = Field(default_factory=list)


class BaseEpisodeManifestRecord(BaseModel):
    """Stable identity for a base episode."""

    model_config = ConfigDict(frozen=True)

    base_episode_id: str
    scenario: str | None = None
    task_seed: int
    task_spec_digest: str
    generation_seed: int
    replicate_index: int
    initial_graph_state_digest: str
    initial_environment_digest: str
    memory_snapshot_id: str
    memory_snapshot_digest: str


class ComparisonRunRecord(BaseModel):
    """A single run record from the comparison experiment."""

    model_config = ConfigDict(frozen=True)

    experiment_id: str
    base_episode_id: str
    episode_id: str
    task_instance_id: str
    method: str
    router_name: str
    scenario: str | None = None
    task_description: str | None = None
    task_seed: int
    environment_seed: int
    generation_seed: int
    replicate_index: int = 0
    traversal_seed: int | None = None
    permutation_id: str | None = None
    permutation_indices: list[int] = Field(default_factory=list)
    permutation_application_policy: str | None = None
    memory_snapshot_id: str
    memory_snapshot_digest: str = ""
    environment_snapshot_digest: str
    team_success: bool = False
    policy_level_transfer_label: str | None = None
    runtime_seconds: float = 0.0
    invocations: list[RoutingInvocationRecord] = Field(default_factory=list)
    unique_selected_memory_ids: list[str] = Field(default_factory=list)
    total_memory_exposures: int = 0
    mean_selected_per_invocation: float = 0.0
    number_of_invocations: int = 0
    all_withhold: bool = True
    # Legacy-derived fields retained only for migration/report readers.
    candidate_memory_ids: list[str] = Field(default_factory=list)
    selected_memory_ids: list[str] = Field(default_factory=list)
    selected_count: int = 0
    router_trace: list[dict[str, Any]] = Field(default_factory=list)


class MethodSummary(BaseModel):
    """Summary statistics for a single method."""

    model_config = ConfigDict(frozen=True)

    method: str
    episode_count: int = 0
    success_rate: float = 0.0
    avg_selected_size: float = 0.0
    median_selected_size: float = 0.0
    all_withhold_rate: float = 0.0
    avg_candidate_count: float = 0.0
    mean_runtime: float = 0.0
    # Transfer metrics (null for B0)
    positive_transfer_rate: float | None = None
    negative_transfer_rate: float | None = None
    neutral_success_rate: float | None = None
    neutral_failure_rate: float | None = None
    success_delta_vs_b0: float | None = None
    # M0 rejection metrics (null for B0, B1)
    share_decision_rate: float | None = None
    tau_mean_rejection_rate: float | None = None
    negative_risk_mean_rejection_rate: float | None = None
    tau_lcb_rejection_rate: float | None = None
    negative_risk_ucb_rejection_rate: float | None = None
    confidence_level: float | None = None
    share_budget_rejection_rate: float | None = None
    low_support_rejection_rate: float | None = None
    effect_condition_pass_rate: float | None = None
    effect_condition_rejection_rate: float | None = None
    risk_condition_pass_rate: float | None = None
    risk_condition_rejection_rate: float | None = None
    opportunity_capture: float | None = None
    safety_preservation: float | None = None
    n_positive_transfer_opportunities: int = 0
    n_negative_transfer_opportunities: int = 0
    mean_exposure_per_invocation: float | None = None
    total_exposure_per_episode: float | None = None
    payload_token_count: int | None = None
    mean_payload_tokens_per_invocation: float | None = None
    all_candidates_shared_rate: float | None = None
    selected_set_conditioning_divergence_rate: float | None = None
    other_reason_counts: dict[str, int] = Field(default_factory=dict)


class ExperimentSummary(BaseModel):
    """Full experiment summary with all methods and comparisons."""

    model_config = ConfigDict(frozen=True)

    methods: dict[str, MethodSummary] = Field(default_factory=dict)
    bootstrap_ci: dict[str, Any] = Field(default_factory=dict)
    n_base_episodes: int = 0
    n_traversal_runs: int = 0
    n_runtime_executions_by_method: dict[str, int] = Field(default_factory=dict)
    experiment_valid: bool = True
    invalid_reason: str | None = None
    infrastructure_failure_rate: float = 0.0
