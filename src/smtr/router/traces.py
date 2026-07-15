from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class CandidateTrace(BaseModel):
    model_config = ConfigDict(frozen=True)

    memory_id: str
    total_score: float = 0.0
    goal_similarity: float = 0.0
    task_tag_overlap: float = 0.0
    environment_compatibility: float = 0.0
    receiver_compatibility: float = 0.0
    explicit_environment_conflict: bool = False
    score_explanation: list[str] = Field(default_factory=list)

    @property
    def score(self) -> float:
        return self.total_score

    @property
    def environment_overlap(self) -> float:
        return self.environment_compatibility

    @property
    def receiver_role_match(self) -> float:
        return self.receiver_compatibility


class RouterDecision(BaseModel):
    model_config = ConfigDict(frozen=True)

    memory_id: str
    action: Literal["share", "withhold"] = "withhold"
    decision: Literal["share", "withhold"] = "withhold"
    score: float = 0.0
    reason: str
    candidate_position: int | None = None
    decision_source: Literal[
        "fixed_prefix",
        "forced_intervention",
        "frozen_continuation",
        "baseline_router",
        "production_router",
        "relevance_topk_router",
    ] = "baseline_router"
    policy_fingerprint: str | None = None
    tau_mean: float | None = None
    tau_lcb: float | None = None
    tau_ucb: float | None = None
    negative_risk_mean: float | None = None
    negative_risk_lcb: float | None = None
    negative_risk_ucb: float | None = None
    epsilon: float | None = None
    accepted: bool | None = None
    decision_reason: str | None = None
    low_support: bool | None = None
    behavior_probability_share: float | None = None
    decision_mode: str | None = None
    gate_name: str | None = None
    conditioning_policy_name: str | None = None
    effect_condition_passed: bool | None = None
    risk_condition_passed: bool | None = None
    effect_condition_status: str | None = None
    risk_condition_status: str | None = None
    exploration_eligible: bool | None = None
    exploration_selected: bool | None = None
    selected_before_actual: list[str] = Field(default_factory=list)
    selected_before_critic: list[str] = Field(default_factory=list)
    selected_before_actual_digest: str | None = None
    selected_before_critic_digest: str | None = None
    support_distance: float | None = None
    support_threshold: float | None = None
    robust_diagnostics: dict[str, float] | None = None
    original_candidate_position: int | None = None
    traversal_position: int | None = None
    traversal_seed: int | None = None
    traversal_policy_name: str | None = None
    proposal_order: list[str] | None = None
    traversal_order: list[str] | None = None
    permutation_indices: list[int] | None = None
    proposal_rank: int | None = None
    """1-based rank in the proposer's relevance ranking."""
    proposal_score: float | None = None
    """Relevance score from the proposer."""

    @model_validator(mode="after")
    def sync_action_and_decision(self) -> "RouterDecision":
        if self.action != self.decision:
            raise ValueError("action and decision must match")
        return self


class RouterTraceEntry(BaseModel):
    model_config = ConfigDict(frozen=True)

    agent: str
    receiver_agent_id: str
    task: str
    task_stage: str
    seed: int
    memory_store_revision: int
    proposer_name: str
    proposer_version: str
    router_name: str
    router_version: str
    candidates: list[CandidateTrace]
    candidate_scores: dict[str, float]
    decisions: list[RouterDecision]
    selected_memory_ids: list[str]
    traversal_seed: int | None = None
    traversal_policy_name: str | None = None
    proposal_order: list[str] = Field(default_factory=list)
    traversal_order: list[str] = Field(default_factory=list)
    permutation_indices: list[int] = Field(default_factory=list)
    graph_node: str | None = None
    receiver_role: str | None = None
    context_fingerprint_digest: str | None = None
    candidate_request_digest: str | None = None
    visible_payload_memory_ids: list[str] = Field(default_factory=list)
