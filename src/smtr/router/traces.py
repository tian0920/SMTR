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
    ] = "baseline_router"
    policy_fingerprint: str | None = None
    tau_mean: float | None = None
    tau_lcb: float | None = None
    tau_ucb: float | None = None
    negative_risk_mean: float | None = None
    negative_risk_ucb: float | None = None
    low_support: bool | None = None
    behavior_probability_share: float | None = None
    decision_mode: Literal[
        "safe_exploit",
        "boundary_explore",
        "risk_veto",
        "hard_ood_veto",
        "budget_exhausted",
        "ordinary_withhold",
        "fixed_prefix",
        "forced_intervention",
        "baseline_router",
    ] | None = None
    exploration_eligible: bool | None = None
    exploration_selected: bool | None = None
    support_distance: float | None = None
    support_threshold: float | None = None

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
