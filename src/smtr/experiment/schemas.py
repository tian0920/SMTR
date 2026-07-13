"""Schemas for B0/B1/M0 comparison experiment runner."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


# Valid method IDs for the ablation experiment
VALID_METHOD_IDS = frozenset({
    "B0", "B1-Top1", "B1-Top3", "B1-Matched", "A1-NoSet", "M0-Full",
})

# Mapping from method_id to internal registry key
METHOD_ID_TO_REGISTRY = {
    "B0": "b0_no_memory",
    "B1-Top1": "b1_top1",
    "B1-Top3": "b1_top3",
    "B1-Matched": "b1_matched",
    "A1-NoSet": "a1_no_selected_set",
    "M0-Full": "m0_full",
}


class ExperimentConfig(BaseModel):
    """Configuration for a comparison experiment."""

    model_config = ConfigDict(frozen=True)

    db_path: str
    critic_checkpoint: str | None = None
    episodes: int = 20
    task_seeds: list[int] = Field(default_factory=lambda: [0])
    generation_seeds: list[int] = Field(default_factory=lambda: [0])
    traversal_seeds: list[int] = Field(default_factory=lambda: [0, 1, 2])
    top_k: int = 4
    max_shares_per_invocation: int | None = 3
    output_dir: str
    overwrite: bool = False
    bootstrap_seed: int = 42
    bootstrap_n: int = 1000
    # Counterfactual scenario name (None = default toy task)
    scenario: str | None = None
    # Methods to run (None = default B0/B1/M0 for backward compat)
    methods: list[str] | None = None
    # A1 critic checkpoint (if different from M0)
    a1_critic_checkpoint: str | None = None
    # B1-Matched budget manifest path
    budget_manifest_path: str | None = None


class ComparisonRunRecord(BaseModel):
    """A single run record from the comparison experiment."""

    model_config = ConfigDict(frozen=True)

    experiment_id: str
    episode_id: str
    task_instance_id: str
    method: str  # One of: B0, B1, B1-Top1, B1-Top3, B1-Matched, A1-NoSet, M0, M0-Full
    router_name: str
    scenario: str | None = None
    task_description: str | None = None
    task_seed: int
    environment_seed: int
    generation_seed: int
    traversal_seed: int | None = None
    memory_snapshot_id: str
    environment_snapshot_digest: str
    candidate_memory_ids: list[str] = Field(default_factory=list)
    selected_memory_ids: list[str] = Field(default_factory=list)
    selected_count: int = 0
    team_success: bool = False
    failure_reason: str | None = None
    policy_level_transfer_label: str | None = None
    runtime_seconds: float = 0.0
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
    tau_lcb_rejection_rate: float | None = None
    negative_risk_ucb_rejection_rate: float | None = None
    share_budget_rejection_rate: float | None = None
    low_support_rejection_rate: float | None = None
    no_critic_rejection_rate: float | None = None
    other_reason_counts: dict[str, int] = Field(default_factory=dict)


class ExperimentSummary(BaseModel):
    """Full experiment summary with all methods and comparisons."""

    model_config = ConfigDict(frozen=True)

    # Legacy fields (kept for backward compatibility)
    b0: MethodSummary = Field(default_factory=lambda: MethodSummary(method="B0"))
    b1: MethodSummary = Field(default_factory=lambda: MethodSummary(method="B1"))
    m0: MethodSummary = Field(default_factory=lambda: MethodSummary(method="M0"))
    m0_vs_b1: dict[str, float] = Field(default_factory=dict)
    # New: dict of all method summaries keyed by method display label
    methods: dict[str, MethodSummary] = Field(default_factory=dict)
    bootstrap_ci: dict[str, Any] = Field(default_factory=dict)
    experiment_invalid: bool = False
    invalid_reason: str | None = None
