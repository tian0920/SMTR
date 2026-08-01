from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, ConfigDict


AgentRole = Literal[
    "planner",
    "executor",
    "critic",
    "verifier",
    "coordinator",
    "unknown",
]


class AgentProfile(BaseModel):
    model_config = ConfigDict(frozen=True)

    agent_id: str
    role: AgentRole
    capabilities: tuple[str, ...] = ()
    model_name: str | None = None
    tool_names: tuple[str, ...] = ()


class ReceiverState(BaseModel):
    model_config = ConfigDict(frozen=True)

    task_id: str
    scenario: str
    task_instruction: str
    receiver: AgentProfile

    subtask: str | None = None
    local_context_summary: str = ""
    team_context_summary: str = ""
    environment_signature: tuple[str, ...] = ()
    selected_memory_ids: tuple[str, ...] = ()


class ProcedurePayload(BaseModel):
    model_config = ConfigDict(frozen=True)

    memory_id: str
    procedure: str
    preconditions: tuple[str, ...] = ()
    postconditions: tuple[str, ...] = ()

    writer: AgentProfile
    source_task_id: str
    source_scenario: str
    version: str = "v1"


class MemoryRoutingCard(BaseModel):
    model_config = ConfigDict(frozen=True)

    memory_id: str
    goal_summary: str
    task_tags: tuple[str, ...] = ()
    environment_constraints: tuple[str, ...] = ()

    positive_transfer_hints: tuple[str, ...] = ()
    negative_transfer_hints: tuple[str, ...] = ()

    writer: AgentProfile
    source_task_id: str
    source_scenario: str

    compatible_receiver_roles: tuple[AgentRole, ...] = ()
    incompatible_receiver_roles: tuple[AgentRole, ...] = ()

    evidence_count: int = 0


class CandidateExposureInput(BaseModel):
    model_config = ConfigDict(frozen=True)

    receiver_state: ReceiverState
    candidate_card: MemoryRoutingCard
    selected_prefix_cards: tuple[MemoryRoutingCard, ...] = ()


class PairedTransferOutcome(BaseModel):
    model_config = ConfigDict(frozen=True)

    y_share_team: bool
    y_withhold_team: bool
    y_share_local: bool | None = None
    y_withhold_local: bool | None = None

    @property
    def four_outcome_label(self) -> str:
        if self.y_share_team and not self.y_withhold_team:
            return "positive_transfer"
        if not self.y_share_team and self.y_withhold_team:
            return "negative_transfer"
        if self.y_share_team and self.y_withhold_team:
            return "neutral_success"
        return "neutral_failure"


class TransferPrediction(BaseModel):
    model_config = ConfigDict(frozen=True)

    q00_neutral_failure: float
    q01_negative_transfer: float
    q10_positive_transfer: float
    q11_neutral_success: float

    @property
    def tau_hat(self) -> float:
        return self.q10_positive_transfer - self.q01_negative_transfer

    @property
    def eta_hat(self) -> float:
        return self.q01_negative_transfer


class RouterDecision(BaseModel):
    model_config = ConfigDict(frozen=True)

    memory_id: str
    action: Literal["share", "withhold"]
    tau_hat: float
    eta_hat: float
    reason: str
