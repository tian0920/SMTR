from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

FactValue = str | bool | int | float


def utc_now() -> datetime:
    return datetime.now(UTC)


class ProcedurePayload(BaseModel):
    model_config = ConfigDict(frozen=True)

    memory_id: str
    version: int = 1
    writer_agent_id: str | None = None
    source_episode_id: str | None = None
    goal: str
    preconditions: list[str] = Field(default_factory=list)
    steps: list[str] = Field(default_factory=list)
    postconditions: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)
    parent_memory_ids: list[str] = Field(default_factory=list)


class ContextFingerprint(BaseModel):
    model_config = ConfigDict(frozen=True)

    task_id: str
    task_tags: list[str] = Field(default_factory=list)
    receiver_agent_id: str
    receiver_role: str
    receiver_capabilities: list[str] = Field(default_factory=list)
    environment_facts: dict[str, FactValue] = Field(default_factory=dict)
    task_stage: str
    selected_memory_ids: list[str] = Field(default_factory=list)
    selected_set_signature: str
    episode_id: str
    decision_index: int | None = None


class ExecutionEvidence(BaseModel):
    model_config = ConfigDict(frozen=True)

    memory_id: str
    payload_version: int
    context: ContextFingerprint
    execution_success: bool
    reward: float | None = None
    failure_category: Literal[
        "precondition_mismatch",
        "environment_constraint_mismatch",
        "invalid_action_sequence",
        "postcondition_not_reached",
        "unknown",
    ] | None = None
    source: Literal["direct_execution", "synthetic_toy"]
    timestamp: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_failure_category(self) -> "ExecutionEvidence":
        if self.execution_success and self.failure_category is not None:
            raise ValueError("failure_category must be None for successful execution evidence")
        if not self.execution_success and self.failure_category is None:
            raise ValueError("failure_category is required for failed execution evidence")
        return self


class MemoryRoutingCard(BaseModel):
    model_config = ConfigDict(frozen=True)

    memory_id: str
    active_payload_version: int = 1
    goal_summary: str
    task_tags: list[str] = Field(default_factory=list)
    precondition_summary: str = ""
    postcondition_summary: str = ""
    required_environment_facts: dict[str, FactValue] = Field(default_factory=dict)
    forbidden_environment_facts: dict[str, FactValue] = Field(default_factory=dict)
    compatible_receiver_roles: list[str] = Field(default_factory=list)
    compatible_receiver_capabilities: list[str] = Field(default_factory=list)
    execution_success_alpha: float = 1.0
    execution_success_beta: float = 1.0
    execution_success_count: int = 0
    execution_failure_count: int = 0
    execution_success_contexts: list[ContextFingerprint] = Field(default_factory=list)
    execution_failure_contexts: list[ContextFingerprint] = Field(default_factory=list)
    paired_positive_transfer_count: int = 0
    paired_negative_transfer_count: int = 0
    paired_neutral_transfer_count: int = 0
    paired_positive_transfer_contexts: list[ContextFingerprint] = Field(default_factory=list)
    paired_negative_transfer_contexts: list[ContextFingerprint] = Field(default_factory=list)
    paired_neutral_transfer_contexts: list[ContextFingerprint] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="before")
    @classmethod
    def accept_prompt0_fields(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        values = dict(data)
        if "version" in values and "active_payload_version" not in values:
            values["active_payload_version"] = values["version"]
        if "receiver_roles" in values and "compatible_receiver_roles" not in values:
            values["compatible_receiver_roles"] = values["receiver_roles"]
        if "environment_constraints" in values:
            constraints = values["environment_constraints"]
            if "task_tags" not in values:
                values["task_tags"] = constraints
            if "precondition_summary" not in values:
                values["precondition_summary"] = ", ".join(map(str, constraints))
        return values

    @property
    def version(self) -> int:
        return self.active_payload_version

    @property
    def receiver_roles(self) -> list[str]:
        return self.compatible_receiver_roles

    @property
    def environment_constraints(self) -> list[str]:
        facts = [f"{key}={value}" for key, value in self.required_environment_facts.items()]
        return [*self.task_tags, *facts]
