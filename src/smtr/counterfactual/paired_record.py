"""Candidate-level paired record schema for MARBLE interventions."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, model_validator

FourOutcomeLabel = Literal[
    "positive_transfer",
    "negative_transfer",
    "neutral_success",
    "neutral_failure",
]


class BranchOutcome(BaseModel):
    model_config = ConfigDict(frozen=True)

    team_success: bool
    local_success: bool | None = None
    environment_valid: bool = True
    native_evaluator_executed: bool = True


class PairedDigests(BaseModel):
    model_config = ConfigDict(frozen=True)

    share_initial_digest: str
    withhold_initial_digest: str
    task_digest: str
    tool_config_digest: str


class CandidateLevelPairedRecord(BaseModel):
    """One candidate-level share vs withhold intervention."""

    model_config = ConfigDict(frozen=True)

    record_type: Literal["marble_candidate_level_pair"] = "marble_candidate_level_pair"
    schema_version: str = "v1"
    scenario: str = "database"

    task_id: str
    receiver_agent_id: str
    receiver_role: str
    receiver_capabilities: tuple[str, ...] = ()

    candidate_memory_id: str
    writer_agent_id: str
    writer_role: str
    writer_capabilities: tuple[str, ...] = ()

    selected_prefix_memory_ids: tuple[str, ...] = ()
    candidate_rank: int = 0
    candidate_score: float = 0.0

    share: BranchOutcome
    withhold: BranchOutcome

    label: FourOutcomeLabel
    valid: bool = True
    invalid_reason: str | None = None

    digests: PairedDigests

    @model_validator(mode="after")
    def check_label_consistency(self) -> CandidateLevelPairedRecord:
        expected = _compute_label(self.share.team_success, self.withhold.team_success)
        if self.label != expected:
            raise ValueError(f"label {self.label} inconsistent with outcomes (expected {expected})")
        return self

    @model_validator(mode="after")
    def reject_payload_leakage(self) -> CandidateLevelPairedRecord:
        serialized = self.model_dump_json().lower()
        forbidden = ("payload", "procedure", "ordered_steps", "raw_action_sequence")
        for token in forbidden:
            if token in serialized:
                raise ValueError(f"paired record contains forbidden field: {token}")
        return self


def _compute_label(y_share: bool, y_withhold: bool) -> FourOutcomeLabel:
    if y_share and not y_withhold:
        return "positive_transfer"
    if not y_share and y_withhold:
        return "negative_transfer"
    if y_share and y_withhold:
        return "neutral_success"
    return "neutral_failure"
