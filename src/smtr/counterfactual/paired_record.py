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


SHARED_CONTROL_SCHEMA_VERSION = "marble_candidate_pair_v3"

SHARED_CONTROL_DEFINITION_VERSION = "shared_no_memory_control_v1"


class PairedDigests(BaseModel):
    model_config = ConfigDict(frozen=True)

    share_initial_digest: str
    withhold_initial_digest: str
    task_digest: str
    tool_config_digest: str

    # Shared-control digests (清单 Shared-Control 第7.5节). They alias the
    # control branch's execution context and are identical across every
    # record of one control group.
    control_group_context_digest: str | None = None
    control_raw_result_digest: str | None = None
    control_initial_digest: str | None = None
    control_agent_config_digest: str | None = None
    control_task_digest: str | None = None
    control_tool_config_digest: str | None = None


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

    # Target vs memory-source provenance (R6 清单 P0-1): the target
    # trajectory is the receiver's execution of the target task, while
    # memory_source_* point back to the train trajectory the candidate
    # memory was extracted from. Legacy artifacts that only persisted
    # ``source_trajectory_id`` are accepted via alias.
    target_trajectory_id: str | None = None
    memory_source_task_id: str | None = None
    memory_source_trajectory_id: str | None = None
    memory_source_split: str | None = None

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

    # Shared-control provenance (清单 Shared-Control 第7章). Present on
    # marble_candidate_pair_v3 records; the withhold block holds the
    # canonical outcome of the group's one shared no-memory control.
    control_group_id: str | None = None
    control_family_id: str | None = None
    control_reused: bool | None = None
    control_definition_version: str | None = None
    control_group_candidate_count: int | None = None
    control_execution_position: str | None = None
    share_execution_rank: int | None = None
    control_artifact_path: str | None = None
    control_raw_result_digest: str | None = None
    candidate_source: str | None = None
    candidate_sources: tuple[str, ...] = ()
    anchor_group_id: str | None = None
    match_type: str | None = None

    digests: PairedDigests

    @model_validator(mode="before")
    @classmethod
    def _accept_legacy_source_trajectory(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if data.get("memory_source_trajectory_id") is None and data.get(
                "source_trajectory_id"
            ) is not None:
                data = dict(data)
                data["memory_source_trajectory_id"] = data["source_trajectory_id"]
                data.pop("source_trajectory_id", None)
        return data

    @model_validator(mode="after")
    def check_label_consistency(self) -> CandidateLevelPairedRecord:
        expected = _compute_label(self.share.team_success, self.withhold.team_success)
        if self.label != expected:
            raise ValueError(f"label {self.label} inconsistent with outcomes (expected {expected})")
        return self

    @model_validator(mode="after")
    def check_shared_control_provenance(self) -> CandidateLevelPairedRecord:
        if self.schema_version != SHARED_CONTROL_SCHEMA_VERSION:
            return self
        if not self.control_group_id:
            raise ValueError("v3 paired record requires control_group_id")
        expected_family = f"{self.task_id}::{self.receiver_agent_id}"
        if self.control_family_id != expected_family:
            raise ValueError(
                f"control_family_id {self.control_family_id!r} does not "
                f"match task::receiver identity {expected_family!r}"
            )
        if self.control_reused is not True:
            raise ValueError("v3 paired record requires control_reused=true")
        if self.control_definition_version != SHARED_CONTROL_DEFINITION_VERSION:
            raise ValueError(
                "v3 paired record requires control_definition_version "
                f"{SHARED_CONTROL_DEFINITION_VERSION!r}"
            )
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


# 清单 4.2: formal paired records must persist complete provenance; a
# missing or empty field fails the audit closed instead of being ignored.
FORMAL_PAIRED_PROVENANCE_FIELDS = {
    "task_id",
    "target_trajectory_id",
    "receiver_agent_id",
    "candidate_memory_id",
    "memory_source_trajectory_id",
    "memory_source_task_id",
    "memory_source_split",
    "generation_seed",
}

# 清单 Shared-Control 第7章: v3 records must additionally persist complete
# shared-control provenance.
FORMAL_SHARED_CONTROL_FIELDS = {
    "control_group_id",
    "control_family_id",
    "control_definition_version",
    "control_artifact_path",
    "control_raw_result_digest",
}


def validate_formal_paired_provenance(
    record: dict[str, Any],
    *,
    record_index: int,
) -> list[str]:
    """Fail-closed provenance schema check for one record (清单 4.3).

    Returns one error per missing (None) or empty-string required field;
    an empty list means the record carries full formal provenance.
    """
    errors: list[str] = []

    for field in FORMAL_PAIRED_PROVENANCE_FIELDS:
        value = record.get(field)

        if value is None:
            errors.append(
                f"record[{record_index}] missing "
                f"required provenance field {field!r}"
            )
            continue

        if isinstance(value, str) and not value.strip():
            errors.append(
                f"record[{record_index}] has empty "
                f"provenance field {field!r}"
            )

    if record.get("schema_version") == SHARED_CONTROL_SCHEMA_VERSION:
        for field in FORMAL_SHARED_CONTROL_FIELDS:
            value = record.get(field)
            if value is None or (isinstance(value, str) and not value.strip()):
                errors.append(
                    f"record[{record_index}] missing shared-control "
                    f"provenance field {field!r}"
                )
        if record.get("control_reused") is not True:
            errors.append(
                f"record[{record_index}] requires control_reused=true"
            )

    return errors
