"""Strict data contracts for real MARBLE database transfer evidence."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, model_validator

from smtr.counterfactual.decision_points import canonical_digest

SplitName = Literal["train", "validation", "test"]


class RealDatabaseTrajectory(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: str = "database_trajectory_v1"
    trajectory_id: str
    task_id: str
    split: SplitName
    generation_seed: int
    model_id: str
    source_dataset_version: str | None = None
    messages: list[dict[str, Any]]
    actions: list[dict[str, Any]]
    tool_calls: list[dict[str, Any]]
    sql_statements: list[str]
    observations: list[dict[str, Any]]
    errors: list[dict[str, Any]]
    final_answer: str
    score: float | None = None
    task_success: bool | None = None
    valid: bool
    failure_reason: str | None = None

    @model_validator(mode="after")
    def validate_real_run(self) -> RealDatabaseTrajectory:
        if self.valid and self.failure_reason is not None:
            raise ValueError("valid trajectory must not carry failure_reason")
        if not self.valid and not self.failure_reason:
            raise ValueError("invalid trajectory must carry failure_reason")
        if self.valid and (self.score is None or self.task_success is None):
            raise ValueError("valid trajectory requires native score and task_success")
        if self.valid and not self.actions and not self.tool_calls and not self.sql_statements:
            raise ValueError("real trajectory must contain structured actions or tool calls")
        return self


class ProceduralRoutingCard(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    goal_summary: str
    task_tags: list[str]
    precondition_summary: str
    expected_effect: str
    known_risks: list[str]


class ProcedurePayload(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    preconditions: list[str]
    steps: list[str]
    failure_signals: list[str]
    recovery_actions: list[str]


class RealProceduralMemory(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: str = "database_procedural_memory_v1"
    memory_id: str
    memory_type: Literal["procedural"] = "procedural"
    source_task_id: str
    source_trajectory_id: str
    routing_card: ProceduralRoutingCard
    procedure_payload: ProcedurePayload

    @model_validator(mode="after")
    def reject_answer_leakage(self) -> RealProceduralMemory:
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True).lower()
        forbidden = ("reference_answer", "y_share", "y_withhold", "recipient_task_outcome")
        if any(token in payload for token in forbidden):
            raise ValueError("procedural memory contains outcome/reference leakage")
        if not self.procedure_payload.steps:
            raise ValueError("procedural memory needs at least one executable step")
        return self


class CandidateSet(BaseModel):
    model_config = ConfigDict(frozen=True)

    recipient_task_id: str
    candidate_memory_ids: list[str]
    retrieval_scores: list[float]

    @model_validator(mode="after")
    def validate_lengths(self) -> CandidateSet:
        if len(self.candidate_memory_ids) != len(self.retrieval_scores):
            raise ValueError("candidate_memory_ids and retrieval_scores length mismatch")
        return self


class DatabaseCandidateManifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: str = "database_candidates_v1"
    retrieval_method: str = "routing_card_lexical_v1"
    candidates: list[CandidateSet]
    candidate_pool_sha256: str
    excluded_self_count: int = 0
    excluded_group_count: int = 0


class RealPairedRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: str = "database_real_pair_v1"
    pair_id: str
    recipient_task_id: str
    memory_id: str
    source_task_id: str
    generation_seed: int
    Y_withhold: int | None = None
    Y_share: int | None = None
    tau: int | None = None
    withhold_score: float | None = None
    share_score: float | None = None
    initial_state_match: bool
    memory_intervention_verified: bool
    valid: bool
    failure_reason: str | None = None

    @model_validator(mode="after")
    def validate_pair(self) -> RealPairedRecord:
        if self.source_task_id == self.recipient_task_id:
            raise ValueError("self-pair is forbidden")
        if self.valid:
            if self.Y_share is None or self.Y_withhold is None or self.tau is None:
                raise ValueError("valid pair requires Y_withhold, Y_share, and tau")
            if self.tau != self.Y_share - self.Y_withhold:
                raise ValueError("tau must equal Y_share-Y_withhold")
            if not (self.initial_state_match and self.memory_intervention_verified):
                raise ValueError("valid pair requires state match and verified intervention")
        if not self.valid and not self.failure_reason:
            raise ValueError("invalid pair must carry failure_reason")
        return self


def build_cross_task_candidates(
    *,
    memories: list[RealProceduralMemory],
    recipients: list[dict[str, str]],
    group_by_task: dict[str, str] | None = None,
    dataset_manifest_sha256: str | None = None,
    split_manifest_sha256: str | None = None,
    created_at: str | None = None,
    top_k: int = 4,
) -> DatabaseCandidateManifest:
    candidate_sets: list[CandidateSet] = []
    excluded_self = 0
    excluded_group = 0
    group_by_task = group_by_task or {}
    for recipient in sorted(recipients, key=lambda item: item["task_id"]):
        scored = []
        recipient_terms = _terms(recipient["instruction"])
        for memory in memories:
            if memory.source_task_id == recipient["task_id"]:
                excluded_self += 1
                continue
            if group_by_task.get(memory.source_task_id) == recipient.get("group_id"):
                excluded_group += 1
                continue
            card = memory.routing_card
            card_terms = _terms(
                " ".join([card.goal_summary, *card.task_tags, card.precondition_summary])
            )
            score = len(recipient_terms & card_terms) / max(1, len(recipient_terms | card_terms))
            scored.append((score, memory))
        top = sorted(scored, key=lambda item: (-item[0], item[1].memory_id))[:top_k]
        candidate_sets.append(
            CandidateSet(
                recipient_task_id=recipient["task_id"],
                candidate_memory_ids=[memory.memory_id for _, memory in top],
                retrieval_scores=[score for score, _ in top],
            )
        )
    pool_digest = canonical_digest([memory.model_dump(mode="json") for memory in memories])
    return DatabaseCandidateManifest(
        candidate_pool_sha256=pool_digest,
        candidates=candidate_sets,
        excluded_self_count=excluded_self,
        excluded_group_count=excluded_group,
    )


def extract_procedural_memories(
    trajectories: list[RealDatabaseTrajectory],
    *,
    group_by_task: dict[str, str],
    created_at: str,
    extraction_seed: int = 0,
) -> list[RealProceduralMemory]:
    """Extract bounded, answer-free diagnostic procedures from real train runs."""
    memories: list[RealProceduralMemory] = []
    for trajectory in sorted(trajectories, key=lambda item: item.trajectory_id):
        if trajectory.split != "train":
            raise ValueError("memory extraction may only read train trajectories")
        action_names = sorted(
            {
                str(action.get("name") or action.get("tool") or action.get("type"))
                for action in [*trajectory.actions, *trajectory.tool_calls]
                if action.get("name") or action.get("tool") or action.get("type")
            }
        )
        if not action_names:
            continue
        steps = [
            "Inspect database health and workload evidence before selecting a hypothesis.",
            "Query the relevant monitoring views with bounded, read-only diagnostics.",
            "Cross-check the suspected cause against at least one independent signal.",
            "Report the supported cause and preserve contradictory evidence.",
        ]
        payload = ProcedurePayload(
            preconditions=["Database performance diagnosis with monitoring access."],
            steps=steps,
            failure_signals=[
                "monitoring view unavailable",
                "query timeout",
                "conflicting evidence",
            ],
            recovery_actions=[
                "use a narrower read-only query",
                "request another agent cross-check",
            ],
        )
        memory_id = f"dbproc-{trajectory.trajectory_id[:16]}"
        memories.append(
            RealProceduralMemory(
                memory_id=memory_id,
                source_task_id=trajectory.task_id,
                source_trajectory_id=trajectory.trajectory_id,
                routing_card=ProceduralRoutingCard(
                    goal_summary="Diagnose database performance using evidence before deciding.",
                    task_tags=["database", "performance", *action_names[:4]],
                    precondition_summary=(
                        "Read-only SQL and database monitoring tools are available."
                    ),
                    expected_effect="More evidence-grounded root-cause selection.",
                    known_risks=[
                        "expensive diagnostic query",
                        "premature single-signal conclusion",
                    ],
                ),
                procedure_payload=payload,
            )
        )
    return memories


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _terms(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_]+", text.lower()))
