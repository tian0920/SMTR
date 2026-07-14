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
    dataset_manifest_sha256: str
    split_manifest_sha256: str
    marble_commit: str
    smtr_commit: str
    initial_database_fingerprint: dict[str, str]
    initial_logical_database_digest: str
    agent_identities: list[dict[str, Any]]
    agent_messages: list[dict[str, Any]]
    agent_actions: list[dict[str, Any]]
    tool_calls: list[dict[str, Any]]
    sql_statements: list[str]
    observations: list[dict[str, Any]]
    errors: list[dict[str, Any]]
    final_answer: str
    raw_result_path: str
    raw_result_sha256: str
    native_evaluator_executed: bool
    native_evaluator_output: dict[str, Any]
    score: float
    task_success: bool
    real_engine_executed: bool
    cleanup_succeeded: bool
    environment_valid: bool
    raw_result_exists: bool
    raw_result_nonempty: bool
    raw_result_fresh: bool
    raw_result_parseable: bool
    started_at: str
    completed_at: str
    stdout_log_path: str
    stderr_log_path: str
    workspace_path: str

    @model_validator(mode="after")
    def validate_real_run(self) -> RealDatabaseTrajectory:
        required = {
            "real_engine_executed": self.real_engine_executed,
            "raw_result_exists": self.raw_result_exists,
            "raw_result_nonempty": self.raw_result_nonempty,
            "raw_result_fresh": self.raw_result_fresh,
            "raw_result_parseable": self.raw_result_parseable,
            "native_evaluator_executed": self.native_evaluator_executed,
            "cleanup_succeeded": self.cleanup_succeeded,
            "environment_valid": self.environment_valid,
        }
        failed = [name for name, passed in required.items() if not passed]
        if failed:
            raise ValueError(f"invalid real trajectory: {','.join(failed)}")
        provenance = [
            self.dataset_manifest_sha256,
            self.split_manifest_sha256,
            self.marble_commit,
            self.smtr_commit,
            self.raw_result_sha256,
            self.initial_logical_database_digest,
        ]
        if any(not value for value in provenance):
            raise ValueError("real trajectory provenance is incomplete")
        if not self.agent_actions and not self.tool_calls:
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

    applicable_context: str
    ordered_steps: list[str]
    failure_signals: list[str]
    recovery_actions: list[str]


class RealProceduralMemory(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: str = "database_procedural_memory_v1"
    memory_id: str
    memory_type: Literal["procedural"] = "procedural"
    source_task_id: str
    source_trajectory_id: str
    source_split: Literal["train"]
    source_group_id: str
    routing_card: ProceduralRoutingCard
    procedure_payload: ProcedurePayload
    extractor_type: str
    extractor_version: str
    extraction_model: str
    extraction_seed: int
    extraction_prompt_hash: str
    dataset_manifest_sha256: str
    split_manifest_sha256: str
    source_trajectory_sha256: str
    created_at: str
    duplicate_cluster_id: str | None = None

    @model_validator(mode="after")
    def reject_answer_leakage(self) -> RealProceduralMemory:
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True).lower()
        forbidden = ("reference_answer", "y_share", "y_withhold", "recipient_task_outcome")
        if any(token in payload for token in forbidden):
            raise ValueError("procedural memory contains outcome/reference leakage")
        if not self.procedure_payload.ordered_steps:
            raise ValueError("procedural memory needs at least one executable step")
        return self


class CandidateEdge(BaseModel):
    model_config = ConfigDict(frozen=True)

    recipient_task_id: str
    recipient_split: Literal["validation"]
    recipient_group_id: str
    memory_id: str
    source_task_id: str
    source_split: Literal["train"]
    source_group_id: str
    retrieval_score: float

    @model_validator(mode="after")
    def validate_cross_task(self) -> CandidateEdge:
        if self.source_task_id == self.recipient_task_id:
            raise ValueError("self-pair is forbidden")
        if self.source_group_id == self.recipient_group_id:
            raise ValueError("same-group candidate is forbidden")
        return self


class DatabaseCandidateManifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: str = "database_candidates_v1"
    retrieval_method: str = "routing_card_lexical_v1"
    candidate_pool_sha256: str
    dataset_manifest_sha256: str
    split_manifest_sha256: str
    edges: list[CandidateEdge]
    excluded_self_count: int = 0
    excluded_group_count: int = 0
    created_at: str


class RealPairedRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: str = "database_real_pair_v1"
    pair_id: str
    recipient_task_id: str
    recipient_split: Literal["validation"]
    memory_id: str
    source_task_id: str
    source_trajectory_id: str
    source_split: Literal["train"]
    generation_seed: int
    branch_order: Literal["share_then_withhold", "withhold_then_share"]
    withhold_trajectory_id: str
    share_trajectory_id: str
    y_withhold: int
    y_share: int
    tau: int
    withhold_score: float
    share_score: float
    withhold_task_success: bool
    share_task_success: bool
    initial_state_match: bool
    initial_logical_digest_match: bool
    memory_intervention_verified: bool
    withhold_real_engine_executed: bool
    share_real_engine_executed: bool
    withhold_native_evaluator_executed: bool
    share_native_evaluator_executed: bool
    withhold_cleanup_succeeded: bool
    share_cleanup_succeeded: bool
    routing_card_snapshot: ProceduralRoutingCard
    memory_payload_sha256: str
    candidate_manifest_sha256: str
    dataset_manifest_sha256: str
    split_manifest_sha256: str
    paired_record_valid: bool
    invalid_reason: str | None = None

    @model_validator(mode="after")
    def validate_pair(self) -> RealPairedRecord:
        if self.source_task_id == self.recipient_task_id:
            raise ValueError("self-pair is forbidden")
        if self.y_share != int(self.share_task_success):
            raise ValueError("Y_share must come from native evaluated share success")
        if self.y_withhold != int(self.withhold_task_success):
            raise ValueError("Y_withhold must come from native evaluated withhold success")
        if self.tau != self.y_share - self.y_withhold:
            raise ValueError("tau must equal Y_share-Y_withhold")
        validity = all(
            (
                self.initial_state_match,
                self.initial_logical_digest_match,
                self.memory_intervention_verified,
                self.withhold_real_engine_executed,
                self.share_real_engine_executed,
                self.withhold_native_evaluator_executed,
                self.share_native_evaluator_executed,
                self.withhold_cleanup_succeeded,
                self.share_cleanup_succeeded,
            )
        )
        if self.paired_record_valid != validity:
            raise ValueError("paired_record_valid does not match branch evidence")
        return self


def build_cross_task_candidates(
    *,
    memories: list[RealProceduralMemory],
    recipients: list[dict[str, str]],
    dataset_manifest_sha256: str,
    split_manifest_sha256: str,
    created_at: str,
    top_k: int = 4,
) -> DatabaseCandidateManifest:
    edges: list[CandidateEdge] = []
    excluded_self = 0
    excluded_group = 0
    for recipient in sorted(recipients, key=lambda item: item["task_id"]):
        scored = []
        recipient_terms = _terms(recipient["instruction"])
        for memory in memories:
            if memory.source_task_id == recipient["task_id"]:
                excluded_self += 1
                continue
            if memory.source_group_id == recipient["group_id"]:
                excluded_group += 1
                continue
            card = memory.routing_card
            card_terms = _terms(
                " ".join([card.goal_summary, *card.task_tags, card.precondition_summary])
            )
            score = len(recipient_terms & card_terms) / max(1, len(recipient_terms | card_terms))
            scored.append((score, memory))
        for score, memory in sorted(scored, key=lambda item: (-item[0], item[1].memory_id))[:top_k]:
            edges.append(
                CandidateEdge(
                    recipient_task_id=recipient["task_id"],
                    recipient_split="validation",
                    recipient_group_id=recipient["group_id"],
                    memory_id=memory.memory_id,
                    source_task_id=memory.source_task_id,
                    source_split="train",
                    source_group_id=memory.source_group_id,
                    retrieval_score=score,
                )
            )
    pool_digest = canonical_digest([memory.model_dump(mode="json") for memory in memories])
    return DatabaseCandidateManifest(
        candidate_pool_sha256=pool_digest,
        dataset_manifest_sha256=dataset_manifest_sha256,
        split_manifest_sha256=split_manifest_sha256,
        edges=edges,
        excluded_self_count=excluded_self,
        excluded_group_count=excluded_group,
        created_at=created_at,
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
    seen_payloads: dict[str, str] = {}
    for trajectory in sorted(trajectories, key=lambda item: item.trajectory_id):
        if trajectory.split != "train":
            raise ValueError("memory extraction may only read train trajectories")
        action_names = sorted(
            {
                str(action.get("name") or action.get("tool") or action.get("type"))
                for action in [*trajectory.agent_actions, *trajectory.tool_calls]
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
            applicable_context="Database performance diagnosis with monitoring access.",
            ordered_steps=steps,
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
        payload_digest = canonical_digest(payload.model_dump(mode="json"))
        duplicate_cluster = seen_payloads.get(payload_digest)
        memory_id = f"dbproc-{trajectory.trajectory_id[:16]}"
        if duplicate_cluster is None:
            seen_payloads[payload_digest] = memory_id
        memories.append(
            RealProceduralMemory(
                memory_id=memory_id,
                source_task_id=trajectory.task_id,
                source_trajectory_id=trajectory.trajectory_id,
                source_split="train",
                source_group_id=group_by_task[trajectory.task_id],
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
                extractor_type="deterministic_real_trajectory",
                extractor_version="1",
                extraction_model="none",
                extraction_seed=extraction_seed,
                extraction_prompt_hash=canonical_digest(
                    {"extractor": "deterministic_real_trajectory", "version": "1"}
                ),
                dataset_manifest_sha256=trajectory.dataset_manifest_sha256,
                split_manifest_sha256=trajectory.split_manifest_sha256,
                source_trajectory_sha256=canonical_digest(trajectory.model_dump(mode="json")),
                created_at=created_at,
                duplicate_cluster_id=duplicate_cluster,
            )
        )
    return memories


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _terms(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_]+", text.lower()))
