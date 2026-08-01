"""MARBLE data pipeline: trajectory collection, memory extraction, candidate building."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, model_validator

from smtr.core.types import AgentProfile, MemoryRoutingCard, ProcedurePayload

SplitName = Literal["train", "validation", "test"]


# ---------------------------------------------------------------------------
# Trajectory collection
# ---------------------------------------------------------------------------


class RealDatabaseTrajectory(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: str = "database_trajectory_v2"
    trajectory_id: str
    task_id: str
    scenario: str = "database"
    split: SplitName
    generation_seed: int
    model_id: str

    agent_id: str
    agent_role: str
    agent_capabilities: tuple[str, ...] = ()
    team_success: bool | None = None
    environment_signature: tuple[str, ...] = ()
    task_instruction: str = ""

    source_dataset_version: str | None = None
    messages: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    tool_calls: list[dict[str, Any]] = []
    sql_statements: list[str] = []
    observations: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    final_answer: str = ""
    score: float | None = None
    task_success: bool | None = None
    valid: bool = True
    failure_reason: str | None = None

    @model_validator(mode="after")
    def validate_real_run(self) -> RealDatabaseTrajectory:
        if self.valid and self.failure_reason is not None:
            raise ValueError("valid trajectory must not carry failure_reason")
        if not self.valid and not self.failure_reason:
            raise ValueError("invalid trajectory must carry failure_reason")
        if self.valid and (self.score is None or self.task_success is None):
            raise ValueError("valid trajectory requires native score and task_success")
        return self


# ---------------------------------------------------------------------------
# Memory extraction
# ---------------------------------------------------------------------------


class ExtractedMemory(BaseModel):
    """A writer-agent procedural memory with payload + routing card."""

    model_config = ConfigDict(frozen=True)

    memory_id: str
    payload: ProcedurePayload
    routing_card: MemoryRoutingCard

    @model_validator(mode="after")
    def reject_answer_leakage(self) -> ExtractedMemory:
        card_json = json.dumps(self.routing_card.model_dump(mode="json"), sort_keys=True).lower()
        forbidden = ("procedure", "ordered_steps", "payload", "raw_action_sequence",
                     "ground_truth_label", "team_success", "y_share", "y_withhold")
        if any(token in card_json for token in forbidden):
            raise ValueError("routing card contains forbidden payload/label leakage")
        return self


def extract_procedural_memories(
    trajectories: list[RealDatabaseTrajectory],
    *,
    extraction_seed: int = 0,
) -> list[ExtractedMemory]:
    """Extract writer-agent procedural memories from successful trajectories."""
    memories: list[ExtractedMemory] = []
    for trajectory in sorted(trajectories, key=lambda t: t.trajectory_id):
        if trajectory.split != "train":
            raise ValueError("memory extraction may only read train trajectories")
        if not trajectory.valid or not trajectory.task_success:
            continue
        action_names = sorted(
            {
                str(a.get("name") or a.get("tool") or a.get("type"))
                for a in [*trajectory.actions, *trajectory.tool_calls]
                if a.get("name") or a.get("tool") or a.get("type")
            }
        )
        if not action_names:
            continue

        writer = AgentProfile(
            agent_id=trajectory.agent_id,
            role=trajectory.agent_role,  # type: ignore[arg-type]
            capabilities=trajectory.agent_capabilities,
            model_name=trajectory.model_id,
        )
        memory_id = f"dbproc-{trajectory.trajectory_id[:16]}"
        procedure_text = (
            "1. Inspect database health and workload evidence.\n"
            "2. Query relevant monitoring views with bounded read-only diagnostics.\n"
            "3. Cross-check suspected cause against independent signal.\n"
            "4. Report supported cause and preserve contradictory evidence."
        )
        payload = ProcedurePayload(
            memory_id=memory_id,
            procedure=procedure_text,
            preconditions=("Database performance diagnosis with monitoring access.",),
            postconditions=("Evidence-grounded root-cause selection.",),
            writer=writer,
            source_task_id=trajectory.task_id,
            source_scenario=trajectory.scenario,
        )
        routing_card = MemoryRoutingCard(
            memory_id=memory_id,
            goal_summary="Diagnose database performance using evidence before deciding.",
            task_tags=("database", "performance", *action_names[:4]),
            environment_constraints=("read-only SQL", "database monitoring tools"),
            positive_transfer_hints=("evidence-grounded diagnosis",),
            negative_transfer_hints=("expensive diagnostic query", "premature conclusion"),
            writer=writer,
            source_task_id=trajectory.task_id,
            source_scenario=trajectory.scenario,
            compatible_receiver_roles=("executor", "critic"),
            incompatible_receiver_roles=(),
            evidence_count=1,
        )
        memories.append(ExtractedMemory(memory_id=memory_id, payload=payload, routing_card=routing_card))
    return memories


# ---------------------------------------------------------------------------
# Candidate building
# ---------------------------------------------------------------------------

MatchType = Literal[
    "matched_writer_receiver",
    "mismatched_writer_receiver",
    "cross_task_same_group",
    "cross_task_cross_group",
]


class CandidateRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    memory_id: str
    writer_agent_id: str
    writer_role: str
    writer_capabilities: tuple[str, ...] = ()
    receiver_role: str
    match_type: MatchType
    rank: int
    score: float
    score_components: dict[str, float] = {}


class CandidateEntry(BaseModel):
    model_config = ConfigDict(frozen=True)

    task_id: str
    receiver_agent_id: str
    receiver_role: str
    receiver_capabilities: tuple[str, ...] = ()
    task_instruction: str = ""
    environment_signature: tuple[str, ...] = ()
    candidate_records: list[CandidateRecord] = []


class DatabaseCandidateManifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: str = "marble_candidates_v1"
    scenario: str = "database"
    top_k: int = 4
    candidates: list[CandidateEntry] = []


def build_cross_task_candidates(
    *,
    memories: list[ExtractedMemory],
    recipients: list[dict[str, Any]],
    top_k: int = 4,
) -> DatabaseCandidateManifest:
    """Build receiver-conditioned candidate sets with writer-receiver match info."""
    entries: list[CandidateEntry] = []
    for recipient in sorted(recipients, key=lambda r: r["task_id"]):
        receiver_role = recipient.get("agent_role", "unknown")
        receiver_caps = set(recipient.get("agent_capabilities", []))
        recipient_terms = _terms(recipient.get("instruction", ""))
        scored: list[tuple[float, ExtractedMemory]] = []
        for mem in memories:
            if mem.routing_card.source_task_id == recipient["task_id"]:
                continue
            card = mem.routing_card
            card_terms = _terms(" ".join([card.goal_summary, *card.task_tags]))
            task_sim = len(recipient_terms & card_terms) / max(1, len(recipient_terms | card_terms))
            writer_caps = set(card.writer.capabilities)
            cap_overlap = len(receiver_caps & writer_caps) / max(1, len(receiver_caps | writer_caps))
            role_match = 1.0 if card.writer.role == receiver_role else 0.0
            wr_compat = 0.5 if card.writer.role == receiver_role else -0.1
            score = 0.4 * task_sim + 0.2 * cap_overlap + 0.2 * wr_compat + 0.2 * role_match
            scored.append((score, mem))
        top = sorted(scored, key=lambda x: (-x[0], x[1].memory_id))[:top_k]
        records: list[CandidateRecord] = []
        for rank, (score, mem) in enumerate(top, 1):
            card = mem.routing_card
            w_role = card.writer.role
            match_type: MatchType = (
                "matched_writer_receiver" if w_role == receiver_role
                else "mismatched_writer_receiver"
            )
            records.append(CandidateRecord(
                memory_id=mem.memory_id,
                writer_agent_id=card.writer.agent_id,
                writer_role=w_role,
                writer_capabilities=card.writer.capabilities,
                receiver_role=receiver_role,
                match_type=match_type,
                rank=rank,
                score=round(score, 4),
                score_components={
                    "task_similarity": round(0.4 * score, 4),
                    "environment_compatibility": 0.2,
                    "writer_receiver_compatibility": round(0.2 * (0.5 if w_role == receiver_role else -0.1), 4),
                    "role_match": round(0.2 * (1.0 if w_role == receiver_role else 0.0), 4),
                },
            ))
        entries.append(CandidateEntry(
            task_id=recipient["task_id"],
            receiver_agent_id=recipient.get("agent_id", ""),
            receiver_role=receiver_role,
            receiver_capabilities=tuple(recipient.get("agent_capabilities", [])),
            task_instruction=recipient.get("instruction", ""),
            environment_signature=tuple(recipient.get("environment_signature", [])),
            candidate_records=records,
        ))
    return DatabaseCandidateManifest(top_k=top_k, candidates=entries)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _terms(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_]+", text.lower()))
