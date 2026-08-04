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


class AgentTrajectorySlice(BaseModel):
    """Agent-specific slice of a team trajectory."""

    model_config = ConfigDict(frozen=True)

    agent_id: str
    agent_role: str
    agent_capabilities: tuple[str, ...] = ()
    tool_names: tuple[str, ...] = ()

    messages: tuple[dict[str, Any], ...] = ()
    actions: tuple[dict[str, Any], ...] = ()
    tool_calls: tuple[dict[str, Any], ...] = ()
    sql_statements: tuple[str, ...] = ()
    observations: tuple[dict[str, Any], ...] = ()


class RealDatabaseTrajectory(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: str = "database_trajectory_v3"

    trajectory_id: str
    task_id: str
    scenario: str = "database"
    split: SplitName
    generation_seed: int
    model_id: str

    team_success: bool | None = None
    score: float | None = None
    task_success: bool | None = None

    task_instruction: str = ""
    environment_signature: tuple[str, ...] = ()

    agents: tuple[AgentTrajectorySlice, ...] = ()

    final_answer: str = ""
    errors: tuple[dict[str, Any], ...] = ()

    source_dataset_version: str | None = None
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
        if self.valid and not self.agents:
            raise ValueError("valid trajectory must have at least one agent slice")
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
    min_actions: int = 2,
) -> list[ExtractedMemory]:
    """Extract writer-agent procedural memories from successful train trajectories.

    Each agent slice with sufficient actions produces one memory.
    Procedure is derived from the agent's actual action/tool order.
    """
    memories: list[ExtractedMemory] = []
    for trajectory in sorted(trajectories, key=lambda t: t.trajectory_id):
        if trajectory.split != "train":
            raise ValueError("memory extraction may only read train trajectories")
        if not trajectory.valid or not trajectory.task_success:
            continue
        for agent_slice in trajectory.agents:
            # Preserve original interleaved order of actions and tool_calls
            ordered_actions = _interleave_by_index(agent_slice.actions, agent_slice.tool_calls)
            if len(ordered_actions) < min_actions:
                continue

            # Generate procedure from real action order
            steps: list[str] = []
            for idx, action in enumerate(ordered_actions, 1):
                step = normalize_action_step(action, idx)
                if step:
                    steps.append(step)
            if not steps:
                continue

            procedure_text = "\n".join(f"{i}. {s}" for i, s in enumerate(steps, 1))

            # Normalize SQL for routing card tags
            sql_ops = sorted({
                _sql_operation_type(sql)
                for sql in agent_slice.sql_statements
                if sql.strip()
            })

            writer = AgentProfile(
                agent_id=agent_slice.agent_id,
                role=agent_slice.agent_role,  # type: ignore[arg-type]
                capabilities=agent_slice.agent_capabilities,
                model_name=trajectory.model_id,
                tool_names=agent_slice.tool_names,
            )
            memory_id = f"dbproc-{trajectory.trajectory_id[:12]}-{agent_slice.agent_id[:8]}"

            # Preconditions from tools/environment
            preconditions: list[str] = []
            if agent_slice.tool_names:
                preconditions.append(f"Requires tools: {', '.join(sorted(agent_slice.tool_names))}")
            if trajectory.environment_signature:
                preconditions.append(f"Environment: {', '.join(sorted(trajectory.environment_signature))}")
            if not preconditions:
                preconditions.append("Database scenario with monitoring access.")

            # Postconditions: state-category descriptions, no answer leakage
            postconditions = [
                "A supported database diagnosis is identified.",
                "The conclusion is backed by independent observations.",
                "No write operation is performed.",
            ]

            action_names = sorted({
                str(a.get("name") or a.get("tool") or a.get("type") or "")
                for a in ordered_actions
                if a.get("name") or a.get("tool") or a.get("type")
            })

            payload = ProcedurePayload(
                memory_id=memory_id,
                procedure=procedure_text,
                preconditions=tuple(preconditions),
                postconditions=tuple(postconditions),
                writer=writer,
                source_task_id=trajectory.task_id,
                source_scenario=trajectory.scenario,
            )
            routing_card = MemoryRoutingCard(
                memory_id=memory_id,
                goal_summary=f"Diagnose database issue using {len(steps)}-step evidence method.",
                task_tags=("database", *action_names[:4], *sql_ops[:2]),
                environment_constraints=tuple(trajectory.environment_signature) or ("read-only SQL",),
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


def normalize_action_step(action: dict[str, Any], index: int) -> str:
    """Normalize a single action/tool call into a procedure step description.

    Extracts tool name, action name, SQL operation type. Never includes
    ground-truth answers, exact entity values, or raw observation dumps.
    """
    tool = str(action.get("tool") or action.get("name") or action.get("type") or "").strip()
    if not tool:
        return ""
    # Determine SQL operation if present
    sql = str(action.get("sql") or action.get("arguments", {}).get("sql") or "").strip()
    if sql:
        op = _sql_operation_type(sql)
        return f"Execute a {op} query via {tool}"
    return f"Call {tool} to gather diagnostic evidence"


def normalize_sql(sql: str) -> str:
    """Normalize SQL by replacing constants with placeholders."""
    result = sql.strip()
    # Replace string literals
    result = re.sub(r"'[^']*'", "<STR>", result)
    result = re.sub(r'"[^"]*"', "<STR>", result)
    # Replace timestamps
    result = re.sub(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}", "<TIMESTAMP>", result)
    # Replace numbers (but not in identifiers)
    result = re.sub(r"\b\d+\.\d+\b", "<NUM>", result)
    result = re.sub(r"(?<=[=<>])\s*\d+", " <NUM>", result)
    result = re.sub(r"LIMIT\s+\d+", "LIMIT <LIMIT>", result, flags=re.IGNORECASE)
    # Normalize whitespace
    result = re.sub(r"\s+", " ", result).strip()
    return result


def _sql_operation_type(sql: str) -> str:
    """Extract the SQL operation type."""
    sql_upper = sql.strip().upper()
    for op in ("SELECT", "INSERT", "UPDATE", "DELETE", "CREATE", "DROP", "ALTER"):
        if sql_upper.startswith(op):
            return op.lower()
    return "query"


# ---------------------------------------------------------------------------
# Candidate building
# ---------------------------------------------------------------------------

MatchType = Literal[
    "matched_writer_receiver",
    "mismatched_writer_receiver",
]

TaskRelation = Literal[
    "cross_task_same_group",
    "cross_task_cross_group",
    "cross_task_unknown_group",
]


class CandidateRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    memory_id: str
    writer_agent_id: str
    writer_role: str
    writer_capabilities: tuple[str, ...] = ()
    writer_tool_names: tuple[str, ...] = ()
    writer_model_name: str | None = None
    receiver_role: str
    match_type: MatchType
    task_relation: TaskRelation = "cross_task_unknown_group"
    rank: int
    score: float
    score_components: dict[str, float] = {}


class CandidateEntry(BaseModel):
    model_config = ConfigDict(frozen=True)

    task_id: str
    receiver_agent_id: str
    receiver_role: str
    receiver_capabilities: tuple[str, ...] = ()
    receiver_tool_names: tuple[str, ...] = ()
    receiver_model_name: str | None = None
    task_instruction: str = ""
    environment_signature: tuple[str, ...] = ()
    candidate_records: list[CandidateRecord] = []


class DatabaseCandidateManifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: str = "marble_candidates_v2"
    scenario: str = "database"
    top_k: int = 4
    target_split: str = ""
    memory_source_split: str = "train"
    candidates: list[CandidateEntry] = []


def build_cross_task_candidates(
    *,
    memories: list[ExtractedMemory],
    recipients: list[dict[str, Any]],
    top_k: int = 4,
    target_split: str = "",
) -> DatabaseCandidateManifest:
    """Build receiver-conditioned candidate sets with writer-receiver match info."""
    entries: list[CandidateEntry] = []
    for recipient in sorted(recipients, key=lambda r: r["task_id"]):
        receiver_role = recipient.get("agent_role", "unknown")
        receiver_caps = set(recipient.get("agent_capabilities", []))
        recipient_terms = _terms(recipient.get("instruction", ""))
        scored: list[tuple[float, ExtractedMemory, dict[str, float]]] = []
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
            # Environment compatibility: constraints satisfied / total constraints
            env_constraints = set(card.environment_constraints)
            receiver_env = set(recipient.get("environment_signature", []))
            if env_constraints:
                env_compat = len(env_constraints & receiver_env) / len(env_constraints)
            else:
                env_compat = 1.0
            score = 0.35 * task_sim + 0.2 * cap_overlap + 0.15 * wr_compat + 0.15 * role_match + 0.15 * env_compat
            components = {
                "task_similarity_raw": round(task_sim, 4),
                "task_similarity_weighted": round(0.35 * task_sim, 4),
                "capability_overlap_raw": round(cap_overlap, 4),
                "capability_overlap_weighted": round(0.2 * cap_overlap, 4),
                "writer_receiver_compatibility_raw": round(wr_compat, 4),
                "writer_receiver_compatibility_weighted": round(0.15 * wr_compat, 4),
                "role_match_raw": round(role_match, 4),
                "role_match_weighted": round(0.15 * role_match, 4),
                "environment_compatibility_raw": round(env_compat, 4),
                "environment_compatibility_weighted": round(0.15 * env_compat, 4),
            }
            scored.append((score, mem, components))
        top = sorted(scored, key=lambda x: (-x[0], x[1].memory_id))[:top_k]
        records: list[CandidateRecord] = []
        for rank, (score, mem, components) in enumerate(top, 1):
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
                writer_tool_names=card.writer.tool_names,
                writer_model_name=card.writer.model_name,
                receiver_role=receiver_role,
                match_type=match_type,
                task_relation="cross_task_unknown_group",
                rank=rank,
                score=round(score, 4),
                score_components=components,
            ))
        entries.append(CandidateEntry(
            task_id=recipient["task_id"],
            receiver_agent_id=recipient.get("agent_id", ""),
            receiver_role=receiver_role,
            receiver_capabilities=tuple(recipient.get("agent_capabilities", [])),
            receiver_tool_names=tuple(recipient.get("tool_names", [])),
            receiver_model_name=recipient.get("model_name"),
            task_instruction=recipient.get("instruction", ""),
            environment_signature=tuple(recipient.get("environment_signature", [])),
            candidate_records=records,
        ))
    return DatabaseCandidateManifest(
        top_k=top_k,
        target_split=target_split,
        memory_source_split="train",
        candidates=entries,
    )


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------


def load_trajectories_from_index(
    *,
    trajectory_index_path: Path,
    split_manifest_path: Path,
    required_split: str = "train",
) -> list[RealDatabaseTrajectory]:
    """Load trajectories from an index file, filtering by split."""
    from smtr.marble.io import load_split_task_ids

    split_task_ids = load_split_task_ids(split_manifest_path, required_split)
    trajectories: list[RealDatabaseTrajectory] = []
    for line in trajectory_index_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        if not entry.get("valid"):
            continue
        if entry.get("split") != required_split:
            continue
        if entry.get("task_id") not in split_task_ids:
            continue
        traj_path = Path(entry["path"])
        if not traj_path.exists():
            continue
        data = json.loads(traj_path.read_text(encoding="utf-8"))
        try:
            trajectories.append(RealDatabaseTrajectory.model_validate(data))
        except Exception:
            continue
    return trajectories


def write_memory_pool(
    *,
    memories: list[ExtractedMemory],
    output_path: Path,
) -> dict[str, Any]:
    """Write memory pool as JSONL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for mem in memories:
        lines.append(json.dumps(mem.model_dump(mode="json"), sort_keys=True))
    output_path.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
    return {"memories_written": len(memories), "output": str(output_path)}


def load_memory_pool(path: Path) -> list[ExtractedMemory]:
    """Load memory pool from JSONL."""
    memories: list[ExtractedMemory] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            memories.append(ExtractedMemory.model_validate(json.loads(line)))
    return memories


def load_receiver_entries(
    *,
    dataset_manifest_path: Path,
    split_manifest_path: Path,
    split: str,
) -> list[dict[str, Any]]:
    """Load receiver entries for a given split from dataset + split manifests."""
    from smtr.marble.io import load_split_task_ids, load_dataset_tasks

    task_ids = load_split_task_ids(split_manifest_path, split)
    tasks = load_dataset_tasks(dataset_manifest_path)
    recipients: list[dict[str, Any]] = []
    for task_id in sorted(task_ids):
        task = tasks.get(task_id)
        if task is None:
            continue
        # Each task may have multiple agents; create one recipient per agent
        agents = task.get("agents", [])
        if agents:
            for agent in agents:
                recipients.append({
                    "task_id": task_id,
                    "agent_id": agent.get("agent_id", ""),
                    "agent_role": agent.get("role", "unknown"),
                    "agent_capabilities": agent.get("capabilities", []),
                    "tool_names": agent.get("tool_names", []),
                    "model_name": agent.get("model_name"),
                    "instruction": task.get("instruction", ""),
                    "environment_signature": task.get("environment_signature", []),
                })
        else:
            recipients.append({
                "task_id": task_id,
                "agent_id": task.get("agent_id", "agent1"),
                "agent_role": task.get("agent_role", "executor"),
                "agent_capabilities": task.get("agent_capabilities", []),
                "tool_names": task.get("tool_names", []),
                "model_name": task.get("model_name"),
                "instruction": task.get("instruction", ""),
                "environment_signature": task.get("environment_signature", []),
            })
    return recipients


def write_candidate_manifest(
    *,
    manifest: DatabaseCandidateManifest,
    output_path: Path,
) -> dict[str, Any]:
    """Write candidate manifest as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "candidates_written": len(manifest.candidates),
        "output": str(output_path),
    }


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _terms(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_]+", text.lower()))


def _interleave_by_index(
    actions: tuple[dict[str, Any], ...],
    tool_calls: tuple[dict[str, Any], ...],
) -> list[dict[str, Any]]:
    """Merge actions and tool_calls preserving original interleaved order.

    Each record may carry an 'index' or 'step' field indicating its position
    in the original execution sequence.  When absent, we fall back to
    positional order within each list, offset so that tool_calls interleave
    with actions rather than being appended after them.
    """
    tagged: list[tuple[float, dict[str, Any]]] = []
    for i, a in enumerate(actions):
        idx = a.get("index", a.get("step", i))
        tagged.append((float(idx), a))
    for i, tc in enumerate(tool_calls):
        idx = tc.get("index", tc.get("step", i))
        # Offset by 0.5 so that a tool_call at the same integer index
        # sorts after the action at that index but before the next one.
        tagged.append((float(idx) + 0.5, tc))
    tagged.sort(key=lambda pair: pair[0])
    return [item for _, item in tagged]


# ---------------------------------------------------------------------------
# Legacy compatibility (used by feature_bridge, router_evaluation, etc.)
# ---------------------------------------------------------------------------


class ProceduralRoutingCard(BaseModel):
    """Legacy routing card schema for backward compatibility."""

    model_config = ConfigDict(frozen=True)

    goal_summary: str = ""
    task_tags: list[str] = []
    precondition_summary: str = ""
    expected_effect: str = ""
    known_risks: list[str] = []


class LegacyProcedurePayload(BaseModel):
    """Legacy procedure payload schema for backward compatibility."""

    model_config = ConfigDict(frozen=True)

    preconditions: list[str] = []
    steps: list[str] = []
    failure_signals: list[str] = []
    recovery_actions: list[str] = []


class RealProceduralMemory(BaseModel):
    """Legacy memory schema for backward compatibility."""

    model_config = ConfigDict(frozen=True)

    memory_id: str
    source_task_id: str = ""
    source_trajectory_id: str = ""
    routing_card: ProceduralRoutingCard = ProceduralRoutingCard()
    payload: Any = None
    procedure_payload: LegacyProcedurePayload | None = None
    expected_role: str = "helpful"


class RealPairedRecord(BaseModel):
    """Legacy paired record schema for backward compatibility."""

    model_config = ConfigDict(frozen=True)

    pair_id: str = ""
    recipient_task_id: str = ""
    memory_id: str = ""
    valid: bool = True
    failure_reason: str | None = None
    Y_share: bool | None = None
    Y_withhold: bool | None = None


class CandidateSet(BaseModel):
    """Legacy candidate set schema for backward compatibility."""

    model_config = ConfigDict(frozen=True)

    recipient_task_id: str = ""
    candidate_memory_ids: list[str] = []
