"""MARBLE data pipeline: trajectory collection, memory extraction, candidate building."""

from __future__ import annotations

import hashlib
import json
import re
import statistics
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
                # Goal summary is derived from observable trajectory
                # structure (operation mix), not human transfer hints, so
                # cards differ across trajectories beyond step count.
                goal_summary=(
                    f"Diagnose database issue via "
                    f"{'/'.join(sql_ops) if sql_ops else 'action'}-based "
                    f"{len(steps)}-step evidence method."
                ),
                task_tags=("database", *action_names[:4], *sql_ops[:2]),
                environment_constraints=tuple(trajectory.environment_signature) or ("read-only SQL",),
                # Deprecated human-authored transfer hints / fixed compatible
                # receiver roles are intentionally never populated: cards may
                # only carry outcome-independent, trajectory-observable
                # attributes.
                writer=writer,
                source_task_id=trajectory.task_id,
                source_scenario=trajectory.scenario,
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

CandidateSource = Literal[
    "semantic_top",
    "role_matched",
    "role_mismatched",
    "cross_receiver_anchor",
]

# 清单 P0-10: canonical cohort tags for the formal manifest. One candidate
# can belong to several cohorts at once (e.g. an anchor that is also
# role-matched), so records carry the full tag list.
CandidateSourceTag = Literal[
    "semantic_topk",
    "role_matched",
    "role_mismatched_hard_negative",
    "cross_receiver_anchor",
]

_SOURCE_TAG_ORDER: tuple[CandidateSourceTag, ...] = (
    "semantic_topk",
    "role_matched",
    "role_mismatched_hard_negative",
    "cross_receiver_anchor",
)


class CandidateCohortQuotas(BaseModel):
    """Per-receiver candidate cohort quotas (configurable, not hardcoded).

    Candidate selection never reads share/withhold outcomes: cohorts are
    built from routing-card / receiver metadata only.
    """

    model_config = ConfigDict(frozen=True)

    semantic_top: int = 2
    role_matched: int = 2
    role_mismatched: int = 2
    cross_receiver_anchor: int = 2
    min_task_relevance: float = 0.0

    @property
    def total(self) -> int:
        return (
            self.semantic_top + self.role_matched
            + self.role_mismatched + self.cross_receiver_anchor
        )


def quotas_from_top_k(top_k: int) -> CandidateCohortQuotas:
    """Derive balanced cohort quotas from a total per-receiver budget."""
    base, rem = divmod(max(0, top_k), 4)
    counts = {
        "semantic_top": base + (1 if rem > 0 else 0),
        "role_matched": base + (1 if rem > 1 else 0),
        "role_mismatched": base + (1 if rem > 2 else 0),
        "cross_receiver_anchor": base,
    }
    return CandidateCohortQuotas(**counts)


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
    candidate_source: CandidateSource = "semantic_top"
    candidate_sources: tuple[CandidateSourceTag, ...] = ()
    anchor_group_id: str | None = None
    anchor_receiver_count: int = 0
    anchor_receiver_role_count: int = 0
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


class CandidateBudgetMetadata(BaseModel):
    """Fixed-budget subsampling provenance (清单 Shared-Control 第13章).

    Budget selection is an analysis-time intervention: it never reads
    share/withhold outcomes, critic predictions, or any adaptive signal.
    """

    model_config = ConfigDict(frozen=True)

    policy_version: str
    requested_fraction: float
    realized_edge_fraction: float
    realized_unit_fraction: float

    parent_manifest_digest: str
    outcome_fields_used: bool = False
    critic_predictions_used: bool = False
    adaptive_sampling_used: bool = False

    parent_edge_count: int
    selected_edge_count: int
    parent_selection_unit_count: int
    selected_selection_unit_count: int

    cohort_counts_before: dict[str, int] = {}
    cohort_counts_after: dict[str, int] = {}


class DatabaseCandidateManifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: str = "marble_candidates_v2"
    scenario: str = "database"
    top_k: int = 4
    cohort_quotas: CandidateCohortQuotas = CandidateCohortQuotas()
    target_split: str = ""
    memory_source_split: str = "train"
    candidates: list[CandidateEntry] = []
    budget_metadata: CandidateBudgetMetadata | None = None


def build_cross_task_candidates(
    *,
    memories: list[ExtractedMemory],
    recipients: list[dict[str, Any]],
    top_k: int = 4,
    target_split: str = "",
    cohort_quotas: CandidateCohortQuotas | None = None,
    min_task_relevance: float | None = None,
    experiment_mode: str | None = None,
) -> DatabaseCandidateManifest:
    """Build receiver-conditioned candidate sets as stratified cohorts.

    Each receiver's candidates come from four cohorts: semantic_top,
    role_matched, role_mismatched hard negatives and cross_receiver_anchor.
    Construction order is anchors first, then mismatched hard negatives,
    then role-matched, then semantic, so semantic candidates never consume
    anchor memories. Cohort selection only reads routing-card / receiver
    metadata and never reads share/withhold outcomes.
    """
    quotas = cohort_quotas if cohort_quotas is not None else quotas_from_top_k(top_k)
    if min_task_relevance is not None:
        quotas = quotas.model_copy(update={"min_task_relevance": min_task_relevance})
    if experiment_mode == "formal" and quotas.min_task_relevance <= 0:
        raise ValueError(
            "experiment_mode='formal' requires min_task_relevance > 0 so that "
            "mismatched hard negatives remain task-relevant."
        )

    sorted_recipients = sorted(recipients, key=lambda r: (r["task_id"], r.get("agent_id", "")))

    # Pass 1: per-receiver eligible pool (outcomes never consulted)
    receiver_pools: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    for recipient in sorted_recipients:
        pool = [
            item
            for mem in memories
            if mem.routing_card.source_task_id != recipient["task_id"]
            for item in (_score_memory_for_recipient(mem, recipient),)
        ]
        receiver_pools.append((recipient, pool))

    # Pass 2: global anchor assignment (same memory to multiple receivers)
    anchor_assignments, anchor_stats = _select_anchor_assignments(receiver_pools, quotas)

    entries: list[CandidateEntry] = []
    for idx, (recipient, pool) in enumerate(receiver_pools):
        receiver_role = recipient.get("agent_role", "unknown")
        assigned_anchor_ids = anchor_assignments.get(idx, [])

        by_score = sorted(pool, key=lambda it: (-it["score"], it["mem"].memory_id))
        relevant_pool = [
            it for it in by_score if it["task_sim"] >= quotas.min_task_relevance
        ]
        matched_pool = [
            it for it in relevant_pool
            if it["matched"]
        ]
        mismatched_pool = [
            it for it in relevant_pool
            if it["mismatched"]
        ]
        anchor_pool = [
            it for mid in assigned_anchor_ids
            for it in relevant_pool
            if it["mem"].memory_id == mid
        ]

        selected: list[tuple[dict[str, Any], CandidateSource]] = []
        chosen_ids: set[str] = set()

        def _fill(source_pool: list[dict[str, Any]], quota: int, source: CandidateSource) -> None:
            for it in source_pool:
                if len([s for s in selected if s[1] == source]) >= quota:
                    break
                mid = it["mem"].memory_id
                if mid in chosen_ids:
                    continue
                chosen_ids.add(mid)
                selected.append((it, source))

        # Construction order: anchors first (so semantic candidates cannot
        # consume anchor memories), then mismatched hard negatives, then
        # role-matched, then semantic top, finally semantic backfill.
        _fill(anchor_pool, quotas.cross_receiver_anchor, "cross_receiver_anchor")
        _fill(mismatched_pool, quotas.role_mismatched, "role_mismatched")
        _fill(matched_pool, quotas.role_matched, "role_matched")
        _fill(relevant_pool, quotas.semantic_top, "semantic_top")
        # Backfill leftover budget by overall relevance (labelled semantic_top)
        _fill(relevant_pool, quotas.total, "semantic_top")

        records: list[CandidateRecord] = []
        for rank, (item, source) in enumerate(selected, 1):
            mem = item["mem"]
            card = mem.routing_card
            w_role = card.writer.role
            match_type: MatchType = (
                "matched_writer_receiver" if w_role == receiver_role
                else "mismatched_writer_receiver"
            )
            is_anchor = mem.memory_id in assigned_anchor_ids
            if source == "cross_receiver_anchor" or is_anchor:
                anchor_receiver_count, anchor_receiver_role_count = anchor_stats.get(
                    mem.memory_id, (0, 0)
                )
            else:
                anchor_receiver_count = 0
                anchor_receiver_role_count = 0
            # 清单 P0-10: mark every cohort the candidate belongs to.
            tags: set[str] = set()
            if source == "semantic_top":
                tags.add("semantic_topk")
            if item["matched"]:
                tags.add("role_matched")
            if item["mismatched"]:
                tags.add("role_mismatched_hard_negative")
            if is_anchor:
                tags.add("cross_receiver_anchor")
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
                candidate_source=source,
                candidate_sources=tuple(
                    tag for tag in _SOURCE_TAG_ORDER if tag in tags
                ),
                anchor_group_id=mem.memory_id if is_anchor else None,
                anchor_receiver_count=anchor_receiver_count,
                anchor_receiver_role_count=anchor_receiver_role_count,
                rank=rank,
                score=round(item["score"], 4),
                score_components=item["components"],
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
        cohort_quotas=quotas,
        target_split=target_split,
        memory_source_split="train",
        candidates=entries,
    )


def _score_memory_for_recipient(
    mem: ExtractedMemory,
    recipient: dict[str, Any],
) -> dict[str, Any]:
    """Score one memory for one receiver using metadata only (no outcomes)."""
    card = mem.routing_card
    receiver_role = recipient.get("agent_role", "unknown")
    receiver_caps = set(recipient.get("agent_capabilities", []))
    receiver_tools = set(recipient.get("tool_names", []))
    recipient_terms = _terms(recipient.get("instruction", ""))
    card_terms = _terms(" ".join([card.goal_summary, *card.task_tags]))
    task_sim = len(recipient_terms & card_terms) / max(1, len(recipient_terms | card_terms))
    writer_caps = set(card.writer.capabilities)
    cap_overlap = len(receiver_caps & writer_caps) / max(1, len(receiver_caps | writer_caps))
    role_match = 1.0 if card.writer.role == receiver_role else 0.0
    wr_compat = 0.5 if card.writer.role == receiver_role else -0.1
    # Environment compatibility: constraints satisfied / total constraints
    env_constraints = set(card.environment_constraints)
    receiver_env = set(recipient.get("environment_signature", []))
    env_compat = (
        len(env_constraints & receiver_env) / len(env_constraints)
        if env_constraints else 1.0
    )
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
    matched, mismatched = _writer_receiver_alignment(card, receiver_role, receiver_caps, receiver_tools)
    return {
        "mem": mem,
        "score": score,
        "components": components,
        "task_sim": task_sim,
        "matched": matched,
        "mismatched": mismatched,
    }


def _writer_receiver_alignment(
    card: MemoryRoutingCard,
    receiver_role: str,
    receiver_caps: set[str],
    receiver_tools: set[str],
) -> tuple[bool, bool]:
    """Return (matched, mismatched) writer-receiver alignment flags.

    matched: same role OR overlapping capabilities OR overlapping tools.
    mismatched (hard negative): different role AND no capability/tool overlap.
    """
    role_same = card.writer.role == receiver_role
    cap_overlaps = bool(receiver_caps & set(card.writer.capabilities))
    tool_overlaps = bool(receiver_tools & set(card.writer.tool_names))
    matched = role_same or cap_overlaps or tool_overlaps
    mismatched = (not role_same) and (not cap_overlaps) and (not tool_overlaps)
    return matched, mismatched


def _select_anchor_assignments(
    receiver_pools: list[tuple[dict[str, Any], list[dict[str, Any]]]],
    quotas: CandidateCohortQuotas,
) -> tuple[dict[int, list[str]], dict[str, tuple[int, int]]]:
    """Return per-receiver anchor ids plus per-memory anchor statistics.

    Anchor memories are task-relevant memories eligible for >=2 receivers,
    ranked to prefer coverage of distinct receiver roles.  Every anchor
    candidate is by construction shared across at least two receivers.
    The second return maps memory_id -> (anchor_receiver_count,
    anchor_receiver_role_count). Selection ignores share/withhold outcomes.
    """
    if quotas.cross_receiver_anchor <= 0:
        return {}, {}
    eligible: dict[str, list[tuple[int, str]]] = {}
    for idx, (recipient, pool) in enumerate(receiver_pools):
        role = recipient.get("agent_role", "unknown")
        for item in pool:
            if item["task_sim"] >= quotas.min_task_relevance:
                eligible.setdefault(item["mem"].memory_id, []).append((idx, role))
    anchor_candidates = [
        (mid, lst) for mid, lst in eligible.items()
        if len({idx for idx, _ in lst}) >= 2
    ]
    anchor_candidates.sort(
        key=lambda p: (
            -len({role for _, role in p[1]}),
            -len({idx for idx, _ in p[1]}),
            p[0],
        )
    )
    anchor_stats = {
        mid: (len({idx for idx, _ in lst}), len({role for _, role in lst}))
        for mid, lst in anchor_candidates
    }
    ranked_anchor_ids = [mid for mid, _ in anchor_candidates]
    assignments: dict[int, list[str]] = {}
    for idx, (_, pool) in enumerate(receiver_pools):
        eligible_ids = {
            it["mem"].memory_id
            for it in pool
            if it["task_sim"] >= quotas.min_task_relevance
        }
        assignments[idx] = [mid for mid in ranked_anchor_ids if mid in eligible_ids]
    return assignments, anchor_stats


def validate_receiver_effect_coverage(
    candidate_manifest: DatabaseCandidateManifest | dict[str, Any],
) -> dict[str, Any]:
    """Audit candidate coverage required for receiver-effect identification.

    Checks every receiver carries matched and mismatched candidates, that at
    least one memory is evaluated by multiple receivers (and multiple
    receiver roles), that anchor candidates meet min task relevance, and that
    the candidate schema never carries outcome labels.
    """
    if isinstance(candidate_manifest, dict):
        manifest = DatabaseCandidateManifest.model_validate(candidate_manifest)
    else:
        manifest = candidate_manifest

    min_relevance = manifest.cohort_quotas.min_task_relevance
    total_records = 0
    matched_records = 0
    mismatched_records = 0
    anchor_records = 0
    receivers_without_matched: list[str] = []
    receivers_without_mismatched: list[str] = []
    memory_receivers: dict[str, set[tuple[str, str]]] = {}
    memory_roles: dict[str, set[str]] = {}
    anchor_memory_receivers: dict[str, set[tuple[str, str]]] = {}
    anchor_memory_roles: dict[str, set[str]] = {}
    cohort_relevances: dict[str, list[float]] = {}
    anchor_relevance_ok = True

    for entry in manifest.candidates:
        key = (entry.task_id, entry.receiver_agent_id)
        recs = entry.candidate_records
        total_records += len(recs)
        has_matched = False
        has_mismatched = False
        for rec in recs:
            memory_receivers.setdefault(rec.memory_id, set()).add(key)
            memory_roles.setdefault(rec.memory_id, set()).add(entry.receiver_role)
            cohort_relevances.setdefault(rec.candidate_source, []).append(
                rec.score_components.get("task_similarity_raw", 0.0)
            )
            if rec.candidate_source == "role_matched":
                matched_records += 1
            elif rec.candidate_source == "role_mismatched":
                mismatched_records += 1
            elif rec.candidate_source == "cross_receiver_anchor":
                anchor_records += 1
                anchor_memory_receivers.setdefault(rec.memory_id, set()).add(key)
                anchor_memory_roles.setdefault(rec.memory_id, set()).add(entry.receiver_role)
                task_sim = rec.score_components.get("task_similarity_raw", 0.0)
                if task_sim < min_relevance:
                    anchor_relevance_ok = False
            if rec.match_type == "matched_writer_receiver":
                has_matched = True
            else:
                has_mismatched = True
        if not has_matched:
            receivers_without_matched.append(f"{key[0]}:{key[1]}")
        if not has_mismatched:
            receivers_without_mismatched.append(f"{key[0]}:{key[1]}")

    total_unique_memories = len(memory_receivers)
    seen_by_2plus_receivers = sum(1 for r in memory_receivers.values() if len(r) >= 2)
    seen_by_2plus_roles = sum(1 for roles in memory_roles.values() if len(roles) >= 2)
    anchor_cross_receiver = sum(
        1 for r in anchor_memory_receivers.values() if len(r) >= 2
    )
    anchor_cross_role = sum(
        1 for roles in anchor_memory_roles.values() if len(roles) >= 2
    )
    forbidden_fields = {"y_share", "y_withhold", "label", "team_success", "outcome"}
    no_outcome_fields = not (set(CandidateRecord.model_fields) & forbidden_fields)

    cohort_relevance_summary = {
        cohort: {
            "mean": round(statistics.mean(values), 4),
            "median": round(statistics.median(values), 4),
            "min": round(min(values), 4),
            "max": round(max(values), 4),
            "count": len(values),
        }
        for cohort, values in sorted(cohort_relevances.items())
    }

    stats = {
        "receiver_count": len(manifest.candidates),
        "unique_memory_count": total_unique_memories,
        "total_unique_memories": total_unique_memories,
        "memories_seen_by_2plus_receivers": seen_by_2plus_receivers,
        "memories_seen_by_2plus_receiver_roles": seen_by_2plus_roles,
        "cross_receiver_anchor_count": anchor_cross_receiver,
        "cross_receiver_role_anchor_count": anchor_cross_role,
        "receiver_effect_coverage": round(
            seen_by_2plus_receivers / total_unique_memories, 4
        ) if total_unique_memories else 0.0,
        "matched_candidate_rate": round(
            matched_records / total_records, 4
        ) if total_records else 0.0,
        "mismatched_candidate_rate": round(
            mismatched_records / total_records, 4
        ) if total_records else 0.0,
        "cross_receiver_anchor_rate": round(
            anchor_records / total_records, 4
        ) if total_records else 0.0,
        "cohort_relevance_summary": cohort_relevance_summary,
    }
    checks = {
        "all_receivers_have_matched_candidate": not receivers_without_matched,
        "all_receivers_have_mismatched_candidate": not receivers_without_mismatched,
        "has_memory_seen_by_2plus_receivers": seen_by_2plus_receivers > 0,
        "has_memory_seen_by_2plus_receiver_roles": seen_by_2plus_roles > 0,
        "has_cross_receiver_anchor": anchor_cross_receiver > 0,
        "has_cross_receiver_role_anchor": anchor_cross_role > 0,
        "anchor_candidates_meet_min_task_relevance": anchor_relevance_ok,
        "candidate_selection_ignores_outcomes": no_outcome_fields,
    }
    return {
        "statistics": stats,
        "checks": checks,
        "receivers_without_matched_candidate": receivers_without_matched,
        "receivers_without_mismatched_candidate": receivers_without_mismatched,
        "ok": all(checks.values()),
    }


class InsufficientReceiverEffectCoverageError(ValueError):
    """Raised when formal candidate generation lacks receiver-effect coverage."""


def compute_proposal_support_metrics(
    manifest: DatabaseCandidateManifest | dict[str, Any],
) -> dict[str, Any]:
    """Proposal support statistics required by 清单 P0-11.

    These are not extra benchmarks; they verify that the proposal actually
    exposes cross-agent heterogeneity (role-matched vs hard negatives,
    cross-receiver anchors) for the receiver-effect analysis.
    """
    if isinstance(manifest, dict):
        manifest = DatabaseCandidateManifest.model_validate(manifest)

    per_receiver_counts: dict[str, int] = {}
    source_distribution: dict[str, int] = {tag: 0 for tag in _SOURCE_TAG_ORDER}
    total_candidates = 0
    matched_candidates = 0
    mismatched_candidates = 0
    anchor_memories: dict[str, set[tuple[str, str]]] = {}
    all_memory_receivers: dict[str, set[tuple[str, str]]] = {}

    for entry in manifest.candidates:
        receiver_key = f"{entry.task_id}:{entry.receiver_agent_id}"
        per_receiver_counts[receiver_key] = len(entry.candidate_records)
        for rec in entry.candidate_records:
            total_candidates += 1
            tags = set(rec.candidate_sources)
            if not tags:
                # Legacy manifests without the multi-source tag list.
                tags = {_legacy_source_tag(rec.candidate_source)}
            for tag in tags:
                source_distribution[tag] = source_distribution.get(tag, 0) + 1
            if "role_matched" in tags:
                matched_candidates += 1
            if "role_mismatched_hard_negative" in tags:
                mismatched_candidates += 1
            if "cross_receiver_anchor" in tags:
                anchor_memories.setdefault(rec.memory_id, set()).add(
                    (entry.task_id, entry.receiver_agent_id)
                )
            all_memory_receivers.setdefault(rec.memory_id, set()).add(
                (entry.task_id, entry.receiver_agent_id)
            )

    anchor_receiver_counts = [len(rs) for rs in anchor_memories.values()]
    return {
        "candidate_count_per_receiver": per_receiver_counts,
        "total_candidate_count": total_candidates,
        "role_matched_candidate_rate": (
            round(matched_candidates / total_candidates, 4) if total_candidates else 0.0
        ),
        "role_mismatched_candidate_rate": (
            round(mismatched_candidates / total_candidates, 4)
            if total_candidates
            else 0.0
        ),
        "cross_receiver_anchor_count": len(anchor_memories),
        "memories_with_multiple_receivers": sum(
            1 for rs in all_memory_receivers.values() if len(rs) >= 2
        ),
        "receivers_per_anchor_memory": {
            memory_id: len(rs) for memory_id, rs in sorted(anchor_memories.items())
        },
        "mean_receivers_per_anchor_memory": (
            round(statistics.mean(anchor_receiver_counts), 4)
            if anchor_receiver_counts
            else 0.0
        ),
        "candidate_source_distribution": source_distribution,
    }


def _legacy_source_tag(source: CandidateSource) -> CandidateSourceTag:
    """Map the legacy single-cohort label onto the canonical tag."""
    if source == "role_mismatched":
        return "role_mismatched_hard_negative"
    if source == "semantic_top":
        return "semantic_topk"
    return source  # role_matched / cross_receiver_anchor keep their names


def require_receiver_effect_coverage(
    coverage: dict[str, Any],
) -> None:
    """Fail fast when coverage checks do not pass (formal data generation).

    Formal experiments must fail, not warn, when any receiver lacks
    matched/mismatched candidates or when no cross-receiver (and
    cross-receiver-role) anchor exists.
    """
    if coverage.get("ok"):
        return
    failed = sorted(
        name for name, passed in coverage.get("checks", {}).items() if not passed
    )
    raise InsufficientReceiverEffectCoverageError(
        "Receiver-effect coverage insufficient for formal data generation; "
        f"failed checks: {failed}; statistics: {coverage.get('statistics', {})}"
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
    """Write candidate manifest as JSON plus proposal support metrics.

    The proposal support statistics (清单 P0-11) are written next to the
    manifest as ``proposal_support_metrics.json``.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )
    support = compute_proposal_support_metrics(manifest)
    support_path = output_path.parent / "proposal_support_metrics.json"
    support_path.write_text(
        json.dumps(support, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "candidates_written": len(manifest.candidates),
        "output": str(output_path),
        "proposal_support_output": str(support_path),
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
