"""Cross-agent transfer feature encoder with writer-receiver blocks."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from sklearn.feature_extraction import FeatureHasher

from smtr.core.types import (
    AgentProfile,
    CandidateExposureInput,
    MemoryRoutingCard,
    ReceiverState,
)
from smtr.marble.core_validity import is_core_valid_pair
from smtr.marble.paired_outcomes import paired_record_label

FORBIDDEN_FEATURE_TOKENS = frozenset({
    "memory_id", "candidate_memory_id", "payload", "procedure", "ordered_steps",
    "label", "team_success", "local_success", "y_share", "y_withhold",
    "q00", "q01", "q10", "q11",
    # Deprecated human-authored transfer hints must never become features.
    "compatible_receiver_role", "incompatible_receiver_role",
    "positive_hint_token", "negative_hint_token",
})

TOKEN_RE = re.compile(r"[a-z0-9]+")


class HashingTransferFeatureEncoder:
    """Deterministic feature encoder for cross-agent transfer prediction.

    Feature blocks:
      - task context block (scenario, task tokens, environment)
      - receiver marginal block (role, capabilities, tools)
      - writer marginal block (role, capabilities, tools, source scenario)
      - writer-receiver interaction block
      - memory card block

    SMTR-v1 action space is A(o_r) in {∅, m_1, ..., m_K}: one receiver
    receives at most one memory, so the selected-memory prefix S is fixed
    to ∅ and is never encoded as features. ``selected_prefix_cards``
    remains a compatibility-only field and must never influence
    predictions.

    Feature modes (``feature_block``):
      - ``full``: all blocks.
      - ``no_pair_interaction``: keep writer and receiver marginals, drop all
        writer-receiver interaction tokens.
      - ``no_receiver``: drop receiver identity/profile and interaction, keep
        task/environment/memory/writer.
      - ``global_transfer``: keep only task context, environment and memory
        card semantics (global transfer critic, 清单 P1-1); drops writer,
        receiver and interaction.
      - ``memory_task_only``: legacy alias of ``global_transfer``.
      - ``no_writer_receiver`` (legacy): historical block kept only for old
        checkpoints; it removes writer and interaction while keeping receiver,
        a mixed definition superseded by the precise modes above.

    Forbidden: payload, procedure, labels, outcomes never enter features.
    """

    schema_version = "2.0"

    def __init__(self, *, n_features: int = 512, feature_block: str = "full") -> None:
        self.n_features = n_features
        self.feature_block = feature_block
        self._hasher = FeatureHasher(
            n_features=n_features,
            alternate_sign=False,
            input_type="string",
        )

    def _mode_flags(self) -> tuple[bool, bool, bool]:
        """Return (include_writer, include_receiver, include_interaction)."""
        mode = self.feature_block
        if mode == "full":
            return True, True, True
        if mode == "no_pair_interaction":
            return True, True, False
        if mode == "no_receiver":
            return True, False, False
        if mode in ("global_transfer", "memory_task_only"):
            return False, False, False
        if mode == "no_writer_receiver":  # legacy mixed block
            return False, True, False
        raise ValueError(f"unknown feature_block: {mode}")

    def tokens(self, item: CandidateExposureInput) -> list[str]:
        """Extract feature tokens from a CandidateExposureInput."""
        include_writer, include_receiver, include_interaction = self._mode_flags()
        tokens: list[str] = []
        rs = item.receiver_state
        card = item.candidate_card

        # --- task context block (scenario / task / environment) ---
        tokens.append(f"scenario:{rs.scenario}")
        for tok in _text_tokens(rs.task_instruction)[:8]:
            tokens.append(f"task_token:{tok}")
        for env in sorted(rs.environment_signature):
            tokens.append(f"env:{env}")

        # --- receiver marginal block ---
        if include_receiver:
            tokens.append(f"receiver_role:{rs.receiver.role}")
            for cap in sorted(rs.receiver.capabilities):
                tokens.append(f"receiver_cap:{cap}")
            for tool in sorted(rs.receiver.tool_names):
                tokens.append(f"receiver_tool:{tool}")

        # --- writer marginal block ---
        if include_writer:
            tokens.append(f"writer_role:{card.writer.role}")
            for cap in sorted(card.writer.capabilities):
                tokens.append(f"writer_cap:{cap}")
            for tool in sorted(card.writer.tool_names):
                tokens.append(f"writer_tool:{tool}")
            tokens.append(f"source_scenario:{card.source_scenario}")

        # --- writer-receiver interaction block ---
        if include_interaction:
            w_role = card.writer.role
            r_role = rs.receiver.role
            tokens.append(f"wr_pair:{w_role}->{r_role}")
            tokens.append(f"wr_same_role:{w_role == r_role}")
            w_caps = set(card.writer.capabilities)
            r_caps = set(rs.receiver.capabilities)
            tokens.append(f"wr_cap_overlap_bucket:{_overlap_bucket(w_caps, r_caps)}")
            w_tools = set(card.writer.tool_names)
            r_tools = set(rs.receiver.tool_names)
            tokens.append(f"wr_tool_overlap_bucket:{_overlap_bucket(w_tools, r_tools)}")
            tokens.append(f"writer_receiver_mismatch:{w_role != r_role}")

        # --- memory card block ---
        # Only outcome-independent, trajectory-observable attributes enter
        # features. Human-authored transfer hints and fixed compatible-role
        # lists are deliberately excluded (deprecated card fields).
        for tok in _text_tokens(card.goal_summary)[:6]:
            tokens.append(f"memory_goal_token:{tok}")
        for tag in sorted(card.task_tags):
            tokens.append(f"task_tag:{tag}")
        for constraint in sorted(card.environment_constraints):
            tokens.append(f"env_constraint:{constraint}")

        # --- v1 action space marker ---
        # The selected-memory prefix S is fixed to ∅ in SMTR-v1; prefix
        # contents are deliberately never encoded so predictions cannot
        # depend on a non-empty prefix passed through the compatibility
        # interface.
        tokens.append("prefix_size:0")

        self._reject_forbidden_tokens(tokens)
        return tokens

    def _reject_forbidden_tokens(self, tokens: list[str]) -> None:
        """Raise if forbidden leakage fields appear in output tokens."""
        for token in tokens:
            token_lower = token.lower()
            prefix = token_lower.split(":", 1)[0]
            if prefix in FORBIDDEN_FEATURE_TOKENS:
                raise ValueError(
                    f"forbidden transfer feature token detected: {token}"
                )

    def encode_one(self, item: CandidateExposureInput) -> Any:
        """Encode a single input to a sparse feature vector."""
        return self._hasher.transform([self.tokens(item)])

    def encode_batch(self, items: list[CandidateExposureInput]) -> Any:
        """Encode a batch of inputs."""
        return self._hasher.transform([self.tokens(item) for item in items])


def build_routing_card_from_pool_entry(mem_entry: dict[str, Any]) -> MemoryRoutingCard:
    """Build a MemoryRoutingCard from a memory-pool JSONL entry.

    This is the single card-construction path shared by the training loader
    and all evaluation builders, so train/inference features stay identical.
    Only routing-card metadata is used; the payload is never read.

    Deprecated human-authored fields (positive/negative transfer hints,
    compatible/incompatible receiver roles) are never restored, even when
    present in old pool entries, so they cannot re-enter features or
    baseline scores.
    """
    routing_card_data = mem_entry.get("routing_card", {})
    writer_data = routing_card_data.get("writer", {})
    writer = AgentProfile(
        agent_id=writer_data.get("agent_id", ""),
        role=writer_data.get("role", "unknown"),
        capabilities=tuple(writer_data.get("capabilities", [])),
        model_name=writer_data.get("model_name"),
        tool_names=tuple(writer_data.get("tool_names", [])),
    )
    return MemoryRoutingCard(
        memory_id=mem_entry["memory_id"],
        goal_summary=routing_card_data.get("goal_summary", ""),
        task_tags=tuple(routing_card_data.get("task_tags", [])),
        environment_constraints=tuple(routing_card_data.get("environment_constraints", [])),
        positive_transfer_hints=(),
        negative_transfer_hints=(),
        writer=writer,
        source_task_id=routing_card_data.get("source_task_id", ""),
        source_scenario=routing_card_data.get("source_scenario", "database"),
        compatible_receiver_roles=(),
        incompatible_receiver_roles=(),
        evidence_count=routing_card_data.get("evidence_count", 0),
        historical_success_count=routing_card_data.get("historical_success_count", 0),
        historical_failure_count=routing_card_data.get("historical_failure_count", 0),
        historical_success_rate=routing_card_data.get("historical_success_rate", 0.0),
    )


def load_paired_records_for_training(
    records_path: Path,
    memory_pool_path: Path,
) -> list[tuple[CandidateExposureInput, str]]:
    """Load paired records and construct (input, label) pairs for critic training.

    Thin wrapper over :func:`load_paired_records_with_metadata` for callers
    that do not need the underlying records.
    """
    return [
        (exposure_input, label)
        for exposure_input, label, _ in load_paired_records_with_metadata(
            records_path, memory_pool_path
        )
    ]


def load_paired_records_with_metadata(
    records_path: Path,
    memory_pool_path: Path,
) -> list[tuple[CandidateExposureInput, str, dict]]:
    """Load paired records into (input, label, record) triples.

    The raw record is kept alongside each training example so downstream
    consumers can group by treatment edge (清单 P0-3): edge-equal sample
    weights, edge-cluster bootstrap, edge-level calibration and split
    auditing all operate on the same key definitions.

    Only routing card metadata is used; payload is never read. All receiver
    context fields stored in the paired record (task instruction, environment
    signature, subtask, context summaries, writer/receiver tool_names and
    model_name) are restored so training features match inference features.

    Records failing the core-validity filter (incomplete branches, missing
    identity fields, cross-branch config mismatches, upstream invalid flag)
    never enter critic training, risk calibration or epsilon selection.
    """
    pool: dict[str, dict] = {}
    for line in memory_pool_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            mem = json.loads(line)
            pool[mem["memory_id"]] = mem

    results: list[tuple[CandidateExposureInput, str, dict]] = []
    for line in records_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if not is_core_valid_pair(rec):
            continue
        mem_entry = pool.get(rec["candidate_memory_id"])
        if mem_entry is None:
            continue
        card = build_routing_card_from_pool_entry(mem_entry)
        # Fall back to record-persisted writer fields only when the pool
        # entry lacks them (older pools).
        routing_card_data = mem_entry.get("routing_card", {})
        writer_data = routing_card_data.get("writer", {})
        if not writer_data.get("tool_names"):
            card = card.model_copy(update={
                "writer": card.writer.model_copy(update={
                    "agent_id": card.writer.agent_id or rec.get("writer_agent_id", ""),
                    "capabilities": card.writer.capabilities or tuple(rec.get("writer_capabilities", [])),
                    "tool_names": tuple(rec.get("writer_tool_names", [])),
                    "model_name": card.writer.model_name or rec.get("writer_model_name"),
                }),
            })
        receiver = AgentProfile(
            agent_id=rec.get("receiver_agent_id", ""),
            role=rec.get("receiver_role", "unknown"),
            capabilities=tuple(rec.get("receiver_capabilities", [])),
            model_name=rec.get("receiver_model_name"),
            tool_names=tuple(rec.get("receiver_tool_names", [])),
        )
        receiver_state = ReceiverState(
            task_id=rec["task_id"],
            scenario=rec.get("scenario", "database"),
            task_instruction=rec.get("task_instruction", ""),
            receiver=receiver,
            subtask=rec.get("subtask"),
            environment_signature=tuple(rec.get("environment_signature", [])),
            local_context_summary=rec.get("local_context_summary", ""),
            team_context_summary=rec.get("team_context_summary", ""),
        )
        exposure_input = CandidateExposureInput(
            receiver_state=receiver_state,
            candidate_card=card,
            selected_prefix_cards=(),
        )
        results.append((exposure_input, paired_record_label(rec), rec))
    return results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _text_tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def _overlap_bucket(set_a: set, set_b: set) -> str:
    if not set_a or not set_b:
        return "none"
    overlap = len(set_a & set_b) / max(1, len(set_a | set_b))
    if overlap == 0:
        return "none"
    if overlap < 0.33:
        return "low"
    if overlap < 0.66:
        return "medium"
    return "high"


def _count_bin(n: int) -> str:
    if n == 0:
        return "0"
    if n <= 2:
        return "1-2"
    if n <= 5:
        return "3-5"
    return "6+"
