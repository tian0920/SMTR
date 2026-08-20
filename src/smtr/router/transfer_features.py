"""Memory-receiver transfer feature encoder (writer-agnostic blocks).

Feature blocks:
  - task context block (scenario, task tokens, environment)
  - receiver marginal block (role, capabilities, tools)
  - memory marginal block (goal, tags, explicit requirements, procedure metadata)
  - memory behavioral procedure signature ψ(m) block (P1-B)
  - memory-receiver compatibility interaction block (φ_rm)
  - task-memory applicability interaction block φ_tm (P1-A)
"""

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

# 清单 Writer-Agnostic 7.1: any feature token whose prefix matches one of
# these provenance/writer names fails the feature audit immediately.
FORBIDDEN_PROVENANCE_FEATURE_PREFIXES = frozenset({
    "writer",
    "writer_role",
    "writer_cap",
    "writer_tool",
    "wr_pair",
    "wr_same_role",
    "source_agent",
    "source_agent_role",
    "memory_source_agent",
    "source_trajectory",
})

TOKEN_RE = re.compile(r"[a-z0-9]+")

# Stop words for domain-keyword extraction from task instructions.
# Removing these leaves domain-specific terms (finance, streaming, etc.).
_STOP_WORDS = frozenset(
    "a an the is are was were to of and in for on with by at from as "
    "be been being have has had do does did will would shall should may "
    "might can could this that these those it its they them their what "
    "which who whom how when where why all each every both some any no "
    "not but or if then so than too very just about up out over under "
    "into through during before after above below between same different "
    "recently recently use uses using used find findout what wrong reason "
    "caused cause root planner assign agent analyze possibility also".split()
)

# Table-name → abstract category mapping for procedure signature.
_TABLE_CATEGORY_MAP = {
    "users": "user_mgmt",
    "accounts": "financial",
    "transactions": "financial",
    "investments": "financial",
    "investment_transactions": "financial",
    "payments": "financial",
    "posts": "content",
    "comments": "content",
    "likes": "content",
    "followers": "social",
    "messages": "communication",
    "media": "content",
    "files": "file_mgmt",
    "shared_files": "file_mgmt",
    "file_access_logs": "logging",
    "access_logs": "logging",
    "orders": "commerce",
    "products": "commerce",
    "inventory": "commerce",
    "suppliers": "commerce",
    "artists": "content",
    "albums": "content",
    "songs": "content",
    "playlists": "content",
    "subscriptions": "subscription",
    "user_activities": "activity",
    "logs": "logging",
    "configurations": "config",
    "devices": "iot",
    "sensors": "iot",
    "sensor_data": "iot",
    "notifications": "communication",
}

# Domain keyword extraction patterns from task content.
_DOMAIN_KEYWORDS = {
    "finance": ["financial", "finance", "account", "transaction", "investment", "payment", "banking"],
    "social_media": ["social media", "post", "comment", "follow", "like", "friend", "share"],
    "file_sharing": ["file sharing", "file", "shared", "upload", "download", "document"],
    "music_streaming": ["music", "song", "album", "artist", "playlist", "streaming"],
    "ecommerce": ["product", "order", "cart", "shopping", "store", "inventory", "supplier"],
    "iot": ["iot", "device", "sensor", "monitor", "telemetry", "configuration"],
    "healthcare": ["patient", "medical", "health", "hospital", "clinic", "diagnosis"],
    "education": ["student", "course", "enrollment", "grade", "university", "school"],
    "manufacturing": ["manufacturing", "production", "factory", "assembly", "supply chain"],
    "transportation": ["transport", "vehicle", "route", "shipment", "logistics", "delivery"],
    "real_estate": ["property", "real estate", "building", "tenant", "lease", "rental"],
    "gaming": ["game", "player", "score", "level", "match", "tournament"],
}


class HashingTransferFeatureEncoder:
    """Deterministic feature encoder for memory-receiver transfer prediction.

    Feature blocks (清单 Writer-Agnostic 第六章 + P1-A/P1-B):
      - task context block (scenario, task tokens, environment)
      - receiver marginal block (role, capabilities, tools)
      - memory marginal block (goal, tags, explicit requirements,
        procedure metadata)
      - memory behavioral procedure signature ψ(m) block (P1-B)
      - memory-receiver compatibility interaction block (φ_rm)
      - task-memory applicability interaction block φ_tm (P1-A)

    Writer/source-agent identity is never encoded: provenance stays in the
    memory payload and never enters features.

    SMTR-v1 action space is A(o_r) in {∅, m_1, ..., m_K}: one receiver
    receives at most one memory, so the selected-memory prefix S = ∅.

    Feature modes (``feature_block``):
      - ``full``: task/environment + receiver marginal + memory marginal
        + procedure signature + memory-receiver compatibility interaction
        + task-memory applicability interaction.
      - ``no_compatibility_interaction``: drop the explicit memory-receiver
        interaction block.
      - ``global_transfer``: task/environment + memory marginal only;
        receiver identity is dropped entirely.

    Forbidden: payload, procedure bodies, labels, outcomes and any
    writer/provenance token never enter features.
    """

    schema_version = "3.1"

    def __init__(self, *, n_features: int = 512, feature_block: str = "full") -> None:
        self.n_features = n_features
        self.feature_block = feature_block
        self._hasher = FeatureHasher(
            n_features=n_features,
            alternate_sign=False,
            input_type="string",
        )

    def _mode_flags(self) -> tuple[bool, bool]:
        """Return (include_receiver, include_compatibility).

        There is no writer flag: writer identity is never a feature
        (清单 Writer-Agnostic 6.2).
        """
        mode = self.feature_block
        if mode == "full":
            return True, True
        if mode == "no_compatibility_interaction":
            return True, False
        if mode == "global_transfer":
            return False, False
        raise ValueError(f"unknown feature_block: {mode}")

    def tokens(self, item: CandidateExposureInput) -> list[str]:
        """Extract feature tokens from a CandidateExposureInput.

        Conditioning is (t, x_r^pre, m, r) only (清单 Writer-Agnostic
        第六章): no writer/provenance token is ever emitted.
        """
        include_receiver, include_compatibility = self._mode_flags()
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

        # --- memory marginal block (explicit requirements, no writer) ---
        for tok in _text_tokens(card.goal_summary)[:6]:
            tokens.append(f"memory_goal_token:{tok}")
        for tag in sorted(card.task_tags):
            tokens.append(f"memory_task_tag:{tag}")
        for tool in sorted(card.required_tools):
            tokens.append(f"memory_required_tool:{tool}")
        for cap in sorted(card.required_capabilities):
            tokens.append(f"memory_required_capability:{cap}")
        for role in sorted(card.execution_role_tags):
            tokens.append(f"memory_execution_role:{role}")
        for constraint in sorted(card.environment_constraints):
            tokens.append(f"memory_environment_constraint:{constraint}")
        for tag in sorted(card.precondition_tags):
            tokens.append(f"memory_precondition_tag:{tag}")
        tokens.append(f"memory_procedure_type:{card.procedure_type}")
        tokens.append(f"memory_length_bucket:{card.procedure_length_bucket}")
        tokens.append(f"memory_read_write_scope:{card.read_write_scope}")

        # --- memory behavioral procedure signature ψ(m) block (P1-B) ---
        # Writer-agnostic, outcome-free, pre-execution available.
        # Describes WHAT kind of procedure this memory encodes.
        for tag in sorted(card.procedure_domain_tags):
            tokens.append(f"psi_domain:{tag}")
        for cat in sorted(card.procedure_table_categories):
            tokens.append(f"psi_table_cat:{cat}")
        for pat in sorted(card.procedure_pattern_tags):
            tokens.append(f"psi_pattern:{pat}")
        if card.procedure_domain_tags:
            tokens.append(
                f"psi_domain_count:{_count_bin(len(card.procedure_domain_tags))}"
            )
        if card.procedure_table_categories:
            tokens.append(
                f"psi_table_cat_count:{_count_bin(len(card.procedure_table_categories))}"
            )

        # --- task-memory applicability interaction block φ_tm (P1-A) ---
        # Directly models task–memory semantic compatibility.
        # This is g(t,m), a deterministic function of task and memory content.
        task_tokens_set = set(_text_tokens(rs.task_instruction))
        memory_tokens_set = (
            set(_text_tokens(card.goal_summary))
            | set(t.lower() for t in card.task_tags)
        )
        # (a) Shared lexical tokens between task instruction and memory
        shared = (task_tokens_set & memory_tokens_set) - _STOP_WORDS
        for tok in sorted(shared)[:6]:
            tokens.append(f"tm_shared_token:{tok}")
        tokens.append(f"tm_shared_token_count:{_count_bin(len(shared))}")
        # (b) Task domain keywords vs memory domain tags
        task_domain_kw = _extract_domain_keywords(rs.task_instruction)
        if task_domain_kw and card.procedure_domain_tags:
            domain_overlap = task_domain_kw & set(card.procedure_domain_tags)
            tokens.append(
                f"tm_domain_match:{_overlap_bucket(domain_overlap, set(card.procedure_domain_tags))}"
            )
        else:
            tokens.append("tm_domain_match:none")
        # (c) Task table categories vs memory table categories
        task_table_cats = _extract_table_categories(rs.task_instruction)
        if task_table_cats and card.procedure_table_categories:
            table_overlap = task_table_cats & set(card.procedure_table_categories)
            tokens.append(
                f"tm_table_cat_match:"
                f"{_overlap_bucket(table_overlap, set(card.procedure_table_categories))}"
            )
        else:
            tokens.append("tm_table_cat_match:none")
        # (d) Task instruction × memory goal lexical overlap
        tm_overlap = _overlap_bucket(
            task_tokens_set - _STOP_WORDS,
            memory_tokens_set,
        )
        tokens.append(f"tm_lexical_overlap:{tm_overlap}")
        # (e) Task environment × memory environment constraints
        task_env = set(rs.environment_signature)
        tokens.append(
            f"tm_environment_match:"
            f"{_satisfaction_bucket(set(card.environment_constraints), task_env)}"
        )
        # (f) Memory required tools × task context (operation match)
        task_tool_tokens = {
            t for t in task_tokens_set
            if t in {"sql", "query", "execute", "select", "insert", "update",
                     "delete", "create", "drop", "alter", "index", "vacuum",
                     "lock", "table", "schema", "database"}
        }
        memory_tool_tokens = set()
        for t in card.required_tools:
            memory_tool_tokens.update(_text_tokens(t))
        if task_tool_tokens and memory_tool_tokens:
            op_overlap = task_tool_tokens & memory_tool_tokens
            tokens.append(
                f"tm_operation_match:{_overlap_bucket(op_overlap, memory_tool_tokens)}"
            )
        else:
            tokens.append("tm_operation_match:none")

        # --- memory-receiver compatibility interaction block (full only) ---
        # Derived from routing card + receiver state only; never payload,
        # procedure, outcomes or provenance (清单 6.5).
        if include_compatibility:
            r_caps = set(rs.receiver.capabilities)
            r_tools = set(rs.receiver.tool_names)
            r_env = set(rs.environment_signature)
            tokens.append(
                "mr_tool_satisfaction:"
                f"{_satisfaction_bucket(set(card.required_tools), r_tools)}"
            )
            tokens.append(
                "mr_capability_satisfaction:"
                f"{_satisfaction_bucket(set(card.required_capabilities), r_caps)}"
            )
            tokens.append(
                "mr_environment_satisfaction:"
                f"{_satisfaction_bucket(set(card.environment_constraints), r_env)}"
            )
            if card.execution_role_tags:
                role_ok = rs.receiver.role in card.execution_role_tags
                tokens.append(f"mr_role_satisfaction:{str(role_ok).lower()}")
            else:
                tokens.append("mr_role_satisfaction:unspecified")
            tokens.append(
                "mr_missing_tool_count:"
                f"{_missing_count_bucket(len(set(card.required_tools) - r_tools))}"
            )
            tokens.append(
                "mr_missing_capability_count:"
                f"{_missing_count_bucket(len(set(card.required_capabilities) - r_caps))}"
            )
            tokens.append(
                "mr_read_write_compatible:"
                f"{str(_read_write_compatible(card, rs.receiver)).lower()}"
            )

        # --- v1 action space marker ---
        # The selected-memory prefix S is fixed to ∅ in SMTR-v1; prefix
        # contents are deliberately never encoded so predictions cannot
        # depend on a non-empty prefix passed through the compatibility
        # interface.
        tokens.append("prefix_size:0")

        self._reject_forbidden_tokens(tokens)
        return tokens

    def _reject_forbidden_tokens(self, tokens: list[str]) -> None:
        """Raise if forbidden leakage or provenance fields appear in tokens."""
        for token in tokens:
            token_lower = token.lower()
            prefix = token_lower.split(":", 1)[0]
            if prefix in FORBIDDEN_FEATURE_TOKENS:
                raise ValueError(
                    f"forbidden transfer feature token detected: {token}"
                )
            if any(prefix.startswith(banned) for banned in FORBIDDEN_PROVENANCE_FEATURE_PREFIXES):
                raise ValueError(
                    f"forbidden provenance/writer feature token detected: {token}"
                )

    def encode_one(self, item: CandidateExposureInput) -> Any:
        """Encode a single input to a sparse feature vector."""
        return self._hasher.transform([self.tokens(item)])

    def encode_batch(self, items: list[CandidateExposureInput]) -> Any:
        """Encode a batch of inputs."""
        return self._hasher.transform([self.tokens(item) for item in items])

    # ------------------------------------------------------------------
    # Baseline-only encoder path (Counterfactual Opportunity v1)
    # ------------------------------------------------------------------
    # Encodes (t, o_r, r) only — strictly NO candidate memory tokens.
    # Used by the baseline head to learn b = P(Y_0=1 | t, o_r, r).

    def baseline_tokens(self, receiver_state: ReceiverState) -> list[str]:
        """Feature tokens for the baseline head — no memory allowed.

        Only scenario, task tokens, environment signature, receiver role,
        capabilities and tools. No memory_goal_token, memory_task_tag,
        psi_*, rm_*, tm_*, source_agent, writer, outcome or label.
        """
        rs = receiver_state
        tokens: list[str] = []
        tokens.append(f"scenario:{rs.scenario}")
        for tok in _text_tokens(rs.task_instruction)[:8]:
            tokens.append(f"task_token:{tok}")
        for env in sorted(rs.environment_signature):
            tokens.append(f"env:{env}")
        tokens.append(f"receiver_role:{rs.receiver.role}")
        for cap in sorted(rs.receiver.capabilities):
            tokens.append(f"receiver_cap:{cap}")
        for tool in sorted(rs.receiver.tool_names):
            tokens.append(f"receiver_tool:{tool}")
        # Forbidden check: baseline tokens must never contain memory,
        # provenance or outcome prefixes.
        self._reject_forbidden_tokens(tokens)
        _reject_baseline_memory_tokens(tokens)
        return tokens

    def encode_baseline_one(self, receiver_state: ReceiverState) -> Any:
        """Encode one receiver state for the baseline head."""
        return self._hasher.transform([self.baseline_tokens(receiver_state)])

    def encode_baseline_batch(
        self, receiver_states: list[ReceiverState]
    ) -> Any:
        """Encode a batch of receiver states for the baseline head."""
        return self._hasher.transform(
            [self.baseline_tokens(rs) for rs in receiver_states]
        )


def build_routing_card_from_pool_entry(mem_entry: dict[str, Any]) -> MemoryRoutingCard:
    """Build a MemoryRoutingCard from a memory-pool JSONL entry.

    This is the single card-construction path shared by the training loader
    and all evaluation builders, so train/inference features stay identical.
    Only routing-card metadata is used; the payload (including provenance)
    is never read.

    Writer-agnostic (清单 Writer-Agnostic 6.6): no ``writer`` profile is
    constructed and no legacy record writer fields are restored. Legacy
    routing-card schemas fail closed instead of silently falling back.
    """
    routing_card_data = mem_entry.get("routing_card", {})
    if "writer" in routing_card_data or "required_tools" not in routing_card_data:
        raise ValueError(
            "legacy routing-card schema detected; rebuild the memory pool "
            "with routing-card schema v3 (writer-agnostic)"
        )
    # --- Behavioral procedure signature ψ(m) (P1-B) ---
    # Derived from source task content when available; otherwise from
    # pre-computed routing_card fields.
    source_task_content = mem_entry.get("source_task_content", "")
    if source_task_content:
        procedure_domain_tags = tuple(sorted(
            _extract_domain_keywords(source_task_content)
        ))
        procedure_table_categories = tuple(sorted(
            _extract_table_categories(source_task_content)
        ))
    else:
        procedure_domain_tags = tuple(
            routing_card_data.get("procedure_domain_tags", [])
        )
        procedure_table_categories = tuple(
            routing_card_data.get("procedure_table_categories", [])
        )
    procedure_pattern_tags = tuple(
        routing_card_data.get("procedure_pattern_tags", [])
    )
    return MemoryRoutingCard(
        memory_id=mem_entry["memory_id"],
        goal_summary=routing_card_data.get("goal_summary", ""),
        task_tags=tuple(routing_card_data.get("task_tags", [])),
        required_tools=tuple(routing_card_data.get("required_tools", [])),
        required_capabilities=tuple(routing_card_data.get("required_capabilities", [])),
        execution_role_tags=tuple(routing_card_data.get("execution_role_tags", [])),
        environment_constraints=tuple(routing_card_data.get("environment_constraints", [])),
        precondition_tags=tuple(routing_card_data.get("precondition_tags", [])),
        procedure_type=routing_card_data.get("procedure_type", "unknown"),
        procedure_length_bucket=routing_card_data.get("procedure_length_bucket", "unknown"),
        read_write_scope=routing_card_data.get("read_write_scope", "unknown"),
        evidence_count=routing_card_data.get("evidence_count", 0),
        procedure_domain_tags=procedure_domain_tags,
        procedure_table_categories=procedure_table_categories,
        procedure_pattern_tags=procedure_pattern_tags,
    )


def load_paired_records_for_training(
    records_path: Path,
    memory_pool_path: Path,
    *,
    marble_source_path: Path | None = None,
) -> list[tuple[CandidateExposureInput, str]]:
    """Load paired records and construct (input, label) pairs for critic training.

    Thin wrapper over :func:`load_paired_records_with_metadata` for callers
    that do not need the underlying records.
    """
    return [
        (exposure_input, label)
        for exposure_input, label, _ in load_paired_records_with_metadata(
            records_path, memory_pool_path,
            marble_source_path=marble_source_path,
        )
    ]


def load_paired_records_with_metadata(
    records_path: Path,
    memory_pool_path: Path,
    *,
    marble_source_path: Path | None = None,
) -> list[tuple[CandidateExposureInput, str, dict]]:
    """Load paired records into (input, label, record) triples.

    The raw record is kept alongside each training example so downstream
    consumers can group by treatment edge (清单 P0-3): edge-equal sample
    weights, edge-cluster bootstrap, edge-level calibration and split
    auditing all operate on the same key definitions.

    Only routing card metadata is used; payload is never read. All receiver
    context fields stored in the paired record (task instruction, environment
    signature, subtask, context summaries, receiver tool_names and
    model_name) are restored so training features match inference features.

    When ``marble_source_path`` is provided, task instructions are injected
    from the MARBLE source database for records whose ``task_instruction``
    field is empty.  This is the P1-A path: the MARBLE source is the
    authoritative task-content store, not the paired record.

    Records failing the core-validity filter (incomplete branches, missing
    identity fields, cross-branch config mismatches, upstream invalid flag)
    never enter critic training, risk calibration or epsilon selection.
    """
    raw_records: list[dict] = []
    for line in records_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            raw_records.append(json.loads(line))
    return build_training_data_from_records(
        raw_records, memory_pool_path,
        marble_source_path=marble_source_path,
    )


def build_training_data_from_records(
    records: list[dict],
    memory_pool_path: Path,
    *,
    marble_source_path: Path | None = None,
) -> list[tuple[CandidateExposureInput, str, dict]]:
    """Build (input, label, record) triples from an explicit record list.

    清单 Fixed-Budget 第7章: budget filtering happens before feature
    construction, so the caller passes the already-filtered effective
    training records and features/labels are never built from the full
    parent record file.

    When ``marble_source_path`` is given, task instructions missing from
    paired records are back-filled from the MARBLE source (P1-A).
    """
    pool: dict[str, dict] = {}
    for line in memory_pool_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            mem = json.loads(line)
            pool[mem["memory_id"]] = mem

    # Optional MARBLE source lookup for task instruction injection (P1-A).
    marble_tasks: dict[str, str] = {}
    if marble_source_path is not None and marble_source_path.exists():
        for line in marble_source_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                task_rec = json.loads(line)
                tid = str(task_rec.get("task_id", ""))
                content = task_rec.get("task", {}).get("content", "")
                if tid and content:
                    marble_tasks[tid] = content

    results: list[tuple[CandidateExposureInput, str, dict]] = []
    for rec in records:
        if not is_core_valid_pair(rec):
            continue
        mem_entry = pool.get(rec["candidate_memory_id"])
        if mem_entry is None:
            continue
        card = build_routing_card_from_pool_entry(mem_entry)
        # 清单 Writer-Agnostic 6.7: training inputs are built from the
        # receiver record fields, the memory routing card and the
        # task/environment context only; record-level writer/provenance
        # fields (memory_source_*) never enter CandidateExposureInput.
        receiver = AgentProfile(
            agent_id=rec.get("receiver_agent_id", ""),
            role=rec.get("receiver_role", "unknown"),
            capabilities=tuple(rec.get("receiver_capabilities", [])),
            model_name=rec.get("receiver_model_name"),
            tool_names=tuple(rec.get("receiver_tool_names", [])),
        )
        # P1-A: inject task instruction from MARBLE source when the
        # paired record has an empty task_instruction field.
        task_instruction = rec.get("task_instruction", "")
        if not task_instruction and marble_tasks:
            task_instruction = marble_tasks.get(str(rec.get("task_id", "")), "")
        receiver_state = ReceiverState(
            task_id=rec["task_id"],
            scenario=rec.get("scenario", "database"),
            task_instruction=task_instruction,
            receiver=receiver,
            subtask=rec.get("subtask"),
            environment_signature=tuple(rec.get("environment_signature", [])),
            local_context_summary=rec.get("local_context_summary", ""),
            team_context_summary=rec.get("team_context_summary", ""),
        )
        exposure_input = CandidateExposureInput(
            receiver_state=receiver_state,
            candidate_card=card,
        )
        results.append((exposure_input, paired_record_label(rec), rec))
    return results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


# Baseline-forbidden prefixes: these must never appear in baseline tokens.
_BASELINE_FORBIDDEN_PREFIXES = frozenset({
    "memory_goal_token", "memory_task_tag", "memory_required_tool",
    "memory_required_capability", "memory_execution_role",
    "memory_environment_constraint", "memory_precondition_tag",
    "memory_procedure_type", "memory_length_bucket", "memory_read_write_scope",
    "psi_domain", "psi_table_cat", "psi_pattern",
    "psi_domain_count", "psi_table_cat_count",
    "tm_shared_token", "tm_shared_token_count", "tm_domain_match",
    "tm_table_cat_match", "tm_lexical_overlap", "tm_environment_match",
    "tm_operation_match",
    "mr_tool_satisfaction", "mr_capability_satisfaction",
    "mr_environment_satisfaction", "mr_role_satisfaction",
    "mr_missing_tool_count", "mr_missing_capability_count",
    "mr_read_write_compatible",
    "prefix_size",
})


def _reject_baseline_memory_tokens(tokens: list[str]) -> None:
    """Raise if any baseline token has a memory/provenance prefix."""
    for token in tokens:
        prefix = token.lower().split(":", 1)[0]
        if prefix in _BASELINE_FORBIDDEN_PREFIXES:
            raise ValueError(
                f"baseline token contains forbidden memory prefix: {token}"
            )


def _text_tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def _satisfaction_bucket(required: set, available: set) -> str:
    """full/partial/none bucket for requirement satisfaction (清单 6.5)."""
    if not required:
        return "full"
    ratio = len(required & available) / len(required)
    if ratio >= 1.0:
        return "full"
    if ratio > 0.0:
        return "partial"
    return "none"


def _missing_count_bucket(count: int) -> str:
    if count == 0:
        return "0"
    if count == 1:
        return "1"
    return "2plus"


def _read_write_compatible(card: MemoryRoutingCard, receiver: AgentProfile) -> bool:
    """Deterministic read/write scope compatibility (清单 6.5).

    Read-only procedures are compatible with every receiver; a write-scope
    procedure requires an explicit write capability or write tool on the
    receiver side.
    """
    if card.read_write_scope != "write":
        return True
    return any("write" in cap.lower() for cap in receiver.capabilities) or any(
        "write" in tool.lower() for tool in receiver.tool_names
    )


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


def _extract_domain_keywords(text: str) -> set[str]:
    """Extract abstract domain labels from a task instruction or memory content.

    Scans ``text`` for keyword patterns defined in ``_DOMAIN_KEYWORDS`` and
    returns the matching domain names (e.g. ``{"finance", "social_media"}``).
    Uses word-boundary matching to avoid substring false positives.
    The result is writer-agnostic and outcome-free.
    """
    if not text:
        return set()
    text_lower = text.lower()
    matched: set[str] = set()
    for domain, keywords in _DOMAIN_KEYWORDS.items():
        for kw in keywords:
            # Use word-boundary regex to avoid substring matches
            # (e.g. "lease" matching inside "please").
            pattern = r"\b" + re.escape(kw) + r"\b"
            if re.search(pattern, text_lower):
                matched.add(domain)
                break
    return matched


def _extract_table_categories(text: str) -> set[str]:
    """Extract abstract table categories from text mentioning table names.

    Maps concrete table names found in ``text`` to their abstract category
    via ``_TABLE_CATEGORY_MAP`` (e.g. ``users`` -> ``user_mgmt``,
    ``transactions`` -> ``financial``).  The result is writer-agnostic.
    """
    if not text:
        return set()
    tokens = set(_text_tokens(text))
    cats: set[str] = set()
    for table_name, category in _TABLE_CATEGORY_MAP.items():
        if table_name in tokens:
            cats.add(category)
    return cats


def prediction_input_from_record(record: Any) -> CandidateExposureInput:
    """Build a CandidateExposureInput from a PairedInterventionRecord.

    Used by the leakage scanner and diagnostic tools to feed records
    through the feature encoder.  Writer/provenance fields are never
    copied into the input (清单 Writer-Agnostic 6.7).
    """
    ctx = record.decision_context
    receiver = AgentProfile(
        agent_id=ctx.receiver_agent_id,
        role=ctx.receiver_role,
        capabilities=tuple(ctx.receiver_capabilities),
    )
    receiver_state = ReceiverState(
        task_id=ctx.task_id,
        scenario="",
        task_instruction="",
        receiver=receiver,
    )
    snap = record.candidate_card_snapshot
    if snap is not None:
        card = MemoryRoutingCard(
            memory_id=snap.memory_id,
            goal_summary=snap.goal_summary,
            task_tags=tuple(snap.task_tags),
        )
    else:
        card = MemoryRoutingCard(
            memory_id=record.candidate_memory_id,
            goal_summary="",
        )
    return CandidateExposureInput(
        receiver_state=receiver_state,
        candidate_card=card,
    )
