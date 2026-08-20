"""Memory card structural feature encoder for TCI ranker.

Encodes MemoryRoutingCard fields into a fixed-dim numeric vector.
Only structural/semantic fields are used; memory_id, digest,
and provenance are excluded.

Feature blocks:
  1. precondition_tags (multi-hot)
  2. environment_constraints (multi-hot)
  3. required_capabilities (multi-hot)
  4. required_tools (multi-hot)
  5. execution_role_tags (multi-hot)
  6. task_tags (multi-hot)
  7. procedure metadata (one-hot / scalar)
  8. complexity scalars (num tags, num constraints, etc.)

No writer identity. No memory_id. No digest. No hash.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ── Fixed vocabularies (determined from database_v1 domain) ──

_PRECONDITION_VOCAB: tuple[str, ...] = (
    "admin_role_required",
    "multi_region_consensus",
    "exclusive_table_lock",
    "audit_logging_disabled",
    "real_time_replication_enabled",
    "privileged_write_access",
)

_ENVIRONMENT_VOCAB: tuple[str, ...] = (
    "database_environment",
    "write_ahead_logging_disabled",
    "real_time_streaming_context",
    "temporary_table_creation_required",
    "gpu_acceleration_required",
    "cross_region_replication_needed",
)

_CAPABILITY_VOCAB: tuple[str, ...] = (
    "database_query",
    "planning",
    "reasoning",
    "tool_use",
    "communication",
    "coordination",
)

_TOOL_VOCAB: tuple[str, ...] = (
    "sql_execute",
    "file_read",
    "file_write",
    "api_call",
    "shell_execute",
)

_ROLE_VOCAB: tuple[str, ...] = (
    "planner",
    "executor",
    "critic",
    "verifier",
    "coordinator",
)

_TASK_TAG_VOCAB: tuple[str, ...] = (
    "database",
    "data_processing",
    "query",
    "migration",
    "backup",
)

_PROCEDURE_TYPE_VOCAB: tuple[str, ...] = (
    "observed_actions",
    "synthesized",
    "curated",
    "unknown",
)

_LENGTH_BUCKET_VOCAB: tuple[str, ...] = (
    "short",
    "medium",
    "long",
    "unknown",
)

_RW_SCOPE_VOCAB: tuple[str, ...] = (
    "read_only",
    "read_write",
    "write_only",
    "unknown",
)


@dataclass
class MemoryCardFeatureEncoder:
    """Encode a MemoryRoutingCard into a fixed-dim numeric vector.

    The vocabulary is fixed at construction time; unknown tokens
    are mapped to an ``<unk>`` slot at the end of each block.
    """

    precondition_vocab: tuple[str, ...] = _PRECONDITION_VOCAB
    environment_vocab: tuple[str, ...] = _ENVIRONMENT_VOCAB
    capability_vocab: tuple[str, ...] = _CAPABILITY_VOCAB
    tool_vocab: tuple[str, ...] = _TOOL_VOCAB
    role_vocab: tuple[str, ...] = _ROLE_VOCAB
    task_tag_vocab: tuple[str, ...] = _TASK_TAG_VOCAB
    procedure_type_vocab: tuple[str, ...] = _PROCEDURE_TYPE_VOCAB
    length_bucket_vocab: tuple[str, ...] = _LENGTH_BUCKET_VOCAB
    rw_scope_vocab: tuple[str, ...] = _RW_SCOPE_VOCAB

    @property
    def feature_dim(self) -> int:
        """Total feature dimension (all blocks)."""
        return (
            len(self.precondition_vocab) + 1  # +1 unk
            + len(self.environment_vocab) + 1
            + len(self.capability_vocab) + 1
            + len(self.tool_vocab) + 1
            + len(self.role_vocab) + 1
            + len(self.task_tag_vocab) + 1
            + len(self.procedure_type_vocab) + 1
            + len(self.length_bucket_vocab) + 1
            + len(self.rw_scope_vocab) + 1
            + 4  # complexity scalars
        )

    @property
    def feature_names(self) -> list[str]:
        """Ordered list of feature names."""
        names: list[str] = []
        for vocab_name, vocab in [
            ("precond", self.precondition_vocab),
            ("env", self.environment_vocab),
            ("cap", self.capability_vocab),
            ("tool", self.tool_vocab),
            ("role", self.role_vocab),
            ("task_tag", self.task_tag_vocab),
            ("proc_type", self.procedure_type_vocab),
            ("len_bucket", self.length_bucket_vocab),
            ("rw_scope", self.rw_scope_vocab),
        ]:
            for tok in vocab:
                names.append(f"{vocab_name}:{tok}")
            names.append(f"{vocab_name}:<unk>")
        names.extend([
            "scalar:num_preconditions",
            "scalar:num_env_constraints",
            "scalar:num_capabilities",
            "scalar:num_tools",
        ])
        return names

    def encode(self, card: Any) -> list[float]:
        """Encode a MemoryRoutingCard (or dict) into a feature vector.

        Parameters
        ----------
        card : MemoryRoutingCard or dict with routing card fields.

        Returns
        -------
        Fixed-length list[float] of feature values.
        """
        if isinstance(card, dict):
            get = card.get
        else:
            get = lambda k, d=(): getattr(card, k, d)

        features: list[float] = []

        # Multi-hot blocks.
        features.extend(self._multi_hot(
            get("precondition_tags", ()), self.precondition_vocab))
        features.extend(self._multi_hot(
            get("environment_constraints", ()), self.environment_vocab))
        features.extend(self._multi_hot(
            get("required_capabilities", ()), self.capability_vocab))
        features.extend(self._multi_hot(
            get("required_tools", ()), self.tool_vocab))
        features.extend(self._multi_hot(
            get("execution_role_tags", ()), self.role_vocab))
        features.extend(self._multi_hot(
            get("task_tags", ()), self.task_tag_vocab))

        # One-hot blocks.
        features.extend(self._one_hot(
            get("procedure_type", "unknown"), self.procedure_type_vocab))
        features.extend(self._one_hot(
            get("procedure_length_bucket", "unknown"), self.length_bucket_vocab))
        features.extend(self._one_hot(
            get("read_write_scope", "unknown"), self.rw_scope_vocab))

        # Complexity scalars (normalized).
        preconds = get("precondition_tags", ())
        envs = get("environment_constraints", ())
        caps = get("required_capabilities", ())
        tools = get("required_tools", ())
        features.append(_normalize_count(len(preconds)))
        features.append(_normalize_count(len(envs)))
        features.append(_normalize_count(len(caps)))
        features.append(_normalize_count(len(tools)))

        return features

    def _multi_hot(
        self, values: Any, vocab: tuple[str, ...]
    ) -> list[float]:
        """Multi-hot encoding with <unk> slot."""
        vec = [0.0] * (len(vocab) + 1)
        if values is None:
            return vec
        for v in values:
            v_str = str(v)
            if v_str in vocab:
                vec[vocab.index(v_str)] = 1.0
            else:
                vec[-1] = 1.0  # <unk>
        return vec

    def _one_hot(
        self, value: Any, vocab: tuple[str, ...]
    ) -> list[float]:
        """One-hot encoding with <unk> slot."""
        vec = [0.0] * (len(vocab) + 1)
        v_str = str(value) if value else "unknown"
        if v_str in vocab:
            vec[vocab.index(v_str)] = 1.0
        else:
            vec[-1] = 1.0
        return vec


def _normalize_count(n: int, max_expected: int = 10) -> float:
    """Normalize a count to [0, 1] range."""
    return min(float(n) / max(max_expected, 1), 1.0)
