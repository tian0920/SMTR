"""Receiver context feature encoder for TCI ranker.

Encodes receiver agent context (role, capabilities, tools,
environment signature) into a fixed-dim numeric vector.

No agent_id. No identity fields. Only structural attributes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


# ── Fixed vocabularies ──

_ROLE_VOCAB: tuple[str, ...] = (
    "planner",
    "executor",
    "critic",
    "verifier",
    "coordinator",
    "unknown",
)

_CAPABILITY_VOCAB: tuple[str, ...] = (
    "database_query",
    "planning",
    "reasoning",
    "tool_use",
    "communication",
    "coordination",
    "file_io",
    "api_call",
)

_TOOL_VOCAB: tuple[str, ...] = (
    "sql_execute",
    "file_read",
    "file_write",
    "api_call",
    "shell_execute",
)

_ENV_VOCAB: tuple[str, ...] = (
    "database_environment",
    "write_ahead_logging_disabled",
    "real_time_streaming_context",
    "temporary_table_creation_required",
    "gpu_acceleration_required",
    "cross_region_replication_needed",
)


@dataclass
class ReceiverFeatureEncoder:
    """Encode receiver context into a fixed-dim numeric vector.

    Uses only structural attributes: role, capabilities, tools,
    environment signature. Excludes agent_id to prevent identity
    memorization.
    """

    role_vocab: tuple[str, ...] = _ROLE_VOCAB
    capability_vocab: tuple[str, ...] = _CAPABILITY_VOCAB
    tool_vocab: tuple[str, ...] = _TOOL_VOCAB
    env_vocab: tuple[str, ...] = _ENV_VOCAB

    @property
    def feature_dim(self) -> int:
        return (
            len(self.role_vocab) + 1        # +1 unk
            + len(self.capability_vocab) + 1
            + len(self.tool_vocab) + 1
            + len(self.env_vocab) + 1
            + 3  # scalar counts
        )

    @property
    def feature_names(self) -> list[str]:
        names: list[str] = []
        for block_name, vocab in [
            ("r_role", self.role_vocab),
            ("r_cap", self.capability_vocab),
            ("r_tool", self.tool_vocab),
            ("r_env", self.env_vocab),
        ]:
            for tok in vocab:
                names.append(f"{block_name}:{tok}")
            names.append(f"{block_name}:<unk>")
        names.extend([
            "r_scalar:num_capabilities",
            "r_scalar:num_tools",
            "r_scalar:num_env",
        ])
        return names

    def encode(self, receiver_context: dict[str, Any]) -> list[float]:
        """Encode receiver context dict into a feature vector.

        Parameters
        ----------
        receiver_context : dict with keys like
            receiver_role, receiver_capabilities,
            receiver_tool_names, environment_signature.

        Returns
        -------
        Fixed-length list[float].
        """
        features: list[float] = []

        # Role one-hot.
        role = receiver_context.get("receiver_role", "unknown")
        features.extend(self._one_hot(str(role), self.role_vocab))

        # Capabilities multi-hot.
        caps = receiver_context.get("receiver_capabilities", [])
        features.extend(self._multi_hot(caps, self.capability_vocab))

        # Tools multi-hot.
        tools = receiver_context.get("receiver_tool_names", [])
        features.extend(self._multi_hot(tools, self.tool_vocab))

        # Environment multi-hot.
        env = receiver_context.get("environment_signature", [])
        features.extend(self._multi_hot(env, self.env_vocab))

        # Scalar counts.
        features.append(_norm(len(caps)))
        features.append(_norm(len(tools)))
        features.append(_norm(len(env)))

        return features

    def _multi_hot(
        self, values: Any, vocab: tuple[str, ...]
    ) -> list[float]:
        vec = [0.0] * (len(vocab) + 1)
        if values is None:
            return vec
        for v in values:
            v_str = str(v)
            if v_str in vocab:
                vec[vocab.index(v_str)] = 1.0
            else:
                vec[-1] = 1.0
        return vec

    def _one_hot(
        self, value: str, vocab: tuple[str, ...]
    ) -> list[float]:
        vec = [0.0] * (len(vocab) + 1)
        if value in vocab:
            vec[vocab.index(value)] = 1.0
        else:
            vec[-1] = 1.0
        return vec


def _norm(n: int, max_expected: int = 10) -> float:
    return min(float(n) / max(max_expected, 1), 1.0)
