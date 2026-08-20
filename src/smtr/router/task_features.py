"""Task context feature encoder for TCI ranker.

Encodes task context (instruction tokens, task tags, scenario)
into a fixed-dim numeric vector.

No task_id. Only structural attributes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


# ── Simple keyword vocabulary for task instruction tokenization ──

_INSTRUCTION_KEYWORDS: tuple[str, ...] = (
    "query",
    "database",
    "table",
    "insert",
    "update",
    "delete",
    "create",
    "schema",
    "migration",
    "backup",
    "restore",
    "replication",
    "transaction",
    "index",
    "constraint",
    "join",
    "aggregate",
    "filter",
    "sort",
    "validate",
)

_SCENARIO_VOCAB: tuple[str, ...] = (
    "database",
    "data_processing",
    "query",
    "migration",
    "backup",
)


@dataclass
class TaskFeatureEncoder:
    """Encode task context into a fixed-dim numeric vector.

    Uses only structural attributes: task tags, scenario type,
    instruction keyword presence. Excludes task_id.
    """

    instruction_keywords: tuple[str, ...] = _INSTRUCTION_KEYWORDS
    scenario_vocab: tuple[str, ...] = _SCENARIO_VOCAB

    @property
    def feature_dim(self) -> int:
        return (
            len(self.instruction_keywords)  # keyword presence
            + len(self.scenario_vocab) + 1  # +1 unk
            + 2  # scalars: instruction length, num tags
        )

    @property
    def feature_names(self) -> list[str]:
        names: list[str] = []
        for kw in self.instruction_keywords:
            names.append(f"instr_kw:{kw}")
        for s in self.scenario_vocab:
            names.append(f"scenario:{s}")
        names.append("scenario:<unk>")
        names.extend([
            "t_scalar:instr_length",
            "t_scalar:num_tags",
        ])
        return names

    def encode(self, task_context: dict[str, Any]) -> list[float]:
        """Encode task context dict into a feature vector.

        Parameters
        ----------
        task_context : dict with keys like
            task_instruction, task_tags, scenario.

        Returns
        -------
        Fixed-length list[float].
        """
        features: list[float] = []

        # Instruction keyword presence (bag-of-words).
        instruction = str(
            task_context.get("task_instruction", "")
        ).lower()
        tokens = set(instruction.split()) if instruction else set()
        for kw in self.instruction_keywords:
            features.append(1.0 if kw in tokens or kw in instruction else 0.0)

        # Scenario one-hot.
        scenario = str(task_context.get("scenario", "unknown"))
        features.extend(self._one_hot(scenario, self.scenario_vocab))

        # Scalar features.
        instr_len = len(instruction.split()) if instruction else 0
        features.append(min(instr_len / 50.0, 1.0))

        tags = task_context.get("task_tags", [])
        features.append(min(len(tags) / 5.0, 1.0))

        return features

    def _one_hot(
        self, value: str, vocab: tuple[str, ...]
    ) -> list[float]:
        vec = [0.0] * (len(vocab) + 1)
        if value in vocab:
            vec[vocab.index(value)] = 1.0
        else:
            vec[-1] = 1.0
        return vec
