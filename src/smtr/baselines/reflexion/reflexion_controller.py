"""Reflexion memory controller (NeurIPS 2023).

Original paper:
    Reflexion: Language Agents with Verbal Reinforcement Learning
    https://github.com/noahshinn/reflexion

Core idea:
    trajectory -> failure/success feedback -> verbal reflection
    -> reflection memory -> future prompt

SMTR mapping:
    Instead of raw experience, we store the *reflection text* as the
    candidate memory.  Every reflection is stored unconditionally (no
    TCI gate) — this matches the original paper where all verbal
    reflections are appended to the agent's episodic memory buffer.

Fairness constraints:
    - No LLM call: reflection text is synthesised deterministically
      from the trajectory (success/failure + topic).
    - No gradient update.
    - Same memory budget as SMTR when capacity is set.
"""

from __future__ import annotations

from typing import Any

from smtr.baselines.base_memory_controller import (
    BaseMemoryController,
    CandidateMemory,
    MemoryQuery,
)


class ReflexionController(BaseMemoryController):
    """Verbal reflection memory baseline.

    Every episode produces one reflection.  Reflections are stored
    unconditionally and retrieved by topic match (same retrieval
    interface as the ``retrieval`` baseline).
    """

    def __init__(self, *, top_k: int = 3) -> None:
        self._store: list[CandidateMemory] = []
        self._topic_index: dict[int, list[CandidateMemory]] = {}
        self._top_k = top_k
        self._reflection_count = 0

    # ------------------------------------------------------------------
    # BaseMemoryController interface
    # ------------------------------------------------------------------
    def extract_memory(self, trajectory: dict[str, Any]) -> list[CandidateMemory]:
        """Generate one reflection memory from the episode trajectory.

        The reflection text is a deterministic summary that encodes:
        success/failure, topic, and episode id.  This replaces the LLM
        reflection generator in the original paper with a fair, cost-free
        deterministic equivalent.
        """
        episode = trajectory["episode"]
        topic = trajectory["topic"]
        success = trajectory["success"]
        reward = trajectory.get("reward", float(success))
        content = trajectory.get("content", "")

        if success:
            reflection = (
                f"Episode {episode}: task on topic {topic} succeeded "
                f"(reward={reward:.2f}). Strategy: {content}. "
                f"Continue using this approach."
            )
        else:
            reflection = (
                f"Episode {episode}: task on topic {topic} failed "
                f"(reward={reward:.2f}). Attempted: {content}. "
                f"Reflection: avoid this strategy for topic {topic}."
            )

        memory_id = f"reflexion_ep{episode}_t{topic}"
        candidate = CandidateMemory(
            memory_id=memory_id,
            type="reflection",
            content=reflection,
            source_episode=episode,
            metadata={"topic": topic, "success": success, "reward": reward},
        )
        self._reflection_count += 1
        return [candidate]

    def update_memory(
        self,
        candidate: CandidateMemory,
        context: dict[str, Any],
    ) -> str:
        """All reflections are stored unconditionally (Reflexion paper).

        The original Reflexion keeps every verbal reflection in the
        episodic memory buffer; there is no discard or modification step.
        """
        self._store.append(candidate)
        topic = candidate.metadata.get("topic", -1)
        self._topic_index.setdefault(topic, []).append(candidate)
        return "store"

    def retrieve_memory(self, query: MemoryQuery) -> list[CandidateMemory]:
        """Retrieve reflections by topic match, most recent first."""
        candidates = self._topic_index.get(query.topic, [])
        # most recent first
        ordered = sorted(candidates, key=lambda m: m.source_episode, reverse=True)
        return ordered[: query.top_k]

    def get_statistics(self) -> dict[str, Any]:
        return {
            "reflection_count": self._reflection_count,
            "memory_size": len(self._store),
            "unique_topics": len(self._topic_index),
        }
