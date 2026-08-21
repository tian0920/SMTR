"""AGILE-inspired experience consolidation controller (NeurIPS 2024).

Original paper:
    AGILE: A Novel Reinforcement Learning Framework of LLM Agents
    https://github.com/bytarnish/AGILE

Core idea:
    trajectory -> experience buffer -> experience summariser
    -> knowledge memory (prioritised by reward, novelty, consequence)

SMTR mapping:
    We cannot run RL gradient updates (would be unfair).  Instead we
    implement only the *experience consolidation* policy:
      - Extract state / action / outcome / lesson from trajectory.
      - Score each experience by (reward + novelty + consequence).
      - Store only the top-k experiences when the budget is limited.

Fairness constraints:
    - No LLM parameter update.
    - No gradient step.
    - Same memory budget as SMTR when capacity is set.
"""

from __future__ import annotations

from typing import Any

from smtr.baselines.base_memory_controller import (
    BaseMemoryController,
    CandidateMemory,
    MemoryQuery,
)


class AgileController(BaseMemoryController):
    """Experience consolidation baseline inspired by AGILE.

    Experiences are scored by a composite of reward, novelty (topic
    uniqueness in the current bank), and consequence magnitude.  When
    the memory budget is exceeded the lowest-scored experience is
    evicted.
    """

    def __init__(self, *, top_k: int | None = None) -> None:
        self._store: list[CandidateMemory] = []
        self._topic_index: dict[int, list[CandidateMemory]] = {}
        self._topic_counts: dict[int, int] = {}
        self._top_k = top_k  # memory budget (None = unlimited)
        self._total_reward = 0.0
        self._experience_count = 0

    # ------------------------------------------------------------------
    # BaseMemoryController interface
    # ------------------------------------------------------------------
    def extract_memory(self, trajectory: dict[str, Any]) -> list[CandidateMemory]:
        """Extract an experience memory: state, action, outcome, lesson."""
        episode = trajectory["episode"]
        topic = trajectory["topic"]
        success = trajectory["success"]
        reward = trajectory.get("reward", float(success))
        content = trajectory.get("content", "")

        lesson = (
            f"Topic {topic}: {'success' if success else 'failure'} — "
            f"learned: {content}"
        )
        novelty = 1.0 / (1.0 + self._topic_counts.get(topic, 0))
        consequence = abs(reward)
        experience_score = reward + 0.3 * novelty + 0.2 * consequence

        memory_id = f"agile_ep{episode}_t{topic}"
        candidate = CandidateMemory(
            memory_id=memory_id,
            type="experience",
            content=lesson,
            source_episode=episode,
            metadata={
                "topic": topic,
                "success": success,
                "reward": reward,
                "novelty": novelty,
                "consequence": consequence,
                "experience_score": experience_score,
            },
        )
        self._experience_count += 1
        self._total_reward += reward
        return [candidate]

    def update_memory(
        self,
        candidate: CandidateMemory,
        context: dict[str, Any],
    ) -> str:
        """Store if within budget; evict lowest-scored if over budget.

        Prioritisation follows the AGILE principle: keep experiences
        with high reward, high novelty, and high consequence.
        """
        self._store.append(candidate)
        topic = candidate.metadata.get("topic", -1)
        self._topic_index.setdefault(topic, []).append(candidate)
        self._topic_counts[topic] = self._topic_counts.get(topic, 0) + 1

        # Enforce memory budget via experience-score eviction
        if self._top_k is not None and len(self._store) > self._top_k:
            self._evict_lowest()
            return "store"  # stored, then one evicted
        return "store"

    def retrieve_memory(self, query: MemoryQuery) -> list[CandidateMemory]:
        """Retrieve by topic match, highest experience score first."""
        candidates = self._topic_index.get(query.topic, [])
        ordered = sorted(
            candidates,
            key=lambda m: m.metadata.get("experience_score", 0.0),
            reverse=True,
        )
        return ordered[: query.top_k]

    def get_statistics(self) -> dict[str, Any]:
        scores = [m.metadata.get("experience_score", 0.0) for m in self._store]
        avg_score = sum(scores) / len(scores) if scores else 0.0
        return {
            "stored_experience": len(self._store),
            "total_experiences_seen": self._experience_count,
            "average_reward": (
                self._total_reward / self._experience_count
                if self._experience_count > 0
                else 0.0
            ),
            "average_experience_score": avg_score,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _evict_lowest(self) -> None:
        """Remove the experience with the lowest score."""
        if not self._store:
            return
        worst_idx = min(
            range(len(self._store)),
            key=lambda i: self._store[i].metadata.get("experience_score", 0.0),
        )
        evicted = self._store.pop(worst_idx)
        topic = evicted.metadata.get("topic", -1)
        topic_list = self._topic_index.get(topic, [])
        self._topic_index[topic] = [m for m in topic_list if m.memory_id != evicted.memory_id]
        self._topic_counts[topic] = max(0, self._topic_counts.get(topic, 0) - 1)
