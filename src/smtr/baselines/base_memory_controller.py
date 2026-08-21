"""Unified memory controller interface for baseline comparisons.

Every baseline memory controller must inherit from
:class:`BaseMemoryController` and implement the four abstract methods.
This guarantees a uniform contract across all baselines so that the
experiment harness can swap controllers without any other code changes.

Data flow::

    Experience
        |
        v
    extract_memory(trajectory)  ->  candidate memories
        |
        v
    update_memory(candidate, context)  ->  store / discard / modify
        |
        v
    PersistentMemoryBank
        |
        v
    retrieve_memory(query)  ->  selected memories for future use
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CandidateMemory:
    """One candidate memory produced by :meth:`extract_memory`.

    Attributes:
        memory_id: Unique identifier (assigned by the controller or env).
        type: Memory type tag (e.g. ``"reflection"``, ``"experience"``).
        content: Natural-language memory content.
        source_episode: Episode index that produced this memory.
        metadata: Arbitrary per-baseline auxiliary data (scores, etc.).
    """

    memory_id: str
    type: str
    content: str
    source_episode: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MemoryQuery:
    """Query object passed to :meth:`retrieve_memory`.

    Attributes:
        topic: Task topic / skill tag to match against.
        episode: Current episode index (for recency-aware retrieval).
        top_k: Maximum number of memories to return.
        extra: Arbitrary per-baseline query context.
    """

    topic: int
    episode: int
    top_k: int = 3
    extra: dict[str, Any] = field(default_factory=dict)


class BaseMemoryController(ABC):
    """Abstract base class that every baseline must implement.

    The controller owns the *memory lifecycle* decisions (what to store,
    what to discard, what to retrieve).  It does **not** own the
    environment, the task sampler, or the evaluation logic.

    Subclasses must implement:

    * :meth:`extract_memory` — turn a trajectory into candidate memories.
    * :meth:`update_memory` — decide store / discard / modify for each
      candidate given the current context.
    * :meth:`retrieve_memory` — select memories for a new query.
    * :meth:`get_statistics` — return a dict of controller-level stats.
    """

    @abstractmethod
    def extract_memory(self, trajectory: dict[str, Any]) -> list[CandidateMemory]:
        """Extract candidate memories from one episode trajectory.

        Parameters:
            trajectory: A dict describing the episode outcome.
                Guaranteed keys: ``episode``, ``topic``, ``success``,
                ``reward``, ``content``.  Baselines may read additional
                keys produced by the environment.

        Returns:
            Zero or more :class:`CandidateMemory` instances.
        """

    @abstractmethod
    def update_memory(
        self,
        candidate: CandidateMemory,
        context: dict[str, Any],
    ) -> str:
        """Decide the fate of one candidate memory.

        Parameters:
            candidate: The candidate produced by :meth:`extract_memory`.
            context: Runtime context (current episode, bank statistics,
                memory budget, etc.).

        Returns:
            One of ``"store"``, ``"discard"``, or ``"modify"``.
        """

    @abstractmethod
    def retrieve_memory(self, query: MemoryQuery) -> list[CandidateMemory]:
        """Retrieve memories relevant to the given query.

        Parameters:
            query: A :class:`MemoryQuery` describing the current task.

        Returns:
            A list of stored :class:`CandidateMemory` instances, ordered
            by relevance (most relevant first).
        """

    @abstractmethod
    def get_statistics(self) -> dict[str, Any]:
        """Return controller-level statistics for logging / audit.

        Returns:
            A flat dict of ``{str: int | float | str}`` statistics.
        """
