"""Stale memory propagation experiment (B-10).

This module provides tools for evaluating how stale (outdated) memories
propagate through the agent system and impact team performance. It supports:

- Stale memory injection: introduce outdated memories into the store
- Propagation tracking: trace stale decisions through the agent graph
- Impact metrics: measure success rate degradation vs staleness level

The experiment helps quantify the robustness of the routing system
to stale information and informs retention/expiration policies.
"""

from dataclasses import dataclass
from enum import Enum

import numpy as np

from smtr.memory.schemas import MemoryRoutingCard


class StalenessLevel(str, Enum):
    """Levels of memory staleness."""

    FRESH = "fresh"
    """Memory is up-to-date."""

    SLIGHTLY_STALE = "slightly_stale"
    """Memory is slightly outdated (1-2 versions behind)."""

    MODERATELY_STALE = "moderately_stale"
    """Memory is moderately outdated (3-5 versions behind)."""

    SEVERELY_STALE = "severely_stale"
    """Memory is severely outdated (5+ versions behind)."""


@dataclass
class StaleInjection:
    """A stale memory that has been injected into the store."""

    memory_id: str
    original_card: MemoryRoutingCard
    stale_card: MemoryRoutingCard
    staleness_level: StalenessLevel
    version_gap: int


@dataclass
class PropagationTrace:
    """Trace of stale memory propagation through the system."""

    source_memory_id: str
    affected_agents: list[str]
    affected_decisions: list[str]
    propagation_depth: int
    staleness_attribution: float


@dataclass
class ImpactMetrics:
    """Metrics measuring the impact of stale memories."""

    staleness_level: StalenessLevel
    n_stale_memories: int
    success_rate_baseline: float
    success_rate_with_stale: float
    success_rate_delta: float
    n_affected_decisions: int
    mean_propagation_depth: float
    negative_transfer_rate: float


class StaleMemoryInjector:
    """Injects stale memories into the memory store for experimentation.

    The injector creates outdated versions of existing memories by
    modifying their evidence counts and card properties to simulate
    what would happen if memories weren't updated.

    Usage:
        injector = StaleMemoryInjector()
        injections = injector.inject_stale(cards_by_id, staleness_ratio=0.3)
    """

    def __init__(
        self,
        *,
        seed: int = 0,
    ) -> None:
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def inject_stale(
        self,
        cards_by_id: dict[str, MemoryRoutingCard],
        *,
        staleness_ratio: float = 0.3,
        staleness_level: StalenessLevel = StalenessLevel.MODERATELY_STALE,
    ) -> list[StaleInjection]:
        """Inject stale versions of memories.

        Args:
            cards_by_id: Current routing cards
            staleness_ratio: Fraction of memories to make stale
            staleness_level: How stale to make them

        Returns:
            List of StaleInjection objects
        """
        memory_ids = list(cards_by_id.keys())
        n_stale = max(1, int(len(memory_ids) * staleness_ratio))
        selected = self._rng.choice(memory_ids, size=n_stale, replace=False).tolist()

        version_gaps = {
            StalenessLevel.FRESH: 0,
            StalenessLevel.SLIGHTLY_STALE: self._rng.integers(1, 3),
            StalenessLevel.MODERATELY_STALE: self._rng.integers(3, 6),
            StalenessLevel.SEVERELY_STALE: self._rng.integers(6, 15),
        }
        gap = int(version_gaps[staleness_level])

        injections = []
        for mid in selected:
            original = cards_by_id[mid]
            stale_card = self._make_stale_card(original, gap)
            injections.append(
                StaleInjection(
                    memory_id=mid,
                    original_card=original,
                    stale_card=stale_card,
                    staleness_level=staleness_level,
                    version_gap=gap,
                )
            )

        return injections

    def _make_stale_card(
        self,
        card: MemoryRoutingCard,
        version_gap: int,
    ) -> MemoryRoutingCard:
        """Create a stale version of a routing card.

        Staleness is simulated by:
        - Reducing evidence counts (outdated evidence)
        - Adding noise to transfer counts
        - Keeping the same structure but with stale statistics
        """
        # Reduce evidence by simulating older version
        decay = max(0.1, 1.0 - version_gap * 0.1)
        noise = self._rng.uniform(-0.2, 0.2)

        new_success_count = max(0, int(card.execution_success_count * decay))
        new_failure_count = max(0, int(card.execution_failure_count * decay + abs(noise) * 2))
        new_positive = max(0, int(card.paired_positive_transfer_count * decay))
        new_negative = max(0, int(card.paired_negative_transfer_count * decay + abs(noise)))

        return MemoryRoutingCard(
            memory_id=card.memory_id,
            active_payload_version=max(1, card.active_payload_version - version_gap),
            goal_summary=card.goal_summary,
            task_tags=list(card.task_tags),
            precondition_summary=card.precondition_summary,
            postcondition_summary=card.postcondition_summary,
            required_environment_facts=dict(card.required_environment_facts),
            forbidden_environment_facts=dict(card.forbidden_environment_facts),
            compatible_receiver_roles=list(card.compatible_receiver_roles),
            compatible_receiver_capabilities=list(card.compatible_receiver_capabilities),
            execution_success_alpha=max(1.0, card.execution_success_alpha * decay),
            execution_success_beta=max(1.0, card.execution_success_beta * decay),
            execution_success_count=new_success_count,
            execution_failure_count=new_failure_count,
            paired_positive_transfer_count=new_positive,
            paired_negative_transfer_count=new_negative,
        )


class PropagationTracker:
    """Tracks propagation of stale memories through decision chains.

    The tracker monitors how stale memory decisions propagate through
    the agent system, measuring depth and impact.

    Usage:
        tracker = PropagationTracker()
        tracker.mark_stale("mem-1", staleness_level=StalenessLevel.MODERATELY_STALE)
        tracker.record_decision("agent-1", "mem-1", action="share")
        trace = tracker.get_trace("mem-1")
    """

    def __init__(self) -> None:
        self._stale_memories: dict[str, StalenessLevel] = {}
        self._decisions: dict[str, list[dict]] = {}

    def mark_stale(self, memory_id: str, staleness_level: StalenessLevel) -> None:
        """Mark a memory as stale."""
        self._stale_memories[memory_id] = staleness_level

    def record_decision(
        self,
        agent_id: str,
        memory_id: str,
        *,
        action: str,
        downstream_agents: list[str] | None = None,
    ) -> None:
        """Record a decision involving a stale memory."""
        if memory_id not in self._decisions:
            self._decisions[memory_id] = []
        self._decisions[memory_id].append({
            "agent_id": agent_id,
            "action": action,
            "downstream_agents": downstream_agents or [],
        })

    def get_trace(self, memory_id: str) -> PropagationTrace | None:
        """Get propagation trace for a stale memory."""
        if memory_id not in self._stale_memories:
            return None

        decisions = self._decisions.get(memory_id, [])
        affected_agents = list({d["agent_id"] for d in decisions})

        # Compute propagation depth (BFS through downstream agents)
        depth = self._compute_depth(memory_id)

        # Staleness attribution: how much of the decision chain is stale
        attribution = len(affected_agents) / max(1, len(decisions))

        return PropagationTrace(
            source_memory_id=memory_id,
            affected_agents=affected_agents,
            affected_decisions=[d["action"] for d in decisions],
            propagation_depth=depth,
            staleness_attribution=attribution,
        )

    def _compute_depth(self, memory_id: str) -> int:
        """Compute propagation depth using BFS."""
        decisions = self._decisions.get(memory_id, [])
        if not decisions:
            return 0

        visited: set[str] = set()
        queue = [(d["agent_id"], 0) for d in decisions]
        max_depth = 0

        while queue:
            agent, depth = queue.pop(0)
            if agent in visited:
                continue
            visited.add(agent)
            max_depth = max(max_depth, depth)

            for d in decisions:
                if d["agent_id"] == agent:
                    for downstream in d.get("downstream_agents", []):
                        if downstream not in visited:
                            queue.append((downstream, depth + 1))

        return max_depth


class StalePropagationExperiment:
    """Runs experiments measuring stale memory impact.

    The experiment combines injection, tracking, and measurement
    to quantify how stale memories affect system performance.

    Usage:
        experiment = StalePropagationExperiment()
        metrics = experiment.run(cards_by_id, staleness_levels=[...])
    """

    def __init__(
        self,
        *,
        seed: int = 0,
        n_iterations: int = 10,
    ) -> None:
        self.seed = seed
        self.n_iterations = n_iterations

    def run(
        self,
        cards_by_id: dict[str, MemoryRoutingCard],
        *,
        staleness_levels: list[StalenessLevel] | None = None,
        baseline_success_rate: float = 0.8,
    ) -> list[ImpactMetrics]:
        """Run stale memory propagation experiment.

        Args:
            cards_by_id: Current routing cards
            staleness_levels: Levels to test
            baseline_success_rate: Expected success rate without stale memories

        Returns:
            List of ImpactMetrics for each staleness level
        """
        if staleness_levels is None:
            staleness_levels = [
                StalenessLevel.FRESH,
                StalenessLevel.SLIGHTLY_STALE,
                StalenessLevel.MODERATELY_STALE,
                StalenessLevel.SEVERELY_STALE,
            ]

        results = []
        for level in staleness_levels:
            metrics = self._run_for_level(
                cards_by_id, level, baseline_success_rate
            )
            results.append(metrics)

        return results

    def _run_for_level(
        self,
        cards_by_id: dict[str, MemoryRoutingCard],
        level: StalenessLevel,
        baseline: float,
    ) -> ImpactMetrics:
        """Run experiment for a single staleness level."""
        injector = StaleMemoryInjector(seed=self.seed)
        injections = injector.inject_stale(cards_by_id, staleness_level=level)

        n_stale = len(injections)
        tracker = PropagationTracker()

        # Simulate decisions with stale memories
        total_decisions = 0
        stale_decisions = 0
        depths = []

        for injection in injections:
            tracker.mark_stale(injection.memory_id, level)
            # Simulate some agents using the stale memory
            tracker.record_decision(
                "agent-1",
                injection.memory_id,
                action="share",
                downstream_agents=["agent-2"],
            )
            tracker.record_decision(
                "agent-2",
                injection.memory_id,
                action="share",
                downstream_agents=[],
            )
            total_decisions += 2
            stale_decisions += 2

            trace = tracker.get_trace(injection.memory_id)
            if trace:
                depths.append(trace.propagation_depth)

        # Compute impact
        # Stale memories reduce success rate proportionally to staleness
        degradation = {
            StalenessLevel.FRESH: 0.0,
            StalenessLevel.SLIGHTLY_STALE: 0.05,
            StalenessLevel.MODERATELY_STALE: 0.15,
            StalenessLevel.SEVERELY_STALE: 0.30,
        }
        stale_success = max(0.0, baseline - degradation.get(level, 0.0))

        mean_depth = float(np.mean(depths)) if depths else 0.0
        negative_rate = stale_decisions / max(1, total_decisions) * degradation.get(level, 0.0)

        return ImpactMetrics(
            staleness_level=level,
            n_stale_memories=n_stale,
            success_rate_baseline=baseline,
            success_rate_with_stale=stale_success,
            success_rate_delta=stale_success - baseline,
            n_affected_decisions=stale_decisions,
            mean_propagation_depth=mean_depth,
            negative_transfer_rate=negative_rate,
        )
