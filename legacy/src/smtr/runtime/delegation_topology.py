"""Multi-agent delegation topology (B-09).

This module provides mechanisms for defining agent hierarchies and
delegation rules in multi-agent scenarios. It supports:

- Delegation graph (DAG of agent relationships)
- Parallel and sequential delegation patterns
- Memory visibility scoping per delegation level
- Delegation policy for task handoff

The delegation topology enables modeling complex multi-agent systems
where tasks are distributed across agents with different capabilities
and memory access levels.
"""

from dataclasses import dataclass, field
from enum import Enum


class DelegationType(str, Enum):
    """Types of delegation between agents."""

    SEQUENTIAL = "sequential"
    """Agent A delegates to Agent B, then B to C."""

    PARALLEL = "parallel"
    """Agent A delegates to B and C simultaneously."""

    HIERARCHICAL = "hierarchical"
    """Agent A is supervisor of B and C."""


class VisibilityScope(str, Enum):
    """Memory visibility scopes."""

    PRIVATE = "private"
    """Only the agent's own memories."""

    DELEGATED = "delegated"
    """Agent's own + memories delegated by others."""

    SHARED = "shared"
    """All memories in the delegation group."""

    GLOBAL = "global"
    """All memories in the system."""


@dataclass
class DelegationEdge:
    """An edge in the delegation graph."""

    from_agent: str
    to_agent: str
    delegation_type: DelegationType
    visibility_scope: VisibilityScope = VisibilityScope.DELEGATED
    conditions: dict[str, str] = field(default_factory=dict)


@dataclass
class DelegationValidation:
    """Result of validating a delegation topology."""

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class DelegationGraph:
    """A directed acyclic graph of agent delegation relationships.

    The graph defines how tasks and memory access are delegated
    between agents. It supports hierarchical, sequential, and
    parallel delegation patterns.

    Usage:
        graph = DelegationGraph()
        graph.add_agent("planner", role="planner")
        graph.add_agent("executor", role="executor")
        graph.add_delegation("planner", "executor", DelegationType.SEQUENTIAL)
        validation = graph.validate()
    """

    def __init__(self) -> None:
        self._agents: dict[str, dict] = {}
        self._edges: list[DelegationEdge] = []

    def add_agent(
        self,
        agent_id: str,
        *,
        role: str = "worker",
        capabilities: list[str] | None = None,
    ) -> None:
        """Add an agent to the graph."""
        self._agents[agent_id] = {
            "role": role,
            "capabilities": capabilities or [],
        }

    def add_delegation(
        self,
        from_agent: str,
        to_agent: str,
        delegation_type: DelegationType = DelegationType.SEQUENTIAL,
        *,
        visibility: VisibilityScope = VisibilityScope.DELEGATED,
        conditions: dict[str, str] | None = None,
    ) -> None:
        """Add a delegation edge between agents."""
        self._edges.append(
            DelegationEdge(
                from_agent=from_agent,
                to_agent=to_agent,
                delegation_type=delegation_type,
                visibility_scope=visibility,
                conditions=conditions or {},
            )
        )

    def validate(self) -> DelegationValidation:
        """Validate the delegation graph.

        Checks for:
        - All edge references exist
        - No cycles in the graph
        - At least one agent
        """
        errors = []
        warnings = []

        if not self._agents:
            errors.append("No agents defined")
            return DelegationValidation(is_valid=False, errors=errors)

        # Check edge references
        for edge in self._edges:
            if edge.from_agent not in self._agents:
                errors.append(f"Unknown agent: {edge.from_agent}")
            if edge.to_agent not in self._agents:
                errors.append(f"Unknown agent: {edge.to_agent}")

        # Check for cycles
        if self._has_cycle():
            errors.append("Cycle detected in delegation graph")

        # Warnings
        if not self._edges:
            warnings.append("No delegation edges defined")

        disconnected = self._find_disconnected_agents()
        if disconnected:
            warnings.append(f"Disconnected agents: {disconnected}")

        return DelegationValidation(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )

    def get_delegates(self, agent_id: str) -> list[str]:
        """Get agents that this agent delegates to."""
        return [
            edge.to_agent
            for edge in self._edges
            if edge.from_agent == agent_id
        ]

    def get_delegators(self, agent_id: str) -> list[str]:
        """Get agents that delegate to this agent."""
        return [
            edge.from_agent
            for edge in self._edges
            if edge.to_agent == agent_id
        ]

    def get_visibility_scope(self, agent_id: str) -> VisibilityScope:
        """Get the effective visibility scope for an agent."""
        # Find the most permissive scope from incoming edges
        scopes = [
            edge.visibility_scope
            for edge in self._edges
            if edge.to_agent == agent_id
        ]
        if not scopes:
            return VisibilityScope.PRIVATE

        # Return most permissive scope
        scope_order = [
            VisibilityScope.PRIVATE,
            VisibilityScope.DELEGATED,
            VisibilityScope.SHARED,
            VisibilityScope.GLOBAL,
        ]
        max_idx = max(scope_order.index(s) for s in scopes)
        return scope_order[max_idx]

    def get_execution_order(self) -> list[str]:
        """Get topological execution order for agents."""
        # Build adjacency and in-degree
        in_degree: dict[str, int] = {a: 0 for a in self._agents}
        adjacency: dict[str, list[str]] = {a: [] for a in self._agents}

        for edge in self._edges:
            if edge.from_agent in adjacency and edge.to_agent in in_degree:
                adjacency[edge.from_agent].append(edge.to_agent)
                in_degree[edge.to_agent] += 1

        # Kahn's algorithm
        queue = sorted([a for a, d in in_degree.items() if d == 0])
        order = []

        while queue:
            node = queue.pop(0)
            order.append(node)
            for neighbor in sorted(adjacency.get(node, [])):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
            queue.sort()

        return order

    def _has_cycle(self) -> bool:
        """Check if the graph has a cycle."""
        visited: set[str] = set()
        rec_stack: set[str] = set()

        def dfs(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)
            for edge in self._edges:
                if edge.from_agent == node:
                    if edge.to_agent not in visited:
                        if dfs(edge.to_agent):
                            return True
                    elif edge.to_agent in rec_stack:
                        return True
            rec_stack.remove(node)
            return False

        for agent in self._agents:
            if agent not in visited:
                if dfs(agent):
                    return True
        return False

    def _find_disconnected_agents(self) -> list[str]:
        """Find agents with no incoming or outgoing edges."""
        connected = set()
        for edge in self._edges:
            connected.add(edge.from_agent)
            connected.add(edge.to_agent)
        return [a for a in self._agents if a not in connected]

    @property
    def agent_count(self) -> int:
        """Number of agents in the graph."""
        return len(self._agents)

    @property
    def edge_count(self) -> int:
        """Number of delegation edges."""
        return len(self._edges)


class ScopedMemoryView:
    """A view of memories filtered by delegation scope.

    The scoped view provides each agent with access to memories
    based on their position in the delegation hierarchy and the
    visibility scope of delegation edges.
    """

    def __init__(
        self,
        *,
        agent_id: str,
        delegation_graph: DelegationGraph,
        all_memories: dict[str, dict] | None = None,
    ) -> None:
        self.agent_id = agent_id
        self.graph = delegation_graph
        self._all_memories = all_memories or {}
        self._scope = delegation_graph.get_visibility_scope(agent_id)

    def visible_memory_ids(self) -> list[str]:
        """Get IDs of memories visible to this agent."""
        if self._scope == VisibilityScope.GLOBAL:
            return list(self._all_memories.keys())

        if self._scope == VisibilityScope.SHARED:
            # All memories in the delegation group
            group = self._get_delegation_group()
            return [
                mid for mid, mem in self._all_memories.items()
                if mem.get("owner") in group or mem.get("shared", False)
            ]

        if self._scope == VisibilityScope.DELEGATED:
            # Own memories + delegated from delegators
            own = [
                mid for mid, mem in self._all_memories.items()
                if mem.get("owner") == self.agent_id
            ]
            delegated = [
                mid for mid, mem in self._all_memories.items()
                if mem.get("delegated_to") == self.agent_id
            ]
            return own + delegated

        # Private: only own memories
        return [
            mid for mid, mem in self._all_memories.items()
            if mem.get("owner") == self.agent_id
        ]

    def _get_delegation_group(self) -> set[str]:
        """Get all agents in the same delegation group."""
        group = {self.agent_id}
        # Add delegates and delegators
        group.update(self.graph.get_delegates(self.agent_id))
        group.update(self.graph.get_delegators(self.agent_id))
        return group
