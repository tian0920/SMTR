"""Meta-procedure composition (B-06).

This module provides mechanisms for composing multiple procedures into
meta-procedures. It supports:

- Dependency graph resolution between procedures
- Conditional execution based on preconditions/postconditions
- Cycle detection and validation
- Composition of procedure sequences into higher-order workflows

Meta-procedures allow the system to build complex workflows from
simpler, well-tested procedures.
"""

from dataclasses import dataclass, field
from enum import Enum

from pydantic import BaseModel, ConfigDict

from smtr.memory.schemas import ProcedurePayload


class DependencyType(str, Enum):
    """Types of dependencies between procedures."""

    PRECONDITION = "precondition"
    """Procedure A's preconditions require Procedure B's postconditions."""

    SEQUENCE = "sequence"
    """Procedure A must execute before Procedure B."""

    OPTIONAL = "optional"
    """Procedure B can optionally follow Procedure A."""


class CompositionMode(str, Enum):
    """Modes of procedure composition."""

    SEQUENTIAL = "sequential"
    """Execute procedures in order."""

    CONDITIONAL = "conditional"
    """Execute based on precondition satisfaction."""

    PARALLEL = "parallel"
    """Execute independent procedures concurrently (not yet supported)."""


@dataclass
class ProcedureDependency:
    """A dependency between two procedures."""

    from_memory_id: str
    to_memory_id: str
    dependency_type: DependencyType
    condition: str | None = None


@dataclass
class CompositionValidation:
    """Result of validating a composition."""

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    resolved_order: list[str] = field(default_factory=list)


class MetaProcedureConfig(BaseModel):
    """Configuration for meta-procedure composition."""

    model_config = ConfigDict(frozen=True)

    max_depth: int = 5
    """Maximum nesting depth for meta-procedures."""

    require_explicit_dependencies: bool = False
    """If True, all dependencies must be explicitly declared."""

    allow_cycles: bool = False
    """If True, allow cyclic dependencies (not recommended)."""


class MetaProcedure:
    """A composed procedure built from multiple sub-procedures.

    Meta-procedures define an ordered sequence of sub-procedures
    with dependencies and conditions. They enable building complex
    workflows from simpler, reusable components.

    Usage:
        meta = MetaProcedure(name="build_and_deploy")
        meta.add_procedure(proc_a)
        meta.add_procedure(proc_b)
        meta.add_dependency(ProcedureDependency(
            from_memory_id=proc_a.memory_id,
            to_memory_id=proc_b.memory_id,
            dependency_type=DependencyType.SEQUENCE,
        ))
        validation = meta.validate()
        if validation.is_valid:
            order = meta.resolve_execution_order()
    """

    def __init__(
        self,
        *,
        name: str,
        config: MetaProcedureConfig | None = None,
    ) -> None:
        self.name = name
        self.config = config or MetaProcedureConfig()
        self._procedures: dict[str, ProcedurePayload] = {}
        self._dependencies: list[ProcedureDependency] = []
        self._entry_points: list[str] = []
        self._exit_points: list[str] = []

    def add_procedure(self, procedure: ProcedurePayload) -> None:
        """Add a procedure to the meta-procedure."""
        self._procedures[procedure.memory_id] = procedure

    def add_dependency(self, dependency: ProcedureDependency) -> None:
        """Add a dependency between procedures."""
        self._dependencies.append(dependency)

    def set_entry_points(self, memory_ids: list[str]) -> None:
        """Set the entry point procedures."""
        self._entry_points = memory_ids

    def set_exit_points(self, memory_ids: list[str]) -> None:
        """Set the exit point procedures."""
        self._exit_points = memory_ids

    def validate(self) -> CompositionValidation:
        """Validate the meta-procedure composition.

        Checks for:
        - All dependency references exist
        - No cycles in dependency graph
        - Entry/exit points are valid
        """
        errors = []
        warnings = []

        # Check dependency references
        for dep in self._dependencies:
            if dep.from_memory_id not in self._procedures:
                errors.append(f"Unknown procedure in dependency: {dep.from_memory_id}")
            if dep.to_memory_id not in self._procedures:
                errors.append(f"Unknown procedure in dependency: {dep.to_memory_id}")

        # Check for cycles
        if not self.config.allow_cycles:
            cycle = self._detect_cycle()
            if cycle:
                errors.append(f"Cycle detected: {' -> '.join(cycle)}")

        # Check entry/exit points
        for ep in self._entry_points:
            if ep not in self._procedures:
                errors.append(f"Unknown entry point: {ep}")
        for ep in self._exit_points:
            if ep not in self._procedures:
                errors.append(f"Unknown exit point: {ep}")

        # Warnings
        if not self._entry_points:
            warnings.append("No entry points defined")
        if not self._exit_points:
            warnings.append("No exit points defined")
        if len(self._procedures) == 0:
            errors.append("No procedures added")

        # Resolve execution order
        resolved = []
        if not errors:
            resolved = self.resolve_execution_order()
            if not resolved:
                errors.append("Could not resolve execution order")

        return CompositionValidation(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            resolved_order=resolved,
        )

    def resolve_execution_order(self) -> list[str]:
        """Resolve the execution order using topological sort.

        Returns:
            List of memory IDs in execution order, or empty if cyclic
        """
        # Build adjacency list
        graph: dict[str, list[str]] = {mid: [] for mid in self._procedures}
        in_degree: dict[str, int] = {mid: 0 for mid in self._procedures}

        for dep in self._dependencies:
            if dep.from_memory_id in graph and dep.to_memory_id in graph:
                graph[dep.from_memory_id].append(dep.to_memory_id)
                in_degree[dep.to_memory_id] += 1

        # Kahn's algorithm for topological sort
        queue = [mid for mid, deg in in_degree.items() if deg == 0]
        order = []

        while queue:
            # Sort for determinism
            queue.sort()
            node = queue.pop(0)
            order.append(node)
            for neighbor in graph.get(node, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(order) != len(self._procedures):
            return []  # Cycle detected

        return order

    def get_procedure(self, memory_id: str) -> ProcedurePayload | None:
        """Get a procedure by memory ID."""
        return self._procedures.get(memory_id)

    def get_preconditions(self, memory_id: str) -> list[str]:
        """Get preconditions for a procedure."""
        proc = self._procedures.get(memory_id)
        return proc.preconditions if proc else []

    def get_postconditions(self, memory_id: str) -> list[str]:
        """Get postconditions for a procedure."""
        proc = self._procedures.get(memory_id)
        return proc.postconditions if proc else []

    def check_precondition_match(
        self,
        from_memory_id: str,
        to_memory_id: str,
    ) -> bool:
        """Check if postconditions of A satisfy preconditions of B."""
        post_a = set(self.get_postconditions(from_memory_id))
        pre_b = set(self.get_preconditions(to_memory_id))
        # Check if any postcondition matches any precondition
        return bool(post_a & pre_b)

    def _detect_cycle(self) -> list[str] | None:
        """Detect cycles using DFS. Returns cycle path or None."""
        visited: set[str] = set()
        rec_stack: set[str] = set()
        path: list[str] = []

        def dfs(node: str) -> list[str] | None:
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for dep in self._dependencies:
                if dep.from_memory_id == node:
                    next_node = dep.to_memory_id
                    if next_node not in visited:
                        result = dfs(next_node)
                        if result:
                            return result
                    elif next_node in rec_stack:
                        path.append(next_node)
                        # Extract cycle
                        cycle_start = path.index(next_node)
                        return path[cycle_start:]

            path.pop()
            rec_stack.remove(node)
            return None

        for mid in self._procedures:
            if mid not in visited:
                cycle = dfs(mid)
                if cycle:
                    return cycle

        return None

    @property
    def procedure_count(self) -> int:
        """Number of procedures in the meta-procedure."""
        return len(self._procedures)

    @property
    def dependency_count(self) -> int:
        """Number of dependencies."""
        return len(self._dependencies)


class MetaProcedureBuilder:
    """Builder for creating meta-procedures from existing procedures.

    The builder provides a fluent interface for constructing
    meta-procedures with automatic dependency inference.
    """

    def __init__(self, *, name: str) -> None:
        self._meta = MetaProcedure(name=name)
        self._last_added: str | None = None

    def add_sequential(self, procedure: ProcedurePayload) -> "MetaProcedureBuilder":
        """Add a procedure that follows the previous one."""
        self._meta.add_procedure(procedure)
        if self._last_added is not None:
            self._meta.add_dependency(
                ProcedureDependency(
                    from_memory_id=self._last_added,
                    to_memory_id=procedure.memory_id,
                    dependency_type=DependencyType.SEQUENCE,
                )
            )
        self._last_added = procedure.memory_id
        return self

    def add_with_dependency(
        self,
        procedure: ProcedurePayload,
        *,
        depends_on: str,
        dep_type: DependencyType = DependencyType.PRECONDITION,
    ) -> "MetaProcedureBuilder":
        """Add a procedure with explicit dependency."""
        self._meta.add_procedure(procedure)
        self._meta.add_dependency(
            ProcedureDependency(
                from_memory_id=depends_on,
                to_memory_id=procedure.memory_id,
                dependency_type=dep_type,
            )
        )
        return self

    def build(self) -> MetaProcedure:
        """Build and return the meta-procedure."""
        if self._meta.procedure_count > 0:
            # Auto-set entry/exit points if not set
            order = self._meta.resolve_execution_order()
            if order:
                self._meta.set_entry_points([order[0]])
                self._meta.set_exit_points([order[-1]])
        return self._meta
