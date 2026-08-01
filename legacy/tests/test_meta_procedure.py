"""Tests for B-06: Meta-Procedure Composition."""

from smtr.memory.meta_procedure import (
    DependencyType,
    MetaProcedure,
    MetaProcedureBuilder,
    ProcedureDependency,
)
from smtr.memory.schemas import ProcedurePayload

# --- Fixtures ---


def _make_procedure(
    memory_id: str,
    *,
    goal: str = "test goal",
    preconditions: list[str] | None = None,
    postconditions: list[str] | None = None,
    steps: list[str] | None = None,
) -> ProcedurePayload:
    """Create a test procedure payload."""
    return ProcedurePayload(
        memory_id=memory_id,
        version=1,
        writer_agent_id="test-agent",
        source_episode_id="ep-1",
        goal=goal,
        preconditions=preconditions or [],
        steps=steps or ["step 1"],
        postconditions=postconditions or [],
    )


# --- MetaProcedure Tests ---


class TestMetaProcedureBasics:
    """Test basic meta-procedure operations."""

    def test_empty_meta_procedure(self):
        meta = MetaProcedure(name="test")
        assert meta.procedure_count == 0
        assert meta.dependency_count == 0

    def test_add_procedure(self):
        meta = MetaProcedure(name="test")
        proc = _make_procedure("mem-1")
        meta.add_procedure(proc)
        assert meta.procedure_count == 1
        assert meta.get_procedure("mem-1") is not None

    def test_add_dependency(self):
        meta = MetaProcedure(name="test")
        meta.add_procedure(_make_procedure("mem-1"))
        meta.add_procedure(_make_procedure("mem-2"))
        meta.add_dependency(
            ProcedureDependency(
                from_memory_id="mem-1",
                to_memory_id="mem-2",
                dependency_type=DependencyType.SEQUENCE,
            )
        )
        assert meta.dependency_count == 1

    def test_get_preconditions_postconditions(self):
        meta = MetaProcedure(name="test")
        proc = _make_procedure(
            "mem-1",
            preconditions=["pre1", "pre2"],
            postconditions=["post1"],
        )
        meta.add_procedure(proc)
        assert meta.get_preconditions("mem-1") == ["pre1", "pre2"]
        assert meta.get_postconditions("mem-1") == ["post1"]


class TestMetaProcedureValidation:
    """Test meta-procedure validation."""

    def test_valid_simple_composition(self):
        meta = MetaProcedure(name="test")
        meta.add_procedure(_make_procedure("mem-1"))
        meta.add_procedure(_make_procedure("mem-2"))
        meta.add_dependency(
            ProcedureDependency(
                from_memory_id="mem-1",
                to_memory_id="mem-2",
                dependency_type=DependencyType.SEQUENCE,
            )
        )
        validation = meta.validate()
        assert validation.is_valid
        assert len(validation.errors) == 0

    def test_invalid_dependency_reference(self):
        meta = MetaProcedure(name="test")
        meta.add_procedure(_make_procedure("mem-1"))
        meta.add_dependency(
            ProcedureDependency(
                from_memory_id="mem-1",
                to_memory_id="mem-unknown",
                dependency_type=DependencyType.SEQUENCE,
            )
        )
        validation = meta.validate()
        assert not validation.is_valid
        assert any("mem-unknown" in e for e in validation.errors)

    def test_cycle_detection(self):
        meta = MetaProcedure(name="test")
        meta.add_procedure(_make_procedure("mem-1"))
        meta.add_procedure(_make_procedure("mem-2"))
        meta.add_procedure(_make_procedure("mem-3"))
        # Create cycle: 1 -> 2 -> 3 -> 1
        meta.add_dependency(
            ProcedureDependency("mem-1", "mem-2", DependencyType.SEQUENCE)
        )
        meta.add_dependency(
            ProcedureDependency("mem-2", "mem-3", DependencyType.SEQUENCE)
        )
        meta.add_dependency(
            ProcedureDependency("mem-3", "mem-1", DependencyType.SEQUENCE)
        )
        validation = meta.validate()
        assert not validation.is_valid
        assert any("Cycle" in e for e in validation.errors)

    def test_empty_procedures_validation(self):
        meta = MetaProcedure(name="test")
        validation = meta.validate()
        assert not validation.is_valid
        assert any("No procedures" in e for e in validation.errors)

    def test_entry_exit_point_warnings(self):
        meta = MetaProcedure(name="test")
        meta.add_procedure(_make_procedure("mem-1"))
        validation = meta.validate()
        # Should have warnings about missing entry/exit points
        assert len(validation.warnings) > 0


class TestExecutionOrder:
    """Test execution order resolution."""

    def test_linear_chain(self):
        meta = MetaProcedure(name="test")
        meta.add_procedure(_make_procedure("mem-1"))
        meta.add_procedure(_make_procedure("mem-2"))
        meta.add_procedure(_make_procedure("mem-3"))
        meta.add_dependency(
            ProcedureDependency("mem-1", "mem-2", DependencyType.SEQUENCE)
        )
        meta.add_dependency(
            ProcedureDependency("mem-2", "mem-3", DependencyType.SEQUENCE)
        )
        order = meta.resolve_execution_order()
        assert order == ["mem-1", "mem-2", "mem-3"]

    def test_diamond_dependency(self):
        meta = MetaProcedure(name="test")
        meta.add_procedure(_make_procedure("mem-1"))
        meta.add_procedure(_make_procedure("mem-2"))
        meta.add_procedure(_make_procedure("mem-3"))
        meta.add_procedure(_make_procedure("mem-4"))
        # Diamond: 1 -> 2,3 -> 4
        meta.add_dependency(
            ProcedureDependency("mem-1", "mem-2", DependencyType.SEQUENCE)
        )
        meta.add_dependency(
            ProcedureDependency("mem-1", "mem-3", DependencyType.SEQUENCE)
        )
        meta.add_dependency(
            ProcedureDependency("mem-2", "mem-4", DependencyType.SEQUENCE)
        )
        meta.add_dependency(
            ProcedureDependency("mem-3", "mem-4", DependencyType.SEQUENCE)
        )
        order = meta.resolve_execution_order()
        assert order[0] == "mem-1"
        assert order[-1] == "mem-4"

    def test_independent_procedures(self):
        meta = MetaProcedure(name="test")
        meta.add_procedure(_make_procedure("mem-a"))
        meta.add_procedure(_make_procedure("mem-b"))
        meta.add_procedure(_make_procedure("mem-c"))
        # No dependencies
        order = meta.resolve_execution_order()
        assert len(order) == 3
        # Should be sorted alphabetically for determinism
        assert order == ["mem-a", "mem-b", "mem-c"]

    def test_cyclic_returns_empty(self):
        meta = MetaProcedure(name="test")
        meta.add_procedure(_make_procedure("mem-1"))
        meta.add_procedure(_make_procedure("mem-2"))
        meta.add_dependency(
            ProcedureDependency("mem-1", "mem-2", DependencyType.SEQUENCE)
        )
        meta.add_dependency(
            ProcedureDependency("mem-2", "mem-1", DependencyType.SEQUENCE)
        )
        order = meta.resolve_execution_order()
        assert order == []


class TestPreconditionMatching:
    """Test precondition/postcondition matching."""

    def test_matching_conditions(self):
        meta = MetaProcedure(name="test")
        meta.add_procedure(
            _make_procedure("mem-1", postconditions=["file_created"])
        )
        meta.add_procedure(
            _make_procedure("mem-2", preconditions=["file_created"])
        )
        assert meta.check_precondition_match("mem-1", "mem-2") is True

    def test_non_matching_conditions(self):
        meta = MetaProcedure(name="test")
        meta.add_procedure(
            _make_procedure("mem-1", postconditions=["file_created"])
        )
        meta.add_procedure(
            _make_procedure("mem-2", preconditions=["database_ready"])
        )
        assert meta.check_precondition_match("mem-1", "mem-2") is False


# --- MetaProcedureBuilder Tests ---


class TestMetaProcedureBuilder:
    """Test the builder interface."""

    def test_sequential_build(self):
        builder = MetaProcedureBuilder(name="pipeline")
        builder.add_sequential(_make_procedure("mem-1", goal="step 1"))
        builder.add_sequential(_make_procedure("mem-2", goal="step 2"))
        builder.add_sequential(_make_procedure("mem-3", goal="step 3"))
        meta = builder.build()
        assert meta.procedure_count == 3
        assert meta.dependency_count == 2
        order = meta.resolve_execution_order()
        assert order == ["mem-1", "mem-2", "mem-3"]

    def test_build_with_explicit_dependency(self):
        builder = MetaProcedureBuilder(name="pipeline")
        builder.add_sequential(_make_procedure("mem-1"))
        builder.add_with_dependency(
            _make_procedure("mem-2"),
            depends_on="mem-1",
            dep_type=DependencyType.PRECONDITION,
        )
        meta = builder.build()
        assert meta.dependency_count == 1

    def test_build_sets_entry_exit(self):
        builder = MetaProcedureBuilder(name="pipeline")
        builder.add_sequential(_make_procedure("mem-1"))
        builder.add_sequential(_make_procedure("mem-2"))
        meta = builder.build()
        validation = meta.validate()
        # Entry/exit should be auto-set
        assert "No entry points" not in validation.warnings
        assert "No exit points" not in validation.warnings
