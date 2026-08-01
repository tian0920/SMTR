"""Tests for B-09: Multi-Agent Delegation Topology."""

from smtr.runtime.delegation_topology import (
    DelegationGraph,
    DelegationType,
    ScopedMemoryView,
    VisibilityScope,
)

# --- DelegationGraph Tests ---


class TestDelegationGraphBasics:
    """Test basic graph operations."""

    def test_empty_graph(self):
        graph = DelegationGraph()
        assert graph.agent_count == 0
        assert graph.edge_count == 0

    def test_add_agent(self):
        graph = DelegationGraph()
        graph.add_agent("planner", role="planner")
        assert graph.agent_count == 1

    def test_add_delegation(self):
        graph = DelegationGraph()
        graph.add_agent("planner")
        graph.add_agent("executor")
        graph.add_delegation("planner", "executor", DelegationType.SEQUENTIAL)
        assert graph.edge_count == 1

    def test_get_delegates(self):
        graph = DelegationGraph()
        graph.add_agent("a")
        graph.add_agent("b")
        graph.add_agent("c")
        graph.add_delegation("a", "b")
        graph.add_delegation("a", "c")
        delegates = graph.get_delegates("a")
        assert set(delegates) == {"b", "c"}

    def test_get_delegators(self):
        graph = DelegationGraph()
        graph.add_agent("a")
        graph.add_agent("b")
        graph.add_delegation("a", "b")
        delegators = graph.get_delegators("b")
        assert delegators == ["a"]


class TestDelegationGraphValidation:
    """Test graph validation."""

    def test_valid_simple_graph(self):
        graph = DelegationGraph()
        graph.add_agent("planner")
        graph.add_agent("executor")
        graph.add_delegation("planner", "executor")
        validation = graph.validate()
        assert validation.is_valid

    def test_invalid_unknown_agent(self):
        graph = DelegationGraph()
        graph.add_agent("a")
        graph.add_delegation("a", "unknown")
        validation = graph.validate()
        assert not validation.is_valid
        assert any("unknown" in e.lower() for e in validation.errors)

    def test_cycle_detection(self):
        graph = DelegationGraph()
        graph.add_agent("a")
        graph.add_agent("b")
        graph.add_agent("c")
        graph.add_delegation("a", "b")
        graph.add_delegation("b", "c")
        graph.add_delegation("c", "a")
        validation = graph.validate()
        assert not validation.is_valid
        assert any("cycle" in e.lower() for e in validation.errors)

    def test_empty_graph_invalid(self):
        graph = DelegationGraph()
        validation = graph.validate()
        assert not validation.is_valid

    def test_disconnected_warning(self):
        graph = DelegationGraph()
        graph.add_agent("a")
        graph.add_agent("b")
        # No edges
        validation = graph.validate()
        assert validation.is_valid
        assert any("disconnected" in w.lower() for w in validation.warnings)


class TestExecutionOrder:
    """Test execution order resolution."""

    def test_linear_chain(self):
        graph = DelegationGraph()
        graph.add_agent("a")
        graph.add_agent("b")
        graph.add_agent("c")
        graph.add_delegation("a", "b")
        graph.add_delegation("b", "c")
        order = graph.get_execution_order()
        assert order == ["a", "b", "c"]

    def test_parallel_delegation(self):
        graph = DelegationGraph()
        graph.add_agent("a")
        graph.add_agent("b")
        graph.add_agent("c")
        graph.add_delegation("a", "b")
        graph.add_delegation("a", "c")
        order = graph.get_execution_order()
        assert order[0] == "a"
        assert set(order[1:]) == {"b", "c"}

    def test_no_edges(self):
        graph = DelegationGraph()
        graph.add_agent("a")
        graph.add_agent("b")
        order = graph.get_execution_order()
        assert len(order) == 2


class TestVisibilityScope:
    """Test visibility scope resolution."""

    def test_default_private(self):
        graph = DelegationGraph()
        graph.add_agent("a")
        scope = graph.get_visibility_scope("a")
        assert scope == VisibilityScope.PRIVATE

    def test_delegated_scope(self):
        graph = DelegationGraph()
        graph.add_agent("a")
        graph.add_agent("b")
        graph.add_delegation("a", "b", visibility=VisibilityScope.DELEGATED)
        scope = graph.get_visibility_scope("b")
        assert scope == VisibilityScope.DELEGATED

    def test_shared_scope(self):
        graph = DelegationGraph()
        graph.add_agent("a")
        graph.add_agent("b")
        graph.add_delegation("a", "b", visibility=VisibilityScope.SHARED)
        scope = graph.get_visibility_scope("b")
        assert scope == VisibilityScope.SHARED

    def test_most_permissive_wins(self):
        graph = DelegationGraph()
        graph.add_agent("a")
        graph.add_agent("b")
        graph.add_agent("c")
        graph.add_delegation("a", "c", visibility=VisibilityScope.PRIVATE)
        graph.add_delegation("b", "c", visibility=VisibilityScope.GLOBAL)
        scope = graph.get_visibility_scope("c")
        assert scope == VisibilityScope.GLOBAL


# --- ScopedMemoryView Tests ---


class TestScopedMemoryView:
    """Test scoped memory access."""

    def test_private_scope(self):
        graph = DelegationGraph()
        graph.add_agent("a")
        memories = {
            "m1": {"owner": "a"},
            "m2": {"owner": "b"},
        }
        view = ScopedMemoryView(agent_id="a", delegation_graph=graph, all_memories=memories)
        visible = view.visible_memory_ids()
        assert visible == ["m1"]

    def test_delegated_scope(self):
        graph = DelegationGraph()
        graph.add_agent("a")
        graph.add_agent("b")
        graph.add_delegation("a", "b", visibility=VisibilityScope.DELEGATED)
        memories = {
            "m1": {"owner": "a"},
            "m2": {"owner": "b"},
            "m3": {"delegated_to": "b"},
        }
        view = ScopedMemoryView(agent_id="b", delegation_graph=graph, all_memories=memories)
        visible = view.visible_memory_ids()
        assert "m2" in visible
        assert "m3" in visible

    def test_global_scope(self):
        graph = DelegationGraph()
        graph.add_agent("a")
        graph.add_agent("b")
        graph.add_delegation("a", "b", visibility=VisibilityScope.GLOBAL)
        memories = {
            "m1": {"owner": "a"},
            "m2": {"owner": "b"},
            "m3": {"owner": "c"},
        }
        view = ScopedMemoryView(agent_id="b", delegation_graph=graph, all_memories=memories)
        visible = view.visible_memory_ids()
        assert len(visible) == 3
