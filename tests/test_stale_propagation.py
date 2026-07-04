"""Tests for B-10: Stale Memory Propagation Experiment."""

from smtr.evaluation.stale_propagation import (
    PropagationTracker,
    StaleMemoryInjector,
    StalenessLevel,
    StalePropagationExperiment,
)
from smtr.memory.schemas import MemoryRoutingCard

# --- Fixtures ---


def _make_card(
    memory_id: str,
    *,
    success_count: int = 10,
    failure_count: int = 2,
    positive_transfer: int = 5,
    negative_transfer: int = 1,
    version: int = 3,
) -> MemoryRoutingCard:
    """Create a test routing card."""
    return MemoryRoutingCard(
        memory_id=memory_id,
        active_payload_version=version,
        goal_summary=f"goal for {memory_id}",
        task_tags=["test"],
        compatible_receiver_roles=["executor"],
        execution_success_count=success_count,
        execution_failure_count=failure_count,
        paired_positive_transfer_count=positive_transfer,
        paired_negative_transfer_count=negative_transfer,
    )


def _make_cards(n: int = 5) -> dict[str, MemoryRoutingCard]:
    """Create n test routing cards."""
    return {f"mem-{i}": _make_card(f"mem-{i}") for i in range(n)}


# --- StaleMemoryInjector Tests ---


class TestStaleMemoryInjector:
    """Test stale memory injection."""

    def test_inject_stale_creates_injections(self):
        injector = StaleMemoryInjector(seed=42)
        cards = _make_cards(10)
        injections = injector.inject_stale(cards, staleness_ratio=0.3)
        assert len(injections) == 3  # 30% of 10

    def test_inject_stale_reduces_evidence(self):
        injector = StaleMemoryInjector(seed=42)
        cards = {"mem-1": _make_card("mem-1", success_count=100)}
        injections = injector.inject_stale(
            cards, staleness_ratio=1.0, staleness_level=StalenessLevel.MODERATELY_STALE
        )
        assert len(injections) == 1
        stale = injections[0].stale_card
        # Stale card should have fewer evidence counts
        assert stale.execution_success_count <= 100

    def test_inject_stale_version_gap(self):
        injector = StaleMemoryInjector(seed=42)
        cards = {"mem-1": _make_card("mem-1", version=10)}
        injections = injector.inject_stale(
            cards, staleness_ratio=1.0, staleness_level=StalenessLevel.SEVERELY_STALE
        )
        stale = injections[0].stale_card
        assert stale.active_payload_version < 10

    def test_fresh_level_no_change(self):
        injector = StaleMemoryInjector(seed=42)
        cards = {"mem-1": _make_card("mem-1", success_count=50)}
        injections = injector.inject_stale(
            cards, staleness_ratio=1.0, staleness_level=StalenessLevel.FRESH
        )
        stale = injections[0].stale_card
        # Fresh should have same version
        assert stale.active_payload_version == 3  # Same as original

    def test_minimum_one_injection(self):
        injector = StaleMemoryInjector(seed=42)
        cards = _make_cards(2)
        injections = injector.inject_stale(cards, staleness_ratio=0.1)
        assert len(injections) >= 1


# --- PropagationTracker Tests ---


class TestPropagationTracker:
    """Test propagation tracking."""

    def test_mark_stale(self):
        tracker = PropagationTracker()
        tracker.mark_stale("mem-1", StalenessLevel.MODERATELY_STALE)
        trace = tracker.get_trace("mem-1")
        assert trace is not None
        assert trace.source_memory_id == "mem-1"

    def test_record_decision(self):
        tracker = PropagationTracker()
        tracker.mark_stale("mem-1", StalenessLevel.MODERATELY_STALE)
        tracker.record_decision("agent-1", "mem-1", action="share")
        trace = tracker.get_trace("mem-1")
        assert trace is not None
        assert "agent-1" in trace.affected_agents
        assert "share" in trace.affected_decisions

    def test_propagation_depth(self):
        tracker = PropagationTracker()
        tracker.mark_stale("mem-1", StalenessLevel.MODERATELY_STALE)
        tracker.record_decision(
            "agent-1", "mem-1", action="share", downstream_agents=["agent-2"]
        )
        tracker.record_decision(
            "agent-2", "mem-1", action="share", downstream_agents=[]
        )
        trace = tracker.get_trace("mem-1")
        assert trace is not None
        # Both agents are affected
        assert len(trace.affected_agents) == 2
        assert trace.propagation_depth >= 0

    def test_non_stale_returns_none(self):
        tracker = PropagationTracker()
        trace = tracker.get_trace("mem-unknown")
        assert trace is None


# --- StalePropagationExperiment Tests ---


class TestStalePropagationExperiment:
    """Test the full experiment."""

    def test_run_experiment(self):
        experiment = StalePropagationExperiment(seed=42, n_iterations=5)
        cards = _make_cards(10)
        results = experiment.run(cards, baseline_success_rate=0.8)
        assert len(results) == 4  # 4 staleness levels

    def test_fresh_no_degradation(self):
        experiment = StalePropagationExperiment(seed=42)
        cards = _make_cards(10)
        results = experiment.run(
            cards,
            staleness_levels=[StalenessLevel.FRESH],
            baseline_success_rate=0.8,
        )
        assert len(results) == 1
        assert results[0].success_rate_delta == 0.0
        assert results[0].success_rate_with_stale == 0.8

    def test_stale_reduces_success(self):
        experiment = StalePropagationExperiment(seed=42)
        cards = _make_cards(10)
        results = experiment.run(
            cards,
            staleness_levels=[StalenessLevel.FRESH, StalenessLevel.SEVERELY_STALE],
            baseline_success_rate=0.8,
        )
        fresh = results[0]
        stale = results[1]
        assert stale.success_rate_with_stale < fresh.success_rate_with_stale

    def test_more_stale_more_degradation(self):
        experiment = StalePropagationExperiment(seed=42)
        cards = _make_cards(10)
        results = experiment.run(
            cards,
            staleness_levels=[
                StalenessLevel.SLIGHTLY_STALE,
                StalenessLevel.SEVERELY_STALE,
            ],
            baseline_success_rate=0.8,
        )
        slight = results[0]
        severe = results[1]
        assert severe.success_rate_delta < slight.success_rate_delta

    def test_metrics_fields(self):
        experiment = StalePropagationExperiment(seed=42)
        cards = _make_cards(5)
        results = experiment.run(cards)
        for metrics in results:
            assert isinstance(metrics.n_stale_memories, int)
            assert isinstance(metrics.success_rate_baseline, float)
            assert isinstance(metrics.success_rate_with_stale, float)
            assert isinstance(metrics.n_affected_decisions, int)
            assert isinstance(metrics.mean_propagation_depth, float)
