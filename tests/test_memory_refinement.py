"""Tests for B-07: Memory Refinement & Contradiction Repair."""

from smtr.counterfactual.schemas import (
    BranchOutcome,
    ContextFingerprint,
    PairedInterventionRecord,
)
from smtr.memory.refinement import (
    ContradictionDetector,
    ContradictionType,
    MemoryRefiner,
    RefinementAction,
)
from smtr.memory.schemas import MemoryRoutingCard

# --- Fixtures ---


def _make_card(
    memory_id: str,
    *,
    task_tags: list[str] | None = None,
    roles: list[str] | None = None,
    success_count: int = 5,
    failure_count: int = 0,
    positive_transfer: int = 3,
    negative_transfer: int = 0,
    with_contexts: bool = False,
) -> MemoryRoutingCard:
    """Create a test routing card."""
    ctx = _make_context() if with_contexts else None
    return MemoryRoutingCard(
        memory_id=memory_id,
        active_payload_version=1,
        goal_summary=f"goal for {memory_id}",
        task_tags=task_tags or ["test"],
        compatible_receiver_roles=roles or ["executor"],
        execution_success_count=success_count,
        execution_failure_count=failure_count,
        execution_success_contexts=[ctx] if with_contexts and success_count > 0 else [],
        execution_failure_contexts=[ctx] if with_contexts and failure_count > 0 else [],
        paired_positive_transfer_count=positive_transfer,
        paired_negative_transfer_count=negative_transfer,
    )


def _make_record(
    *,
    memory_id: str = "mem-1",
    y_share: int = 1,
    y_withhold: int = 0,
    transfer_class: str | None = None,
    record_id: str = "rec-1",
) -> PairedInterventionRecord:
    """Create a minimal paired intervention record."""
    if transfer_class is None:
        if y_share == 1 and y_withhold == 0:
            transfer_class = "positive"
        elif y_share == 0 and y_withhold == 1:
            transfer_class = "negative"
        elif y_share == 1 and y_withhold == 1:
            transfer_class = "neutral_success"
        else:
            transfer_class = "neutral_failure"

    context = ContextFingerprint(
        task_id="task-1",
        receiver_agent_id="agent-1",
        receiver_role="executor",
        task_stage="test",
        selected_memory_ids=[],
        selected_set_signature="empty",
        episode_id="ep-1",
    )

    branch = BranchOutcome(
        team_success=bool(y_share),
        team_reward=0.0,
        team_summary="test",
        final_environment_observation={},
        selected_memory_ids_by_agent={},
        router_trace=[],
        target_memory_visible_to_receiver=True,
        selected_final_at_target_node=[],
    )

    return PairedInterventionRecord(
        record_id=record_id,
        episode_id="ep-1",
        task_id="task-1",
        graph_node="node-1",
        receiver_agent_id="agent-1",
        receiver_role="executor",
        task_stage="test",
        candidate_memory_id=memory_id,
        candidate_payload_version=1,
        candidate_order=[memory_id],
        target_index=0,
        selected_before=[],
        decision_context=context,
        memory_store_revision=1,
        memory_snapshot_digest="abc",
        runtime_snapshot_digest="def",
        continuation_policy_name="test",
        continuation_policy_version="1",
        common_seed=42,
        share_outcome=branch,
        withhold_outcome=branch,
        y_share=y_share,
        y_withhold=y_withhold,
        transfer_class=transfer_class,
        target_selection_probability=0.5,
    )


def _make_context(task_id: str = "task-1") -> ContextFingerprint:
    return ContextFingerprint(
        task_id=task_id,
        receiver_agent_id="agent-1",
        receiver_role="executor",
        task_stage="test",
        selected_memory_ids=[],
        selected_set_signature="empty",
        episode_id="ep-1",
    )


# --- ContradictionDetector Tests ---


class TestContradictionDetector:
    """Test contradiction detection."""

    def test_no_contradiction_consistent_memories(self):
        """Memories with same transfer direction should not contradict."""
        detector = ContradictionDetector()
        records = [
            _make_record(memory_id="mem-1", y_share=1, y_withhold=0, record_id="r1"),
            _make_record(memory_id="mem-1", y_share=1, y_withhold=0, record_id="r2"),
            _make_record(memory_id="mem-2", y_share=1, y_withhold=0, record_id="r3"),
            _make_record(memory_id="mem-2", y_share=1, y_withhold=0, record_id="r4"),
        ]
        cards = {
            "mem-1": _make_card("mem-1", task_tags=["test"], roles=["executor"]),
            "mem-2": _make_card("mem-2", task_tags=["test"], roles=["executor"]),
        }
        contradictions = detector.detect(records, cards)
        # Both positive, no transfer contradiction
        transfer_contradictions = [
            c
            for c in contradictions
            if c.contradiction_type == ContradictionType.TRANSFER_CONTRADICTION
        ]
        assert len(transfer_contradictions) == 0

    def test_transfer_contradiction_detected(self):
        """Memories with opposite transfer effects should contradict."""
        detector = ContradictionDetector(
            similarity_threshold=0.5, contradiction_threshold=0.3
        )
        records = [
            # mem-1: consistently positive
            _make_record(memory_id="mem-1", y_share=1, y_withhold=0, record_id="r1"),
            _make_record(memory_id="mem-1", y_share=1, y_withhold=0, record_id="r2"),
            _make_record(memory_id="mem-1", y_share=1, y_withhold=0, record_id="r3"),
            # mem-2: consistently negative
            _make_record(memory_id="mem-2", y_share=0, y_withhold=1, record_id="r4"),
            _make_record(memory_id="mem-2", y_share=0, y_withhold=1, record_id="r5"),
            _make_record(memory_id="mem-2", y_share=0, y_withhold=1, record_id="r6"),
        ]
        cards = {
            "mem-1": _make_card("mem-1", task_tags=["test"], roles=["executor"]),
            "mem-2": _make_card("mem-2", task_tags=["test"], roles=["executor"]),
        }
        contradictions = detector.detect(records, cards)
        transfer_contradictions = [
            c
            for c in contradictions
            if c.contradiction_type == ContradictionType.TRANSFER_CONTRADICTION
        ]
        assert len(transfer_contradictions) == 1
        assert transfer_contradictions[0].severity > 0

    def test_execution_contradiction_mixed_evidence(self):
        """Memory with both success and failure evidence should contradict."""
        detector = ContradictionDetector()
        records = []
        cards = {
            "mem-1": _make_card(
                "mem-1",
                success_count=5,
                failure_count=5,
                with_contexts=True,
            ),
        }
        contradictions = detector.detect(records, cards)
        exec_contradictions = [
            c
            for c in contradictions
            if c.contradiction_type == ContradictionType.EXECUTION_CONTRADICTION
        ]
        assert len(exec_contradictions) == 1

    def test_no_execution_contradiction_pure_success(self):
        """Memory with only success evidence should not contradict."""
        detector = ContradictionDetector()
        records = []
        cards = {
            "mem-1": _make_card("mem-1", success_count=10, failure_count=0),
        }
        contradictions = detector.detect(records, cards)
        exec_contradictions = [
            c
            for c in contradictions
            if c.contradiction_type == ContradictionType.EXECUTION_CONTRADICTION
        ]
        assert len(exec_contradictions) == 0

    def test_dissimilar_contexts_no_contradiction(self):
        """Memories with dissimilar contexts should not contradict."""
        detector = ContradictionDetector(
            similarity_threshold=0.9, contradiction_threshold=0.3
        )
        records = [
            _make_record(memory_id="mem-1", y_share=1, y_withhold=0, record_id="r1"),
            _make_record(memory_id="mem-2", y_share=0, y_withhold=1, record_id="r2"),
        ]
        cards = {
            "mem-1": _make_card("mem-1", task_tags=["alpha"], roles=["planner"]),
            "mem-2": _make_card("mem-2", task_tags=["beta"], roles=["executor"]),
        }
        contradictions = detector.detect(records, cards)
        transfer_contradictions = [
            c
            for c in contradictions
            if c.contradiction_type == ContradictionType.TRANSFER_CONTRADICTION
        ]
        assert len(transfer_contradictions) == 0

    def test_empty_records_and_cards(self):
        detector = ContradictionDetector()
        contradictions = detector.detect([], {})
        assert len(contradictions) == 0


# --- MemoryRefiner Tests ---


class TestMemoryRefiner:
    """Test memory refinement suggestions."""

    def test_suggest_deprecate_for_strong_contradiction(self):
        """Strong contradiction with clear evidence winner → deprecate."""
        refiner = MemoryRefiner(deprecate_threshold=0.3)
        from smtr.memory.refinement import Contradiction

        contradiction = Contradiction(
            memory_id_a="mem-1",
            memory_id_b="mem-2",
            contradiction_type=ContradictionType.TRANSFER_CONTRADICTION,
            severity=0.8,
            description="test",
            evidence_a={"tau_mean": 0.5, "n_records": 20},
            evidence_b={"tau_mean": -0.5, "n_records": 3},
        )
        cards = {
            "mem-1": _make_card("mem-1"),
            "mem-2": _make_card("mem-2"),
        }
        suggestions = refiner.suggest_refinements([contradiction], cards)
        assert len(suggestions) == 1
        assert suggestions[0].action == RefinementAction.DEPRECATE
        assert "mem-2" in suggestions[0].memory_ids

    def test_suggest_flag_for_equal_evidence(self):
        """Equal evidence contradiction → flag for review."""
        refiner = MemoryRefiner(deprecate_threshold=0.3)
        from smtr.memory.refinement import Contradiction

        contradiction = Contradiction(
            memory_id_a="mem-1",
            memory_id_b="mem-2",
            contradiction_type=ContradictionType.TRANSFER_CONTRADICTION,
            severity=0.9,
            description="test",
            evidence_a={"tau_mean": 0.5, "n_records": 10},
            evidence_b={"tau_mean": -0.5, "n_records": 10},
        )
        cards = {
            "mem-1": _make_card("mem-1"),
            "mem-2": _make_card("mem-2"),
        }
        suggestions = refiner.suggest_refinements([contradiction], cards)
        assert len(suggestions) == 1
        assert suggestions[0].action == RefinementAction.FLAG

    def test_suggest_update_for_mild_contradiction(self):
        """Mild contradiction → update evidence."""
        refiner = MemoryRefiner(deprecate_threshold=0.8)
        from smtr.memory.refinement import Contradiction

        contradiction = Contradiction(
            memory_id_a="mem-1",
            memory_id_b="mem-2",
            contradiction_type=ContradictionType.TRANSFER_CONTRADICTION,
            severity=0.4,
            description="test",
            evidence_a={"tau_mean": 0.2, "n_records": 5},
            evidence_b={"tau_mean": -0.2, "n_records": 5},
        )
        cards = {
            "mem-1": _make_card("mem-1"),
            "mem-2": _make_card("mem-2"),
        }
        suggestions = refiner.suggest_refinements([contradiction], cards)
        assert len(suggestions) == 1
        assert suggestions[0].action == RefinementAction.UPDATE_EVIDENCE

    def test_execution_contradiction_suggests_update(self):
        """Execution contradiction → update evidence."""
        refiner = MemoryRefiner()
        from smtr.memory.refinement import Contradiction

        contradiction = Contradiction(
            memory_id_a="mem-1",
            memory_id_b="mem-1",
            contradiction_type=ContradictionType.EXECUTION_CONTRADICTION,
            severity=0.5,
            description="mixed evidence",
        )
        cards = {"mem-1": _make_card("mem-1", success_count=5, failure_count=5)}
        suggestions = refiner.suggest_refinements([contradiction], cards)
        assert len(suggestions) == 1
        assert suggestions[0].action == RefinementAction.UPDATE_EVIDENCE

    def test_apply_refinement_returns_updated_cards(self):
        """Apply refinement should return updated cards dict."""
        refiner = MemoryRefiner()
        from smtr.memory.refinement import RefinementSuggestion

        suggestion = RefinementSuggestion(
            memory_ids=["mem-1"],
            action=RefinementAction.UPDATE_EVIDENCE,
            confidence=0.5,
            reason="test",
        )
        cards = {"mem-1": _make_card("mem-1")}
        updated = refiner.apply_refinement(suggestion, cards)
        assert "mem-1" in updated

    def test_empty_contradictions(self):
        refiner = MemoryRefiner()
        suggestions = refiner.suggest_refinements([], {})
        assert len(suggestions) == 0
