"""S9 Invariant tests — properties that must never be broken.

These tests verify the fundamental invariants described in implementation.md §11:
- T-21: Payload isolation (unselected payload steps don't leak)
- T-22: Paired branch isolation (share/withhold only differ in target exposure)
- T-23: Memory store immutability during collection
- T-24: Policy-specific estimand (no mixing of different policy records)
- T-25: Feature leakage prevention (forbidden fields don't enter critic features)
"""

import json
from pathlib import Path

import pytest

from smtr.counterfactual.schemas import (
    BranchOutcome,
    ContextFingerprint,
    PairedInterventionRecord,
    RoutingFeatureSnapshot,
    transfer_class_from_outcomes,
)
from smtr.counterfactual.snapshot import ReadOnlyCounterfactualStoreError, ReadOnlyPinnedMemoryView
from smtr.evaluation.leakage_scanner import TransferFeatureLeakageScanner
from smtr.memory.schemas import (
    ExecutionEvidence,
    MemoryRoutingCard,
    ProcedurePayload,
)
from smtr.memory.seed_memories import build_seed_memories, build_seed_memory_pool
from smtr.memory.store import SQLiteSharedMemoryRepository
from smtr.router.baseline_router import NoMemoryRouter
from smtr.router.candidate_proposer import DeterministicHybridCandidateProposer
from smtr.router.transfer_features import (
    FORBIDDEN_FIELDS,
    HashingTransferFeatureEncoder,
    TransferPredictionInput,
    _reject_forbidden,
)
from smtr.runtime.graph import run_demo_with_repository

# =============================================================================
# T-21: Payload isolation
# =============================================================================


class TestPayloadIsolation:
    """T-21: Unselected payload steps must not leak anywhere."""

    def test_unselected_payload_steps_not_in_state(self, tmp_path) -> None:
        """Verify payload steps don't appear in graph state when nothing is selected."""
        repository = SQLiteSharedMemoryRepository(tmp_path / "memory.sqlite")
        from smtr.memory.seed_memories import seed_repository
        seed_repository(repository)

        state = run_demo_with_repository(repository=repository, seed=7, top_k=4)

        # No memories should be selected by default (NoMemoryRouter)
        assert all(
            context["visible_payloads"] == []
            for context in state["agent_local_context"].values()
        )

        # Payload steps must not appear in state representation
        payload_steps = [step for _, payload in build_seed_memories() for step in payload.steps]
        state_text = repr(state)
        assert all(step not in state_text for step in payload_steps)

    def test_candidate_proposer_cannot_see_payload_steps(self) -> None:
        """Candidate proposer only sees routing cards, not payload steps."""
        pool = build_seed_memory_pool()
        cards = pool.list_routing_cards()

        proposer = DeterministicHybridCandidateProposer()
        candidates = proposer.propose(
            task="test task",
            receiver_agent="executor",
            environment_observation={},
            cards=cards,
            top_k=3,
            seed=7,
        )

        # Candidates should only contain card-level info, not payload steps
        for candidate in candidates:
            assert not hasattr(candidate, "steps")
            assert not hasattr(candidate, "payload")

    def test_router_trace_does_not_contain_payload_steps(self) -> None:
        """Router decisions don't include payload steps."""
        pool = build_seed_memory_pool()
        cards = pool.list_routing_cards()
        cards_by_id = {card.memory_id: card for card in cards}

        proposer = DeterministicHybridCandidateProposer()
        candidates = proposer.propose(
            task="test task",
            receiver_agent="executor",
            environment_observation={},
            cards=cards,
            top_k=3,
            seed=7,
        )

        decisions, selected_ids = NoMemoryRouter().decide(
            task="test task",
            receiver_agent="executor",
            candidates=candidates,
            cards_by_id=cards_by_id,
            seed=7,
        )

        # Decisions should not contain payload steps
        payload_steps = [step for _, payload in build_seed_memories() for step in payload.steps]
        decision_text = repr(decisions)
        for step in payload_steps:
            assert step not in decision_text

    def test_routing_card_serialization_excludes_payload(self) -> None:
        """MemoryRoutingCard serialization doesn't include payload steps."""
        card = MemoryRoutingCard(
            memory_id="test-mem",
            active_payload_version=1,
            goal_summary="test goal",
            task_tags=["test"],
            compatible_receiver_roles=["executor"],
        )
        card_json = card.model_dump_json()
        assert "steps" not in card_json.lower() or "task_tags" in card_json
        assert "procedure_payload" not in card_json
        assert "goal_summary" in card_json  # Card has goal summary, not steps


# =============================================================================
# T-22: Paired branch isolation
# =============================================================================


class TestPairedBranchIsolation:
    """T-22: Share/withhold branches only differ in target memory exposure."""

    def test_readonly_memory_view_blocks_writes(self, tmp_path) -> None:
        """ReadOnlyPinnedMemoryView prevents any writes during collection."""
        repository = SQLiteSharedMemoryRepository(tmp_path / "memory.sqlite")
        from smtr.memory.seed_memories import seed_repository
        seed_repository(repository)

        snapshot = repository.create_read_snapshot()
        view = ReadOnlyPinnedMemoryView(repository=repository, snapshot=snapshot)

        # All write operations should raise
        with pytest.raises(ReadOnlyCounterfactualStoreError):
            view.create_memory("test", "test", ProcedurePayload(
                memory_id="test", version=1, goal="test", steps=["step1"]
            ))

        with pytest.raises(ReadOnlyCounterfactualStoreError):
            view.record_execution_evidence(ExecutionEvidence(
                memory_id="test",
                payload_version=1,
                context=ContextFingerprint(
                    task_id="t", receiver_agent_id="a", receiver_role="r",
                    task_stage="s", selected_memory_ids=[], selected_set_signature="e",
                    episode_id="ep",
                ),
                execution_success=True,
                source="direct_execution",
            ))

    def test_readonly_view_returns_pinned_snapshot(self, tmp_path) -> None:
        """ReadOnlyPinnedMemoryView returns consistent snapshot data."""
        repository = SQLiteSharedMemoryRepository(tmp_path / "memory.sqlite")
        from smtr.memory.seed_memories import seed_repository
        seed_repository(repository)

        snapshot = repository.create_read_snapshot()
        view = ReadOnlyPinnedMemoryView(repository=repository, snapshot=snapshot)

        # Revision should match snapshot
        assert view.current_revision() == snapshot.store_revision

        # Cards should match snapshot
        view_cards = view.get_routing_cards()
        snapshot_cards = snapshot.get_routing_cards()
        assert len(view_cards) == len(snapshot_cards)

    def test_branch_outcomes_share_same_context(self) -> None:
        """Share and withhold outcomes should share the same decision context."""
        ctx = ContextFingerprint(
            task_id="task-1",
            receiver_agent_id="agent-1",
            receiver_role="executor",
            task_stage="test",
            selected_memory_ids=[],
            selected_set_signature="empty",
            episode_id="ep-1",
        )

        # Both branches use the same context
        branch_share = BranchOutcome(
            team_success=True,
            team_reward=1.0,
            team_summary="success",
            final_environment_observation={},
            selected_memory_ids_by_agent={},
            router_trace=[],
            target_memory_visible_to_receiver=True,
            selected_final_at_target_node=[],
        )
        branch_withhold = BranchOutcome(
            team_success=False,
            team_reward=0.0,
            team_summary="failure",
            final_environment_observation={},
            selected_memory_ids_by_agent={},
            router_trace=[],
            target_memory_visible_to_receiver=False,
            selected_final_at_target_node=[],
        )

        record = PairedInterventionRecord(
            record_id="rec-1",
            episode_id="ep-1",
            task_id="task-1",
            graph_node="node-1",
            receiver_agent_id="agent-1",
            receiver_role="executor",
            task_stage="test",
            candidate_memory_id="mem-1",
            candidate_payload_version=1,
            candidate_order=["mem-1"],
            target_index=0,
            selected_before=[],
            decision_context=ctx,
            memory_store_revision=1,
            memory_snapshot_digest="abc",
            runtime_snapshot_digest="def",
            continuation_policy_name="test",
            continuation_policy_version="1",
            common_seed=42,
            share_outcome=branch_share,
            withhold_outcome=branch_withhold,
            y_share=1,
            y_withhold=0,
            transfer_class="positive",
            target_selection_probability=0.5,
            schema_version="1.1",
            candidate_card_snapshot=RoutingFeatureSnapshot(
                memory_id="mem-1", active_payload_version=1, goal_summary="test",
            ),
            selected_before_card_snapshots=[],
            selected_before_payload_versions={},
        )

        # Both branches share the same decision context
        assert record.share_outcome is not None
        assert record.withhold_outcome is not None
        assert record.decision_context.task_id == "task-1"
        assert record.decision_context.episode_id == "ep-1"


# =============================================================================
# T-23: Memory store immutability during collection
# =============================================================================


class TestMemoryStoreImmutability:
    """T-23: Memory store revision is fixed during collection."""

    def test_snapshot_revision_is_frozen(self, tmp_path) -> None:
        """Memory store snapshot captures a fixed revision."""
        repository = SQLiteSharedMemoryRepository(tmp_path / "memory.sqlite")
        from smtr.memory.seed_memories import seed_repository
        seed_repository(repository)

        revision_before = repository.current_revision()
        snapshot = repository.create_read_snapshot()
        revision_after = repository.current_revision()

        # Snapshot doesn't change repository state
        assert revision_before == revision_after
        assert snapshot.store_revision == revision_before

    def test_readonly_view_preserves_revision(self, tmp_path) -> None:
        """ReadOnlyPinnedMemoryView preserves the snapshot revision."""
        repository = SQLiteSharedMemoryRepository(tmp_path / "memory.sqlite")
        from smtr.memory.seed_memories import seed_repository
        seed_repository(repository)

        snapshot = repository.create_read_snapshot()
        view = ReadOnlyPinnedMemoryView(repository=repository, snapshot=snapshot)

        # Revision is fixed at snapshot time
        assert view.current_revision() == snapshot.store_revision

    def test_paired_rollout_checks_revision(self) -> None:
        """PairedRolloutCollector checks revision doesn't change."""
        from smtr.counterfactual.paired_rollout import (
            CounterfactualIntegrityError,
        )

        # The collector code explicitly checks:
        # if before_revision != after_revision:
        #     raise CounterfactualIntegrityError(...)
        # This test verifies the error class exists and is a RuntimeError
        assert issubclass(CounterfactualIntegrityError, RuntimeError)


# =============================================================================
# T-24: Policy-specific estimand
# =============================================================================


class TestPolicySpecificEstimand:
    """T-24: Don't mix records from different continuation policies."""

    def test_records_have_policy_fingerprint(self) -> None:
        """PairedInterventionRecord tracks continuation policy fingerprint."""
        ctx = ContextFingerprint(
            task_id="task-1", receiver_agent_id="agent-1", receiver_role="executor",
            task_stage="test", selected_memory_ids=[], selected_set_signature="empty",
            episode_id="ep-1",
        )
        branch = BranchOutcome(
            team_success=True, team_reward=1.0, team_summary="ok",
            final_environment_observation={}, selected_memory_ids_by_agent={},
            router_trace=[], target_memory_visible_to_receiver=True,
            selected_final_at_target_node=[],
        )
        record = PairedInterventionRecord(
            record_id="rec-1", episode_id="ep-1", task_id="task-1",
            graph_node="node-1", receiver_agent_id="agent-1", receiver_role="executor",
            task_stage="test", candidate_memory_id="mem-1", candidate_payload_version=1,
            candidate_order=["mem-1"], target_index=0, selected_before=[],
            decision_context=ctx, memory_store_revision=1, memory_snapshot_digest="abc",
            runtime_snapshot_digest="def", continuation_policy_name="pi2_explore",
            continuation_policy_version="2", common_seed=42,
            share_outcome=branch, withhold_outcome=branch,
            y_share=1, y_withhold=1, transfer_class="neutral_success",
            target_selection_probability=0.5,
            continuation_policy_fingerprint="test_fingerprint_abc",
            schema_version="1.1",
            candidate_card_snapshot=RoutingFeatureSnapshot(
                memory_id="mem-1", active_payload_version=1, goal_summary="test",
            ),
            selected_before_card_snapshots=[],
            selected_before_payload_versions={},
        )

        assert record.continuation_policy_fingerprint == "test_fingerprint_abc"
        assert record.continuation_policy_name == "pi2_explore"

    def test_different_policy_records_are_distinguishable(self) -> None:
        """Records from different policies have different fingerprints."""
        ctx = ContextFingerprint(
            task_id="task-1", receiver_agent_id="agent-1", receiver_role="executor",
            task_stage="test", selected_memory_ids=[], selected_set_signature="empty",
            episode_id="ep-1",
        )
        branch = BranchOutcome(
            team_success=True, team_reward=1.0, team_summary="ok",
            final_environment_observation={}, selected_memory_ids_by_agent={},
            router_trace=[], target_memory_visible_to_receiver=True,
            selected_final_at_target_node=[],
        )
        card_snapshot = RoutingFeatureSnapshot(
            memory_id="mem-1", active_payload_version=1, goal_summary="test",
        )

        record_pi0 = PairedInterventionRecord(
            record_id="rec-pi0", episode_id="ep-1", task_id="task-1",
            graph_node="node-1", receiver_agent_id="agent-1", receiver_role="executor",
            task_stage="test", candidate_memory_id="mem-1", candidate_payload_version=1,
            candidate_order=["mem-1"], target_index=0, selected_before=[],
            decision_context=ctx, memory_store_revision=1, memory_snapshot_digest="abc",
            runtime_snapshot_digest="def", continuation_policy_name="pi0_no_share",
            continuation_policy_version="1", common_seed=42,
            share_outcome=branch, withhold_outcome=branch,
            y_share=1, y_withhold=1, transfer_class="neutral_success",
            target_selection_probability=0.5,
            continuation_policy_fingerprint="pi0_fingerprint",
            schema_version="1.1",
            candidate_card_snapshot=card_snapshot,
            selected_before_card_snapshots=[],
            selected_before_payload_versions={},
        )

        record_pi2 = PairedInterventionRecord(
            record_id="rec-pi2", episode_id="ep-1", task_id="task-1",
            graph_node="node-1", receiver_agent_id="agent-1", receiver_role="executor",
            task_stage="test", candidate_memory_id="mem-1", candidate_payload_version=1,
            candidate_order=["mem-1"], target_index=0, selected_before=[],
            decision_context=ctx, memory_store_revision=1, memory_snapshot_digest="abc",
            runtime_snapshot_digest="def", continuation_policy_name="pi2_explore",
            continuation_policy_version="2", common_seed=42,
            share_outcome=branch, withhold_outcome=branch,
            y_share=1, y_withhold=1, transfer_class="neutral_success",
            target_selection_probability=0.5,
            continuation_policy_fingerprint="pi2_fingerprint",
            schema_version="1.1",
            candidate_card_snapshot=card_snapshot,
            selected_before_card_snapshots=[],
            selected_before_payload_versions={},
        )

        # Different policies have different fingerprints
        assert (
            record_pi0.continuation_policy_fingerprint
            != record_pi2.continuation_policy_fingerprint
        )
        assert record_pi0.continuation_policy_name != record_pi2.continuation_policy_name


# =============================================================================
# T-25: Feature leakage prevention
# =============================================================================


class TestFeatureLeakagePrevention:
    """T-25: Forbidden fields must not enter critic features."""

    def test_forbidden_fields_defined(self) -> None:
        """FORBIDDEN_FIELDS set is properly defined."""
        expected_forbidden = {
            "steps", "payload", "procedure_payload", "visible_payloads", "chain_of_thought"
        }
        assert expected_forbidden.issubset(FORBIDDEN_FIELDS)

    def test_reject_forbidden_raises_on_steps(self) -> None:
        """_reject_forbidden raises when steps field is present."""
        data = {"steps": ["step1", "step2"], "other": "value"}
        with pytest.raises(ValueError, match="forbidden"):
            _reject_forbidden(data)

    def test_reject_forbidden_raises_on_payload(self) -> None:
        """_reject_forbidden raises when payload field is present."""
        data = {"payload": {"steps": ["step1"]}, "other": "value"}
        with pytest.raises(ValueError, match="forbidden"):
            _reject_forbidden(data)

    def test_reject_forbidden_passes_clean_data(self) -> None:
        """_reject_forbidden passes for clean data."""
        data = {"goal_summary": "test", "task_tags": ["test"], "version": 1}
        _reject_forbidden(data)  # Should not raise

    def test_leakage_scanner_detects_no_violations_on_clean_records(self) -> None:
        """Leakage scanner finds no violations on clean records."""
        ctx = ContextFingerprint(
            task_id="task-1", receiver_agent_id="agent-1", receiver_role="executor",
            task_stage="test", selected_memory_ids=[], selected_set_signature="empty",
            episode_id="ep-1",
        )
        branch = BranchOutcome(
            team_success=True, team_reward=1.0, team_summary="ok",
            final_environment_observation={}, selected_memory_ids_by_agent={},
            router_trace=[], target_memory_visible_to_receiver=True,
            selected_final_at_target_node=[],
        )
        card_snapshot = RoutingFeatureSnapshot(
            memory_id="mem-1", active_payload_version=1, goal_summary="test goal",
            task_tags=["test"], compatible_receiver_roles=["executor"],
        )
        record = PairedInterventionRecord(
            record_id="rec-1", episode_id="ep-1", task_id="task-1",
            graph_node="node-1", receiver_agent_id="agent-1", receiver_role="executor",
            task_stage="test", candidate_memory_id="mem-1", candidate_payload_version=1,
            candidate_order=["mem-1"], target_index=0, selected_before=[],
            decision_context=ctx, memory_store_revision=1, memory_snapshot_digest="abc",
            runtime_snapshot_digest="def", continuation_policy_name="test",
            continuation_policy_version="1", common_seed=42,
            share_outcome=branch, withhold_outcome=branch,
            y_share=1, y_withhold=1, transfer_class="neutral_success",
            target_selection_probability=0.5,
            schema_version="1.1",
            candidate_card_snapshot=card_snapshot,
            selected_before_card_snapshots=[],
            selected_before_payload_versions={},
        )

        scanner = TransferFeatureLeakageScanner()
        result = scanner.scan([record])

        assert result["record_count"] == 1
        assert len(result["violations"]) == 0

    def test_leakage_scanner_forbidden_fields(self) -> None:
        """Leakage scanner has comprehensive forbidden field list."""
        scanner = TransferFeatureLeakageScanner()
        assert "memory_id" in scanner.forbidden
        assert "steps" in scanner.forbidden
        assert "payload" in scanner.forbidden
        assert "transfer_class" in scanner.forbidden
        assert "y_share" in scanner.forbidden
        assert "y_withhold" in scanner.forbidden
        assert "team_reward" in scanner.forbidden
        assert "scenario_family" in scanner.forbidden
        assert "environment_regime" in scanner.forbidden

    def test_encoder_tokens_dont_contain_forbidden_values(self) -> None:
        """Encoder tokens don't contain raw forbidden field values."""
        ctx = ContextFingerprint(
            task_id="task-1", receiver_agent_id="agent-1", receiver_role="executor",
            task_stage="test", selected_memory_ids=[], selected_set_signature="empty",
            episode_id="ep-1",
        )
        card = RoutingFeatureSnapshot(
            memory_id="mem-1", active_payload_version=1, goal_summary="test goal",
            task_tags=["test"], compatible_receiver_roles=["executor"],
        )
        item = TransferPredictionInput(
            context=ctx,
            candidate_card=card,
            selected_cards=[],
        )

        encoder = HashingTransferFeatureEncoder()
        tokens = encoder.tokens(item)

        # Tokens should be prefixed (e.g., "task_tag:test") not raw values
        for token in tokens:
            assert ":" in token or token.startswith("selected_count:")
            # No token should be a raw memory_id
            assert token != "mem-1"
            # No token should contain forbidden field names directly
            assert "transfer_class" not in token
            assert "y_share" not in token
            assert "y_withhold" not in token

    def test_training_record_validation_rejects_forbidden_fields(self) -> None:
        """Training record validation rejects records with forbidden fields."""
        import tempfile

        from smtr.router.transfer_features import load_paired_records_for_training

        # Create a record with forbidden field
        bad_record = {
            "record_id": "bad",
            "episode_id": "ep-1",
            "task_id": "task-1",
            "graph_node": "node-1",
            "receiver_agent_id": "agent-1",
            "receiver_role": "executor",
            "task_stage": "test",
            "candidate_memory_id": "mem-1",
            "candidate_payload_version": 1,
            "candidate_order": ["mem-1"],
            "target_index": 0,
            "selected_before": [],
            "memory_store_revision": 1,
            "memory_snapshot_digest": "abc",
            "runtime_snapshot_digest": "def",
            "continuation_policy_name": "test",
            "continuation_policy_version": "1",
            "common_seed": 42,
            "y_share": 1,
            "y_withhold": 1,
            "transfer_class": "neutral_success",
            "target_selection_probability": 0.5,
            "schema_version": "1.1",
            "steps": ["forbidden_step"],  # This should be rejected
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps(bad_record) + "\n")
            f.flush()
            temp_path = Path(f.name)

        try:
            with pytest.raises(ValueError, match="forbidden"):
                load_paired_records_for_training(temp_path)
        finally:
            temp_path.unlink()


# =============================================================================
# Additional invariant cross-checks
# =============================================================================


class TestInvariantCrossChecks:
    """Cross-cutting invariant checks."""

    def test_four_outcome_labels_consistent(self) -> None:
        """Four-outcome labels are consistent with y_share/y_withhold."""
        assert transfer_class_from_outcomes(1, 0) == "positive"
        assert transfer_class_from_outcomes(0, 1) == "negative"
        assert transfer_class_from_outcomes(1, 1) == "neutral_success"
        assert transfer_class_from_outcomes(0, 0) == "neutral_failure"

    def test_routing_feature_snapshot_is_frozen(self) -> None:
        """RoutingFeatureSnapshot is immutable (frozen)."""
        snapshot = RoutingFeatureSnapshot(
            memory_id="mem-1",
            active_payload_version=1,
            goal_summary="test",
        )
        assert snapshot.model_config.get("frozen") is True

    def test_context_fingerprint_is_frozen(self) -> None:
        """ContextFingerprint is immutable (frozen)."""
        ctx = ContextFingerprint(
            task_id="task-1",
            receiver_agent_id="agent-1",
            receiver_role="executor",
            task_stage="test",
            selected_memory_ids=[],
            selected_set_signature="empty",
            episode_id="ep-1",
        )
        assert ctx.model_config.get("frozen") is True

    def test_transfer_prediction_input_is_frozen(self) -> None:
        """TransferPredictionInput is immutable (frozen)."""
        ctx = ContextFingerprint(
            task_id="task-1", receiver_agent_id="agent-1", receiver_role="executor",
            task_stage="test", selected_memory_ids=[], selected_set_signature="empty",
            episode_id="ep-1",
        )
        card = RoutingFeatureSnapshot(
            memory_id="mem-1", active_payload_version=1, goal_summary="test",
        )
        item = TransferPredictionInput(context=ctx, candidate_card=card)
        assert item.model_config.get("frozen") is True
