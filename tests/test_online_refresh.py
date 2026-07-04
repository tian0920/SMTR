"""Tests for B-03: Online Policy Refresh & Active Data Acquisition."""


from smtr.counterfactual.schemas import (
    BranchOutcome,
    ContextFingerprint,
    PairedInterventionRecord,
    RoutingFeatureSnapshot,
)
from smtr.policy.online_refresh import (
    ActiveAcquisitionConfig,
    ActiveDataAcquisition,
    OnlinePolicyRefresher,
    RefreshConfig,
    RefreshTrigger,
)
from smtr.router.transfer_critic import FourOutcomeTransferCritic
from smtr.router.transfer_features import TransferPredictionInput

# --- Fixtures ---


def _make_record(
    *,
    y_share: int = 1,
    y_withhold: int = 0,
    transfer_class: str | None = None,
    record_id: str = "test-record-001",
    candidate_memory_id: str = "mem-1",
) -> PairedInterventionRecord:
    """Create a minimal PairedInterventionRecord for testing."""
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

    snapshot = RoutingFeatureSnapshot(
        memory_id=candidate_memory_id,
        active_payload_version=1,
        goal_summary="test goal",
        task_tags=["test"],
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
        candidate_memory_id=candidate_memory_id,
        candidate_payload_version=1,
        candidate_card_snapshot=snapshot,
        candidate_order=[candidate_memory_id],
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


def _make_records(n: int, *, positive_ratio: float = 0.5) -> list[PairedInterventionRecord]:
    """Create n records with mixed transfer classes."""
    records = []
    for i in range(n):
        if i < int(n * positive_ratio):
            records.append(_make_record(
                y_share=1, y_withhold=0, record_id=f"pos-{i}", candidate_memory_id=f"mem-{i}"
            ))
        else:
            records.append(_make_record(
                y_share=0, y_withhold=1, record_id=f"neg-{i}", candidate_memory_id=f"mem-{i}"
            ))
    return records


def _make_prediction_input(memory_id: str = "mem-1") -> TransferPredictionInput:
    """Create a test prediction input."""
    context = ContextFingerprint(
        task_id="task-1",
        receiver_agent_id="agent-1",
        receiver_role="executor",
        task_stage="test",
        selected_memory_ids=[],
        selected_set_signature="empty",
        episode_id="ep-1",
    )
    card = RoutingFeatureSnapshot(
        memory_id=memory_id,
        active_payload_version=1,
        goal_summary="test goal",
        task_tags=["test"],
    )
    return TransferPredictionInput(context=context, candidate_card=card, selected_cards=[])


# --- OnlinePolicyRefresher Tests ---


class TestRefresherBasics:
    """Test OnlinePolicyRefresher basic operations."""

    def test_initial_state(self):
        refresher = OnlinePolicyRefresher()
        assert refresher.state.current_version == 0
        assert refresher.state.total_records_seen == 0
        assert refresher.state.new_records_since_refresh == 0

    def test_add_records(self):
        refresher = OnlinePolicyRefresher()
        records = _make_records(5)
        added = refresher.add_records(records)
        assert added == 5
        assert refresher.state.total_records_seen == 5
        assert refresher.state.new_records_since_refresh == 5

    def test_should_refresh_record_count_mode(self):
        config = RefreshConfig(
            trigger_mode=RefreshTrigger.RECORD_COUNT, min_new_records=5
        )
        refresher = OnlinePolicyRefresher(config=config)
        assert refresher.should_refresh() is False
        refresher.add_records(_make_records(3))
        assert refresher.should_refresh() is False
        refresher.add_records(_make_records(2))
        assert refresher.should_refresh() is True

    def test_should_refresh_manual_mode(self):
        config = RefreshConfig(trigger_mode=RefreshTrigger.MANUAL)
        refresher = OnlinePolicyRefresher(config=config)
        refresher.add_records(_make_records(100))
        assert refresher.should_refresh() is False


class TestRefresherRefresh:
    """Test OnlinePolicyRefresher refresh operations."""

    def test_refresh_with_records(self):
        config = RefreshConfig(
            trigger_mode=RefreshTrigger.RECORD_COUNT, min_new_records=3
        )
        refresher = OnlinePolicyRefresher(config=config)
        records = _make_records(5)
        refresher.add_records(records)
        assert refresher.should_refresh() is True

        new_critic, success = refresher.refresh(seed=42)
        assert success is True
        assert refresher.state.current_version == 1
        assert refresher.state.refresh_count == 1
        assert refresher.state.new_records_since_refresh == 0

    def test_refresh_without_trigger(self):
        config = RefreshConfig(min_new_records=10)
        refresher = OnlinePolicyRefresher(config=config)
        refresher.add_records(_make_records(3))
        critic, success = refresher.refresh()
        assert success is False
        assert refresher.state.current_version == 0

    def test_force_refresh(self):
        refresher = OnlinePolicyRefresher()
        refresher.add_records(_make_records(5))
        critic, success = refresher.refresh(force=True, seed=42)
        assert success is True

    def test_refresh_no_records_fails(self):
        refresher = OnlinePolicyRefresher()
        critic, success = refresher.refresh(force=True)
        assert success is False

    def test_version_info(self):
        refresher = OnlinePolicyRefresher()
        refresher.add_records(_make_records(5))
        refresher.refresh(force=True, seed=42)
        info = refresher.get_version_info()
        assert info["current_version"] == 1
        assert info["total_records_seen"] == 5
        assert info["refresh_count"] == 1


class TestRefresherUncertainty:
    """Test uncertainty-based refresh trigger."""

    def test_uncertainty_trigger(self):
        config = RefreshConfig(
            trigger_mode=RefreshTrigger.UNCERTAINTY,
            uncertainty_threshold=0.3,
        )
        refresher = OnlinePolicyRefresher(config=config)
        # Initially uncertainty is 0
        assert refresher.should_refresh() is False

        # Set high uncertainty
        refresher.state.last_uncertainty_score = 0.5
        assert refresher.should_refresh() is True

    def test_estimate_uncertainty(self):
        # Train a critic first
        records = _make_records(10)
        critic = FourOutcomeTransferCritic()
        critic.fit(records, seed=42, n_bootstrap=5)

        refresher = OnlinePolicyRefresher(initial_critic=critic)
        samples = [_make_prediction_input(f"mem-{i}") for i in range(5)]
        uncertainty = refresher.estimate_uncertainty(samples)
        assert isinstance(uncertainty, float)
        assert uncertainty >= 0.0


# --- ActiveDataAcquisition Tests ---


class TestActiveAcquisition:
    """Test ActiveDataAcquisition."""

    def test_score_candidates(self):
        records = _make_records(10)
        critic = FourOutcomeTransferCritic()
        critic.fit(records, seed=42, n_bootstrap=5)

        acquisition = ActiveDataAcquisition(critic=critic)
        candidates = [_make_prediction_input(f"mem-{i}") for i in range(5)]
        scored = acquisition.score_candidates(candidates)
        assert len(scored) == 5
        # Should be sorted by uncertainty descending
        for i in range(len(scored) - 1):
            assert scored[i][1] >= scored[i + 1][1]

    def test_suggest_acquisitions(self):
        records = _make_records(10)
        critic = FourOutcomeTransferCritic()
        critic.fit(records, seed=42, n_bootstrap=5)

        config = ActiveAcquisitionConfig(max_candidates_per_round=3)
        acquisition = ActiveDataAcquisition(critic=critic, config=config)
        candidates = [_make_prediction_input(f"mem-{i}") for i in range(10)]
        suggestions = acquisition.suggest_acquisitions(candidates)
        assert len(suggestions) <= 3
        # Suggestions should be sorted by priority
        for i in range(len(suggestions) - 1):
            assert suggestions[i].priority_score >= suggestions[i + 1].priority_score

    def test_find_boundary_regions(self):
        records = _make_records(10)
        critic = FourOutcomeTransferCritic()
        critic.fit(records, seed=42, n_bootstrap=5)

        acquisition = ActiveDataAcquisition(critic=critic)
        candidates = [_make_prediction_input(f"mem-{i}") for i in range(10)]
        boundary = acquisition.find_boundary_regions(candidates, margin=0.5)
        # With large margin, should find some boundary candidates
        assert isinstance(boundary, list)

    def test_empty_candidates(self):
        critic = FourOutcomeTransferCritic()
        acquisition = ActiveDataAcquisition(critic=critic)
        scored = acquisition.score_candidates([])
        assert len(scored) == 0
        suggestions = acquisition.suggest_acquisitions([])
        assert len(suggestions) == 0
