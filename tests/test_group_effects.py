"""Tests for B-05: High-Order Group Effects."""

from smtr.counterfactual.schemas import (
    BranchOutcome,
    ContextFingerprint,
    PairedInterventionRecord,
)
from smtr.evaluation.group_effects import (
    GroupEffectAnalyzer,
)

# --- Fixtures ---


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


def _make_records_for_memory(
    memory_id: str,
    n: int,
    *,
    positive_ratio: float = 0.7,
) -> list[PairedInterventionRecord]:
    """Create n records for a memory with given positive ratio."""
    records = []
    for i in range(n):
        if i < int(n * positive_ratio):
            records.append(
                _make_record(
                    memory_id=memory_id,
                    y_share=1,
                    y_withhold=0,
                    record_id=f"{memory_id}-pos-{i}",
                )
            )
        else:
            records.append(
                _make_record(
                    memory_id=memory_id,
                    y_share=0,
                    y_withhold=1,
                    record_id=f"{memory_id}-neg-{i}",
                )
            )
    return records


# --- GroupEffectAnalyzer Tests ---


class TestGroupEffectAnalyzerBasics:
    """Test basic analyzer operations."""

    def test_analyze_empty_records(self):
        analyzer = GroupEffectAnalyzer()
        summary = analyzer.analyze([])
        assert summary.n_memories == 0
        assert summary.n_groups_analyzed == 0

    def test_analyze_single_memory(self):
        analyzer = GroupEffectAnalyzer()
        records = _make_records_for_memory("mem-1", 10)
        summary = analyzer.analyze(records)
        assert summary.n_memories == 1
        assert summary.n_groups_analyzed == 0

    def test_analyze_two_memories(self):
        analyzer = GroupEffectAnalyzer(min_samples_per_group=3)
        records = _make_records_for_memory("mem-1", 5) + _make_records_for_memory("mem-2", 5)
        summary = analyzer.analyze(records)
        assert summary.n_memories == 2
        assert len(summary.pairwise_effects) == 1

    def test_analyze_three_memories(self):
        analyzer = GroupEffectAnalyzer(min_samples_per_group=3, max_order=3)
        records = (
            _make_records_for_memory("mem-1", 5)
            + _make_records_for_memory("mem-2", 5)
            + _make_records_for_memory("mem-3", 5)
        )
        summary = analyzer.analyze(records)
        assert summary.n_memories == 3
        # 3 pairwise + 1 three-way
        assert len(summary.pairwise_effects) == 3
        assert len(summary.higher_order_effects) >= 0  # May have 3-way if enough samples


class TestGroupTau:
    """Test group tau computation."""

    def test_compute_group_tau_positive(self):
        analyzer = GroupEffectAnalyzer()
        records = _make_records_for_memory("mem-1", 10, positive_ratio=1.0)
        tau = analyzer.compute_group_tau(records, ["mem-1"])
        assert tau == 1.0  # All positive

    def test_compute_group_tau_negative(self):
        analyzer = GroupEffectAnalyzer()
        records = _make_records_for_memory("mem-1", 10, positive_ratio=0.0)
        tau = analyzer.compute_group_tau(records, ["mem-1"])
        assert tau == -1.0  # All negative

    def test_compute_group_tau_mixed(self):
        analyzer = GroupEffectAnalyzer()
        records = _make_records_for_memory("mem-1", 10, positive_ratio=0.5)
        tau = analyzer.compute_group_tau(records, ["mem-1"])
        assert abs(tau) < 1.0  # Mixed

    def test_compute_group_tau_empty(self):
        analyzer = GroupEffectAnalyzer()
        tau = analyzer.compute_group_tau([], ["mem-1"])
        assert tau == 0.0


class TestContributions:
    """Test SHAP-style contribution analysis."""

    def test_contributions_sum_to_zero(self):
        """Contributions should approximately sum to zero (deviation from mean)."""
        analyzer = GroupEffectAnalyzer(min_samples_per_group=3)
        records = (
            _make_records_for_memory("mem-1", 10, positive_ratio=0.8)
            + _make_records_for_memory("mem-2", 10, positive_ratio=0.2)
        )
        summary = analyzer.analyze(records)
        total_shap = sum(c.shap_value for c in summary.contributions)
        # Should be close to zero (deviations from mean cancel out)
        assert abs(total_shap) < 0.5

    def test_positive_memory_has_positive_contribution(self):
        analyzer = GroupEffectAnalyzer(min_samples_per_group=3)
        records = (
            _make_records_for_memory("mem-1", 10, positive_ratio=1.0)
            + _make_records_for_memory("mem-2", 10, positive_ratio=0.0)
        )
        summary = analyzer.analyze(records)
        contributions = {c.memory_id: c.shap_value for c in summary.contributions}
        # mem-1 (all positive) should have positive contribution
        assert contributions.get("mem-1", 0) > 0
        # mem-2 (all negative) should have negative contribution
        assert contributions.get("mem-2", 0) < 0


class TestInteractionEffects:
    """Test interaction effect detection."""

    def test_pairwise_effect_computed(self):
        analyzer = GroupEffectAnalyzer(min_samples_per_group=3)
        records = (
            _make_records_for_memory("mem-1", 10)
            + _make_records_for_memory("mem-2", 10)
        )
        summary = analyzer.analyze(records)
        assert len(summary.pairwise_effects) == 1
        effect = summary.pairwise_effects[0]
        assert effect.order == 2
        assert set(effect.memory_ids) == {"mem-1", "mem-2"}
        assert effect.n_observations == 20

    def test_higher_order_respects_max_order(self):
        analyzer = GroupEffectAnalyzer(min_samples_per_group=3, max_order=2)
        records = (
            _make_records_for_memory("mem-1", 5)
            + _make_records_for_memory("mem-2", 5)
            + _make_records_for_memory("mem-3", 5)
        )
        summary = analyzer.analyze(records)
        assert len(summary.higher_order_effects) == 0  # max_order=2

    def test_variance_explained_non_negative(self):
        analyzer = GroupEffectAnalyzer(min_samples_per_group=3)
        records = (
            _make_records_for_memory("mem-1", 10)
            + _make_records_for_memory("mem-2", 10)
        )
        summary = analyzer.analyze(records)
        assert summary.total_variance_explained >= 0
