"""Tests for B-04: Off-Policy Correction."""

import numpy as np
import pytest

from smtr.counterfactual.schemas import (
    BranchOutcome,
    ContextFingerprint,
    PairedInterventionRecord,
)
from smtr.router.off_policy_correction import (
    OffPolicyConfig,
    OffPolicyCorrector,
    PolicyRatioEstimator,
    WeightingScheme,
)

# --- Fixtures ---


def _make_record(
    *,
    y_share: int = 1,
    y_withhold: int = 0,
    transfer_class: str | None = None,
    target_selection_probability: float | None = 0.5,
    record_id: str = "test-record-001",
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
        candidate_memory_id="mem-1",
        candidate_payload_version=1,
        candidate_order=["mem-1"],
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
        target_selection_probability=target_selection_probability,
    )


# --- OffPolicyCorrector Tests ---


class TestComputeWeight:
    """Test importance weight computation."""

    def test_equal_probabilities_give_weight_one(self):
        corrector = OffPolicyCorrector()
        raw, clipped = corrector.compute_weight(
            target_probability=0.5, behavior_probability=0.5
        )
        assert abs(raw - 1.0) < 1e-6
        assert abs(clipped - 1.0) < 1e-6

    def test_higher_target_gives_weight_above_one(self):
        corrector = OffPolicyCorrector()
        raw, clipped = corrector.compute_weight(
            target_probability=0.8, behavior_probability=0.4
        )
        assert raw > 1.0
        assert abs(raw - 2.0) < 1e-6

    def test_lower_target_gives_weight_below_one(self):
        corrector = OffPolicyCorrector()
        raw, clipped = corrector.compute_weight(
            target_probability=0.2, behavior_probability=0.8
        )
        assert raw < 1.0
        assert abs(raw - 0.25) < 1e-6

    def test_clipping_upper_bound(self):
        config = OffPolicyConfig(max_weight=5.0)
        corrector = OffPolicyCorrector(config=config)
        raw, clipped = corrector.compute_weight(
            target_probability=0.99, behavior_probability=0.01
        )
        assert raw > 5.0
        assert clipped == 5.0

    def test_clipping_lower_bound(self):
        config = OffPolicyConfig(min_weight=0.1)
        corrector = OffPolicyCorrector(config=config)
        raw, clipped = corrector.compute_weight(
            target_probability=0.001, behavior_probability=0.99
        )
        assert raw < 0.1
        assert clipped == 0.1

    def test_no_clipping_with_ratio_scheme(self):
        config = OffPolicyConfig(weighting_scheme=WeightingScheme.RATIO, max_weight=2.0)
        corrector = OffPolicyCorrector(config=config)
        raw, clipped = corrector.compute_weight(
            target_probability=0.9, behavior_probability=0.1
        )
        # Ratio scheme doesn't clip
        assert raw == clipped
        assert raw > 2.0

    def test_zero_behavior_probability_uses_min_weight(self):
        corrector = OffPolicyCorrector(config=OffPolicyConfig(min_weight=0.01))
        raw, clipped = corrector.compute_weight(
            target_probability=0.5, behavior_probability=0.0
        )
        # Should use min_weight as floor for behavior_probability
        assert raw > 0
        assert np.isfinite(raw)


class TestCorrectRecord:
    """Test single record correction."""

    def test_positive_record_positive_tau(self):
        corrector = OffPolicyCorrector()
        record = _make_record(y_share=1, y_withhold=0)
        result = corrector.correct_record(record, target_probability=0.5)
        assert result.tau_observed == 1.0  # y_share - y_withhold = 1 - 0
        assert result.tau_corrected > 0
        assert result.transfer_class == "positive"

    def test_negative_record_negative_tau(self):
        corrector = OffPolicyCorrector()
        record = _make_record(y_share=0, y_withhold=1)
        result = corrector.correct_record(record, target_probability=0.5)
        assert result.tau_observed == -1.0
        assert result.tau_corrected < 0
        assert result.transfer_class == "negative"

    def test_neutral_success_zero_tau(self):
        corrector = OffPolicyCorrector()
        record = _make_record(y_share=1, y_withhold=1)
        result = corrector.correct_record(record, target_probability=0.5)
        assert result.tau_observed == 0.0
        assert result.tau_corrected == 0.0

    def test_weight_affects_corrected_tau(self):
        corrector = OffPolicyCorrector()
        record = _make_record(y_share=1, y_withhold=0)
        # With equal probabilities, weight = 1.0
        result_equal = corrector.correct_record(record, target_probability=0.5)
        # With higher target prob, weight > 1.0
        result_higher = corrector.correct_record(record, target_probability=0.8)
        assert result_higher.tau_corrected > result_equal.tau_corrected

    def test_clipping_detected(self):
        config = OffPolicyConfig(max_weight=2.0)
        corrector = OffPolicyCorrector(config=config)
        record = _make_record(
            y_share=1, y_withhold=0, target_selection_probability=0.01
        )
        result = corrector.correct_record(record, target_probability=0.9)
        assert result.is_clipped is True

    def test_missing_probability_uses_default(self):
        config = OffPolicyConfig(default_selection_probability=0.3)
        corrector = OffPolicyCorrector(config=config)
        record = _make_record(target_selection_probability=None)
        result = corrector.correct_record(record, target_probability=0.3)
        # weight = 0.3 / 0.3 = 1.0
        assert abs(result.clipped_weight - 1.0) < 1e-6

    def test_require_recorded_probabilities_raises(self):
        config = OffPolicyConfig(require_recorded_probabilities=True)
        corrector = OffPolicyCorrector(config=config)
        record = _make_record(target_selection_probability=None)
        with pytest.raises(ValueError, match="no target_selection_probability"):
            corrector.correct_record(record, target_probability=0.5)


class TestCorrectBatch:
    """Test batch correction."""

    def test_empty_batch(self):
        corrector = OffPolicyCorrector()
        summary = corrector.correct_batch([])
        assert summary.n_records == 0
        assert summary.effective_sample_size == 0.0

    def test_uniform_weights_ess_equals_n(self):
        corrector = OffPolicyCorrector()
        records = [
            _make_record(record_id=f"r{i}", target_selection_probability=0.5)
            for i in range(10)
        ]
        summary = corrector.correct_batch(records, target_probability=0.5)
        assert summary.n_records == 10
        # All weights = 1.0, ESS = n
        assert abs(summary.effective_sample_size - 10.0) < 1e-3

    def test_mixed_records(self):
        corrector = OffPolicyCorrector()
        records = [
            _make_record(record_id="r1", y_share=1, y_withhold=0, target_selection_probability=0.3),
            _make_record(record_id="r2", y_share=0, y_withhold=1, target_selection_probability=0.7),
            _make_record(record_id="r3", y_share=1, y_withhold=1, target_selection_probability=0.5),
        ]
        summary = corrector.correct_batch(records, target_probability=0.5)
        assert summary.n_records == 3
        assert summary.n_clipped == 0  # No clipping with default config
        assert summary.mean_weight > 0
        assert len(summary.weights) == 3

    def test_self_normalized_weights(self):
        config = OffPolicyConfig(weighting_scheme=WeightingScheme.SELF_NORMALIZED)
        corrector = OffPolicyCorrector(config=config)
        records = [
            _make_record(record_id=f"r{i}", target_selection_probability=0.5)
            for i in range(5)
        ]
        summary = corrector.correct_batch(records, target_probability=0.5)
        # Self-normalized: mean weight should be ~1.0
        assert abs(summary.mean_weight - 1.0) < 1e-6


class TestComputeESS:
    """Test effective sample size computation."""

    def test_uniform_weights(self):
        corrector = OffPolicyCorrector()
        ess = corrector.compute_ess([1.0, 1.0, 1.0, 1.0])
        assert abs(ess - 4.0) < 1e-6

    def test_single_dominant_weight(self):
        corrector = OffPolicyCorrector()
        # One very large weight reduces ESS
        ess = corrector.compute_ess([100.0, 1.0, 1.0, 1.0])
        assert ess < 4.0
        assert ess > 1.0

    def test_empty_weights(self):
        corrector = OffPolicyCorrector()
        assert corrector.compute_ess([]) == 0.0


class TestPolicyRatioEstimator:
    """Test policy ratio estimation."""

    def test_greedy_action_higher_ratio(self):
        estimator = PolicyRatioEstimator(epsilon=0.1, n_candidates=5)
        ratio = estimator.estimate_ratio(old_probability=0.2, new_rank=0)
        # Greedy: new_prob = 0.9 + 0.1/5 = 0.92
        assert abs(ratio - 0.92 / 0.2) < 1e-6

    def test_non_greedy_action_lower_ratio(self):
        estimator = PolicyRatioEstimator(epsilon=0.1, n_candidates=5)
        ratio = estimator.estimate_ratio(old_probability=0.8, new_rank=2)
        # Non-greedy: new_prob = 0.1/5 = 0.02
        assert abs(ratio - 0.02 / 0.8) < 1e-6

    def test_uniform_probability(self):
        estimator = PolicyRatioEstimator(n_candidates=4)
        assert abs(estimator.uniform_probability() - 0.25) < 1e-6

    def test_zero_old_probability_handled(self):
        estimator = PolicyRatioEstimator(epsilon=0.1)
        ratio = estimator.estimate_ratio(old_probability=0.0, new_rank=0)
        assert np.isfinite(ratio)
