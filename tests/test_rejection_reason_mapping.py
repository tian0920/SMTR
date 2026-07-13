"""Tests for canonical rejection reason mapping (Section 1)."""

import pytest

from smtr.experiment.summary import (
    CANONICAL_REASONS,
    _add_m0_rejection_metrics,
    canonicalize_reason,
    compute_summary,
)
from smtr.experiment.schemas import (
    ComparisonRunRecord,
    ExperimentConfig,
    MethodSummary,
)


class TestCanonicalizeReason:
    """Test canonicalize_reason() mapping."""

    def test_shared_reasons(self):
        assert canonicalize_reason("accepted") == "shared"
        assert canonicalize_reason("critic_guided_share") == "shared"
        assert canonicalize_reason("epsilon_exploration") == "shared"
        assert canonicalize_reason("relevance_topk_selected") == "shared"
        assert canonicalize_reason("baseline_no_memory_router") == "shared"

    def test_tau_lcb_nonpositive(self):
        """tau_lcb_nonpositive must NOT go to 'other'."""
        assert canonicalize_reason("tau_lcb_nonpositive") == "tau_lcb_nonpositive"
        # Legacy reason also maps correctly
        assert canonicalize_reason("tau_lcb_below_threshold") == "tau_lcb_nonpositive"
        assert canonicalize_reason("tau_below_threshold") == "tau_lcb_nonpositive"

    def test_negative_risk_ucb_exceeded(self):
        assert canonicalize_reason("negative_risk_ucb_exceeds_epsilon") == "negative_risk_ucb_exceeded"
        assert canonicalize_reason("negative_risk_veto") == "negative_risk_ucb_exceeded"

    def test_low_support(self):
        assert canonicalize_reason("low_support") == "low_support"

    def test_share_budget_exceeded(self):
        assert canonicalize_reason("budget_exhausted") == "share_budget_exceeded"
        assert canonicalize_reason("relevance_topk_budget_exceeded") == "share_budget_exceeded"

    def test_no_critic_available(self):
        assert canonicalize_reason("no_critic_available") == "no_critic_available"
        assert canonicalize_reason("no_critic_estimate") == "no_critic_available"

    def test_other(self):
        assert canonicalize_reason("high_uncertainty") == "other"
        assert canonicalize_reason("missing_routing_card") == "other"
        assert canonicalize_reason("safety_guard_veto") == "other"
        assert canonicalize_reason("safety_guard_payload_conflict") == "other"

    def test_unknown_reason_goes_to_other(self):
        assert canonicalize_reason("some_future_reason") == "other"
        assert canonicalize_reason("") == "other"

    def test_all_mapped_reasons_in_canonical_set(self):
        """Every mapped reason must produce a canonical category."""
        from smtr.experiment.summary import _REASON_MAP
        for raw, canonical in _REASON_MAP.items():
            assert canonical in CANONICAL_REASONS, f"{raw} -> {canonical} not in canonical set"


class TestRejectionMetricsConsistency:
    """Test that rejection metrics sum correctly."""

    def _make_run(self, decisions: list[dict]) -> ComparisonRunRecord:
        """Create a minimal M0 run record with given decisions."""
        return ComparisonRunRecord(
            experiment_id="test",
            episode_id="ep0",
            task_instance_id="task_0",
            method="M0",
            router_name="ProductionSequentialRouter",
            task_seed=0,
            environment_seed=0,
            generation_seed=0,
            memory_snapshot_id="snap0",
            environment_snapshot_digest="digest0",
            router_trace=[
                {
                    "agent": "planner",
                    "decisions": decisions,
                }
            ],
        )

    def test_tau_lcb_nonpositive_counted_correctly(self):
        """tau_lcb_nonpositive must be counted, not put in other_reason_counts."""
        run = self._make_run([
            {"memory_id": "m1", "action": "withhold", "reason": "tau_lcb_nonpositive"},
            {"memory_id": "m2", "action": "withhold", "reason": "tau_lcb_nonpositive"},
            {"memory_id": "m3", "action": "share", "reason": "accepted"},
        ])
        summary = _add_m0_rejection_metrics(MethodSummary(method="M0"), [run])

        assert summary.tau_lcb_rejection_rate == pytest.approx(2 / 3)
        assert summary.share_decision_rate == pytest.approx(1 / 3)
        assert summary.other_reason_counts == {}

    def test_rates_sum_to_one(self):
        """share_rate + sum(rejection_rates) == 1 within float tolerance."""
        run = self._make_run([
            {"memory_id": "m1", "action": "withhold", "reason": "tau_lcb_nonpositive"},
            {"memory_id": "m2", "action": "withhold", "reason": "negative_risk_ucb_exceeds_epsilon"},
            {"memory_id": "m3", "action": "withhold", "reason": "budget_exhausted"},
            {"memory_id": "m4", "action": "withhold", "reason": "low_support"},
            {"memory_id": "m5", "action": "share", "reason": "accepted"},
        ])
        summary = _add_m0_rejection_metrics(MethodSummary(method="M0"), [run])

        total = (
            (summary.share_decision_rate or 0)
            + (summary.tau_lcb_rejection_rate or 0)
            + (summary.negative_risk_ucb_rejection_rate or 0)
            + (summary.share_budget_rejection_rate or 0)
            + (summary.low_support_rejection_rate or 0)
            + (summary.no_critic_rejection_rate or 0)
        )
        other_count = sum(summary.other_reason_counts.values())
        other_rate = other_count / 5  # 5 total decisions
        total += other_rate

        assert total == pytest.approx(1.0, abs=1e-9)

    def test_decision_count_equals_share_plus_rejection(self):
        """share_count + rejection_count == total_decisions."""
        run = self._make_run([
            {"memory_id": "m1", "action": "withhold", "reason": "tau_lcb_nonpositive"},
            {"memory_id": "m2", "action": "share", "reason": "critic_guided_share"},
            {"memory_id": "m3", "action": "withhold", "reason": "high_uncertainty"},
        ])
        summary = _add_m0_rejection_metrics(MethodSummary(method="M0"), [run])

        total_decisions = 3
        share = int(round((summary.share_decision_rate or 0) * total_decisions))
        tau_lcb = int(round((summary.tau_lcb_rejection_rate or 0) * total_decisions))
        neg_risk = int(round((summary.negative_risk_ucb_rejection_rate or 0) * total_decisions))
        budget = int(round((summary.share_budget_rejection_rate or 0) * total_decisions))
        low_sup = int(round((summary.low_support_rejection_rate or 0) * total_decisions))
        no_critic = int(round((summary.no_critic_rejection_rate or 0) * total_decisions))
        other = sum(summary.other_reason_counts.values())

        assert share + tau_lcb + neg_risk + budget + low_sup + no_critic + other == total_decisions

    def test_legacy_tau_lcb_below_threshold_mapped_correctly(self):
        """Legacy 'tau_lcb_below_threshold' must not appear in other_reason_counts."""
        run = self._make_run([
            {"memory_id": "m1", "action": "withhold", "reason": "tau_lcb_below_threshold"},
        ])
        summary = _add_m0_rejection_metrics(MethodSummary(method="M0"), [run])

        assert summary.tau_lcb_rejection_rate == pytest.approx(1.0)
        assert summary.other_reason_counts == {}

    def test_empty_runs(self):
        summary = _add_m0_rejection_metrics(MethodSummary(method="M0"), [])
        assert summary.share_decision_rate is None
        assert summary.tau_lcb_rejection_rate is None
