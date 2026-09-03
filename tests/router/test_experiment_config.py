"""Tests for experiment configuration (§18).

Covers:
    1. MethodVariant frozen dataclass
    2. Six method variant definitions and invariants
    3. Method registry lookup
    4. Pilot protocol (Phase A / Phase B)
    5. PilotReportMetrics computation
    6. Early/late score decomposition
"""

from __future__ import annotations

import pytest

from smtr.rima.experiment_config import (
    ALL_METHOD_VARIANTS,
    RIMA_RECEIVER,
    RIMA_STATIC_SAME_PROBE_BUDGET,
    RIMA_TRANSFER_ADAPTIVE,
    RIMA_TRANSFER_FROZEN,
    RIMA_TRANSFER_NO_UNCERTAINTY,
    RIMA_TRANSFER_POSITIVE_STOP,
    MethodVariant,
    PilotProtocol,
    PilotReportMetrics,
    build_pilot_protocol,
    build_smoke_protocol,
    compute_early_late_scores,
    compute_pilot_report_metrics,
    get_method_variant,
)


# ---------------------------------------------------------------------------
# MethodVariant dataclass
# ---------------------------------------------------------------------------


class TestMethodVariant:
    def test_frozen(self):
        with pytest.raises(AttributeError):
            RIMA_RECEIVER.method_id = "changed"  # type: ignore

    def test_beta_fixed_at_164(self):
        """All variants using uncertainty must have beta=1.64 (§16.1)."""
        for v in ALL_METHOD_VARIANTS.values():
            if v.use_uncertainty:
                assert v.beta == 1.64, f"{v.method_id} has beta != 1.64"

    def test_delta_fixed_at_zero(self):
        """delta is fixed at 0.0 for all variants."""
        for v in ALL_METHOD_VARIANTS.values():
            assert v.delta == 0.0, f"{v.method_id} has delta != 0.0"


# ---------------------------------------------------------------------------
# Six method variants
# ---------------------------------------------------------------------------


class TestMethodVariants:
    def test_six_variants(self):
        assert len(ALL_METHOD_VARIANTS) == 6

    def test_static_rima(self):
        v = RIMA_RECEIVER
        assert v.method_id == "rima_receiver"
        assert v.use_transfer_state is False
        assert v.conditional_global_retrieval is False
        assert v.use_causal_probe is False
        assert v.use_critic_update is False
        assert v.is_baseline is True

    def test_frozen_transfer(self):
        v = RIMA_TRANSFER_FROZEN
        assert v.method_id == "rima_transfer_frozen"
        assert v.use_transfer_state is True
        assert v.conditional_global_retrieval is True
        assert v.use_causal_probe is False
        assert v.use_critic_update is False
        assert v.use_uncertainty is True
        assert v.is_baseline is True

    def test_adaptive_continual(self):
        v = RIMA_TRANSFER_ADAPTIVE
        assert v.method_id == "rima_transfer_adaptive"
        assert v.use_transfer_state is True
        assert v.conditional_global_retrieval is True
        assert v.use_causal_probe is True
        assert v.use_critic_update is True
        assert v.use_uncertainty is True
        assert v.is_baseline is False  # main method

    def test_positive_stop(self):
        v = RIMA_TRANSFER_POSITIVE_STOP
        assert v.method_id == "rima_transfer_positive_stop"
        assert v.gamma_mode == "positive_stop"
        assert v.use_causal_probe is True
        assert v.use_critic_update is True
        assert v.is_baseline is True

    def test_no_uncertainty(self):
        v = RIMA_TRANSFER_NO_UNCERTAINTY
        assert v.method_id == "rima_transfer_no_uncertainty"
        assert v.use_uncertainty is False
        assert v.use_causal_probe is True
        assert v.is_baseline is True

    def test_static_same_probe_budget(self):
        v = RIMA_STATIC_SAME_PROBE_BUDGET
        assert v.method_id == "rima_static_same_probe_budget"
        assert v.use_transfer_state is False
        assert v.conditional_global_retrieval is False
        assert v.use_causal_probe is True
        assert v.use_critic_update is False
        assert v.is_baseline is True

    def test_only_adaptive_is_main(self):
        """Exactly one variant should be non-baseline (the main method)."""
        main_methods = [v for v in ALL_METHOD_VARIANTS.values() if not v.is_baseline]
        assert len(main_methods) == 1
        assert main_methods[0].method_id == "rima_transfer_adaptive"

    def test_display_labels_unique(self):
        labels = [v.display_label for v in ALL_METHOD_VARIANTS.values()]
        assert len(labels) == len(set(labels))

    def test_method_ids_unique(self):
        ids = [v.method_id for v in ALL_METHOD_VARIANTS.values()]
        assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# Method registry lookup
# ---------------------------------------------------------------------------


class TestGetMethodVariant:
    def test_valid_lookup(self):
        v = get_method_variant("rima_transfer_adaptive")
        assert v.method_id == "rima_transfer_adaptive"
        assert v.use_critic_update is True

    def test_invalid_lookup_raises(self):
        with pytest.raises(ValueError, match="Unknown method_id"):
            get_method_variant("nonexistent_method")

    def test_all_ids_lookupable(self):
        for mid in ALL_METHOD_VARIANTS:
            v = get_method_variant(mid)
            assert v.method_id == mid


# ---------------------------------------------------------------------------
# Pilot protocol
# ---------------------------------------------------------------------------


class TestPilotProtocol:
    def test_smoke_protocol(self):
        p = build_smoke_protocol()
        assert p.phase == "A"
        assert p.n_scenarios == 1
        assert p.n_tasks_per_stream == 15
        assert p.stream_seeds == (0,)
        assert p.execution_seeds == (0,)
        assert p.probe_generation_seeds == (0,)

    def test_pilot_protocol_defaults(self):
        p = build_pilot_protocol()
        assert p.phase == "B"
        assert p.n_scenarios == 2
        assert p.n_tasks_per_stream == 30
        assert p.stream_seeds == (0, 1, 2)
        assert p.execution_seeds == (0, 1, 2)

    def test_pilot_protocol_custom(self):
        p = build_pilot_protocol(
            n_scenarios=3,
            n_tasks_per_stream=50,
            n_stream_seeds=5,
            n_execution_seeds=4,
        )
        assert p.n_scenarios == 3
        assert p.n_tasks_per_stream == 50
        assert len(p.stream_seeds) == 5
        assert len(p.execution_seeds) == 4

    def test_protocol_frozen(self):
        p = build_smoke_protocol()
        with pytest.raises(AttributeError):
            p.phase = "B"  # type: ignore

    def test_task_order_robustness(self):
        """Phase B must have >= 3 stream seeds (§18 task-order robustness)."""
        p = build_pilot_protocol()
        assert len(p.stream_seeds) >= 3


# ---------------------------------------------------------------------------
# PilotReportMetrics
# ---------------------------------------------------------------------------


class TestPilotReportMetrics:
    def test_defaults_zero(self):
        m = PilotReportMetrics()
        assert m.mean_team_task_score == 0.0
        assert m.global_retrieval_calls_per_task == 0.0
        assert m.known_transfer_selection_rate == 0.0
        assert m.extra_causal_probe_episodes_per_task == 0.0
        assert m.critic_calls_per_task == 0.0
        assert m.continual_gain_delta_score_late == 0.0

    def test_frozen(self):
        with pytest.raises(AttributeError):
            PilotReportMetrics().mean_team_task_score = 1.0  # type: ignore


class TestComputePilotReportMetrics:
    def test_empty(self):
        m = compute_pilot_report_metrics(task_scores=[], n_tasks=0)
        assert m.mean_team_task_score == 0.0

    def test_basic_computation(self):
        scores = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        m = compute_pilot_report_metrics(
            task_scores=scores,
            n_tasks=6,
            n_global_retrieval_calls=3,
            n_known_selections=4,
            n_causal_probe_episodes=2,
            n_critic_calls=30,
        )
        assert m.mean_team_task_score == pytest.approx(0.75)
        assert m.global_retrieval_calls_per_task == pytest.approx(0.5)
        assert m.known_transfer_selection_rate == pytest.approx(4 / 6)
        assert m.extra_causal_probe_episodes_per_task == pytest.approx(2 / 6)
        assert m.critic_calls_per_task == pytest.approx(5.0)

    def test_continual_gain(self):
        scores = [0.3, 0.4, 0.5, 0.6, 0.7, 0.9]
        # late_start = 2*6/3 = 4, late = scores[4:] = [0.7, 0.9]
        # adaptive_late = 0.8, baseline_late = 0.5
        # delta = 0.8 - 0.5 = 0.3
        m = compute_pilot_report_metrics(
            task_scores=scores,
            n_tasks=6,
            baseline_late_score=0.5,
        )
        assert m.continual_gain_delta_score_late == pytest.approx(0.3)

    def test_no_baseline_no_gain(self):
        m = compute_pilot_report_metrics(
            task_scores=[0.5, 0.6],
            n_tasks=2,
        )
        assert m.continual_gain_delta_score_late == 0.0


# ---------------------------------------------------------------------------
# Early/late score decomposition
# ---------------------------------------------------------------------------


class TestComputeEarlyLateScores:
    def test_empty(self):
        result = compute_early_late_scores([])
        assert result["score_early"] == 0.0
        assert result["score_late"] == 0.0
        assert result["delta_score"] == 0.0

    def test_basic_decomposition(self):
        # T=9: early = t<=3 → [0,1,2], late = t>6 → [7,8]
        scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        result = compute_early_late_scores(scores)
        # early_end = 9/3 = 3 → scores[:3] = [0.1, 0.2, 0.3]
        assert result["score_early"] == pytest.approx(0.2)
        # late_start = 2*9/3 = 6 → scores[6:] = [0.7, 0.8, 0.9]
        assert result["score_late"] == pytest.approx(0.8)
        assert result["delta_score"] == pytest.approx(0.6)

    def test_single_element(self):
        result = compute_early_late_scores([0.5])
        # early_end = max(1, 0) = 1, late_start = max(1, 0) = 1
        assert result["score_early"] == pytest.approx(0.5)
        assert result["score_late"] == 0.0  # no late elements

    def test_delta_positive_for_improving_stream(self):
        """Score_late > Score_early for an improving stream."""
        scores = [0.2, 0.3, 0.4, 0.6, 0.7, 0.8]
        result = compute_early_late_scores(scores)
        assert result["delta_score"] > 0


# ---------------------------------------------------------------------------
# Invariants across all variants
# ---------------------------------------------------------------------------


class TestCrossVariantInvariants:
    def test_gamma_mode_values(self):
        """Only 'train_q75' and 'positive_stop' are valid gamma modes."""
        valid_modes = {"train_q75", "positive_stop"}
        for v in ALL_METHOD_VARIANTS.values():
            assert v.gamma_mode in valid_modes

    def test_positive_stop_is_only_non_standard_gamma(self):
        """Only positive_stop variant uses non-standard gamma_mode."""
        non_standard = [
            v for v in ALL_METHOD_VARIANTS.values()
            if v.gamma_mode != "train_q75"
        ]
        assert len(non_standard) == 1
        assert non_standard[0].method_id == "rima_transfer_positive_stop"

    def test_cost_matched_baseline_has_probe_but_no_state(self):
        """Static same-probe-budget: has probe, no state, no update."""
        v = RIMA_STATIC_SAME_PROBE_BUDGET
        assert v.use_causal_probe is True
        assert v.use_transfer_state is False
        assert v.use_critic_update is False
