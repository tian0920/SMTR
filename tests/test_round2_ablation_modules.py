"""Tests for Round 2 ablation analysis modules.

Covers: bottleneck_funnel, paired_comparisons, prefix_matched_pair, rejection_analysis.
"""

from __future__ import annotations

import numpy as np
import pytest

from smtr.experiment.bottleneck_funnel import (
    BottleneckFunnelResult,
    FunnelStage,
    compute_bottleneck_funnel,
    compute_all_bottleneck_funnels,
)
from smtr.experiment.paired_comparisons import (
    PairedComparisonResult,
    compute_paired_comparisons,
    _group_runs_by_episode,
    _episode_metric,
)
from smtr.experiment.prefix_matched_pair import (
    PrefixMatchedPairResult,
    compute_prefix_matched_pair_audit,
    _extract_target_prediction,
)
from smtr.experiment.rejection_analysis import (
    MatchedCase,
    ReasonProportions,
    RejectionAnalysisResult,
    compute_rejection_analysis,
    _compute_reason_proportions,
    _extract_decision_for_memory,
    _get_selected_before,
)


# ── Fixtures ──────────────────────────────────────────────────────────


def _make_run(
    method: str = "M0-Full",
    episode_id: str = "ep1",
    task_instance_id: str = "task1",
    generation_seed: int = 0,
    traversal_seed: int = 0,
    candidate_memory_ids: list[str] | None = None,
    router_trace: list[dict] | None = None,
    team_success: bool = True,
    selected_count: int = 1,
    selected_memory_ids: list[str] | None = None,
    policy_level_transfer_label: str = "positive_transfer",
) -> dict:
    return {
        "method": method,
        "episode_id": episode_id,
        "task_instance_id": task_instance_id,
        "generation_seed": generation_seed,
        "traversal_seed": traversal_seed,
        "candidate_memory_ids": candidate_memory_ids or [],
        "router_trace": router_trace or [],
        "team_success": team_success,
        "selected_count": selected_count,
        "selected_memory_ids": selected_memory_ids or [],
        "policy_level_transfer_label": policy_level_transfer_label,
    }


def _make_trace(
    decisions: list[dict],
) -> list[dict]:
    return [{"decisions": decisions}]


def _make_decision(
    memory_id: str,
    action: str = "share",
    reason: str = "accepted",
    tau_mean: float = 0.5,
    tau_lcb: float = 0.1,
    tau_ucb: float = 0.9,
    negative_risk_ucb: float = 0.1,
) -> dict:
    return {
        "memory_id": memory_id,
        "action": action,
        "reason": reason,
        "tau_mean": tau_mean,
        "tau_lcb": tau_lcb,
        "tau_ucb": tau_ucb,
        "negative_risk_ucb": negative_risk_ucb,
    }


# ── Bottleneck Funnel ─────────────────────────────────────────────────


class TestBottleneckFunnel:
    def test_empty_runs(self):
        result = compute_bottleneck_funnel([], scenario="positive", method="M0-Full")
        assert isinstance(result, BottleneckFunnelResult)
        assert result.scenario == "positive"
        assert result.ground_truth_opportunity == 0

    def test_positive_scenario_all_pass(self):
        """All stages pass when target is recalled, traversed, selected, shared, and task succeeds."""
        trace = _make_trace([
            _make_decision("mem_cf_positive", action="share"),
        ])
        runs = [_make_run(
            method="M0-Full",
            candidate_memory_ids=["mem_cf_positive"],
            router_trace=trace,
            team_success=True,
        )]
        result = compute_bottleneck_funnel(runs, scenario="positive", method="M0-Full")
        assert result.ground_truth_opportunity == 1
        stage_names = [s.name for s in result.stages]
        assert "target_prefix_recalled" in stage_names
        assert "task_succeeds" in stage_names
        # Target recalled
        assert result.stages[1].count == 1
        # Task succeeds
        assert result.stages[5].count == 1

    def test_positive_scenario_target_not_recalled(self):
        """Target not in candidates → stage 1 fails."""
        trace = _make_trace([
            _make_decision("mem_other", action="share"),
        ])
        runs = [_make_run(
            method="M0-Full",
            candidate_memory_ids=["mem_other"],
            router_trace=trace,
            team_success=True,
        )]
        result = compute_bottleneck_funnel(runs, scenario="positive", method="M0-Full")
        assert result.stages[1].count == 0  # target_prefix_recalled = 0
        assert result.stages[5].count == 0  # task_succeeds = 0

    def test_negative_scenario_withhold(self):
        """Negative target correctly withheld → stage 4 passes."""
        trace = _make_trace([
            _make_decision("mem_cf_negative", action="withhold", reason="tau_lcb_nonpositive"),
        ])
        runs = [_make_run(
            method="M0-Full",
            candidate_memory_ids=["mem_cf_negative"],
            router_trace=trace,
            team_success=True,
        )]
        result = compute_bottleneck_funnel(runs, scenario="negative", method="M0-Full")
        assert result.stages[4].count == 1  # target_correctly_routed

    def test_prefix_sensitive_scenario(self):
        """Prefix-sensitive scenario: prefix must be recalled and traversed before target."""
        trace = _make_trace([
            _make_decision("mem_prefix_lock", action="share"),
            _make_decision("mem_cf_prefix_recover", action="share"),
        ])
        runs = [_make_run(
            method="M0-Full",
            candidate_memory_ids=["mem_prefix_lock", "mem_cf_prefix_recover"],
            router_trace=trace,
            team_success=True,
        )]
        result = compute_bottleneck_funnel(runs, scenario="prefix_sensitive", method="M0-Full")
        assert result.stages[1].count == 1  # both target and prefix recalled
        assert result.stages[2].count == 1  # prefix before target in traversal

    def test_all_methods(self):
        """compute_all_bottleneck_funnels returns results for each method."""
        runs = [
            _make_run(method="M0-Full", candidate_memory_ids=["mem_cf_positive"],
                      router_trace=_make_trace([_make_decision("mem_cf_positive", action="share")]),
                      team_success=True),
            _make_run(method="B0", candidate_memory_ids=["mem_cf_positive"],
                      router_trace=_make_trace([_make_decision("mem_cf_positive", action="share")]),
                      team_success=False),
        ]
        results = compute_all_bottleneck_funnels(runs, scenario="positive")
        assert "M0-Full" in results
        assert "B0" in results

    def test_funnel_rates(self):
        """Funnel rates should be count/total."""
        runs = [
            _make_run(method="M0-Full", episode_id=f"ep{i}",
                      candidate_memory_ids=["mem_cf_positive"] if i < 5 else [],
                      router_trace=_make_trace([_make_decision("mem_cf_positive", action="share")]),
                      team_success=True)
            for i in range(10)
        ]
        result = compute_bottleneck_funnel(runs, scenario="positive", method="M0-Full")
        assert result.stages[1].count == 5
        assert result.stages[1].total == 10
        assert abs(result.stages[1].rate - 0.5) < 1e-9


# ── Paired Comparisons ────────────────────────────────────────────────


class TestPairedComparisons:
    def test_empty_runs(self):
        result = compute_paired_comparisons([])
        assert isinstance(result, dict)
        assert "M0-Full vs B1-Top1" in result

    def test_grouping(self):
        runs = [
            _make_run(method="M0-Full", task_instance_id="t1", generation_seed=0, traversal_seed=0),
            _make_run(method="M0-Full", task_instance_id="t1", generation_seed=0, traversal_seed=1),
            _make_run(method="M0-Full", task_instance_id="t1", generation_seed=1, traversal_seed=0),
        ]
        groups = _group_runs_by_episode(runs, "M0-Full")
        assert len(groups) == 2  # (t1,0) and (t1,1)
        assert len(groups[("t1", 0)]) == 2

    def test_episode_metric_success(self):
        runs = [
            _make_run(method="M0-Full", team_success=True),
            _make_run(method="M0-Full", team_success=False),
        ]
        metric = _episode_metric(runs, "success")
        assert abs(metric - 0.5) < 1e-9

    def test_episode_metric_selected(self):
        runs = [
            _make_run(method="M0-Full", selected_count=2),
            _make_run(method="M0-Full", selected_count=4),
        ]
        metric = _episode_metric(runs, "selected_count")
        assert abs(metric - 3.0) < 1e-9

    def test_paired_comparison_basic(self):
        """M0-Full always succeeds, B1-Top1 always fails → positive diff."""
        runs = []
        for i in range(10):
            runs.append(_make_run(
                method="M0-Full", task_instance_id=f"t{i}", generation_seed=0,
                team_success=True, selected_count=1,
                policy_level_transfer_label="positive_transfer",
            ))
            runs.append(_make_run(
                method="B1-Top1", task_instance_id=f"t{i}", generation_seed=0,
                team_success=False, selected_count=1,
                policy_level_transfer_label="no_transfer",
            ))
        result = compute_paired_comparisons(runs, pairs=[("M0-Full", "B1-Top1")])
        pc = result["M0-Full vs B1-Top1"]
        assert pc.success_diff_mean > 0.5
        assert pc.success_diff_ci_low > 0.0
        assert pc.n_base_episodes == 10

    def test_paired_comparison_identical(self):
        """Identical methods → diff ≈ 0."""
        runs = []
        for i in range(20):
            for method in ["M0-Full", "A1-NoSet"]:
                runs.append(_make_run(
                    method=method, task_instance_id=f"t{i}", generation_seed=0,
                    team_success=True, selected_count=2,
                ))
        result = compute_paired_comparisons(runs, pairs=[("M0-Full", "A1-NoSet")])
        pc = result["M0-Full vs A1-NoSet"]
        assert abs(pc.success_diff_mean) < 1e-9
        assert abs(pc.avg_selected_diff_mean) < 1e-9


# ── Prefix Matched-Pair ──────────────────────────────────────────────


class TestPrefixMatchedPair:
    def test_empty_runs(self):
        result = compute_prefix_matched_pair_audit([], scenario="prefix_sensitive")
        assert result.n_paired_episodes == 0

    def test_extract_target_prediction_found(self):
        run = _make_run(router_trace=_make_trace([
            _make_decision("mem_target", tau_mean=0.7, action="share"),
        ]))
        pred = _extract_target_prediction(run, "mem_target")
        assert abs(pred["tau_mean"] - 0.7) < 1e-9
        assert pred["action"] == "share"

    def test_extract_target_prediction_missing(self):
        run = _make_run(router_trace=_make_trace([
            _make_decision("mem_other", tau_mean=0.3),
        ]))
        pred = _extract_target_prediction(run, "mem_target")
        assert pred["tau_mean"] == 0.0
        assert pred["action"] == "withhold"

    def test_prefix_audit_basic(self):
        """M0 has higher tau for positive target → positive delta."""
        runs = []
        for i in range(5):
            # M0-Full: share target with high tau
            runs.append(_make_run(
                method="M0-Full", episode_id=f"ep{i}", generation_seed=0,
                router_trace=_make_trace([
                    _make_decision("mem_cf_prefix_recover", tau_mean=0.8, action="share"),
                ]),
            ))
            # A1-NoSet: withhold target with lower tau
            runs.append(_make_run(
                method="A1-NoSet", episode_id=f"ep{i}", generation_seed=0,
                router_trace=_make_trace([
                    _make_decision("mem_cf_prefix_recover", tau_mean=0.2, action="withhold"),
                ]),
            ))
        result = compute_prefix_matched_pair_audit(
            runs, scenario="prefix_sensitive",
        )
        assert result.n_paired_episodes == 5
        # Correlation is None because all ground truth taus are identical (constant)
        assert result.delta_tau_correlation is None
        # But MAE and direction accuracy should still work
        assert result.delta_tau_mae is not None
        assert abs(result.delta_tau_mae - 0.4) < 0.1  # |0.6 - 1.0| = 0.4
        assert result.effect_direction_accuracy is not None
        assert result.effect_direction_accuracy == 1.0  # all correct direction

    def test_flip_scenario_accuracy(self):
        """flip_pos_to_neg: M0 should withhold target."""
        runs = []
        for i in range(3):
            runs.append(_make_run(
                method="M0-Full", episode_id=f"ep{i}", generation_seed=0,
                router_trace=_make_trace([
                    _make_decision("mem_cf_positive", tau_mean=0.1, action="withhold"),
                ]),
            ))
            runs.append(_make_run(
                method="A1-NoSet", episode_id=f"ep{i}", generation_seed=0,
                router_trace=_make_trace([
                    _make_decision("mem_cf_positive", tau_mean=0.5, action="share"),
                ]),
            ))
        result = compute_prefix_matched_pair_audit(runs, scenario="flip_pos_to_neg")
        assert result.positive_to_negative_accuracy is not None
        assert result.positive_to_negative_accuracy == 1.0  # all withheld


# ── Rejection Analysis ───────────────────────────────────────────────


class TestRejectionAnalysis:
    def test_empty_runs(self):
        result = compute_rejection_analysis([], scenario="positive")
        assert isinstance(result, RejectionAnalysisResult)
        assert result.m0_reasons.total_decisions == 0

    def test_reason_proportions_sum_to_one(self):
        runs = [_make_run(
            method="M0-Full",
            router_trace=_make_trace([
                _make_decision("mem1", action="share", reason="accepted"),
                _make_decision("mem2", action="withhold", reason="tau_lcb_nonpositive"),
                _make_decision("mem3", action="withhold", reason="negative_risk_ucb_exceeds_epsilon"),
            ]),
        )]
        props = _compute_reason_proportions(runs, method="M0-Full")
        assert props.total_decisions == 3
        assert abs(props.sum_check - 1.0) < 1e-9

    def test_extract_decision_for_memory(self):
        run = _make_run(router_trace=_make_trace([
            _make_decision("mem_a", action="share"),
            _make_decision("mem_b", action="withhold"),
        ]))
        dec = _extract_decision_for_memory(run, "mem_b")
        assert dec is not None
        assert dec["action"] == "withhold"

    def test_extract_decision_missing(self):
        run = _make_run(router_trace=_make_trace([
            _make_decision("mem_a", action="share"),
        ]))
        dec = _extract_decision_for_memory(run, "mem_z")
        assert dec is None

    def test_get_selected_before(self):
        run = _make_run(router_trace=_make_trace([
            _make_decision("mem_a", action="share"),
            _make_decision("mem_b", action="withhold"),
            _make_decision("mem_target", action="share"),
        ]))
        selected = _get_selected_before(run, "mem_target")
        assert selected == ["mem_a"]

    def test_matched_discordant_cases(self):
        """A1 shares but M0 withholds on the target memory."""
        runs = []
        for i in range(3):
            # A1-NoSet: share target
            runs.append(_make_run(
                method="A1-NoSet", episode_id=f"ep{i}", generation_seed=0,
                candidate_memory_ids=["mem_cf_positive"],
                router_trace=_make_trace([
                    _make_decision("mem_cf_positive", action="share", reason="accepted"),
                ]),
                team_success=True,
            ))
            # M0-Full: withhold target
            runs.append(_make_run(
                method="M0-Full", episode_id=f"ep{i}", generation_seed=0,
                candidate_memory_ids=["mem_cf_positive"],
                router_trace=_make_trace([
                    _make_decision("mem_cf_positive", action="withhold", reason="tau_lcb_nonpositive"),
                ]),
                team_success=False,
            ))
        result = compute_rejection_analysis(runs, scenario="positive")
        assert result.n_a1_share_m0_withhold == 3
        assert result.n_a1_withhold_m0_share == 0
        assert len(result.matched_cases) == 3
        assert all(c.case_type == "A1_share_M0_withhold" for c in result.matched_cases)

    def test_reverse_discordant(self):
        """M0 shares but A1 withholds."""
        runs = []
        for i in range(2):
            runs.append(_make_run(
                method="A1-NoSet", episode_id=f"ep{i}", generation_seed=0,
                router_trace=_make_trace([
                    _make_decision("mem_cf_positive", action="withhold", reason="tau_lcb_nonpositive"),
                ]),
            ))
            runs.append(_make_run(
                method="M0-Full", episode_id=f"ep{i}", generation_seed=0,
                router_trace=_make_trace([
                    _make_decision("mem_cf_positive", action="share", reason="accepted"),
                ]),
            ))
        result = compute_rejection_analysis(runs, scenario="positive")
        assert result.n_a1_withhold_m0_share == 2
        assert result.n_a1_share_m0_withhold == 0

    def test_no_discordant_when_agree(self):
        """Both methods agree → no discordant cases."""
        runs = []
        for i in range(2):
            for method in ["M0-Full", "A1-NoSet"]:
                runs.append(_make_run(
                    method=method, episode_id=f"ep{i}", generation_seed=0,
                    router_trace=_make_trace([
                        _make_decision("mem_cf_positive", action="share", reason="accepted"),
                    ]),
                ))
        result = compute_rejection_analysis(runs, scenario="positive")
        assert result.n_a1_share_m0_withhold == 0
        assert result.n_a1_withhold_m0_share == 0
        assert len(result.matched_cases) == 0
