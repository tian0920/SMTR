"""Tests for routing semantics refactor (§17).

Covers:
    1. RoutingSemanticsLog dataclass (frozen, fields)
    2. build_routing_semantics_log from plans + decision
    3. compute_episode_metrics aggregation
    4. compute_continual_learning_metrics
    5. compute_three_way_cost breakdown
"""

from __future__ import annotations

from dataclasses import asdict

import pytest

from smtr.rima.transfer_controller import (
    EpisodeTransferDecision,
    RoutingSemanticsLog,
    TransferCandidateDecision,
    TransferRoutingPlan,
    build_routing_semantics_log,
)
from smtr.rima.transfer_metrics import (
    compute_continual_learning_metrics,
    compute_episode_metrics,
    compute_three_way_cost,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_candidate(
    memory_id: str = "m1",
    receiver_id: str = "r1",
    task_id: str = "t1",
    source: str = "known",
    lcb: float | None = 0.5,
    eligible: bool = True,
) -> TransferCandidateDecision:
    return TransferCandidateDecision(
        memory_id=memory_id,
        receiver_id=receiver_id,
        task_id=task_id,
        candidate_source=source,
        mu_tau=0.6 if lcb is not None else None,
        sigma_tau=0.1 if lcb is not None else None,
        lcb=lcb,
        eligible_for_context=eligible,
        selected_for_context=False,
        status="positive" if eligible else "negative",
    )


def _make_plan(
    receiver_id: str = "r1",
    task_id: str = "t1",
    routing_mode: str = "exploit_only",
    known: list[TransferCandidateDecision] | None = None,
    global_: list[TransferCandidateDecision] | None = None,
) -> TransferRoutingPlan:
    return TransferRoutingPlan(
        receiver_id=receiver_id,
        task_id=task_id,
        routing_mode=routing_mode,
        best_known_lcb=0.5 if known else None,
        known_candidates=known or [],
        global_candidates=global_ or [],
        selected_memory_ids=[],
        global_retrieval_triggered=(routing_mode != "exploit_only"),
    )


def _make_episode_decision(
    task_id: str = "t1",
    receiver_id: str | None = "r1",
    memory_id: str | None = "m1",
    lcb: float | None = 0.5,
) -> EpisodeTransferDecision:
    return EpisodeTransferDecision(
        task_id=task_id,
        selected_receiver_id=receiver_id,
        selected_memory_id=memory_id,
        mu_tau=0.6 if lcb is not None else None,
        sigma_tau=0.1 if lcb is not None else None,
        lcb=lcb,
        source="known" if memory_id else "none",
    )


def _make_diag(
    *,
    global_triggered: bool = False,
    n_known: int = 5,
    n_global: int = 0,
) -> dict:
    return {
        "global_retrieval_triggered": global_triggered,
        "n_known_candidates_considered": n_known,
        "n_global_candidates_considered": n_global,
    }


# ---------------------------------------------------------------------------
# RoutingSemanticsLog
# ---------------------------------------------------------------------------


class TestRoutingSemanticsLog:
    def test_frozen(self):
        log = RoutingSemanticsLog(
            task_id="t1",
            episode_selected_receiver="r1",
            episode_selected_memory="m1",
            episode_selected_lcb=0.5,
            candidate_receivers_considered=3,
            receiver_plans_generated=3,
        )
        with pytest.raises(AttributeError):
            log.task_id = "t2"  # type: ignore

    def test_joint_exposure_default_zero(self):
        log = RoutingSemanticsLog(
            task_id="t1",
            episode_selected_receiver=None,
            episode_selected_memory=None,
            episode_selected_lcb=None,
            candidate_receivers_considered=0,
            receiver_plans_generated=0,
        )
        assert log.joint_exposure_count == 0

    def test_no_selection(self):
        log = RoutingSemanticsLog(
            task_id="t1",
            episode_selected_receiver=None,
            episode_selected_memory=None,
            episode_selected_lcb=None,
            candidate_receivers_considered=2,
            receiver_plans_generated=3,
        )
        assert log.episode_selected_receiver is None
        assert log.episode_selected_memory is None
        assert log.episode_selected_lcb is None


# ---------------------------------------------------------------------------
# build_routing_semantics_log
# ---------------------------------------------------------------------------


class TestBuildRoutingSemanticsLog:
    def test_basic_build(self):
        plans = {
            "r1": _make_plan("r1", known=[_make_candidate("m1", "r1")]),
            "r2": _make_plan("r2", known=[_make_candidate("m2", "r2")]),
        }
        decision = _make_episode_decision("t1", "r1", "m1", 0.7)

        log = build_routing_semantics_log(plans, decision)

        assert log.task_id == "t1"
        assert log.episode_selected_receiver == "r1"
        assert log.episode_selected_memory == "m1"
        assert log.episode_selected_lcb == 0.7
        assert log.receiver_plans_generated == 2
        assert log.candidate_receivers_considered == 2
        assert log.joint_exposure_count == 0

    def test_empty_plans_no_selection(self):
        plans: dict[str, TransferRoutingPlan] = {}
        decision = _make_episode_decision("t1", None, None, None)

        log = build_routing_semantics_log(plans, decision)

        assert log.episode_selected_receiver is None
        assert log.episode_selected_memory is None
        assert log.receiver_plans_generated == 0
        assert log.candidate_receivers_considered == 0

    def test_receivers_without_candidates_excluded(self):
        """Receivers with no candidates should not count as 'considered'."""
        plans = {
            "r1": _make_plan("r1", known=[_make_candidate("m1", "r1")]),
            "r2": _make_plan("r2", known=[], global_=[]),  # no candidates
            "r3": _make_plan("r3", known=[_make_candidate("m3", "r3")]),
        }
        decision = _make_episode_decision("t1", "r1", "m1", 0.5)

        log = build_routing_semantics_log(plans, decision)

        assert log.receiver_plans_generated == 3
        assert log.candidate_receivers_considered == 2  # r2 has no candidates


# ---------------------------------------------------------------------------
# compute_episode_metrics
# ---------------------------------------------------------------------------


class TestComputeEpisodeMetrics:
    def test_empty(self):
        result = compute_episode_metrics([])
        assert result["n_episodes"] == 0
        assert result["selection_rate"] == 0.0
        assert result["joint_exposure_violations"] == 0

    def test_all_selected(self):
        logs = [
            asdict(RoutingSemanticsLog(
                task_id=f"t{i}",
                episode_selected_receiver="r1",
                episode_selected_memory=f"m{i}",
                episode_selected_lcb=0.5,
                candidate_receivers_considered=3,
                receiver_plans_generated=3,
            ))
            for i in range(5)
        ]
        result = compute_episode_metrics(logs)
        assert result["n_episodes"] == 5
        assert result["selection_rate"] == 1.0
        assert result["mean_candidate_receivers_considered"] == 3.0
        assert result["mean_receiver_plans_generated"] == 3.0
        assert result["joint_exposure_violations"] == 0

    def test_mixed_selection(self):
        logs = [
            asdict(RoutingSemanticsLog(
                task_id="t0",
                episode_selected_receiver="r1",
                episode_selected_memory="m1",
                episode_selected_lcb=0.6,
                candidate_receivers_considered=2,
                receiver_plans_generated=3,
            )),
            asdict(RoutingSemanticsLog(
                task_id="t1",
                episode_selected_receiver=None,
                episode_selected_memory=None,
                episode_selected_lcb=None,
                candidate_receivers_considered=1,
                receiver_plans_generated=3,
            )),
        ]
        result = compute_episode_metrics(logs)
        assert result["n_episodes"] == 2
        assert result["selection_rate"] == pytest.approx(0.5)
        assert result["mean_candidate_receivers_considered"] == pytest.approx(1.5)

    def test_joint_exposure_violation_detected(self):
        """Non-zero joint_exposure_count must be flagged."""
        logs = [
            asdict(RoutingSemanticsLog(
                task_id="t0",
                episode_selected_receiver="r1",
                episode_selected_memory="m1",
                episode_selected_lcb=0.5,
                candidate_receivers_considered=2,
                receiver_plans_generated=2,
                joint_exposure_count=1,  # VIOLATION
            )),
            asdict(RoutingSemanticsLog(
                task_id="t1",
                episode_selected_receiver="r1",
                episode_selected_memory="m2",
                episode_selected_lcb=0.4,
                candidate_receivers_considered=2,
                receiver_plans_generated=2,
            )),
        ]
        result = compute_episode_metrics(logs)
        assert result["joint_exposure_violations"] == 1


# ---------------------------------------------------------------------------
# compute_continual_learning_metrics
# ---------------------------------------------------------------------------


class TestComputeContinualLearningMetrics:
    def test_defaults_all_zero(self):
        result = compute_continual_learning_metrics()
        assert result["causal_probe_count"] == 0
        assert result["causal_probe_episode_count"] == 0
        assert result["causal_observed_edge_count"] == 0
        assert result["predicted_only_state_size"] == 0
        assert result["causal_observed_state_size"] == 0
        assert result["online_critic_refit_count"] == 0
        assert result["critic_version"] == 0
        assert result["online_causal_evidence_used"] == 0

    def test_with_values(self):
        result = compute_continual_learning_metrics(
            causal_probe_count=10,
            causal_probe_episode_count=8,
            causal_observed_edge_count=6,
            predicted_only_state_size=50,
            causal_observed_state_size=20,
            online_critic_refit_count=3,
            critic_version=4,
            online_causal_evidence_used=15,
        )
        assert result["causal_probe_count"] == 10
        assert result["causal_probe_episode_count"] == 8
        assert result["causal_observed_edge_count"] == 6
        assert result["predicted_only_state_size"] == 50
        assert result["causal_observed_state_size"] == 20
        assert result["online_critic_refit_count"] == 3
        assert result["critic_version"] == 4
        assert result["online_causal_evidence_used"] == 15

    def test_all_keys_present(self):
        result = compute_continual_learning_metrics()
        expected_keys = {
            "causal_probe_count",
            "causal_probe_episode_count",
            "causal_observed_edge_count",
            "predicted_only_state_size",
            "causal_observed_state_size",
            "online_critic_refit_count",
            "critic_version",
            "online_causal_evidence_used",
        }
        assert set(result.keys()) == expected_keys


# ---------------------------------------------------------------------------
# compute_three_way_cost
# ---------------------------------------------------------------------------


class TestComputeThreeWayCost:
    def test_empty_diagnostics(self):
        result = compute_three_way_cost([])
        assert result["retrieval_cost"]["global_retrieval_calls"] == 0
        assert result["retrieval_cost"]["known_retrieval_calls"] == 0
        assert result["model_cost"]["known_critic_predictions"] == 0
        assert result["model_cost"]["global_critic_predictions"] == 0
        assert result["environment_learning_cost"]["post_task_probe_expose_episodes"] == 0
        assert result["environment_learning_cost"]["post_task_probe_control_episodes"] == 0

    def test_three_way_breakdown(self):
        diags = [
            _make_diag(global_triggered=True, n_known=5, n_global=3),
            _make_diag(global_triggered=False, n_known=8, n_global=0),
            _make_diag(global_triggered=True, n_known=4, n_global=2),
        ]
        result = compute_three_way_cost(
            diags,
            post_task_probe_expose_episodes=6,
            post_task_probe_control_episodes=3,
        )

        # Retrieval cost
        assert result["retrieval_cost"]["global_retrieval_calls"] == 2
        assert result["retrieval_cost"]["known_retrieval_calls"] == 3

        # Model cost
        assert result["model_cost"]["known_critic_predictions"] == 17
        assert result["model_cost"]["global_critic_predictions"] == 5

        # Environment learning cost
        assert result["environment_learning_cost"]["post_task_probe_expose_episodes"] == 6
        assert result["environment_learning_cost"]["post_task_probe_control_episodes"] == 3

    def test_cost_separation_invariant(self):
        """Cost must be in three separate buckets, not collapsed."""
        diags = [
            _make_diag(global_triggered=True, n_known=10, n_global=5),
        ]
        result = compute_three_way_cost(diags)

        # Must have three separate top-level keys
        assert "retrieval_cost" in result
        assert "model_cost" in result
        assert "environment_learning_cost" in result

        # Each must have its own sub-keys
        assert "global_retrieval_calls" in result["retrieval_cost"]
        assert "known_retrieval_calls" in result["retrieval_cost"]
        assert "known_critic_predictions" in result["model_cost"]
        assert "global_critic_predictions" in result["model_cost"]
        assert "post_task_probe_expose_episodes" in result["environment_learning_cost"]
        assert "post_task_probe_control_episodes" in result["environment_learning_cost"]

    def test_probe_cost_defaults_to_zero(self):
        """Formal run (no probes) must have zero environment-learning cost."""
        diags = [_make_diag(n_known=3, n_global=0)]
        result = compute_three_way_cost(diags)
        assert result["environment_learning_cost"]["post_task_probe_expose_episodes"] == 0
        assert result["environment_learning_cost"]["post_task_probe_control_episodes"] == 0
