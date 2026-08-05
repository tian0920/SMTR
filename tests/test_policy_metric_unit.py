"""Commit 5: separated candidate-level vs receiver-episode-level metrics.

The policy statistical unit is (target_task_id, receiver_agent_id,
generation_seed); a receiver episode contributes exactly one policy
outcome regardless of how many candidates the router inspected.
"""

import pytest

from smtr.evaluation.metrics import (
    compute_candidate_transfer_metrics,
    compute_method_metrics,
    compute_receiver_policy_metrics,
)


def _paired_record(memory_id: str, *, share_success: bool, withhold_success: bool,
                   label: str = "neutral_success", task_id: str = "t1",
                   receiver: str = "r1", seed: int = 0) -> dict:
    return {
        "task_id": task_id,
        "generation_seed": seed,
        "receiver_agent_id": receiver,
        "candidate_memory_id": memory_id,
        "label": label,
        "valid": True,
        "share": {"team_success": share_success},
        "withhold": {"team_success": withhold_success},
    }


def _decision(memory_id: str, action: str, task_id: str = "t1",
              receiver: str = "r1", seed: int = 0) -> dict:
    return {
        "task_id": task_id,
        "generation_seed": seed,
        "receiver_agent_id": receiver,
        "receiver_role": "executor",
        "writer_role": "executor",
        "candidate_memory_id": memory_id,
        "action": action,
        "eta_hat": 0.0,
    }


class TestPolicyMetricUnit:
    def test_policy_metric_counts_one_receiver_episode_once(self):
        """1 task, 1 receiver, 4 candidates, router shares exactly 1 ->
        policy_total must be 1, not 4."""
        outcomes = [
            _paired_record("m1", share_success=True, withhold_success=False,
                           label="positive_transfer"),
            _paired_record("m2", share_success=False, withhold_success=True,
                           label="negative_transfer"),
            _paired_record("m3", share_success=False, withhold_success=False),
            _paired_record("m4", share_success=False, withhold_success=False),
        ]
        decisions = [
            _decision("m1", "share"),
            _decision("m2", "withhold"),
            _decision("m3", "withhold"),
            _decision("m4", "withhold"),
        ]
        policy = compute_receiver_policy_metrics(
            method="test", decisions=decisions, paired_outcomes=outcomes,
        )
        assert policy["policy_total"] == 1
        assert policy["episodes_with_share"] == 1
        assert policy["episodes_no_memory"] == 0
        # Y_pi = Y_1(m1) = success
        assert policy["paired_policy_success_rate"] == pytest.approx(1.0)

        # Candidate-level view still sees all 4 decisions
        cand = compute_candidate_transfer_metrics(
            method="test", decisions=decisions, paired_outcomes=outcomes,
        )
        assert cand["n_candidates"] == 4

    def test_no_memory_episode_counts_one_withhold_outcome(self):
        outcomes = [
            _paired_record("m1", share_success=True, withhold_success=True),
            _paired_record("m2", share_success=False, withhold_success=True),
        ]
        decisions = [_decision("m1", "withhold"), _decision("m2", "withhold")]
        policy = compute_receiver_policy_metrics(
            method="test", decisions=decisions, paired_outcomes=outcomes,
        )
        assert policy["policy_total"] == 1
        assert policy["episodes_no_memory"] == 1
        assert policy["paired_policy_success_rate"] == pytest.approx(1.0)

    def test_multiple_selections_are_forbidden_in_v1(self):
        outcomes = [
            _paired_record("m1", share_success=True, withhold_success=False),
            _paired_record("m2", share_success=True, withhold_success=False),
        ]
        decisions = [_decision("m1", "share"), _decision("m2", "share")]
        with pytest.raises(ValueError, match="forbids selecting multiple"):
            compute_receiver_policy_metrics(
                method="test", decisions=decisions, paired_outcomes=outcomes,
            )

    def test_inconsistent_withhold_outcomes_are_reported(self):
        outcomes = [
            _paired_record("m1", share_success=True, withhold_success=True),
            _paired_record("m2", share_success=False, withhold_success=False),
        ]
        decisions = [_decision("m1", "withhold"), _decision("m2", "withhold")]
        with pytest.raises(ValueError, match="inconsistent no-memory outcome"):
            compute_receiver_policy_metrics(
                method="test", decisions=decisions, paired_outcomes=outcomes,
            )

    def test_episodes_are_separated_by_task_receiver_seed(self):
        outcomes = [
            _paired_record("m1", share_success=True, withhold_success=False),
            _paired_record("m1", share_success=False, withhold_success=False, seed=1),
            _paired_record("m1", share_success=True, withhold_success=False,
                           receiver="r2"),
        ]
        decisions = [
            _decision("m1", "share", seed=0),
            _decision("m1", "share", seed=1),
            _decision("m1", "share", receiver="r2"),
        ]
        policy = compute_receiver_policy_metrics(
            method="test", decisions=decisions, paired_outcomes=outcomes,
        )
        assert policy["policy_total"] == 3
        assert policy["policy_success_count"] == 2

    def test_combined_method_metrics_keep_both_levels(self):
        outcomes = [
            _paired_record("m1", share_success=True, withhold_success=False,
                           label="positive_transfer"),
            _paired_record("m2", share_success=False, withhold_success=True,
                           label="negative_transfer"),
        ]
        decisions = [_decision("m1", "share"), _decision("m2", "withhold")]
        metrics = compute_method_metrics(
            method="test", decisions=decisions, paired_outcomes=outcomes,
        )
        assert metrics["policy_total"] == 1
        assert metrics["candidate_share_rate"] == pytest.approx(0.5)
        assert metrics["negative_transfer_rejection_rate"] == pytest.approx(1.0)
