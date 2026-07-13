"""Tests for prefix formation trace."""

import pytest

from smtr.experiment.prefix_trace import (
    PrefixTraceSummary,
    compute_prefix_trace,
)
from smtr.experiment.candidate_diagnostics import (
    SCENARIO_PREFIX_MEMORIES,
    SCENARIO_TARGET_MEMORY,
)


class TestPrefixTrace:
    """Test prefix formation trace computation."""

    def _make_run(self, scenario: str, prefix_selected: bool, target_shared: bool):
        """Create a mock run for prefix scenario."""
        target = SCENARIO_TARGET_MEMORY.get(scenario, "target")
        prefix_mems = SCENARIO_PREFIX_MEMORIES.get(scenario, [])

        # Build decisions: prefix first, then target
        decisions = []
        for pm in prefix_mems:
            decisions.append({
                "memory_id": pm,
                "action": "share" if prefix_selected else "withhold",
                "reason": "test",
                "candidate_position": 0,
                "proposal_rank": 1,
                "tau_mean": 0.5,
                "tau_lcb": 0.3,
                "negative_risk_ucb": 0.1,
            })
        decisions.append({
            "memory_id": target,
            "action": "share" if target_shared else "withhold",
            "reason": "test",
            "candidate_position": 1,
            "proposal_rank": 2,
            "tau_mean": 0.6,
            "tau_lcb": 0.4,
            "negative_risk_ucb": 0.1,
        })

        all_mems = prefix_mems + [target]
        return {
            "method": "M0-Full",
            "episode_id": "ep0",
            "candidate_memory_ids": all_mems,
            "team_success": target_shared and prefix_selected,
            "router_trace": [
                {"agent": "planner", "decisions": decisions},
            ],
        }

    def test_prefix_sensitive_with_correct_prefix(self):
        """Prefix trace detects correct prefix selection."""
        run = self._make_run("prefix_sensitive", prefix_selected=True, target_shared=True)
        result = compute_prefix_trace([run], scenario="prefix_sensitive", method="M0-Full")
        assert result.n_episodes == 1
        assert result.prefix_selection_success_rate == 1.0
        assert result.success_given_correct_prefix == 1.0

    def test_prefix_sensitive_with_rejected_prefix(self):
        """Prefix trace detects rejected prefix."""
        run = self._make_run("prefix_sensitive", prefix_selected=False, target_shared=False)
        result = compute_prefix_trace([run], scenario="prefix_sensitive", method="M0-Full")
        assert result.prefix_selection_success_rate == 0.0
        assert result.success_without_correct_prefix == 0.0

    def test_non_prefix_scenario_returns_empty(self):
        """Non-prefix scenario returns empty summary."""
        result = compute_prefix_trace([], scenario="positive", method="M0-Full")
        assert result.n_episodes == 0
        assert result.traces == []

    def test_prefix_candidate_recall(self):
        """Prefix candidate recall is 1.0 when all prefix memories are in candidates."""
        run = self._make_run("flip_pos_to_neg", prefix_selected=True, target_shared=False)
        result = compute_prefix_trace([run], scenario="flip_pos_to_neg", method="M0-Full")
        assert result.prefix_candidate_recall == 1.0

    def test_empty_runs(self):
        """Empty runs return default summary."""
        result = compute_prefix_trace([], scenario="prefix_sensitive", method="M0-Full")
        assert result.n_episodes == 0
