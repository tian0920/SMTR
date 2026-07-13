"""Tests for candidate-level diagnostics."""

import pytest

from smtr.experiment.candidate_diagnostics import (
    CandidateDiagnosticsSummary,
    SCENARIO_TARGET_EFFECT,
    SCENARIO_TARGET_MEMORY,
    compute_candidate_diagnostics,
)


class TestScenarioGroundTruth:
    """Test scenario ground-truth mappings."""

    def test_all_scenarios_have_target(self):
        """All 9 scenarios have a target memory mapping."""
        expected = {
            "positive", "negative", "neutral_success", "neutral_failure",
            "prefix_sensitive", "flip_pos_to_neg", "flip_neg_to_pos",
            "flip_neu_to_neg", "flip_neu_to_pos",
        }
        assert set(SCENARIO_TARGET_MEMORY.keys()) == expected
        assert set(SCENARIO_TARGET_EFFECT.keys()) == expected

    def test_positive_scenario(self):
        """Positive scenario target is positive transfer."""
        assert SCENARIO_TARGET_EFFECT["positive"] == "positive"
        assert SCENARIO_TARGET_MEMORY["positive"] == "mem_cf_positive"

    def test_negative_scenario(self):
        """Negative scenario target is negative transfer."""
        assert SCENARIO_TARGET_EFFECT["negative"] == "negative"
        assert SCENARIO_TARGET_MEMORY["negative"] == "mem_cf_negative"


class TestCandidateDiagnostics:
    """Test candidate diagnostics computation."""

    def _make_runs(self, scenario: str, target_in_candidates: bool, target_shared: bool):
        """Create mock runs for testing."""
        target = SCENARIO_TARGET_MEMORY[scenario]
        candidates = [target, "other_1", "other_2"] if target_in_candidates else ["other_1", "other_2"]
        action = "share" if target_shared else "withhold"
        return [
            {
                "method": "M0-Full",
                "episode_id": "ep0",
                "candidate_memory_ids": candidates,
                "team_success": target_shared,
                "router_trace": [
                    {
                        "agent": "planner",
                        "decisions": [
                            {"memory_id": target, "action": action, "reason": "test"},
                            {"memory_id": "other_1", "action": "withhold", "reason": "test"},
                        ],
                    }
                ],
            }
        ]

    def test_positive_recall_when_target_in_candidates(self):
        """Positive target recall is 1.0 when target is always in candidates."""
        runs = self._make_runs("positive", target_in_candidates=True, target_shared=True)
        result = compute_candidate_diagnostics(runs, scenario="positive", method="M0-Full")
        assert result.positive_target_recall_at_k == 1.0

    def test_router_positive_recall(self):
        """Router positive recall = 1.0 when target is shared."""
        runs = self._make_runs("positive", target_in_candidates=True, target_shared=True)
        result = compute_candidate_diagnostics(runs, scenario="positive", method="M0-Full")
        assert result.router_positive_recall == 1.0

    def test_harmful_rejection(self):
        """Harmful memory rejection = 1.0 when negative target is withheld."""
        runs = self._make_runs("negative", target_in_candidates=True, target_shared=False)
        result = compute_candidate_diagnostics(runs, scenario="negative", method="M0-Full")
        assert result.harmful_memory_rejection == 1.0

    def test_empty_runs(self):
        """Empty runs return default summary."""
        result = compute_candidate_diagnostics([], scenario="positive", method="M0-Full")
        assert result.n_episodes == 0

    def test_unknown_scenario(self):
        """Unknown scenario returns default summary."""
        result = compute_candidate_diagnostics([], scenario="unknown", method="M0-Full")
        assert result.scenario == "unknown"
