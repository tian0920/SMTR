"""Tests for B1-Matched budget manifest and router."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from smtr.experiment.budget_manifest import (
    ShareBudgetManifest,
    build_manifest_from_runs,
    load_manifest,
    save_manifest,
)
from smtr.router.baselines import BudgetMatchedTopKRouter, BudgetManifestConfig
from smtr.router.candidate_proposer import CandidateProposal, CandidateScore, CandidateRequest


def _make_proposal(n: int = 5) -> CandidateProposal:
    request = CandidateRequest(
        task="test", task_stage="test",
        receiver_agent_id="a1", receiver_role="planner",
        receiver_capabilities=[], environment_observation={},
        local_context_summary="", top_k=n, seed=0,
    )
    candidates = [
        CandidateScore(memory_id=f"m{i}", total_score=1.0 - i * 0.1)
        for i in range(n)
    ]
    return CandidateProposal(request=request, ranked_candidates=candidates, pool_revision=0)


class TestShareBudgetManifest:
    """Test budget manifest creation and sampling."""

    def test_manifest_frozen(self):
        """Manifest is immutable."""
        m = ShareBudgetManifest(count_distribution={"0": 0.5, "1": 0.3, "2": 0.2}, seed=7)
        with pytest.raises(Exception):
            m.seed = 99  # type: ignore

    def test_sample_budget_distribution(self):
        """Budget sampling respects the distribution."""
        m = ShareBudgetManifest(
            count_distribution={"0": 0.5, "1": 0.3, "2": 0.2},
            max_shares_per_invocation=2,
            seed=42,
        )
        rng = np.random.default_rng(0)
        samples = [m.sample_budget(rng) for _ in range(1000)]
        # Check approximate distribution (within 5% tolerance)
        p0 = sum(1 for s in samples if s == 0) / 1000
        assert abs(p0 - 0.5) < 0.05

    def test_empty_distribution_returns_max(self):
        """Empty distribution returns max_shares."""
        m = ShareBudgetManifest(count_distribution={}, max_shares_per_invocation=3)
        rng = np.random.default_rng(0)
        assert m.sample_budget(rng) == 3

    def test_save_and_load(self):
        """Manifest round-trips through JSON."""
        m = ShareBudgetManifest(
            count_distribution={"0": 0.6, "1": 0.4},
            max_shares_per_invocation=3,
            seed=7,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            save_manifest(m, path)
            loaded = load_manifest(path)
            assert loaded.count_distribution == m.count_distribution
            assert loaded.seed == m.seed


class TestBuildManifestFromRuns:
    """Test manifest construction from M0 run records."""

    def test_build_from_mock_runs(self):
        """Build manifest from mock M0 runs."""
        runs = [
            {
                "router_trace": [
                    {"decisions": [{"action": "share"}, {"action": "share"}]},
                    {"decisions": [{"action": "withhold"}]},
                ]
            },
            {
                "router_trace": [
                    {"decisions": [{"action": "share"}]},
                ]
            },
        ]
        manifest = build_manifest_from_runs(runs, max_shares_per_invocation=3, seed=7)
        assert manifest.total_invocations == 3
        # 2-share: 1 invocation, 0-share: 1, 1-share: 1
        assert abs(manifest.count_distribution["0"] - 1/3) < 0.01
        assert abs(manifest.count_distribution["1"] - 1/3) < 0.01
        assert abs(manifest.count_distribution["2"] - 1/3) < 0.01


class TestBudgetMatchedTopKRouter:
    """Test B1-Matched router behavior."""

    def test_budget_respected(self):
        """B1-Matched respects sampled budget."""
        config = BudgetManifestConfig(
            count_distribution={"0": 0.0, "1": 0.0, "2": 1.0},
            max_shares=3,
            seed=42,
        )
        router = BudgetMatchedTopKRouter(manifest_config=config, invocation_seed=0)
        proposal = _make_proposal(5)
        result = router.decide_from_proposal(
            receiver_agent_id="a1",
            proposal=proposal,
            traversal_seed=0,
        )
        # With distribution {2: 1.0}, budget should always be 2
        assert len(result.selected_memory_ids) == 2

    def test_no_test_leakage(self):
        """B1-Matched does not read test M0 outcome."""
        config = BudgetManifestConfig(
            count_distribution={"0": 0.5, "1": 0.5},
            max_shares=3,
            seed=42,
        )
        router = BudgetMatchedTopKRouter(manifest_config=config, invocation_seed=0)
        # Router has no reference to test outcomes
        assert not hasattr(router, "test_outcomes")
        assert not hasattr(router, "m0_results")

    def test_deterministic(self):
        """B1-Matched is deterministic given same seed."""
        config = BudgetManifestConfig(
            count_distribution={"0": 0.3, "1": 0.3, "2": 0.4},
            max_shares=3,
            seed=42,
        )
        results = []
        for _ in range(2):
            router = BudgetMatchedTopKRouter(manifest_config=config, invocation_seed=0)
            proposal = _make_proposal(5)
            result = router.decide_from_proposal(
                receiver_agent_id="a1",
                proposal=proposal,
                traversal_seed=0,
            )
            results.append(result.selected_memory_ids)
        assert results[0] == results[1]
