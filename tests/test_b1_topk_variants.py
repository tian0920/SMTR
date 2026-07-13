"""Tests for B1-Top1 and B1-Top3 router variants."""

import pytest

from smtr.router.baselines import RelevanceTopKRouter, RelevanceTopKRouterConfig
from smtr.router.candidate_proposer import CandidateProposal, CandidateScore, CandidateRequest


def _make_proposal(n_candidates: int = 5) -> CandidateProposal:
    """Create a test proposal with n_candidates."""
    request = CandidateRequest(
        task="test task",
        task_stage="test",
        receiver_agent_id="agent_1",
        receiver_role="planner",
        receiver_capabilities=[],
        environment_observation={},
        local_context_summary="",
        top_k=n_candidates,
        seed=0,
    )
    candidates = [
        CandidateScore(
            memory_id=f"mem_{i}",
            total_score=1.0 - i * 0.1,
            goal_similarity=0.9 - i * 0.1,
        )
        for i in range(n_candidates)
    ]
    return CandidateProposal(
        request=request,
        ranked_candidates=candidates,
        pool_revision=0,
    )


class TestB1Top1:
    """B1-Top1: max_shares_per_invocation=1."""

    def test_selects_at_most_one(self):
        """B1-Top1 selects at most 1 memory per invocation."""
        router = RelevanceTopKRouter(
            config=RelevanceTopKRouterConfig(max_shares_per_invocation=1)
        )
        proposal = _make_proposal(5)
        result = router.decide_from_proposal(
            receiver_agent_id="agent_1",
            proposal=proposal,
        )
        assert len(result.selected_memory_ids) == 1
        assert result.selected_memory_ids[0] == "mem_0"

    def test_proposal_rank_and_score_recorded(self):
        """B1-Top1 records proposal_rank and proposal_score."""
        router = RelevanceTopKRouter(
            config=RelevanceTopKRouterConfig(max_shares_per_invocation=1)
        )
        proposal = _make_proposal(3)
        result = router.decide_from_proposal(
            receiver_agent_id="agent_1",
            proposal=proposal,
        )
        for dec in result.decisions:
            assert dec.proposal_rank is not None
            assert dec.proposal_score is not None
            assert dec.proposal_rank == dec.candidate_position + 1

    def test_no_critic_called(self):
        """B1-Top1 does not use critic fields."""
        router = RelevanceTopKRouter(
            config=RelevanceTopKRouterConfig(max_shares_per_invocation=1)
        )
        proposal = _make_proposal(3)
        result = router.decide_from_proposal(
            receiver_agent_id="agent_1",
            proposal=proposal,
        )
        for dec in result.decisions:
            assert dec.tau_mean is None
            assert dec.tau_lcb is None


class TestB1Top3:
    """B1-Top3: max_shares_per_invocation=3."""

    def test_selects_at_most_three(self):
        """B1-Top3 selects at most 3 memories per invocation."""
        router = RelevanceTopKRouter(
            config=RelevanceTopKRouterConfig(max_shares_per_invocation=3)
        )
        proposal = _make_proposal(5)
        result = router.decide_from_proposal(
            receiver_agent_id="agent_1",
            proposal=proposal,
        )
        assert len(result.selected_memory_ids) == 3
        assert result.selected_memory_ids == ["mem_0", "mem_1", "mem_2"]

    def test_fewer_candidates_than_limit(self):
        """B1-Top3 selects all when fewer candidates than limit."""
        router = RelevanceTopKRouter(
            config=RelevanceTopKRouterConfig(max_shares_per_invocation=3)
        )
        proposal = _make_proposal(2)
        result = router.decide_from_proposal(
            receiver_agent_id="agent_1",
            proposal=proposal,
        )
        assert len(result.selected_memory_ids) == 2

    def test_withhold_reason(self):
        """B1-Top3 withhold decisions have correct reason."""
        router = RelevanceTopKRouter(
            config=RelevanceTopKRouterConfig(max_shares_per_invocation=3)
        )
        proposal = _make_proposal(5)
        result = router.decide_from_proposal(
            receiver_agent_id="agent_1",
            proposal=proposal,
        )
        withheld = [d for d in result.decisions if d.action == "withhold"]
        assert len(withheld) == 2
        for d in withheld:
            assert d.reason == "relevance_topk_budget_exceeded"
