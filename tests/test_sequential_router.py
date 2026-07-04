"""Tests for B-01: Production Sequential Router."""


from smtr.memory.schemas import ContextFingerprint, MemoryRoutingCard
from smtr.router.candidate_proposer import CandidateProposal, CandidateRequest, CandidateScore
from smtr.router.sequential_router import (
    ProductionSequentialRouter,
    SequentialRouterConfig,
)
from smtr.router.transfer_critic import FourOutcomeTransferCritic, TransferEstimate

# --- Fixtures ---


def _make_card(
    memory_id: str,
    *,
    goal: str = "test goal",
    roles: list[str] | None = None,
    required_env: dict | None = None,
) -> MemoryRoutingCard:
    """Create a test routing card."""
    return MemoryRoutingCard(
        memory_id=memory_id,
        active_payload_version=1,
        goal_summary=goal,
        task_tags=["test"],
        compatible_receiver_roles=roles or ["planner", "executor"],
        required_environment_facts=required_env or {},
    )


def _make_proposal(
    memory_ids: list[str],
    *,
    task: str = "execute test action",
    receiver: str = "executor",
) -> CandidateProposal:
    """Create a test candidate proposal."""
    request = CandidateRequest(
        task=task,
        task_stage="test",
        receiver_agent_id=receiver,
        receiver_role=receiver,
        receiver_capabilities=[],
        environment_observation={},
        top_k=len(memory_ids),
        seed=42,
    )
    candidates = [
        CandidateScore(
            memory_id=mid,
            total_score=0.5 - i * 0.1,
            goal_similarity=0.5,
            task_tag_overlap=0.5,
            environment_compatibility=0.5,
            receiver_compatibility=1.0,
        )
        for i, mid in enumerate(memory_ids)
    ]
    return CandidateProposal(
        request=request,
        ranked_candidates=candidates,
        pool_revision=1,
    )


def _make_cards_by_id(memory_ids: list[str]) -> dict[str, MemoryRoutingCard]:
    """Create a mapping of memory ID to routing card."""
    return {mid: _make_card(mid) for mid in memory_ids}


def _make_critic_with_estimate(
    tau_mean: float = 0.3,
    negative_risk: float = 0.1,
    tau_lcb: float = 0.2,
    tau_ucb: float = 0.4,
) -> FourOutcomeTransferCritic:
    """Create a mock critic that returns fixed estimates."""

    class MockCritic:
        critic_version = "mock_v1"

        def predict(self, item):
            return TransferEstimate(
                q00_mean=0.1,
                q01_mean=negative_risk,
                q10_mean=tau_mean + negative_risk,
                q11_mean=0.1,
                tau_mean=tau_mean,
                tau_lcb=tau_lcb,
                tau_ucb=tau_ucb,
                negative_risk_mean=negative_risk,
                negative_risk_ucb=negative_risk * 1.5,
                support_distance=0.1,
                support_threshold=0.5,
                low_support=False,
                ensemble_size=31,
                critic_version=self.critic_version,
            )

    return MockCritic()


def _make_context(
    receiver: str = "executor",
    selected_ids: list[str] | None = None,
) -> ContextFingerprint:
    """Create a test context fingerprint."""
    from smtr.memory.execution_evidence import selected_set_signature

    return ContextFingerprint(
        task_id="test_task",
        task_tags=["test"],
        receiver_agent_id=receiver,
        receiver_role=receiver,
        receiver_capabilities=[],
        environment_facts={},
        task_stage="test",
        selected_memory_ids=selected_ids or [],
        selected_set_signature=selected_set_signature(selected_ids or []),
        episode_id="test_episode",
    )


# --- Tests: Basic Router Functionality ---


class TestSequentialRouterBasics:
    """Test basic sequential router functionality."""

    def test_router_creation_without_critic(self):
        """Router can be created without a critic."""
        router = ProductionSequentialRouter()
        assert router.critic is None
        assert router.router_name == "ProductionSequentialRouter"

    def test_router_creation_with_config(self):
        """Router accepts custom configuration."""
        config = SequentialRouterConfig(
            tau_threshold=0.2,
            max_shares_per_invocation=5,
        )
        router = ProductionSequentialRouter(config=config)
        assert router.config.tau_threshold == 0.2
        assert router.config.max_shares_per_invocation == 5

    def test_router_without_critic_withholds_all(self):
        """Router without critic falls back to no-share."""
        router = ProductionSequentialRouter()
        proposal = _make_proposal(["mem_1", "mem_2", "mem_3"])
        result = router.decide_from_proposal(
            receiver_agent_id="executor",
            proposal=proposal,
        )
        assert result.selected_memory_ids == []
        assert all(d.action == "withhold" for d in result.decisions)

    def test_router_with_positive_tau_shares(self):
        """Router shares memories with positive tau estimate."""
        critic = _make_critic_with_estimate(tau_mean=0.3, negative_risk=0.1)
        router = ProductionSequentialRouter(critic=critic)
        memory_ids = ["mem_1", "mem_2"]
        proposal = _make_proposal(memory_ids)
        cards_by_id = _make_cards_by_id(memory_ids)
        result = router.decide_from_proposal(
            receiver_agent_id="executor",
            proposal=proposal,
            cards_by_id=cards_by_id,
        )
        # Should share both (tau > 0)
        assert len(result.selected_memory_ids) == 2

    def test_router_with_negative_tau_withholds(self):
        """Router withholds memories with negative tau estimate."""
        critic = _make_critic_with_estimate(tau_mean=-0.2, negative_risk=0.1)
        router = ProductionSequentialRouter(critic=critic)
        memory_ids = ["mem_1"]
        proposal = _make_proposal(memory_ids)
        cards_by_id = _make_cards_by_id(memory_ids)
        result = router.decide_from_proposal(
            receiver_agent_id="executor",
            proposal=proposal,
            cards_by_id=cards_by_id,
        )
        assert result.selected_memory_ids == []
        assert result.decisions[0].reason == "tau_below_threshold"


# --- Tests: Decision Rules ---


class TestSequentialRouterDecisionRules:
    """Test decision rules for sequential routing."""

    def test_negative_risk_veto(self):
        """Router vetoes high negative risk memories."""
        critic = _make_critic_with_estimate(tau_mean=0.3, negative_risk=0.6)
        router = ProductionSequentialRouter(critic=critic)
        memory_ids = ["mem_1"]
        proposal = _make_proposal(memory_ids)
        cards_by_id = _make_cards_by_id(memory_ids)
        result = router.decide_from_proposal(
            receiver_agent_id="executor",
            proposal=proposal,
            cards_by_id=cards_by_id,
        )
        assert result.selected_memory_ids == []
        assert result.decisions[0].decision_mode == "risk_veto"

    def test_uncertainty_veto(self):
        """Router vetoes high uncertainty memories."""
        critic = _make_critic_with_estimate(
            tau_mean=0.3,
            negative_risk=0.1,
            tau_lcb=-0.3,
            tau_ucb=0.9,  # High uncertainty
        )
        config = SequentialRouterConfig(uncertainty_veto=0.3)
        router = ProductionSequentialRouter(critic=critic, config=config)
        memory_ids = ["mem_1"]
        proposal = _make_proposal(memory_ids)
        cards_by_id = _make_cards_by_id(memory_ids)
        result = router.decide_from_proposal(
            receiver_agent_id="executor",
            proposal=proposal,
            cards_by_id=cards_by_id,
        )
        assert result.selected_memory_ids == []
        assert result.decisions[0].decision_mode == "hard_ood_veto"

    def test_max_shares_limit(self):
        """Router respects max shares per invocation."""
        critic = _make_critic_with_estimate(tau_mean=0.3, negative_risk=0.1)
        config = SequentialRouterConfig(max_shares_per_invocation=2)
        router = ProductionSequentialRouter(critic=critic, config=config)
        memory_ids = ["mem_1", "mem_2", "mem_3", "mem_4"]
        proposal = _make_proposal(memory_ids)
        cards_by_id = _make_cards_by_id(memory_ids)
        result = router.decide_from_proposal(
            receiver_agent_id="executor",
            proposal=proposal,
            cards_by_id=cards_by_id,
        )
        assert len(result.selected_memory_ids) == 2
        # Remaining should be withheld due to budget
        assert result.decisions[2].decision_mode == "budget_exhausted"
        assert result.decisions[3].decision_mode == "budget_exhausted"

    def test_tau_threshold_configurable(self):
        """Router respects configurable tau threshold."""
        critic = _make_critic_with_estimate(tau_mean=0.15, negative_risk=0.1)
        config = SequentialRouterConfig(tau_threshold=0.2)
        router = ProductionSequentialRouter(critic=critic, config=config)
        memory_ids = ["mem_1"]
        proposal = _make_proposal(memory_ids)
        cards_by_id = _make_cards_by_id(memory_ids)
        result = router.decide_from_proposal(
            receiver_agent_id="executor",
            proposal=proposal,
            cards_by_id=cards_by_id,
        )
        # tau_mean (0.15) < threshold (0.2), should withhold
        assert result.selected_memory_ids == []


# --- Tests: Sequential State Tracking ---


class TestSequentialRouterStateTracking:
    """Test state tracking during sequential routing."""

    def test_selected_set_grows_with_shares(self):
        """Selected set grows as memories are shared."""
        critic = _make_critic_with_estimate(tau_mean=0.3, negative_risk=0.1)
        router = ProductionSequentialRouter(critic=critic)
        memory_ids = ["mem_1", "mem_2", "mem_3"]
        proposal = _make_proposal(memory_ids)
        cards_by_id = _make_cards_by_id(memory_ids)
        result = router.decide_from_proposal(
            receiver_agent_id="executor",
            proposal=proposal,
            cards_by_id=cards_by_id,
        )
        assert result.selected_memory_ids == ["mem_1", "mem_2", "mem_3"]

    def test_decisions_include_position(self):
        """Each decision includes candidate position."""
        critic = _make_critic_with_estimate(tau_mean=0.3, negative_risk=0.1)
        router = ProductionSequentialRouter(critic=critic)
        proposal = _make_proposal(["mem_a", "mem_b", "mem_c"])
        result = router.decide_from_proposal(
            receiver_agent_id="executor",
            proposal=proposal,
        )
        for i, decision in enumerate(result.decisions):
            assert decision.candidate_position == i

    def test_decisions_include_tau_estimates(self):
        """Decisions include critic tau estimates."""
        critic = _make_critic_with_estimate(
            tau_mean=0.25,
            tau_lcb=0.1,
            tau_ucb=0.4,
            negative_risk=0.15,
        )
        router = ProductionSequentialRouter(critic=critic)
        memory_ids = ["mem_1"]
        proposal = _make_proposal(memory_ids)
        cards_by_id = _make_cards_by_id(memory_ids)
        result = router.decide_from_proposal(
            receiver_agent_id="executor",
            proposal=proposal,
            cards_by_id=cards_by_id,
        )
        decision = result.decisions[0]
        assert decision.tau_mean == 0.25
        assert decision.tau_lcb == 0.1
        assert decision.tau_ucb == 0.4
        assert decision.negative_risk_mean == 0.15


# --- Tests: Legacy Interface ---


class TestSequentialRouterLegacyInterface:
    """Test legacy decide() interface compatibility."""

    def test_decide_interface_works(self):
        """Legacy decide() interface works with candidate traces."""
        from smtr.router.traces import CandidateTrace

        critic = _make_critic_with_estimate(tau_mean=0.3, negative_risk=0.1)
        router = ProductionSequentialRouter(critic=critic)

        candidates = [
            CandidateTrace(
                memory_id="mem_1",
                total_score=0.5,
                goal_similarity=0.5,
                task_tag_overlap=0.5,
                environment_compatibility=0.5,
                receiver_compatibility=1.0,
            ),
        ]
        cards_by_id = {"mem_1": _make_card("mem_1")}

        decisions, selected = router.decide(
            task="execute test",
            receiver_agent="executor",
            candidates=candidates,
            cards_by_id=cards_by_id,
            seed=42,
        )
        assert len(decisions) == 1
        assert selected == ["mem_1"]


# --- Tests: Edge Cases ---


class TestSequentialRouterEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_proposal(self):
        """Router handles empty proposal gracefully."""
        critic = _make_critic_with_estimate()
        router = ProductionSequentialRouter(critic=critic)
        proposal = _make_proposal([])
        result = router.decide_from_proposal(
            receiver_agent_id="executor",
            proposal=proposal,
        )
        assert result.selected_memory_ids == []
        assert result.decisions == []

    def test_single_candidate(self):
        """Router handles single candidate correctly."""
        critic = _make_critic_with_estimate(tau_mean=0.3, negative_risk=0.1)
        router = ProductionSequentialRouter(critic=critic)
        memory_ids = ["only_one"]
        proposal = _make_proposal(memory_ids)
        cards_by_id = _make_cards_by_id(memory_ids)
        result = router.decide_from_proposal(
            receiver_agent_id="executor",
            proposal=proposal,
            cards_by_id=cards_by_id,
        )
        assert len(result.decisions) == 1
        assert result.selected_memory_ids == ["only_one"]

    def test_zero_tau_with_positive_threshold(self):
        """Router withholds when tau equals threshold (not strictly greater)."""
        critic = _make_critic_with_estimate(tau_mean=0.0, negative_risk=0.1)
        config = SequentialRouterConfig(tau_threshold=0.0, require_positive_tau=True)
        router = ProductionSequentialRouter(critic=critic, config=config)
        memory_ids = ["mem_1"]
        proposal = _make_proposal(memory_ids)
        cards_by_id = _make_cards_by_id(memory_ids)
        result = router.decide_from_proposal(
            receiver_agent_id="executor",
            proposal=proposal,
            cards_by_id=cards_by_id,
        )
        # tau_mean (0.0) <= threshold (0.0), should withhold
        assert result.selected_memory_ids == []

    def test_router_preserves_router_metadata(self):
        """Router includes name and version in result."""
        router = ProductionSequentialRouter()
        proposal = _make_proposal(["mem_1"])
        result = router.decide_from_proposal(
            receiver_agent_id="executor",
            proposal=proposal,
        )
        assert result.router_name == "ProductionSequentialRouter"
        assert result.router_version == "1"
