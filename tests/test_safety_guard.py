"""Tests for B-02: Runtime Safety Guard & Fallback Router."""

from smtr.memory.schemas import MemoryRoutingCard
from smtr.router.candidate_proposer import CandidateProposal, CandidateRequest, CandidateScore
from smtr.router.gate_protocol import TransferPointEstimate
from smtr.router.safety_guard import (
    FallbackRouter,
    SafetyGuard,
    SafetyGuardConfig,
    SafetyVetoReason,
)
from smtr.router.transfer_critic import FourOutcomeTransferCritic, TransferEstimate

# --- Fixtures ---


def _make_estimate(**overrides) -> TransferEstimate:
    """Create a TransferEstimate with sensible defaults."""
    defaults = dict(
        q00_mean=0.1,
        q01_mean=0.1,
        q10_mean=0.3,
        q11_mean=0.5,
        tau_mean=0.2,
        tau_lcb=0.1,
        tau_ucb=0.3,
        negative_risk_mean=0.1,
        negative_risk_ucb=0.15,
        support_distance=0.2,
        support_threshold=0.5,
        low_support=False,
        ensemble_size=31,
        critic_version="test_v1",
    )
    defaults.update(overrides)
    return TransferEstimate(**defaults)


def _make_card(memory_id: str) -> MemoryRoutingCard:
    return MemoryRoutingCard(
        memory_id=memory_id,
        active_payload_version=1,
        goal_summary="test goal",
        task_tags=["test"],
        compatible_receiver_roles=["planner", "executor"],
        required_environment_facts={},
    )


def _make_proposal(
    memory_ids: list[str],
    *,
    task: str = "execute test action",
    receiver: str = "executor",
) -> CandidateProposal:
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
    return {mid: _make_card(mid) for mid in memory_ids}


def _make_critic_with_estimate(
    tau_mean: float = 0.3,
    negative_risk: float = 0.1,
    tau_lcb: float = 0.2,
    tau_ucb: float = 0.4,
    support_distance: float = 0.1,
) -> FourOutcomeTransferCritic:
    """Create a mock critic that returns fixed estimates."""

    class MockCritic(FourOutcomeTransferCritic):
        def __init__(self):
            super().__init__()

        def predict(self, item):
            return _make_estimate(
                tau_mean=tau_mean,
                tau_lcb=tau_lcb,
                tau_ucb=tau_ucb,
                negative_risk_mean=negative_risk,
                negative_risk_ucb=negative_risk + 0.05,
                support_distance=support_distance,
            )

        def predict_point(self, item):
            return TransferPointEstimate(
                tau_mean=tau_mean,
                negative_risk_mean=negative_risk,
            )

    return MockCritic()


# --- SafetyGuard Tests ---


class TestSafetyGuardBasics:
    """Test SafetyGuard veto logic."""

    def test_safe_estimate_passes(self):
        guard = SafetyGuard()
        estimate = _make_estimate()
        is_safe, reason = guard.check_estimate(estimate)
        assert is_safe is True
        assert reason == SafetyVetoReason.NONE

    def test_high_negative_risk_vetoed(self):
        guard = SafetyGuard(config=SafetyGuardConfig(max_negative_risk_ucb=0.3))
        estimate = _make_estimate(negative_risk_ucb=0.5)
        is_safe, reason = guard.check_estimate(estimate)
        assert is_safe is False
        assert reason == SafetyVetoReason.HIGH_NEGATIVE_RISK

    def test_high_uncertainty_vetoed(self):
        guard = SafetyGuard(config=SafetyGuardConfig(max_uncertainty=0.3))
        # tau_ucb - tau_lcb = 0.5 - 0.1 = 0.4 > 0.3
        estimate = _make_estimate(tau_lcb=0.1, tau_ucb=0.5)
        is_safe, reason = guard.check_estimate(estimate)
        assert is_safe is False
        assert reason == SafetyVetoReason.HIGH_UNCERTAINTY

    def test_low_support_vetoed(self):
        guard = SafetyGuard(config=SafetyGuardConfig(min_support_distance=0.3))
        estimate = _make_estimate(support_distance=0.5)
        is_safe, reason = guard.check_estimate(estimate)
        assert is_safe is False
        assert reason == SafetyVetoReason.LOW_SUPPORT

    def test_risk_veto_disabled(self):
        guard = SafetyGuard(config=SafetyGuardConfig(enable_risk_veto=False))
        estimate = _make_estimate(negative_risk_ucb=0.9)
        is_safe, reason = guard.check_estimate(estimate)
        # Should pass since risk veto is disabled (other checks may still pass)
        # uncertainty = 0.3 - 0.1 = 0.2 < 0.5, support = 0.2 < 0.8
        assert is_safe is True

    def test_uncertainty_veto_disabled(self):
        guard = SafetyGuard(config=SafetyGuardConfig(enable_uncertainty_veto=False))
        estimate = _make_estimate(tau_lcb=0.0, tau_ucb=1.0)
        is_safe, reason = guard.check_estimate(estimate)
        assert is_safe is True

    def test_ood_veto_disabled(self):
        guard = SafetyGuard(config=SafetyGuardConfig(enable_ood_veto=False))
        estimate = _make_estimate(support_distance=1.0)
        is_safe, reason = guard.check_estimate(estimate)
        assert is_safe is True


class TestSafetyGuardState:
    """Test SafetyGuard state tracking."""

    def test_consecutive_vetoes_increment(self):
        guard = SafetyGuard(config=SafetyGuardConfig(max_negative_risk_ucb=0.1))
        estimate = _make_estimate(negative_risk_ucb=0.5)
        guard.check_estimate(estimate)
        guard.check_estimate(estimate)
        assert guard.state.consecutive_vetoes == 2
        assert guard.state.total_vetoes == 2

    def test_share_resets_consecutive_vetoes(self):
        guard = SafetyGuard(config=SafetyGuardConfig(max_negative_risk_ucb=0.3))
        # Veto
        guard.check_estimate(_make_estimate(negative_risk_ucb=0.5))
        assert guard.state.consecutive_vetoes == 1
        # Share
        guard.check_estimate(_make_estimate(negative_risk_ucb=0.1))
        assert guard.state.consecutive_vetoes == 0
        assert guard.state.total_shares == 1

    def test_conservative_mode_after_threshold(self):
        guard = SafetyGuard(
            config=SafetyGuardConfig(max_consecutive_vetoes=3, max_negative_risk_ucb=0.1)
        )
        estimate = _make_estimate(negative_risk_ucb=0.5)
        assert guard.should_enter_conservative_mode() is False
        guard.check_estimate(estimate)
        guard.check_estimate(estimate)
        assert guard.should_enter_conservative_mode() is False
        guard.check_estimate(estimate)
        assert guard.should_enter_conservative_mode() is True

    def test_stats(self):
        guard = SafetyGuard(config=SafetyGuardConfig(max_negative_risk_ucb=0.3))
        guard.check_estimate(_make_estimate(negative_risk_ucb=0.5))  # veto
        guard.check_estimate(_make_estimate(negative_risk_ucb=0.1))  # share
        stats = guard.get_stats()
        assert stats["total_vetoes"] == 1
        assert stats["total_shares"] == 1
        assert stats["consecutive_vetoes"] == 0

    def test_reset_state(self):
        guard = SafetyGuard(config=SafetyGuardConfig(max_negative_risk_ucb=0.1))
        guard.check_estimate(_make_estimate(negative_risk_ucb=0.5))
        guard.reset_state()
        assert guard.state.total_vetoes == 0
        assert guard.state.consecutive_vetoes == 0


# --- FallbackRouter Tests ---


class TestFallbackRouterBasics:
    """Test FallbackRouter normal operation."""

    def test_normal_operation_passes_safe_decisions(self):
        critic = _make_critic_with_estimate(
            tau_mean=0.3, tau_lcb=0.2, tau_ucb=0.4, negative_risk=0.05
        )
        router = FallbackRouter(critic=critic)
        proposal = _make_proposal(["m1"])
        cards_by_id = _make_cards_by_id(["m1"])
        result = router.decide_from_proposal(
            receiver_agent_id="agent1",
            proposal=proposal,
            cards_by_id=cards_by_id,
        )
        assert result.router_name == "FallbackRouter"
        # Should have one decision
        assert len(result.decisions) == 1
        # The decision should be share (tau=0.3 > 0.0 threshold, low risk)
        assert result.decisions[0].action == "share"

    def test_safety_guard_veto_overrides_to_withhold(self):
        critic = _make_critic_with_estimate(
            tau_mean=0.3, tau_lcb=0.2, tau_ucb=0.4, negative_risk=0.1
        )
        # Safety guard with very low risk threshold
        router = FallbackRouter(
            critic=critic,
            safety_config=SafetyGuardConfig(max_negative_risk_ucb=0.05),
        )
        proposal = _make_proposal(["m1"])
        cards_by_id = _make_cards_by_id(["m1"])
        result = router.decide_from_proposal(
            receiver_agent_id="agent1",
            proposal=proposal,
            cards_by_id=cards_by_id,
        )
        # negative_risk_ucb from critic = 0.15 > 0.05, so safety guard vetoes
        assert result.decisions[0].action == "withhold"
        assert "safety_guard" in result.decisions[0].reason

    def test_withhold_decisions_pass_through(self):
        # Critic returns negative tau → router withholds
        critic = _make_critic_with_estimate(tau_mean=-0.5, tau_lcb=-0.6, tau_ucb=-0.4)
        router = FallbackRouter(critic=critic)
        proposal = _make_proposal(["m1"])
        cards_by_id = _make_cards_by_id(["m1"])
        result = router.decide_from_proposal(
            receiver_agent_id="agent1",
            proposal=proposal,
            cards_by_id=cards_by_id,
        )
        assert result.decisions[0].action == "withhold"


class TestFallbackRouterMode:
    """Test FallbackRouter entering/exiting fallback mode."""

    def test_starts_not_in_fallback(self):
        router = FallbackRouter()
        assert router.in_fallback_mode is False

    def test_enters_fallback_after_consecutive_vetoes(self):
        # Create critic with high negative risk to trigger vetoes
        critic = _make_critic_with_estimate(
            tau_mean=0.3, tau_lcb=0.2, tau_ucb=0.4, negative_risk=0.1
        )
        router = FallbackRouter(
            critic=critic,
            safety_config=SafetyGuardConfig(
                max_negative_risk_ucb=0.05,
                max_consecutive_vetoes=2,
            ),
        )
        proposal = _make_proposal(["m1"])
        cards_by_id = _make_cards_by_id(["m1"])

        # First veto
        router.decide_from_proposal(
            receiver_agent_id="agent1", proposal=proposal, cards_by_id=cards_by_id
        )
        assert router.in_fallback_mode is False

        # Second veto triggers fallback on next call
        router.decide_from_proposal(
            receiver_agent_id="agent1", proposal=proposal, cards_by_id=cards_by_id
        )
        # Now safety_guard has 2 consecutive vetoes = max_consecutive_vetoes
        assert router.safety_guard.should_enter_conservative_mode() is True

        # Third call should enter fallback mode
        router.decide_from_proposal(
            receiver_agent_id="agent1", proposal=proposal, cards_by_id=cards_by_id
        )
        assert router.in_fallback_mode is True

    def test_reset_exits_fallback(self):
        router = FallbackRouter()
        router._in_fallback_mode = True
        router.reset()
        assert router.in_fallback_mode is False

    def test_stats_include_fallback_mode(self):
        router = FallbackRouter()
        stats = router.get_stats()
        assert "in_fallback_mode" in stats
        assert stats["in_fallback_mode"] is False
        assert "safety_guard" in stats


class TestFallbackRouterEdgeCases:
    """Edge cases for FallbackRouter."""

    def test_empty_proposal(self):
        router = FallbackRouter()
        proposal = _make_proposal([])
        result = router.decide_from_proposal(
            receiver_agent_id="agent1",
            proposal=proposal,
        )
        assert len(result.decisions) == 0
        assert len(result.selected_memory_ids) == 0

    def test_no_critic_withholds_all(self):
        router = FallbackRouter(critic=None)
        proposal = _make_proposal(["m1", "m2"])
        cards_by_id = _make_cards_by_id(["m1", "m2"])
        result = router.decide_from_proposal(
            receiver_agent_id="agent1",
            proposal=proposal,
            cards_by_id=cards_by_id,
        )
        # Without critic, all should be withheld
        assert all(d.action == "withhold" for d in result.decisions)

    def test_multiple_candidates(self):
        critic = _make_critic_with_estimate(
            tau_mean=0.3, tau_lcb=0.2, tau_ucb=0.4, negative_risk=0.05
        )
        router = FallbackRouter(critic=critic)
        proposal = _make_proposal(["m1", "m2", "m3"])
        cards_by_id = _make_cards_by_id(["m1", "m2", "m3"])
        result = router.decide_from_proposal(
            receiver_agent_id="agent1",
            proposal=proposal,
            cards_by_id=cards_by_id,
        )
        assert len(result.decisions) == 3
