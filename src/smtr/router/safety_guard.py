"""Runtime safety guard and fallback router (B-02).

This module provides safety mechanisms for the sequential router:
- SafetyGuard: Monitors and blocks high-risk share decisions
- FallbackRouter: Switches to conservative policy when uncertainty is high

These components wrap the ProductionSequentialRouter to provide
defense-in-depth against negative transfer.
"""

from dataclasses import dataclass, field
from enum import Enum

from pydantic import BaseModel, ConfigDict

from smtr.router.baseline_router import RoutingResult
from smtr.router.sequential_router import (
    ProductionSequentialRouter,
    SequentialRouterConfig,
    SequentialRouterDecision,
)
from smtr.router.transfer_critic import FourOutcomeTransferCritic, TransferEstimate


class SafetyVetoReason(str, Enum):
    """Reasons for safety guard veto."""

    NONE = "none"
    HIGH_NEGATIVE_RISK = "high_negative_risk"
    HIGH_UNCERTAINTY = "high_uncertainty"
    LOW_SUPPORT = "low_support"
    CONSECUTIVE_NEGATIVES = "consecutive_negatives"
    BUDGET_EXCEEDED = "budget_exceeded"


class SafetyGuardConfig(BaseModel):
    """Configuration for the safety guard."""

    model_config = ConfigDict(frozen=True)

    max_negative_risk_ucb: float = 0.4
    """Maximum allowed negative risk UCB before veto."""

    max_uncertainty: float = 0.5
    """Maximum allowed uncertainty (tau_ucb - tau_lcb) before veto."""

    min_support_distance: float = 0.8
    """Minimum support distance threshold for OOD veto."""

    max_consecutive_vetoes: int = 3
    """After this many consecutive vetoes, force conservative mode."""

    enable_ood_veto: bool = True
    """Enable out-of-distribution veto based on support distance."""

    enable_risk_veto: bool = True
    """Enable negative risk veto."""

    enable_uncertainty_veto: bool = True
    """Enable uncertainty-based veto."""


@dataclass
class SafetyGuardState:
    """Tracks safety guard state across invocations."""

    consecutive_vetoes: int = 0
    total_vetoes: int = 0
    total_shares: int = 0
    veto_history: list[SafetyVetoReason] = field(default_factory=list)


class SafetyGuard:
    """Runtime safety guard that monitors and blocks high-risk decisions.

    The safety guard wraps a router and applies additional safety checks
    before allowing share decisions. It can veto decisions based on:
    - High negative transfer risk
    - High uncertainty (low confidence)
    - Low support (out of distribution)
    - Consecutive veto patterns
    """

    def __init__(
        self,
        *,
        config: SafetyGuardConfig | None = None,
    ) -> None:
        self.config = config or SafetyGuardConfig()
        self.state = SafetyGuardState()

    def check_estimate(self, estimate: TransferEstimate) -> tuple[bool, SafetyVetoReason]:
        """Check if a transfer estimate passes safety checks.

        Returns:
            Tuple of (is_safe, veto_reason)
        """
        # Check negative risk
        if self.config.enable_risk_veto:
            if estimate.negative_risk_ucb > self.config.max_negative_risk_ucb:
                self._record_veto(SafetyVetoReason.HIGH_NEGATIVE_RISK)
                return False, SafetyVetoReason.HIGH_NEGATIVE_RISK

        # Check uncertainty
        if self.config.enable_uncertainty_veto:
            uncertainty = estimate.tau_ucb - estimate.tau_lcb
            if uncertainty > self.config.max_uncertainty:
                self._record_veto(SafetyVetoReason.HIGH_UNCERTAINTY)
                return False, SafetyVetoReason.HIGH_UNCERTAINTY

        # Check support distance (OOD detection)
        if self.config.enable_ood_veto:
            if estimate.support_distance > self.config.min_support_distance:
                self._record_veto(SafetyVetoReason.LOW_SUPPORT)
                return False, SafetyVetoReason.LOW_SUPPORT

        # All checks passed
        self._record_share()
        return True, SafetyVetoReason.NONE

    def should_enter_conservative_mode(self) -> bool:
        """Check if we should enter conservative mode due to consecutive vetoes."""
        return self.state.consecutive_vetoes >= self.config.max_consecutive_vetoes

    def reset_state(self) -> None:
        """Reset the safety guard state."""
        self.state = SafetyGuardState()

    def get_stats(self) -> dict:
        """Get safety guard statistics."""
        return {
            "total_vetoes": self.state.total_vetoes,
            "total_shares": self.state.total_shares,
            "consecutive_vetoes": self.state.consecutive_vetoes,
            "conservative_mode": self.should_enter_conservative_mode(),
        }

    def _record_veto(self, reason: SafetyVetoReason) -> None:
        """Record a veto decision."""
        self.state.consecutive_vetoes += 1
        self.state.total_vetoes += 1
        self.state.veto_history.append(reason)

    def _record_share(self) -> None:
        """Record a share decision."""
        self.state.consecutive_vetoes = 0
        self.state.total_shares += 1


class FallbackRouterConfig(BaseModel):
    """Configuration for the fallback router."""

    model_config = ConfigDict(frozen=True)

    conservative_tau_threshold: float = 0.2
    """Higher tau threshold for conservative mode."""

    conservative_negative_risk_veto: float = 0.3
    """Lower negative risk threshold for conservative mode."""

    fallback_after_consecutive_vetoes: int = 3
    """Enter fallback mode after this many consecutive vetoes."""

    max_shares_in_fallback: int = 1
    """Maximum shares allowed in fallback mode."""


class FallbackRouter:
    """Router that falls back to conservative policy when safety is uncertain.

    The fallback router wraps a ProductionSequentialRouter and monitors
    its decisions. When too many decisions are vetoed by the safety guard,
    it switches to a more conservative configuration.
    """

    router_name = "FallbackRouter"
    router_version = "1"

    def __init__(
        self,
        *,
        critic: FourOutcomeTransferCritic | None = None,
        normal_config: SequentialRouterConfig | None = None,
        fallback_config: FallbackRouterConfig | None = None,
        safety_config: SafetyGuardConfig | None = None,
        seed: int = 0,
    ) -> None:
        self.critic = critic
        self.normal_config = normal_config or SequentialRouterConfig()
        self.fallback_config = fallback_config or FallbackRouterConfig()
        self.safety_guard = SafetyGuard(config=safety_config)
        self.seed = seed
        self._in_fallback_mode = False
        self._primary_router = self._create_router(self.normal_config)

    def _create_router(self, config: SequentialRouterConfig) -> ProductionSequentialRouter:
        """Create a sequential router with the given config."""
        return ProductionSequentialRouter(
            critic=self.critic,
            config=config,
            seed=self.seed,
        )

    def decide_from_proposal(
        self,
        *,
        receiver_agent_id: str,
        proposal,
        cards_by_id=None,
        context=None,
    ) -> RoutingResult:
        """Make routing decisions with fallback logic.

        The router first tries normal routing. If too many decisions are
        vetoed by the safety guard, it switches to conservative mode.
        """
        # Check if we should enter/exit fallback mode
        if self.safety_guard.should_enter_conservative_mode():
            if not self._in_fallback_mode:
                self._enter_fallback_mode()
        elif self._in_fallback_mode:
            # Recovery check: if consecutive vetoes dropped, exit fallback
            if self.safety_guard.state.consecutive_vetoes == 0:
                self._exit_fallback_mode()

        # Get routing result from primary router
        result = self._primary_router.decide_from_proposal(
            receiver_agent_id=receiver_agent_id,
            proposal=proposal,
            cards_by_id=cards_by_id,
            context=context,
        )

        # Apply safety guard checks to each decision
        guarded_decisions = []
        guarded_selected_ids = []

        for decision in result.decisions:
            if decision.action == "share" and decision.tau_mean is not None:
                # Create a synthetic estimate for safety check
                estimate = TransferEstimate(
                    q00_mean=0.1,
                    q01_mean=decision.negative_risk_mean or 0.0,
                    q10_mean=decision.tau_mean or 0.0,
                    q11_mean=0.1,
                    tau_mean=decision.tau_mean or 0.0,
                    tau_lcb=decision.tau_lcb or 0.0,
                    tau_ucb=decision.tau_ucb or 0.0,
                    negative_risk_mean=decision.negative_risk_mean or 0.0,
                    negative_risk_ucb=decision.negative_risk_ucb or 0.0,
                    support_distance=decision.support_distance or 0.0,
                    support_threshold=decision.support_threshold or 0.5,
                    low_support=decision.low_support or False,
                    ensemble_size=31,
                    critic_version="fallback_v1",
                )

                is_safe, veto_reason = self.safety_guard.check_estimate(estimate)
                if is_safe:
                    guarded_decisions.append(decision)
                    guarded_selected_ids.append(decision.memory_id)
                else:
                    # Override to withhold
                    guarded_decision = SequentialRouterDecision(
                        memory_id=decision.memory_id,
                        action="withhold",
                        decision="withhold",
                        score=decision.score,
                        reason=f"safety_guard_{veto_reason.value}",
                        candidate_position=decision.candidate_position,
                        decision_source="baseline_router",
                        tau_mean=decision.tau_mean,
                        tau_lcb=decision.tau_lcb,
                        tau_ucb=decision.tau_ucb,
                        negative_risk_mean=decision.negative_risk_mean,
                        negative_risk_ucb=decision.negative_risk_ucb,
                        low_support=decision.low_support,
                        decision_mode=f"safety_veto_{veto_reason.value}",
                        support_distance=decision.support_distance,
                        support_threshold=decision.support_threshold,
                    )
                    guarded_decisions.append(guarded_decision)
            else:
                guarded_decisions.append(decision)

        return RoutingResult(
            receiver_agent_id=receiver_agent_id,
            candidate_proposal=proposal,
            decisions=guarded_decisions,
            selected_memory_ids=guarded_selected_ids,
            router_name=self.router_name,
            router_version=self.router_version,
        )

    def _enter_fallback_mode(self) -> None:
        """Switch to conservative fallback configuration."""
        self._in_fallback_mode = True
        conservative_config = SequentialRouterConfig(
            tau_threshold=self.fallback_config.conservative_tau_threshold,
            negative_risk_veto=self.fallback_config.conservative_negative_risk_veto,
            max_shares_per_invocation=self.fallback_config.max_shares_in_fallback,
            require_positive_tau=True,
        )
        self._primary_router = self._create_router(conservative_config)

    def _exit_fallback_mode(self) -> None:
        """Exit fallback mode and return to normal configuration."""
        self._in_fallback_mode = False
        self._primary_router = self._create_router(self.normal_config)

    @property
    def in_fallback_mode(self) -> bool:
        """Check if router is in fallback mode."""
        return self._in_fallback_mode

    def get_stats(self) -> dict:
        """Get router and safety guard statistics."""
        return {
            "in_fallback_mode": self._in_fallback_mode,
            "safety_guard": self.safety_guard.get_stats(),
        }

    def reset(self) -> None:
        """Reset router state."""
        self.safety_guard.reset_state()
        self._in_fallback_mode = False
        self._primary_router = self._create_router(self.normal_config)
