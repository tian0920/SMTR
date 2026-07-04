"""Production sequential router with critic-guided candidate selection (B-01).

This module implements a sequential router that uses the transfer critic to
make share/withhold decisions for each candidate memory in order. Unlike the
NoMemoryRouter which always withholds, the SequentialRouter uses critic
estimates of tau(m|o,S) to decide whether sharing a memory is beneficial.
"""


from pydantic import BaseModel, ConfigDict, Field

from smtr.counterfactual.schemas import RoutingFeatureSnapshot
from smtr.memory.schemas import ContextFingerprint, MemoryRoutingCard
from smtr.router.baseline_router import RoutingResult
from smtr.router.candidate_proposer import CandidateProposal
from smtr.router.traces import CandidateTrace, RouterDecision
from smtr.router.transfer_critic import FourOutcomeTransferCritic, TransferEstimate
from smtr.router.transfer_features import TransferPredictionInput


class SequentialRouterConfig(BaseModel):
    """Configuration for the sequential router."""

    model_config = ConfigDict(frozen=True)

    tau_threshold: float = 0.0
    """Minimum tau estimate to share a memory."""

    negative_risk_veto: float = 0.5
    """Veto share if negative risk exceeds this threshold."""

    uncertainty_veto: float = 0.3
    """Veto share if uncertainty (tau_ucb - tau_lcb) exceeds this."""

    max_shares_per_invocation: int = 3
    """Maximum number of memories to share per router invocation."""

    epsilon: float = 0.0
    """Probability of random exploration (epsilon-greedy)."""

    require_positive_tau: bool = True
    """If True, only share memories with tau_mean > tau_threshold."""

    use_support_distance: bool = False
    """If True, veto shares with low support (high support_distance)."""

    support_threshold: float = 0.5
    """Maximum support distance to allow sharing."""


class SequentialRouterDecision(RouterDecision):
    """Extended router decision with critic estimates."""

    tau_mean: float | None = None
    tau_lcb: float | None = None
    tau_ucb: float | None = None
    negative_risk_mean: float | None = None
    negative_risk_ucb: float | None = None
    support_distance: float | None = None
    decision_mode: str | None = None


class SequentialRouterState(BaseModel):
    """Tracks state during sequential routing."""

    model_config = ConfigDict(frozen=False)

    selected_memory_ids: list[str] = Field(default_factory=list)
    selected_cards: list[RoutingFeatureSnapshot] = Field(default_factory=list)
    decisions: list[RouterDecision] = Field(default_factory=list)
    share_count: int = 0


class ProductionSequentialRouter:
    """Production router using critic-guided sequential selection.

    The router processes candidates in order and for each candidate:
    1. Constructs a prediction input with current context and selected set
    2. Queries the critic for tau(m|o,S) estimate
    3. Applies decision rules (threshold, risk veto, uncertainty veto)
    4. Updates selected set if sharing

    This implements the "learned router" concept from the research spec,
    using the offline-trained critic to guide online routing decisions.
    """

    router_name = "ProductionSequentialRouter"
    router_version = "1"

    def __init__(
        self,
        *,
        critic: FourOutcomeTransferCritic | None = None,
        config: SequentialRouterConfig | None = None,
        seed: int = 0,
    ) -> None:
        self.critic = critic
        self.config = config or SequentialRouterConfig()
        self.seed = seed
        self._rng = None

    def _get_rng(self):
        """Lazy initialization of random number generator."""
        if self._rng is None:
            import numpy as np
            self._rng = np.random.default_rng(self.seed)
        return self._rng

    def decide_from_proposal(
        self,
        *,
        receiver_agent_id: str,
        proposal: CandidateProposal,
        cards_by_id: dict[str, MemoryRoutingCard] | None = None,
        context: ContextFingerprint | None = None,
    ) -> RoutingResult:
        """Make sequential routing decisions for all candidates.

        Args:
            receiver_agent_id: ID of the agent receiving memories
            proposal: Candidate proposal with ranked candidates
            cards_by_id: Optional mapping of memory ID to routing card
            context: Optional context fingerprint for critic prediction

        Returns:
            RoutingResult with decisions and selected memory IDs
        """
        if self.critic is None:
            # Fall back to no-share if no critic available
            return self._fallback_no_share(receiver_agent_id, proposal)

        state = SequentialRouterState()

        for position, candidate in enumerate(proposal.ranked_candidates):
            if state.share_count >= self.config.max_shares_per_invocation:
                # Budget exhausted - withhold remaining
                decision = self._make_decision(
                    candidate=candidate,
                    position=position,
                    action="withhold",
                    reason="budget_exhausted",
                    decision_mode="budget_exhausted",
                    estimate=None,
                )
                state.decisions.append(decision)
                continue

            # Get critic estimate for this candidate
            estimate = self._estimate_transfer_effect(
                candidate=candidate,
                receiver_agent_id=receiver_agent_id,
                proposal=proposal,
                selected_cards=state.selected_cards,
                cards_by_id=cards_by_id,
                context=context,
            )

            # Make decision based on estimate
            action, reason, mode = self._apply_decision_rules(estimate, state)

            decision = self._make_decision(
                candidate=candidate,
                position=position,
                action=action,
                reason=reason,
                decision_mode=mode,
                estimate=estimate,
            )
            state.decisions.append(decision)

            # Update selected set if sharing
            if action == "share":
                state.selected_memory_ids.append(candidate.memory_id)
                state.share_count += 1
                # Add card snapshot to selected set for future predictions
                card_snapshot = self._get_card_snapshot(
                    candidate.memory_id, cards_by_id
                )
                if card_snapshot is not None:
                    state.selected_cards.append(card_snapshot)

        return RoutingResult(
            receiver_agent_id=receiver_agent_id,
            candidate_proposal=proposal,
            decisions=state.decisions,
            selected_memory_ids=state.selected_memory_ids,
            router_name=self.router_name,
            router_version=self.router_version,
        )

    def decide(
        self,
        *,
        task: str,
        receiver_agent: str,
        candidates: list[CandidateTrace],
        cards_by_id: dict[str, MemoryRoutingCard],
        seed: int,
    ) -> tuple[list[RouterDecision], list[str]]:
        """Legacy interface for compatibility with NoMemoryRouter.

        This method converts the legacy candidate format to the new proposal
        format and calls decide_from_proposal.
        """
        from smtr.router.candidate_proposer import CandidateRequest, CandidateScore

        request = CandidateRequest(
            task=task,
            task_stage="legacy",
            receiver_agent_id=receiver_agent,
            receiver_role=receiver_agent,
            receiver_capabilities=[],
            environment_observation={},
            local_context_summary="",
            top_k=len(candidates),
            seed=seed,
        )

        scored_candidates = [
            CandidateScore(
                memory_id=c.memory_id,
                total_score=c.total_score,
                goal_similarity=c.goal_similarity,
                task_tag_overlap=c.task_tag_overlap,
                environment_compatibility=c.environment_compatibility,
                receiver_compatibility=c.receiver_compatibility,
                explicit_environment_conflict=c.explicit_environment_conflict,
                score_explanation=c.score_explanation,
            )
            for c in candidates
        ]

        proposal = CandidateProposal(
            request=request,
            ranked_candidates=scored_candidates,
            pool_revision=0,
        )

        result = self.decide_from_proposal(
            receiver_agent_id=receiver_agent,
            proposal=proposal,
            cards_by_id=cards_by_id,
        )

        return result.decisions, result.selected_memory_ids

    def _estimate_transfer_effect(
        self,
        *,
        candidate: CandidateTrace,
        receiver_agent_id: str,
        proposal: CandidateProposal,
        selected_cards: list[RoutingFeatureSnapshot],
        cards_by_id: dict[str, MemoryRoutingCard] | None,
        context: ContextFingerprint | None,
    ) -> TransferEstimate | None:
        """Get critic estimate for a candidate given current selected set."""
        if self.critic is None:
            return None

        card_snapshot = self._get_card_snapshot(candidate.memory_id, cards_by_id)
        if card_snapshot is None:
            return None

        if context is None:
            # Build minimal context from proposal
            context = self._build_context_from_proposal(
                proposal=proposal,
                receiver_agent_id=receiver_agent_id,
                selected_memory_ids=[c.memory_id for c in selected_cards],
            )

        prediction_input = TransferPredictionInput(
            context=context,
            candidate_card=card_snapshot,
            selected_cards=selected_cards,
        )

        return self.critic.predict(prediction_input)

    def _apply_decision_rules(
        self,
        estimate: TransferEstimate | None,
        state: SequentialRouterState,
    ) -> tuple[str, str, str]:
        """Apply decision rules to determine share/withhold action.

        Returns:
            Tuple of (action, reason, decision_mode)
        """
        if estimate is None:
            return "withhold", "no_critic_estimate", "ordinary_withhold"

        # Check negative risk veto
        if estimate.negative_risk_ucb > self.config.negative_risk_veto:
            return "withhold", "negative_risk_veto", "risk_veto"

        # Check uncertainty veto
        uncertainty = estimate.tau_ucb - estimate.tau_lcb
        if uncertainty > self.config.uncertainty_veto:
            return "withhold", "high_uncertainty", "hard_ood_veto"

        # Check support distance
        if self.config.use_support_distance and estimate.low_support:
            if estimate.support_distance > self.config.support_threshold:
                return "withhold", "low_support", "hard_ood_veto"

        # Check tau threshold
        if self.config.require_positive_tau:
            if estimate.tau_mean <= self.config.tau_threshold:
                return "withhold", "tau_below_threshold", "ordinary_withhold"

        # Epsilon-greedy exploration
        if self.config.epsilon > 0:
            rng = self._get_rng()
            if rng.random() < self.config.epsilon:
                return "share", "epsilon_exploration", "boundary_explore"

        # Share if all checks pass
        return "share", "critic_guided_share", "safe_exploit"

    def _make_decision(
        self,
        *,
        candidate: CandidateTrace,
        position: int,
        action: str,
        reason: str,
        decision_mode: str,
        estimate: TransferEstimate | None,
    ) -> RouterDecision:
        """Create a router decision with critic estimates."""
        return RouterDecision(
            memory_id=candidate.memory_id,
            action=action,
            decision=action,
            score=candidate.total_score,
            reason=reason,
            candidate_position=position,
            decision_source="baseline_router",
            tau_mean=estimate.tau_mean if estimate else None,
            tau_lcb=estimate.tau_lcb if estimate else None,
            tau_ucb=estimate.tau_ucb if estimate else None,
            negative_risk_mean=estimate.negative_risk_mean if estimate else None,
            negative_risk_ucb=estimate.negative_risk_ucb if estimate else None,
            low_support=estimate.low_support if estimate else None,
            decision_mode=decision_mode,
            support_distance=estimate.support_distance if estimate else None,
            support_threshold=estimate.support_threshold if estimate else None,
        )

    def _get_card_snapshot(
        self,
        memory_id: str,
        cards_by_id: dict[str, MemoryRoutingCard] | None,
    ) -> RoutingFeatureSnapshot | None:
        """Get routing feature snapshot for a memory ID."""
        if cards_by_id is None or memory_id not in cards_by_id:
            return None
        card = cards_by_id[memory_id]
        return RoutingFeatureSnapshot(
            memory_id=card.memory_id,
            active_payload_version=card.active_payload_version,
            goal_summary=card.goal_summary,
            task_tags=list(card.task_tags),
            precondition_summary=card.precondition_summary,
            postcondition_summary=card.postcondition_summary,
            required_environment_facts=dict(card.required_environment_facts),
            forbidden_environment_facts=dict(card.forbidden_environment_facts),
            compatible_receiver_roles=list(card.compatible_receiver_roles),
            compatible_receiver_capabilities=list(card.compatible_receiver_capabilities),
            execution_success_alpha=card.execution_success_alpha,
            execution_success_beta=card.execution_success_beta,
            execution_success_count=card.execution_success_count,
            execution_failure_count=card.execution_failure_count,
            paired_positive_transfer_count=card.paired_positive_transfer_count,
            paired_negative_transfer_count=card.paired_negative_transfer_count,
            paired_neutral_transfer_count=card.paired_neutral_transfer_count,
        )

    def _build_context_from_proposal(
        self,
        *,
        proposal: CandidateProposal,
        receiver_agent_id: str,
        selected_memory_ids: list[str],
    ) -> ContextFingerprint:
        """Build a context fingerprint from a candidate proposal."""
        from smtr.memory.execution_evidence import selected_set_signature

        request = proposal.request
        return ContextFingerprint(
            task_id=request.task[:32],
            task_tags=list(_extract_task_tags(request.task)),
            receiver_agent_id=receiver_agent_id,
            receiver_role=request.receiver_role,
            receiver_capabilities=list(request.receiver_capabilities),
            environment_facts=dict(request.environment_observation),
            task_stage=request.task_stage,
            selected_memory_ids=selected_memory_ids,
            selected_set_signature=selected_set_signature(selected_memory_ids),
            episode_id=f"router_{receiver_agent_id}",
        )

    def _fallback_no_share(
        self,
        receiver_agent_id: str,
        proposal: CandidateProposal,
    ) -> RoutingResult:
        """Fall back to no-share when critic is not available."""
        decisions = [
            RouterDecision(
                memory_id=candidate.memory_id,
                action="withhold",
                decision="withhold",
                score=candidate.total_score,
                reason="no_critic_available",
            )
            for candidate in proposal.ranked_candidates
        ]
        return RoutingResult(
            receiver_agent_id=receiver_agent_id,
            candidate_proposal=proposal,
            decisions=decisions,
            selected_memory_ids=[],
            router_name=self.router_name,
            router_version=self.router_version,
        )


def _extract_task_tags(task: str) -> set[str]:
    """Extract task tags from task description."""
    tokens = set(task.lower().split())
    tags = set()
    if {"artifact", "target"} & tokens:
        tags.add("artifact")
    if {"plan", "sequence", "ordered"} & tokens:
        tags.add("ordered-actions")
    if {"execute", "action", "actions", "tool"} & tokens:
        tags.add("tool-chain")
    if {"judge", "verify", "check", "critic"} & tokens:
        tags.add("verification")
    return tags
