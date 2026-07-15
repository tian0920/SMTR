"""Production sequential router with critic-guided candidate selection (B-01).

This module implements a sequential router that uses the transfer critic to
make share/withhold decisions for each candidate memory in order. Unlike the
NoMemoryRouter which always withholds, the SequentialRouter uses critic
estimates of tau(m|o,S) to decide whether sharing a memory is beneficial.
"""


from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from smtr.counterfactual.schemas import RoutingFeatureSnapshot
from smtr.memory.execution_evidence import selected_set_signature
from smtr.memory.schemas import ContextFingerprint, MemoryRoutingCard
from smtr.router.baseline_router import RoutingResult
from smtr.router.candidate_proposer import CandidateProposal
from smtr.router.conditioning import (
    DynamicSelectedSetConditioning,
    SelectedSetConditioningPolicy,
)
from smtr.router.gate_protocol import GateDecision, RoutingGate, TransferPointEstimate
from smtr.router.smtr_gate import SMTRGate, SMTRGateConfig
from smtr.router.traces import CandidateTrace, RouterDecision
from smtr.router.transfer_critic import FourOutcomeTransferCritic
from smtr.router.transfer_features import TransferPredictionInput
from smtr.router.traversal import (
    RandomTraversal,
    TraversalPolicy,
    traversal_permutation_indices,
)


class SequentialRouterConfig(BaseModel):
    """Operational configuration for the sequential router."""

    model_config = ConfigDict(frozen=True)


class SequentialRouterDecision(RouterDecision):
    """Extended router decision with critic estimates."""

    tau_mean: float | None = None
    tau_lcb: float | None = None
    tau_ucb: float | None = None
    negative_risk_mean: float | None = None
    negative_risk_ucb: float | None = None
    support_distance: float | None = None
    decision_mode: str | None = None
    gate_name: str | None = None
    effect_condition_passed: bool | None = None
    risk_condition_passed: bool | None = None


class SequentialRouterState(BaseModel):
    """Tracks state during sequential routing."""

    model_config = ConfigDict(frozen=False)

    selected_memory_ids: list[str] = Field(default_factory=list)
    selected_cards: list[RoutingFeatureSnapshot] = Field(default_factory=list)
    initial_selected_memory_ids: list[str] = Field(default_factory=list)
    initial_selected_cards: list[RoutingFeatureSnapshot] = Field(default_factory=list)
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
        critic: FourOutcomeTransferCritic | Any,
        gate: RoutingGate | None = None,
        conditioning_policy: SelectedSetConditioningPolicy | None = None,
        traversal_policy: TraversalPolicy | None = None,
        config: SequentialRouterConfig | None = None,
        seed: int = 0,
    ) -> None:
        if critic is None:
            raise TypeError(
                "ProductionSequentialRouter requires a trained critic; "
                "use NoMemoryRouter for the no-memory baseline"
            )
        self.critic = critic
        self.gate = gate or SMTRGate(SMTRGateConfig())
        self.conditioning_policy = conditioning_policy or DynamicSelectedSetConditioning()
        self.traversal_policy = traversal_policy or RandomTraversal()
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
        traversal_seed: int | None = None,
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
        state = SequentialRouterState()
        traversal_seed = (
            traversal_seed
            if traversal_seed is not None
            else proposal.request.seed
            if proposal.request.seed is not None
            else self.seed
        )
        proposal_order = [candidate.memory_id for candidate in proposal.ranked_candidates]
        traversal_order = list(
            self.traversal_policy.order(proposal_order, seed=traversal_seed)
        )
        permutation_indices = list(
            traversal_permutation_indices(proposal_order, traversal_order)
        )
        candidate_by_id = {
            candidate.memory_id: candidate for candidate in proposal.ranked_candidates
        }
        original_position_by_id = {
            candidate.memory_id: index
            for index, candidate in enumerate(proposal.ranked_candidates)
        }

        for position, memory_id in enumerate(traversal_order):
            candidate = candidate_by_id[memory_id]
            card_snapshot = self._get_card_snapshot(candidate.memory_id, cards_by_id)
            if card_snapshot is None:
                decision = self._make_decision(
                    candidate=candidate,
                    position=position,
                    original_position=original_position_by_id[candidate.memory_id],
                    action="withhold",
                    reason="missing_routing_card",
                    decision_mode="ordinary_withhold",
                    effect_condition_passed=None,
                    risk_condition_passed=None,
                    effect_condition_status=None,
                    risk_condition_status=None,
                    selected_before_actual=state.selected_memory_ids,
                    selected_before_critic=state.selected_memory_ids,
                    estimate=None,
                    traversal_seed=traversal_seed,
                    traversal_policy_name=self.traversal_policy.policy_name,
                    proposal_order=proposal_order,
                    traversal_order=traversal_order,
                    permutation_indices=permutation_indices,
                    accepted=False,
                )
                state.decisions.append(decision)
                continue

            actual_selected_set = tuple(state.selected_memory_ids)
            critic_selected_set = self.conditioning_policy.critic_selected_set(
                initial_selected_set=tuple(state.initial_selected_memory_ids),
                current_selected_set=actual_selected_set,
            )
            critic_selected_cards = _cards_for_selected_ids(
                critic_selected_set,
                current_cards=state.selected_cards,
                initial_cards=state.initial_selected_cards,
            )

            estimate = self._estimate_transfer_effect(
                candidate=candidate,
                receiver_agent_id=receiver_agent_id,
                proposal=proposal,
                card_snapshot=card_snapshot,
                selected_cards=critic_selected_cards,
                cards_by_id=cards_by_id,
                context=context,
                decision_index=position,
            )

            action, reason, mode, gate_decision = self._apply_decision_rules(estimate)
            accepted = action == "share"

            decision = self._make_decision(
                candidate=candidate,
                position=position,
                original_position=original_position_by_id[candidate.memory_id],
                action=action,
                reason=reason,
                decision_mode=mode,
                effect_condition_passed=gate_decision.effect_condition_passed,
                risk_condition_passed=gate_decision.risk_condition_passed,
                effect_condition_status=gate_decision.effect_condition_status,
                risk_condition_status=gate_decision.risk_condition_status,
                selected_before_actual=actual_selected_set,
                selected_before_critic=critic_selected_set,
                estimate=estimate,
                traversal_seed=traversal_seed,
                traversal_policy_name=self.traversal_policy.policy_name,
                proposal_order=proposal_order,
                traversal_order=traversal_order,
                permutation_indices=permutation_indices,
                accepted=accepted,
            )
            state.decisions.append(decision)

            # Update selected set if sharing
            if action == "share":
                state.selected_memory_ids.append(candidate.memory_id)
                state.share_count += 1
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
            traversal_seed=seed,
        )

        return result.decisions, result.selected_memory_ids

    def _estimate_transfer_effect(
        self,
        *,
        candidate: CandidateTrace,
        receiver_agent_id: str,
        proposal: CandidateProposal,
        card_snapshot: RoutingFeatureSnapshot,
        selected_cards: list[RoutingFeatureSnapshot],
        cards_by_id: dict[str, MemoryRoutingCard] | None,
        context: ContextFingerprint | None,
        decision_index: int | None,
    ) -> Any | None:
        """Get critic estimate for a candidate given current selected set."""
        del candidate, cards_by_id
        selected_memory_ids = [card.memory_id for card in selected_cards]
        context = self._context_for_selected_set(
            base_context=context,
            proposal=proposal,
            receiver_agent_id=receiver_agent_id,
            selected_memory_ids=selected_memory_ids,
            decision_index=decision_index,
        )

        prediction_input = TransferPredictionInput(
            context=context,
            candidate_card=card_snapshot,
            selected_cards=selected_cards,
        )

        return self.critic.predict_point(prediction_input)

    def _apply_decision_rules(
        self,
        estimate: Any | None,
    ) -> tuple[str, str, str, GateDecision]:
        """Apply the configured gate to determine share/withhold action.

        Returns:
            Tuple of (action, reason, decision_mode)
        """
        if estimate is None:
            raise RuntimeError("critic did not return a transfer estimate")

        gate_decision = self.gate.decide(estimate)
        if gate_decision.share:
            return (
                "share",
                gate_decision.reason,
                "safe_exploit",
                gate_decision,
            )
        mode = (
            "risk_veto"
            if gate_decision.reason
            in {"negative_risk_ucb_exceeded", "negative_risk_mean_exceeded"}
            else "ordinary_withhold"
        )
        return (
            "withhold",
            gate_decision.reason,
            mode,
            gate_decision,
        )

    def _make_decision(
        self,
        *,
        candidate: CandidateTrace,
        position: int,
        original_position: int | None,
        action: str,
        reason: str,
        decision_mode: str,
        effect_condition_passed: bool | None,
        risk_condition_passed: bool | None,
        effect_condition_status: str | None,
        risk_condition_status: str | None,
        selected_before_actual: tuple[str, ...] | list[str],
        selected_before_critic: tuple[str, ...] | list[str],
        estimate: TransferPointEstimate | None,
        traversal_seed: int | None,
        traversal_policy_name: str | None,
        proposal_order: list[str] | None,
        traversal_order: list[str] | None,
        permutation_indices: list[int] | None,
        accepted: bool,
    ) -> RouterDecision:
        """Create a router decision with critic estimates."""
        return RouterDecision(
            memory_id=candidate.memory_id,
            action=action,
            decision=action,
            score=candidate.total_score,
            reason=reason,
            candidate_position=position,
            decision_source="production_router",
            tau_mean=getattr(estimate, "tau_mean", None) if estimate else None,
            tau_lcb=getattr(estimate, "tau_lcb", None) if estimate else None,
            tau_ucb=getattr(estimate, "tau_ucb", None) if estimate else None,
            negative_risk_mean=(
                getattr(estimate, "negative_risk_mean", None) if estimate else None
            ),
            negative_risk_lcb=(
                getattr(estimate, "negative_risk_lcb", None) if estimate else None
            ),
            negative_risk_ucb=(
                getattr(estimate, "negative_risk_ucb", None) if estimate else None
            ),
            epsilon=_gate_negative_risk_budget(self.gate),
            accepted=accepted,
            decision_reason=reason,
            low_support=getattr(estimate, "low_support", None) if estimate else None,
            decision_mode=decision_mode,
            gate_name=self.gate.gate_name,
            conditioning_policy_name=self.conditioning_policy.policy_name,
            effect_condition_passed=effect_condition_passed,
            risk_condition_passed=risk_condition_passed,
            effect_condition_status=effect_condition_status,
            risk_condition_status=risk_condition_status,
            selected_before_actual=list(selected_before_actual),
            selected_before_critic=list(selected_before_critic),
            selected_before_actual_digest=selected_set_signature(
                list(selected_before_actual)
            ),
            selected_before_critic_digest=selected_set_signature(
                list(selected_before_critic)
            ),
            support_distance=(
                getattr(estimate, "support_distance", None) if estimate else None
            ),
            support_threshold=(
                getattr(estimate, "support_threshold", None) if estimate else None
            ),
            robust_diagnostics=_robust_diagnostics_from_estimate(estimate),
            original_candidate_position=original_position,
            traversal_position=position,
            traversal_seed=traversal_seed,
            traversal_policy_name=traversal_policy_name,
            proposal_order=proposal_order,
            traversal_order=traversal_order,
            permutation_indices=permutation_indices,
            proposal_rank=original_position + 1 if original_position is not None else None,
            proposal_score=candidate.total_score,
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

    def _context_for_selected_set(
        self,
        *,
        base_context: ContextFingerprint | None,
        proposal: CandidateProposal,
        receiver_agent_id: str,
        selected_memory_ids: list[str],
        decision_index: int | None,
    ) -> ContextFingerprint:
        """Build/update context so critic sees the current selected set."""
        if base_context is not None:
            return base_context.model_copy(
                update={
                    "selected_memory_ids": list(selected_memory_ids),
                    "selected_set_signature": selected_set_signature(selected_memory_ids),
                    "decision_index": decision_index,
                }
            )
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
            decision_index=decision_index,
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


def _gate_negative_risk_budget(gate: RoutingGate) -> float | None:
    config = getattr(gate, "config", None)
    budget = getattr(config, "negative_risk_budget", None)
    return float(budget) if budget is not None else None


def _cards_for_selected_ids(
    selected_ids: tuple[str, ...],
    *,
    current_cards: list[RoutingFeatureSnapshot],
    initial_cards: list[RoutingFeatureSnapshot],
) -> list[RoutingFeatureSnapshot]:
    cards_by_id = {card.memory_id: card for card in [*initial_cards, *current_cards]}
    return [cards_by_id[memory_id] for memory_id in selected_ids if memory_id in cards_by_id]


def _robust_diagnostics_from_estimate(
    estimate: TransferPointEstimate | None,
) -> dict[str, float] | None:
    if estimate is None:
        return None
    required = (
        "confidence_level",
        "tau_lcb",
        "tau_ucb",
        "negative_risk_lcb",
        "negative_risk_ucb",
    )
    if not all(hasattr(estimate, field) for field in required):
        return None
    return {field: float(getattr(estimate, field)) for field in required}
