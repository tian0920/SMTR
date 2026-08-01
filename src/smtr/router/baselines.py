"""Paper-required baselines and ablation methods."""

from __future__ import annotations

from typing import Any

from smtr.core.types import (
    CandidateExposureInput,
    MemoryRoutingCard,
    ReceiverState,
    RouterDecision,
)
from smtr.router.transfer_critic import FourOutcomeTransferCritic


class NoMemoryRouter:
    """B0-NoMemory: never share any memory."""

    def decide(
        self,
        receiver_state: ReceiverState,
        candidate_cards: list[MemoryRoutingCard],
        selected_prefix_cards: tuple[MemoryRoutingCard, ...] = (),
    ) -> list[RouterDecision]:
        return [
            RouterDecision(memory_id=c.memory_id, action="withhold", tau_hat=0.0, eta_hat=0.0, reason="no_memory_baseline")
            for c in candidate_cards
        ]


class Top1RelevanceRouter:
    """B1-Top1Relevance: share the top-1 most relevant candidate (by card similarity)."""

    def decide(
        self,
        receiver_state: ReceiverState,
        candidate_cards: list[MemoryRoutingCard],
        selected_prefix_cards: tuple[MemoryRoutingCard, ...] = (),
    ) -> list[RouterDecision]:
        if not candidate_cards:
            return []
        # Use first candidate as top-1 (assumes pre-sorted by relevance)
        top = candidate_cards[0]
        decisions = []
        for c in candidate_cards:
            if c.memory_id == top.memory_id:
                decisions.append(RouterDecision(memory_id=c.memory_id, action="share", tau_hat=0.0, eta_hat=0.0, reason="top1_relevance"))
            else:
                decisions.append(RouterDecision(memory_id=c.memory_id, action="withhold", tau_hat=0.0, eta_hat=0.0, reason="not_top1"))
        return decisions


class AllShareRouter:
    """B2-AllShare: share all candidates."""

    def decide(
        self,
        receiver_state: ReceiverState,
        candidate_cards: list[MemoryRoutingCard],
        selected_prefix_cards: tuple[MemoryRoutingCard, ...] = (),
    ) -> list[RouterDecision]:
        return [
            RouterDecision(memory_id=c.memory_id, action="share", tau_hat=0.0, eta_hat=0.0, reason="all_share_baseline")
            for c in candidate_cards
        ]


class FactualSuccessRouter:
    """B3-FactualSuccess: share only memories with sufficient evidence and success rate."""

    def __init__(self, min_evidence: int = 2, min_success_rate: float = 0.7) -> None:
        self.min_evidence = min_evidence
        self.min_success_rate = min_success_rate

    def decide(
        self,
        receiver_state: ReceiverState,
        candidate_cards: list[MemoryRoutingCard],
        selected_prefix_cards: tuple[MemoryRoutingCard, ...] = (),
    ) -> list[RouterDecision]:
        decisions = []
        for c in candidate_cards:
            if c.evidence_count >= self.min_evidence and c.historical_success_rate >= self.min_success_rate:
                decisions.append(RouterDecision(memory_id=c.memory_id, action="share", tau_hat=0.0, eta_hat=0.0, reason="factual_success_evidence"))
            else:
                decisions.append(RouterDecision(memory_id=c.memory_id, action="withhold", tau_hat=0.0, eta_hat=0.0, reason="insufficient_evidence"))
        return decisions


class SMTRNoRiskRouter:
    """SMTR-no-risk: use tau_hat only, ignore eta_hat (no risk constraint)."""

    def __init__(
        self,
        critic: FourOutcomeTransferCritic,
        max_shared_memories_per_receiver: int = 1,
    ) -> None:
        self.critic = critic
        self.max_shared_memories_per_receiver = max_shared_memories_per_receiver

    def decide(
        self,
        receiver_state: ReceiverState,
        candidate_cards: list[MemoryRoutingCard],
        selected_prefix_cards: tuple[MemoryRoutingCard, ...] = (),
    ) -> list[RouterDecision]:
        scored: list[tuple[float, float, MemoryRoutingCard]] = []
        for card in candidate_cards:
            exposure_input = CandidateExposureInput(
                receiver_state=receiver_state,
                candidate_card=card,
                selected_prefix_cards=selected_prefix_cards,
            )
            pred = self.critic.predict(exposure_input)
            scored.append((pred.tau_hat, pred.eta_hat, card))

        # Share top tau>0 without risk constraint
        positive = [(tau, eta, c) for tau, eta, c in scored if tau > 0]
        positive_sorted = sorted(positive, key=lambda x: -x[0])
        share_set = {c.memory_id for _, _, c in positive_sorted[:self.max_shared_memories_per_receiver]}

        decisions = []
        for tau, eta, card in scored:
            if card.memory_id in share_set:
                decisions.append(RouterDecision(memory_id=card.memory_id, action="share", tau_hat=tau, eta_hat=eta, reason="tau>0_no_risk_constraint"))
            else:
                decisions.append(RouterDecision(memory_id=card.memory_id, action="withhold", tau_hat=tau, eta_hat=eta, reason="tau<=0_or_not_top"))
        return decisions


class SMTRNoWriterReceiverRouter:
    """SMTR-no-writer-receiver: critic trained without writer-receiver features."""

    def __init__(
        self,
        critic: FourOutcomeTransferCritic,
        negative_risk_budget: float = 0.2,
        max_shared_memories_per_receiver: int = 1,
    ) -> None:
        self.critic = critic
        self.negative_risk_budget = negative_risk_budget
        self.max_shared_memories_per_receiver = max_shared_memories_per_receiver

    def decide(
        self,
        receiver_state: ReceiverState,
        candidate_cards: list[MemoryRoutingCard],
        selected_prefix_cards: tuple[MemoryRoutingCard, ...] = (),
    ) -> list[RouterDecision]:
        from smtr.router.exposure_router import SMTRExposureRouter

        # Same logic as SMTR but critic was trained with no_writer_receiver features
        router = SMTRExposureRouter(
            critic=self.critic,
            negative_risk_budget=self.negative_risk_budget,
            max_shared_memories_per_receiver=self.max_shared_memories_per_receiver,
        )
        return router.decide(receiver_state, candidate_cards, selected_prefix_cards)


# Method registry
METHOD_REGISTRY: dict[str, type] = {
    "b0_no_memory": NoMemoryRouter,
    "top1_relevance": Top1RelevanceRouter,
    "all_share": AllShareRouter,
    "factual_success": FactualSuccessRouter,
    "smtr_no_risk": SMTRNoRiskRouter,
    "smtr_no_writer_receiver": SMTRNoWriterReceiverRouter,
}
