"""Paper-required baselines and ablation methods (清单第十一章).

Main-table methods (SMTR-v1 single-memory setting):

* NoMemory            — never share any memory;
* RoleAwareTop1       — top-1 by task relevance + role compatibility +
                        capability/tool overlap, no paired transfer labels;
* AllShare            — always share the single highest-relevance memory
                        (v1 constraint: one memory per receiver episode);
* GlobalTransferCritic— critic with task/env/memory-card features only
                        (no writer, receiver or interaction features);
* SMTRNoPairInteraction — SMTR without writer-receiver interaction features;
* SMTRNoRisk          — SMTR decision with tau_hat > 0 only (no risk gate);
* SMTR                — full method.

FactualSuccess is deliberately not part of the main table until reliable
memory-level historical aggregates exist (清单 11).
"""

from __future__ import annotations

from typing import Any

from smtr.core.types import (
    CandidateExposureInput,
    MemoryRoutingCard,
    ReceiverState,
    RouterDecision,
)
from smtr.router.transfer_features import _overlap_bucket, _text_tokens
from smtr.router.transfer_critic import FourOutcomeTransferCritic


def _heuristic_relevance_score(receiver_state: ReceiverState, card: MemoryRoutingCard) -> float:
    """Task relevance between the receiver task context and the memory card.

    Uses only pre-execution metadata (task tags, goal tokens, scenario);
    never paired transfer labels.
    """
    rs = receiver_state
    task_tokens = set(_text_tokens(rs.task_instruction)) | {tok.lower() for tok in rs.task_id.split("_")}
    card_tokens = set(_text_tokens(card.goal_summary)) | {tok.lower() for tok in card.task_tags}
    if not task_tokens or not card_tokens:
        return 0.0
    return len(task_tokens & card_tokens) / len(task_tokens | card_tokens)


def _role_compatibility_score(receiver_state: ReceiverState, card: MemoryRoutingCard) -> float:
    role = receiver_state.receiver.role
    if role in card.incompatible_receiver_roles:
        return 0.0
    if card.compatible_receiver_roles and role in card.compatible_receiver_roles:
        return 1.0
    return 0.5


def role_aware_top1_score(receiver_state: ReceiverState, card: MemoryRoutingCard) -> float:
    """Combined heuristic score for the label-free baselines."""
    receiver = receiver_state.receiver
    cap_overlap = _overlap_bucket(set(card.writer.capabilities), set(receiver.capabilities))
    tool_overlap = _overlap_bucket(set(card.writer.tool_names), set(receiver.tool_names))
    overlap_bonus = {"high": 0.2, "medium": 0.1}.get(cap_overlap, 0.0)
    overlap_bonus += {"high": 0.2, "medium": 0.1}.get(tool_overlap, 0.0)
    return (
        _heuristic_relevance_score(receiver_state, card)
        + _role_compatibility_score(receiver_state, card)
        + overlap_bonus
    )


def _select_top1(receiver_state: ReceiverState, candidate_cards: list[MemoryRoutingCard]) -> str:
    scored = sorted(
        candidate_cards,
        key=lambda c: (-role_aware_top1_score(receiver_state, c), c.memory_id),
    )
    return scored[0].memory_id


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


class RoleAwareTop1Router:
    """RoleAwareTop1: share the top-1 candidate by task relevance, role
    compatibility and capability/tool overlap. No paired transfer labels."""

    def decide(
        self,
        receiver_state: ReceiverState,
        candidate_cards: list[MemoryRoutingCard],
        selected_prefix_cards: tuple[MemoryRoutingCard, ...] = (),
    ) -> list[RouterDecision]:
        if not candidate_cards:
            return []
        top_id = _select_top1(receiver_state, candidate_cards)
        decisions = []
        for c in candidate_cards:
            if c.memory_id == top_id:
                decisions.append(RouterDecision(memory_id=c.memory_id, action="share", tau_hat=0.0, eta_hat=0.0, reason="role_aware_top1"))
            else:
                decisions.append(RouterDecision(memory_id=c.memory_id, action="withhold", tau_hat=0.0, eta_hat=0.0, reason="not_top1"))
        return decisions


class AllShareRouter:
    """AllShare in the SMTR-v1 single-memory setting.

    Always shares exactly one memory — the highest-relevance candidate
    (same label-free score as RoleAwareTop1). Sharing all candidates is
    forbidden in v1 because one receiver episode receives one treatment.
    """

    def decide(
        self,
        receiver_state: ReceiverState,
        candidate_cards: list[MemoryRoutingCard],
        selected_prefix_cards: tuple[MemoryRoutingCard, ...] = (),
    ) -> list[RouterDecision]:
        if not candidate_cards:
            return []
        top_id = _select_top1(receiver_state, candidate_cards)
        decisions = []
        for c in candidate_cards:
            if c.memory_id == top_id:
                decisions.append(RouterDecision(memory_id=c.memory_id, action="share", tau_hat=0.0, eta_hat=0.0, reason="all_share_top1_relevance"))
            else:
                decisions.append(RouterDecision(memory_id=c.memory_id, action="withhold", tau_hat=0.0, eta_hat=0.0, reason="all_share_single_memory_limit"))
        return decisions


class FactualSuccessRouter:
    """B3-FactualSuccess: share only memories with sufficient evidence and success rate.

    Kept for legacy pipelines only; removed from the main table until
    reliable memory-level historical aggregates exist.
    """

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


def _critic_router(
    critic: FourOutcomeTransferCritic,
    expected_feature_block: str,
    negative_risk_budget: float,
    max_shared_memories_per_receiver: int = 1,
):
    from smtr.router.exposure_router import SMTRExposureRouter

    assert critic.feature_block == expected_feature_block, (
        f"{expected_feature_block} method requires a critic trained with "
        f"feature_block='{expected_feature_block}', got '{critic.feature_block}'"
    )
    return SMTRExposureRouter(
        critic=critic,
        negative_risk_budget=negative_risk_budget,
        max_shared_memories_per_receiver=max_shared_memories_per_receiver,
    )


class GlobalTransferCriticRouter:
    """GlobalTransferCritic: tau^global(m | task) without writer/receiver.

    The critic sees task, environment and memory-card features only
    (feature_block='memory_task_only'); decisions follow the SMTR rule so
    the comparison isolates receiver conditioning.
    """

    def __init__(
        self,
        critic: FourOutcomeTransferCritic,
        negative_risk_budget: float = 0.2,
        max_shared_memories_per_receiver: int = 1,
    ) -> None:
        self._router = _critic_router(critic, "memory_task_only", negative_risk_budget, max_shared_memories_per_receiver)

    def decide(
        self,
        receiver_state: ReceiverState,
        candidate_cards: list[MemoryRoutingCard],
        selected_prefix_cards: tuple[MemoryRoutingCard, ...] = (),
    ) -> list[RouterDecision]:
        return self._router.decide(receiver_state, candidate_cards, selected_prefix_cards)


class SMTRNoPairInteractionRouter:
    """SMTRNoPairInteraction: writer+receiver marginals, no pair interaction."""

    def __init__(
        self,
        critic: FourOutcomeTransferCritic,
        negative_risk_budget: float = 0.2,
        max_shared_memories_per_receiver: int = 1,
    ) -> None:
        self._router = _critic_router(critic, "no_pair_interaction", negative_risk_budget, max_shared_memories_per_receiver)

    def decide(
        self,
        receiver_state: ReceiverState,
        candidate_cards: list[MemoryRoutingCard],
        selected_prefix_cards: tuple[MemoryRoutingCard, ...] = (),
    ) -> list[RouterDecision]:
        return self._router.decide(receiver_state, candidate_cards, selected_prefix_cards)


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
    """Legacy SMTR-no-writer-receiver (kept for old checkpoint compatibility)."""

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
    "role_aware_top1": RoleAwareTop1Router,
    "all_share": AllShareRouter,
    "factual_success": FactualSuccessRouter,
    "global_transfer_critic": GlobalTransferCriticRouter,
    "smtr_no_pair_interaction": SMTRNoPairInteractionRouter,
    "smtr_no_risk": SMTRNoRiskRouter,
    "smtr_no_writer_receiver": SMTRNoWriterReceiverRouter,
}
