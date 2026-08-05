"""Paper-required baselines and ablation methods (清单 P0-2).

Main-table methods (SMTR-v1 single-memory setting):

* NoMemory            — never share any memory;
* SemanticTop1        — top-1 by task-memory semantic similarity only;
                        ignores writer, receiver, role, capability,
                        environment and the transfer critic;
* RoleAwareTop1       — top-1 by task relevance + role compatibility +
                        capability/tool overlap, no paired transfer labels;
* GlobalTransferCritic— critic with task/env/memory-card features only
                        (no writer, receiver or interaction features);
* SMTRNoPairInteraction — SMTR without writer-receiver interaction features;
* SMTRNoRisk          — SMTR decision with tau_hat > 0 only (no risk gate);
* SMTR                — full method.

AllShare and FactualSuccess were removed: in the v1 single-memory action
space AllShare is behaviorally identical to a top-1 heuristic baseline,
and FactualSuccess has no reliable memory-level historical aggregates.
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
from smtr.router.exposure_router import SMTRUCBRouter


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
    """Role compatibility from observable writer/receiver roles only.

    Deprecated human-authored ``compatible_receiver_roles`` /
    ``incompatible_receiver_roles`` card fields are deliberately ignored so
    the baseline never benefits from manually pre-annotated transfer hints.
    """
    writer_role = card.writer.role
    receiver_role = receiver_state.receiver.role
    if writer_role == receiver_role:
        return 1.0
    if writer_role in ("unknown", "") or receiver_role in ("unknown", ""):
        return 0.5
    return 0.25


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
            RouterDecision(memory_id=c.memory_id, action="withhold", tau_hat=0.0, eta_raw=0.0, reason="no_memory_baseline")
            for c in candidate_cards
        ]


class SemanticTop1Router:
    """B1-SemanticTop1: share the top-1 candidate by task-memory semantic
    similarity only.

    Deliberately ignores writer identity, receiver role, capability/tool
    compatibility, paired transfer labels and the transfer critic; the
    top-1 candidate is always exposed.
    """

    def decide(
        self,
        receiver_state: ReceiverState,
        candidate_cards: list[MemoryRoutingCard],
        selected_prefix_cards: tuple[MemoryRoutingCard, ...] = (),
    ) -> list[RouterDecision]:
        if not candidate_cards:
            return []
        scored = sorted(
            candidate_cards,
            key=lambda c: (-_heuristic_relevance_score(receiver_state, c), c.memory_id),
        )
        top_id = scored[0].memory_id
        decisions = []
        for c in candidate_cards:
            if c.memory_id == top_id:
                decisions.append(RouterDecision(memory_id=c.memory_id, action="share", tau_hat=0.0, eta_raw=0.0, reason="semantic_top1"))
            else:
                decisions.append(RouterDecision(memory_id=c.memory_id, action="withhold", tau_hat=0.0, eta_raw=0.0, reason="not_semantic_top1"))
        return decisions


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
                decisions.append(RouterDecision(memory_id=c.memory_id, action="share", tau_hat=0.0, eta_raw=0.0, reason="role_aware_top1"))
            else:
                decisions.append(RouterDecision(memory_id=c.memory_id, action="withhold", tau_hat=0.0, eta_raw=0.0, reason="not_top1"))
        return decisions


def _critic_router(
    critic: FourOutcomeTransferCritic,
    expected_feature_blocks: str | tuple[str, ...],
    negative_risk_budget: float | None = None,
    max_shared_memories_per_receiver: int = 1,
    allow_risk_budget_override: bool = False,
):
    from smtr.router.exposure_router import SMTRExposureRouter

    if isinstance(expected_feature_blocks, str):
        allowed = (expected_feature_blocks,)
    else:
        allowed = tuple(expected_feature_blocks)

    if critic.feature_block not in allowed:
        raise ValueError(
            "critic feature block mismatch: "
            f"expected one of {allowed}, "
            f"got {critic.feature_block!r}"
        )

    return SMTRExposureRouter(
        critic=critic,
        negative_risk_budget=negative_risk_budget,
        max_shared_memories_per_receiver=max_shared_memories_per_receiver,
        allow_risk_budget_override=allow_risk_budget_override,
    )


class GlobalTransferCriticRouter:
    """GlobalTransferCritic.

    Uses task, environment and memory-card features, but removes writer,
    receiver and pair-interaction features. Requires
    feature_block='global_transfer'.
    """

    def __init__(
        self,
        critic: FourOutcomeTransferCritic,
        negative_risk_budget: float | None = None,
        max_shared_memories_per_receiver: int = 1,
        allow_risk_budget_override: bool = False,
    ) -> None:
        self._router = _critic_router(
            critic, "global_transfer", negative_risk_budget,
            max_shared_memories_per_receiver, allow_risk_budget_override,
        )

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
        negative_risk_budget: float | None = None,
        max_shared_memories_per_receiver: int = 1,
        allow_risk_budget_override: bool = False,
    ) -> None:
        self._router = _critic_router(
            critic, "no_pair_interaction", negative_risk_budget,
            max_shared_memories_per_receiver, allow_risk_budget_override,
        )

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
        from smtr.router.exposure_router import _require_single_memory_action_space

        _require_single_memory_action_space(max_shared_memories_per_receiver)
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
                decisions.append(RouterDecision(memory_id=card.memory_id, action="share", tau_hat=tau, eta_raw=eta, reason="tau>0_no_risk_constraint"))
            else:
                decisions.append(RouterDecision(memory_id=card.memory_id, action="withhold", tau_hat=tau, eta_raw=eta, reason="tau<=0_or_not_top"))
        return decisions


class SMTRNoWriterReceiverRouter:
    """Legacy SMTR-no-writer-receiver (kept for old checkpoint compatibility)."""

    def __init__(
        self,
        critic: FourOutcomeTransferCritic,
        negative_risk_budget: float | None = None,
        max_shared_memories_per_receiver: int = 1,
        allow_risk_budget_override: bool = False,
    ) -> None:
        self.critic = critic
        self.negative_risk_budget = negative_risk_budget
        self.allow_risk_budget_override = allow_risk_budget_override
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
            allow_risk_budget_override=self.allow_risk_budget_override,
        )
        return router.decide(receiver_state, candidate_cards, selected_prefix_cards)


# Formal method registry (清单 P0-2). AllShare and FactualSuccess are
# deliberately absent: AllShare duplicates a top-1 heuristic baseline under
# the v1 single-memory action space, and FactualSuccess lacks reliable
# memory-level historical aggregates.
METHOD_REGISTRY: dict[str, type] = {
    "b0_no_memory": NoMemoryRouter,
    "semantic_top1": SemanticTop1Router,
    "role_aware_top1": RoleAwareTop1Router,
    "global_transfer_critic": GlobalTransferCriticRouter,
    "smtr_no_pair_interaction": SMTRNoPairInteractionRouter,
    "smtr_no_risk": SMTRNoRiskRouter,
    "smtr_no_writer_receiver": SMTRNoWriterReceiverRouter,
    "smtr_ucb": SMTRUCBRouter,
}
