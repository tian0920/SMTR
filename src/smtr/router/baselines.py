"""Paper-required baselines and ablation methods (清单 P0-2).

Main-table methods (清单 Writer-Agnostic 第九章; SMTR-v1
single-memory setting):

* B0-NoMemory            — never expose any memory;
* B1-SemanticTop1        — top-1 by task-memory semantic similarity only;
                           ignores receiver, requirements and the critic;
* B2-ReceiverCompatibleTop1 — semantic relevance plus explicit memory
                           requirement satisfaction, no paired labels;
* B3-GlobalTransferCritic— critic on q(Y | task, m): no receiver marginal
                           and no compatibility interaction;
* B4-SMTR-no-compatibility-interaction — task/memory/receiver marginals,
                           no explicit compatibility interactions;
* SMTR                   — full memory-receiver critic with pure
                           tau_hat > 0 selective exposure.

Writer identity is never a conditioning variable in any baseline (清单
Writer-Agnostic 第二章). AllShare and FactualSuccess were removed: in the
v1 single-memory action space AllShare is behaviorally identical to a
top-1 heuristic baseline, and FactualSuccess has no reliable memory-level
historical aggregates.

SMTR-no-risk was removed: the new SMTR-v1 definition uses tau > 0 as the
only selective-exposure gate, so SMTR-no-risk is behaviorally identical
to SMTR.
"""

from __future__ import annotations

from typing import Any

from smtr.core.types import (
    CandidateExposureInput,
    MemoryRoutingCard,
    ReceiverState,
    RouterDecision,
)
from smtr.router.transfer_critic import FourOutcomeTransferCritic
from smtr.router.transfer_features import _text_tokens


def _heuristic_relevance_score(receiver_state: ReceiverState, card: MemoryRoutingCard) -> float:
    """Task relevance between the receiver task context and the memory card.

    Uses only pre-execution metadata (task tags, goal tokens, scenario);
    never paired transfer labels.
    """
    rs = receiver_state
    task_tokens = set(_text_tokens(rs.task_instruction)) | {
        tok.lower() for tok in rs.task_id.split("_")
    }
    card_tokens = set(_text_tokens(card.goal_summary)) | {tok.lower() for tok in card.task_tags}
    if not task_tokens or not card_tokens:
        return 0.0
    return len(task_tokens & card_tokens) / len(task_tokens | card_tokens)


def _requirement_satisfaction(required: set[str], available: set[str]) -> float:
    """Fraction of explicit memory requirements satisfied by the receiver."""
    if not required:
        return 1.0
    return len(required & available) / len(required)


def memory_receiver_compatibility(
    card: MemoryRoutingCard,
    receiver_state: ReceiverState,
) -> dict[str, float]:
    """Memory-requirement vs receiver-state satisfaction (清单 8.3).

    Derived from the routing card and pre-execution receiver state only;
    writer identity never participates.
    """
    rs = receiver_state
    r_caps = set(rs.receiver.capabilities)
    r_tools = set(rs.receiver.tool_names)
    r_env = set(rs.environment_signature)

    if card.execution_role_tags:
        role_satisfaction = (
            1.0 if rs.receiver.role in card.execution_role_tags else 0.0
        )
    else:
        # Unspecified execution role: no evidence of a role constraint.
        role_satisfaction = 0.5

    return {
        "tool_satisfaction": _requirement_satisfaction(
            set(card.required_tools), r_tools
        ),
        "capability_satisfaction": _requirement_satisfaction(
            set(card.required_capabilities), r_caps
        ),
        "environment_satisfaction": _requirement_satisfaction(
            set(card.environment_constraints), r_env
        ),
        "role_satisfaction": role_satisfaction,
    }


def receiver_compatible_top1_score(
    receiver_state: ReceiverState,
    card: MemoryRoutingCard,
) -> float:
    """Combined heuristic score for the label-free B2 baseline (清单 8.3).

    Mean of task relevance and the four explicit requirement satisfactions;
    never paired transfer labels and never writer identity.
    """
    compatibility = memory_receiver_compatibility(card, receiver_state)
    return sum([
        _heuristic_relevance_score(receiver_state, card),
        compatibility["tool_satisfaction"],
        compatibility["capability_satisfaction"],
        compatibility["environment_satisfaction"],
        compatibility["role_satisfaction"],
    ]) / 5.0


def _select_top1(receiver_state: ReceiverState, candidate_cards: list[MemoryRoutingCard]) -> str:
    scored = sorted(
        candidate_cards,
        key=lambda c: (-receiver_compatible_top1_score(receiver_state, c), c.memory_id),
    )
    return scored[0].memory_id


class NoMemoryRouter:
    """B0-NoMemory: never share any memory."""

    def decide(
        self,
        receiver_state: ReceiverState,
        candidate_cards: list[MemoryRoutingCard],
        candidate_context: dict[str, dict[str, Any]] | None = None,
    ) -> list[RouterDecision]:
        return [
            RouterDecision(
                memory_id=c.memory_id, action="withhold",
                tau_hat=0.0, eta_raw=0.0, reason="no_memory_baseline",
            )
            for c in candidate_cards
        ]


class SemanticTop1Router:
    """B1-SemanticTop1: share the top-1 candidate by task-memory semantic
    similarity only.

    Deliberately ignores receiver profile, memory requirements, paired
    transfer labels and the transfer critic; the top-1 candidate is always
    exposed.
    """

    def decide(
        self,
        receiver_state: ReceiverState,
        candidate_cards: list[MemoryRoutingCard],
        candidate_context: dict[str, dict[str, Any]] | None = None,
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
                decisions.append(RouterDecision(
                    memory_id=c.memory_id, action="share",
                    tau_hat=0.0, eta_raw=0.0, reason="semantic_top1",
                ))
            else:
                decisions.append(RouterDecision(
                    memory_id=c.memory_id, action="withhold",
                    tau_hat=0.0, eta_raw=0.0, reason="not_semantic_top1",
                ))
        return decisions


class ReceiverCompatibleTop1Router:
    """B2-ReceiverCompatibleTop1: share the top-1 candidate by semantic
    relevance plus explicit memory requirement satisfaction.

    No paired transfer labels; writer identity is never used (清单
    Writer-Agnostic 8.3)."""

    def decide(
        self,
        receiver_state: ReceiverState,
        candidate_cards: list[MemoryRoutingCard],
        candidate_context: dict[str, dict[str, Any]] | None = None,
    ) -> list[RouterDecision]:
        if not candidate_cards:
            return []
        top_id = _select_top1(receiver_state, candidate_cards)
        decisions = []
        for c in candidate_cards:
            if c.memory_id == top_id:
                decisions.append(RouterDecision(
                    memory_id=c.memory_id, action="share",
                    tau_hat=0.0, eta_raw=0.0,
                    reason="receiver_compatible_top1",
                ))
            else:
                decisions.append(RouterDecision(
                    memory_id=c.memory_id, action="withhold",
                    tau_hat=0.0, eta_raw=0.0,
                    reason="not_receiver_compatible_top1",
                ))
        return decisions


def _critic_router(
    critic: FourOutcomeTransferCritic,
    expected_feature_blocks: str | tuple[str, ...],
    max_shared_memories_per_receiver: int = 1,
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
        max_shared_memories_per_receiver=max_shared_memories_per_receiver,
    )


class GlobalTransferCriticRouter:
    """B3-GlobalTransferCritic: q(Y^share, Y^withhold | task, m).

    Removes the receiver marginal and the compatibility interaction, used
    to show receiver conditioning is necessary. Requires
    feature_block='global_transfer'.
    """

    def __init__(
        self,
        critic: FourOutcomeTransferCritic,
        max_shared_memories_per_receiver: int = 1,
    ) -> None:
        self._router = _critic_router(
            critic, "global_transfer",
            max_shared_memories_per_receiver,
        )

    def decide(
        self,
        receiver_state: ReceiverState,
        candidate_cards: list[MemoryRoutingCard],
        candidate_context: dict[str, dict[str, Any]] | None = None,
    ) -> list[RouterDecision]:
        return self._router.decide(receiver_state, candidate_cards, candidate_context)


class SMTRNoCompatibilityInteractionRouter:
    """B4-SMTR-no-compatibility-interaction: task/memory/receiver marginals
    without the explicit compatibility interaction block (清单 8.4)."""

    def __init__(
        self,
        critic: FourOutcomeTransferCritic,
        max_shared_memories_per_receiver: int = 1,
    ) -> None:
        self._router = _critic_router(
            critic, "no_compatibility_interaction",
            max_shared_memories_per_receiver,
        )

    def decide(
        self,
        receiver_state: ReceiverState,
        candidate_cards: list[MemoryRoutingCard],
        candidate_context: dict[str, dict[str, Any]] | None = None,
    ) -> list[RouterDecision]:
        return self._router.decide(receiver_state, candidate_cards, candidate_context)


# Formal method registry (清单 Writer-Agnostic 第九章). AllShare and
# FactualSuccess are deliberately absent: AllShare duplicates a top-1
# heuristic baseline under the v1 single-memory action space, and
# FactualSuccess lacks reliable memory-level historical aggregates.
# smtr_no_writer_receiver was removed because the new full SMTR never
# conditions on writer identity (清单 8.7).
# smtr_no_risk was removed: SMTR-v1 uses tau > 0 as the only gate,
# making SMTR-no-risk behaviorally identical to SMTR.
METHOD_REGISTRY: dict[str, type] = {
    "b0_no_memory": NoMemoryRouter,
    "semantic_top1": SemanticTop1Router,
    "receiver_compatible_top1": ReceiverCompatibleTop1Router,
    "global_transfer_critic": GlobalTransferCriticRouter,
    "smtr_no_compatibility_interaction": SMTRNoCompatibilityInteractionRouter,
}

# 清单最终闭环 §22: the formal method set is defined once here so the CLI
# never hard-codes a second, diverging collection. "smtr" is appended even
# though it lives in smtr.router.exposure_router (it needs live critic
# instances rather than the zero-arg construction METHOD_REGISTRY assumes).
FORMAL_METHOD_NAMES: tuple[str, ...] = (
    *tuple(METHOD_REGISTRY),
    "smtr",
)
