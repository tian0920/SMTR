"""SMTR exposure router: receiver-specific share/withhold decisions."""

from __future__ import annotations

from typing import Any

from smtr.core.types import (
    CandidateExposureInput,
    MemoryRoutingCard,
    ReceiverState,
    RouterDecision,
)
from smtr.router.transfer_critic import FourOutcomeTransferCritic


class SMTRExposureRouter:
    """Receiver-specific exposure router using four-outcome transfer critic.

    Decision rule:
      For each candidate:
        pred = critic.predict(receiver_state, candidate_card, selected_prefix_cards)
        tau = pred.q10 - pred.q01
        eta = pred.q01
      Safe candidate: tau > 0 and eta <= negative_risk_budget
      Decision: share the safe candidate with largest tau; withhold all others.
      If no safe candidate: share nothing.
    """

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
        """Make share/withhold decisions for all candidates."""
        scored: list[tuple[float, float, MemoryRoutingCard]] = []
        for card in candidate_cards:
            exposure_input = CandidateExposureInput(
                receiver_state=receiver_state,
                candidate_card=card,
                selected_prefix_cards=selected_prefix_cards,
            )
            pred = self.critic.predict(exposure_input)
            tau = pred.tau_hat
            eta = pred.eta_hat
            scored.append((tau, eta, card))

        decisions: list[RouterDecision] = []
        # Find safe candidates
        safe = [(tau, eta, card) for tau, eta, card in scored if tau > 0 and eta <= self.negative_risk_budget]
        safe_sorted = sorted(safe, key=lambda x: -x[0])
        share_set = {card.memory_id for _, _, card in safe_sorted[:self.max_shared_memories_per_receiver]}

        for tau, eta, card in scored:
            if card.memory_id in share_set:
                decisions.append(RouterDecision(
                    memory_id=card.memory_id,
                    action="share",
                    tau_hat=tau,
                    eta_hat=eta,
                    reason="tau>0 and eta<=budget",
                ))
            else:
                reason = "no safe candidate" if not safe else "not top safe candidate"
                if tau <= 0:
                    reason = "tau<=0"
                elif eta > self.negative_risk_budget:
                    reason = "eta>budget"
                decisions.append(RouterDecision(
                    memory_id=card.memory_id,
                    action="withhold",
                    tau_hat=tau,
                    eta_hat=eta,
                    reason=reason,
                ))
        return decisions

    def trace(
        self,
        receiver_state: ReceiverState,
        candidate_cards: list[MemoryRoutingCard],
        selected_prefix_cards: tuple[MemoryRoutingCard, ...] = (),
    ) -> list[dict[str, Any]]:
        """Produce router trace (no payload/procedure included)."""
        decisions = self.decide(receiver_state, candidate_cards, selected_prefix_cards)
        traces = []
        for dec in decisions:
            card = next(c for c in candidate_cards if c.memory_id == dec.memory_id)
            traces.append({
                "receiver_agent_id": receiver_state.receiver.agent_id,
                "receiver_role": receiver_state.receiver.role,
                "candidate_memory_id": dec.memory_id,
                "writer_role": card.writer.role,
                "tau_hat": round(dec.tau_hat, 4),
                "eta_hat": round(dec.eta_hat, 4),
                "action": dec.action,
                "reason": dec.reason,
            })
        return traces
