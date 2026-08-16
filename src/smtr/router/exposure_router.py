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


def _require_single_memory_action_space(max_shared_memories_per_receiver: int) -> None:
    """SMTR-v1 action space is A(o_r) in {∅, m_1, ..., m_K}.

    Each receiver may be exposed to at most one memory; the parameter stays
    in signatures for compatibility but any value other than 1 is rejected
    so no experiment can silently use a multi-memory action space.
    """
    if max_shared_memories_per_receiver != 1:
        raise ValueError(
            "SMTR-v1 fixes the action space to single-memory exposure: "
            "max_shared_memories_per_receiver must be 1, got "
            f"{max_shared_memories_per_receiver}."
        )


class SMTRExposureRouter:
    """Receiver-specific exposure router using four-outcome transfer critic.

    Decision rule (SMTR-v1):
      For each candidate:
        pred = critic.predict_calibrated(receiver_state, candidate_card)
        tau = pred.tau_hat  (= q10 - q01, net transfer effect)
        eta = pred.eta_hat_calibrated  (= q01, diagnostic only)
      Select m* = argmax tau among candidates with tau > 0.
      If tau(m*) > 0: expose m*. Otherwise: withhold all.

    eta (= q01, harmful-transfer probability) is reported for diagnostics
    but never used as a routing gate. No epsilon* risk threshold.
    """

    def __init__(
        self,
        critic: FourOutcomeTransferCritic,
        max_shared_memories_per_receiver: int = 1,
    ) -> None:
        _require_single_memory_action_space(max_shared_memories_per_receiver)
        self.critic = critic
        self.max_shared_memories_per_receiver = max_shared_memories_per_receiver

    def decide(
        self,
        receiver_state: ReceiverState,
        candidate_cards: list[MemoryRoutingCard],
        candidate_context: dict[str, dict[str, Any]] | None = None,
    ) -> list[RouterDecision]:
        """Make share/withhold decisions for all candidates.

        SMTR-v1: pure tau selective exposure. eta is diagnostic only.
        """
        scored: list[tuple[float, float, float, MemoryRoutingCard]] = []
        for card in candidate_cards:
            exposure_input = CandidateExposureInput(
                receiver_state=receiver_state,
                candidate_card=card,
            )
            pred = self.critic.predict_calibrated(exposure_input)
            tau = pred.tau_hat
            eta = float(pred.eta_hat_calibrated)
            raw = float(getattr(pred, "eta_hat_raw", pred.eta_hat))
            scored.append((tau, eta, raw, card))

        decisions: list[RouterDecision] = []
        # Find candidates with positive net transfer effect
        positive = [
            (tau, eta, raw, card)
            for tau, eta, raw, card in scored
            if tau > 0
        ]
        positive_sorted = sorted(positive, key=lambda x: -x[0])
        share_set = {
            card.memory_id
            for *_, card in positive_sorted[:self.max_shared_memories_per_receiver]
        }

        for tau, eta, raw, card in scored:
            if card.memory_id in share_set:
                decisions.append(RouterDecision(
                    memory_id=card.memory_id,
                    action="share",
                    tau_hat=tau,
                    eta_raw=raw,
                    eta_calibrated=eta,
                    reason="tau>0",
                ))
            else:
                reason = "no positive tau candidate" if not positive else "not top tau candidate"
                if tau <= 0:
                    reason = "tau<=0"
                decisions.append(RouterDecision(
                    memory_id=card.memory_id,
                    action="withhold",
                    tau_hat=tau,
                    eta_raw=raw,
                    eta_calibrated=eta,
                    reason=reason,
                ))
        return decisions

    def trace(
        self,
        receiver_state: ReceiverState,
        candidate_cards: list[MemoryRoutingCard],
        candidate_context: dict[str, dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Produce router trace (no payload/procedure included)."""
        decisions = self.decide(receiver_state, candidate_cards, candidate_context)
        traces = []
        for dec in decisions:
            card = next(c for c in candidate_cards if c.memory_id == dec.memory_id)
            traces.append({
                "receiver_agent_id": receiver_state.receiver.agent_id,
                "receiver_role": receiver_state.receiver.role,
                "candidate_memory_id": dec.memory_id,
                # 清单 Writer-Agnostic 11.3: explicit memory requirements
                # replace writer identity in traces.
                "memory_required_tools": list(card.required_tools),
                "memory_required_capabilities": list(card.required_capabilities),
                "memory_execution_role_tags": list(card.execution_role_tags),
                "tau_hat": round(dec.tau_hat, 4),
                "eta_raw": round(dec.eta_raw, 4),
                "eta_calibrated": (
                    round(dec.eta_calibrated, 4)
                    if dec.eta_calibrated is not None
                    else None
                ),
                "action": dec.action,
                "reason": dec.reason,
            })
        return traces
