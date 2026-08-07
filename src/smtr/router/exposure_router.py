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

    Decision rule (清单第三章):
      For each candidate:
        pred = critic.predict_calibrated(receiver_state, candidate_card, ...)
        tau = pred.tau_hat
        eta = pred.eta_hat_calibrated   (never the raw eta)
      Safe candidate: tau > 0 and eta <= epsilon_star
      Decision: share the safe candidate with largest tau; withhold all others.
      If no safe candidate: share nothing.

    The risk budget epsilon_star is read from the critic checkpoint
    (selected on validation, never re-selected on test). An explicit
    ``negative_risk_budget`` overrides the checkpoint only in debug mode.
    """

    def __init__(
        self,
        critic: FourOutcomeTransferCritic,
        negative_risk_budget: float | None = None,
        max_shared_memories_per_receiver: int = 1,
        allow_risk_budget_override: bool = False,
    ) -> None:
        if negative_risk_budget is not None and not allow_risk_budget_override:
            raise ValueError(
                "An explicit negative_risk_budget overrides the checkpoint "
                "epsilon_star and is only allowed in debug mode "
                "(allow_risk_budget_override=True)."
            )
        _require_single_memory_action_space(max_shared_memories_per_receiver)
        self.critic = critic
        self.negative_risk_budget = negative_risk_budget
        self.allow_risk_budget_override = allow_risk_budget_override
        self.max_shared_memories_per_receiver = max_shared_memories_per_receiver

    def _effective_risk_budget(self) -> float:
        """Risk budget for decisions: checkpoint epsilon_star unless overridden."""
        if self.negative_risk_budget is not None:
            return float(self.negative_risk_budget)
        epsilon_star = getattr(self.critic, "epsilon_star", None)
        if epsilon_star is None:
            raise ValueError(
                "Checkpoint does not contain validation-selected epsilon_star."
            )
        return float(epsilon_star)

    def decide(
        self,
        receiver_state: ReceiverState,
        candidate_cards: list[MemoryRoutingCard],
    ) -> list[RouterDecision]:
        """Make share/withhold decisions for all candidates."""
        budget = self._effective_risk_budget()
        scored: list[tuple[float, float, float, MemoryRoutingCard]] = []
        for card in candidate_cards:
            exposure_input = CandidateExposureInput(
                receiver_state=receiver_state,
                candidate_card=card,
            )
            pred = self.critic.predict_calibrated(exposure_input)
            tau = pred.tau_hat
            eta = float(pred.eta_hat_calibrated)
            # eta_hat is the raw q01 estimand; kept as fallback for fakes
            # that expose the raw value only through eta_hat.
            raw = float(getattr(pred, "eta_hat_raw", pred.eta_hat))
            scored.append((tau, eta, raw, card))

        decisions: list[RouterDecision] = []
        # Find safe candidates
        safe = [
            (tau, eta, raw, card)
            for tau, eta, raw, card in scored
            if tau > 0 and eta <= budget
        ]
        safe_sorted = sorted(safe, key=lambda x: -x[0])
        share_set = {card.memory_id for *_, card in safe_sorted[:self.max_shared_memories_per_receiver]}

        for tau, eta, raw, card in scored:
            if card.memory_id in share_set:
                decisions.append(RouterDecision(
                    memory_id=card.memory_id,
                    action="share",
                    tau_hat=tau,
                    eta_raw=raw,
                    eta_calibrated=eta,
                    risk_budget=budget,
                    reason="tau>0 and eta_calibrated<=epsilon_star",
                ))
            else:
                reason = "no safe candidate" if not safe else "not top safe candidate"
                if tau <= 0:
                    reason = "tau<=0"
                elif eta > budget:
                    reason = "eta_calibrated>epsilon_star"
                decisions.append(RouterDecision(
                    memory_id=card.memory_id,
                    action="withhold",
                    tau_hat=tau,
                    eta_raw=raw,
                    eta_calibrated=eta,
                    risk_budget=budget,
                    reason=reason,
                ))
        return decisions

    def trace(
        self,
        receiver_state: ReceiverState,
        candidate_cards: list[MemoryRoutingCard],
    ) -> list[dict[str, Any]]:
        """Produce router trace (no payload/procedure included)."""
        decisions = self.decide(receiver_state, candidate_cards)
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
                "risk_budget": dec.risk_budget,
                "action": dec.action,
                "reason": dec.reason,
            })
        return traces


class SMTRUCBRouter:
    """SMTR-UCB: uncertainty-aware ablation variant (清单第九章 9.2).

    Uses the bootstrap-ensemble distribution instead of point estimates:
    share iff tau_lower > 0 and eta_upper <= epsilon_star, where
    tau_lower = quantile(member_tau, 0.10) and
    eta_upper = quantile(member_eta, 0.90). Among safe candidates the one
    with the largest ensemble-mean tau is shared (v1 single memory).
    This is an additional ablation; it never replaces the main SMTR rule.
    """

    def __init__(
        self,
        critic: FourOutcomeTransferCritic,
        negative_risk_budget: float | None = None,
        max_shared_memories_per_receiver: int = 1,
        allow_risk_budget_override: bool = False,
    ) -> None:
        if negative_risk_budget is not None and not allow_risk_budget_override:
            raise ValueError(
                "An explicit negative_risk_budget overrides the checkpoint "
                "epsilon_star and is only allowed in debug mode "
                "(allow_risk_budget_override=True)."
            )
        _require_single_memory_action_space(max_shared_memories_per_receiver)
        self.critic = critic
        self.negative_risk_budget = negative_risk_budget
        self.max_shared_memories_per_receiver = max_shared_memories_per_receiver

    def _effective_risk_budget(self) -> float:
        if self.negative_risk_budget is not None:
            return float(self.negative_risk_budget)
        epsilon_star = getattr(self.critic, "epsilon_star", None)
        if epsilon_star is None:
            raise ValueError(
                "Checkpoint does not contain validation-selected epsilon_star."
            )
        return float(epsilon_star)

    def decide(
        self,
        receiver_state: ReceiverState,
        candidate_cards: list[MemoryRoutingCard],
    ) -> list[RouterDecision]:
        budget = self._effective_risk_budget()
        scored: list[tuple[float, float, float, float, MemoryRoutingCard]] = []
        for card in candidate_cards:
            exposure_input = CandidateExposureInput(
                receiver_state=receiver_state,
                candidate_card=card,
            )
            dist = self.critic.predict_distribution(exposure_input)
            scored.append((
                dist.mean.tau_hat, dist.mean.eta_hat,
                dist.tau_lower, dist.eta_upper, card,
            ))

        safe = [
            (tau_mean, eta_mean, tau_lower, eta_upper, card)
            for tau_mean, eta_mean, tau_lower, eta_upper, card in scored
            if tau_lower > 0 and eta_upper <= budget
        ]
        safe_sorted = sorted(safe, key=lambda x: -x[0])
        share_set = {card.memory_id for *_, card in safe_sorted[:self.max_shared_memories_per_receiver]}

        decisions: list[RouterDecision] = []
        for tau_mean, eta_mean, tau_lower, eta_upper, card in scored:
            if card.memory_id in share_set:
                decisions.append(RouterDecision(
                    memory_id=card.memory_id,
                    action="share",
                    tau_hat=tau_mean,
                    eta_raw=eta_mean,
                    eta_calibrated=eta_mean,
                    risk_budget=budget,
                    reason="tau_lower>0 and eta_upper<=epsilon_star",
                ))
            else:
                if tau_lower <= 0:
                    reason = "tau_lower<=0"
                elif eta_upper > budget:
                    reason = "eta_upper>epsilon_star"
                else:
                    reason = "no safe candidate" if not safe else "not top safe candidate"
                decisions.append(RouterDecision(
                    memory_id=card.memory_id,
                    action="withhold",
                    tau_hat=tau_mean,
                    eta_raw=eta_mean,
                    eta_calibrated=eta_mean,
                    risk_budget=budget,
                    reason=reason,
                ))
        return decisions
