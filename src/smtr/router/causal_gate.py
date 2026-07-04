from dataclasses import dataclass

from smtr.router.transfer_critic import TransferEstimate


@dataclass(frozen=True)
class CausalGateDecision:
    accepted: bool
    reason: str


def strict_lcb_ucb_gate(
    estimate: TransferEstimate,
    *,
    epsilon: float,
) -> CausalGateDecision:
    """Apply the default SMTR online gate.

    A memory is exposed only when the conservative lower bound on marginal
    transfer is strictly positive and the conservative upper bound on harm is at
    or below the allowed risk epsilon.
    """
    if estimate.tau_lcb <= 0.0:
        return CausalGateDecision(False, "tau_lcb_nonpositive")
    if estimate.negative_risk_ucb > epsilon:
        return CausalGateDecision(False, "negative_risk_ucb_exceeds_epsilon")
    return CausalGateDecision(True, "accepted")
