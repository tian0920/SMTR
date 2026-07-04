from smtr.counterfactual.schemas import RoutingFeatureSnapshot
from smtr.memory.schemas import ContextFingerprint
from smtr.policy.manifests import ContinuationPolicyManifest, validate_policy_manifest
from smtr.router.causal_gate import strict_lcb_ucb_gate
from smtr.router.traces import RouterDecision
from smtr.router.transfer_critic import FourOutcomeTransferCritic
from smtr.router.transfer_features import TransferPredictionInput


class FrozenCriticSequentialContinuationPolicy:
    policy_name = "FrozenCriticSequentialContinuationPolicy"
    policy_version = "1"

    def __init__(
        self,
        *,
        manifest: ContinuationPolicyManifest,
        critic: FourOutcomeTransferCritic,
    ) -> None:
        validate_policy_manifest(manifest)
        self.manifest = manifest
        self.critic = critic

    def decide(
        self,
        *,
        context: ContextFingerprint | None = None,
        candidate_card: RoutingFeatureSnapshot | None = None,
        selected_cards: list[RoutingFeatureSnapshot] | None = None,
        candidate_id: str | None = None,
        candidate_position: int | None = None,
        target_index: int | None = None,
        selected_so_far: list[str] | None = None,
        decision_context: dict | None = None,
    ) -> RouterDecision | str:
        del target_index, selected_so_far, decision_context
        if context is None or candidate_card is None:
            return "withhold"
        estimate = self.critic.predict(
            TransferPredictionInput(
                context=context,
                candidate_card=candidate_card,
                selected_cards=selected_cards or [],
            )
        )
        epsilon = float(self.manifest.negative_risk_ucb_threshold or 0.2)
        tau_lcb_threshold = float(self.manifest.tau_lcb_threshold or 0.0)
        if estimate.low_support and self.manifest.reject_low_support:
            action = "withhold"
            reason = "withhold_low_support"
        else:
            gate_estimate = estimate
            if tau_lcb_threshold != 0.0:
                gate_estimate = estimate.model_copy(
                    update={"tau_lcb": estimate.tau_lcb - tau_lcb_threshold}
                )
            gate = strict_lcb_ucb_gate(gate_estimate, epsilon=epsilon)
            action = "share" if gate.accepted else "withhold"
            reason = (
                "share_positive_lcb_and_safe_risk"
                if gate.accepted
                else gate.reason
            )
        return RouterDecision(
            memory_id=candidate_id or candidate_card.memory_id,
            action=action,
            decision=action,
            reason=reason,
            candidate_position=candidate_position,
            decision_source="frozen_continuation",
            policy_fingerprint=self.manifest.fingerprint,
            tau_mean=estimate.tau_mean,
            tau_lcb=estimate.tau_lcb,
            tau_ucb=estimate.tau_ucb,
            negative_risk_mean=estimate.negative_risk_mean,
            negative_risk_ucb=estimate.negative_risk_ucb,
            epsilon=epsilon,
            accepted=action == "share",
            decision_reason=reason,
            low_support=estimate.low_support,
        )
