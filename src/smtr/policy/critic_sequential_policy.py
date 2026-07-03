from smtr.counterfactual.schemas import RoutingFeatureSnapshot
from smtr.memory.schemas import ContextFingerprint
from smtr.policy.manifests import ContinuationPolicyManifest, validate_policy_manifest
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
        if estimate.low_support and self.manifest.reject_low_support:
            action = "withhold"
            reason = "withhold_low_support"
        elif estimate.tau_lcb <= float(self.manifest.tau_lcb_threshold or 0.0):
            action = "withhold"
            reason = "withhold_nonpositive_tau_lcb"
        elif estimate.negative_risk_ucb > float(
            self.manifest.negative_risk_ucb_threshold or 0.2
        ):
            action = "withhold"
            reason = "withhold_negative_risk_ucb_exceeds_threshold"
        else:
            action = "share"
            reason = "share_positive_lcb_and_safe_risk"
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
            low_support=estimate.low_support,
        )
