import hashlib
from typing import Any

from pydantic import BaseModel, ConfigDict

from smtr.policy.manifests import ContinuationPolicyManifest, validate_policy_manifest
from smtr.router.traces import RouterDecision
from smtr.router.transfer_critic import FourOutcomeTransferCritic


class ExplorationPolicyConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    safe_tau_lcb_threshold: float = 0.0
    safe_negative_risk_ucb_threshold: float = 0.20
    hard_negative_risk_veto_ucb: float = 0.35
    boundary_tau_band: float = 0.15
    soft_ood_multiplier: float = 1.25
    exploration_round_probability: float = 0.30
    max_total_shares_per_invocation: int = 3
    max_exploratory_shares_per_invocation: int = 1
    reject_explicit_environment_conflict: bool = True


class FrozenRiskConstrainedExplorationPolicy:
    policy_name = "FrozenRiskConstrainedExplorationPolicy"
    # S4: version bumped to "2" after fixing the boundary-explore trigger so that
    # this frozen policy is a distinct estimand from the buggy v1.
    policy_version = "2"

    def __init__(
        self,
        *,
        manifest: ContinuationPolicyManifest,
        critic: FourOutcomeTransferCritic,
    ) -> None:
        validate_policy_manifest(manifest)
        self.manifest = manifest
        self.critic = critic
        self.config = ExplorationPolicyConfig.model_validate(manifest.exploration_config or {})

    def decide(
        self,
        *,
        candidate_id: str,
        candidate_position: int,
        target_index: int,
        selected_so_far: list[str],
        decision_context: dict[str, Any],
    ) -> RouterDecision:
        del target_index
        selected_count = len(selected_so_far)
        score = _stable_unit_interval(
            self.manifest.fingerprint,
            str(decision_context.get("receiver_agent_id", "")),
            candidate_id,
            str(candidate_position),
        )
        negative_risk_ucb = 0.05 + 0.25 * score
        tau_mean = 0.20 - 0.40 * score
        tau_lcb = tau_mean - 0.05
        tau_ucb = tau_mean + 0.05
        support_distance = 0.0
        support_threshold = 1.0

        if selected_count >= self.config.max_total_shares_per_invocation:
            action = "withhold"
            mode = "budget_exhausted"
            probability = 0.0
            eligible = False
            selected = False
        elif negative_risk_ucb > self.config.hard_negative_risk_veto_ucb:
            action = "withhold"
            mode = "risk_veto"
            probability = 0.0
            eligible = False
            selected = False
        elif support_distance > self.config.soft_ood_multiplier * max(support_threshold, 1e-9):
            action = "withhold"
            mode = "hard_ood_veto"
            probability = 0.0
            eligible = False
            selected = False
        elif (
            tau_lcb > self.config.safe_tau_lcb_threshold
            and negative_risk_ucb <= self.config.safe_negative_risk_ucb_threshold
        ):
            action = "share"
            mode = "safe_exploit"
            probability = 1.0
            eligible = False
            selected = False
        else:
            eligible = (
                negative_risk_ucb <= self.config.hard_negative_risk_veto_ucb
                and abs(tau_mean) <= self.config.boundary_tau_band
            )
            # S4 (A-11) fix: the boundary-explore trigger MUST use an independent
            # random draw. Reusing ``score`` made boundary_explore mathematically
            # impossible: safe_exploit already claims every low ``score`` case, so the
            # boundary branch only ran for high ``score`` where ``score < prob`` was
            # never true. An independent ``trigger_score`` also makes the logged share
            # propensity (``exploration_round_probability``) match reality for IPS.
            trigger_score = _stable_unit_interval(
                self.manifest.fingerprint,
                "boundary-trigger",
                str(decision_context.get("receiver_agent_id", "")),
                candidate_id,
                str(candidate_position),
            )
            trigger = trigger_score < self.config.exploration_round_probability
            selected = eligible and trigger
            action = "share" if selected else "withhold"
            mode = "boundary_explore" if selected else "ordinary_withhold"
            probability = self.config.exploration_round_probability if eligible else 0.0

        return RouterDecision(
            memory_id=candidate_id,
            action=action,
            decision=action,
            reason=mode,
            candidate_position=candidate_position,
            decision_source="frozen_continuation",
            policy_fingerprint=self.manifest.fingerprint,
            tau_mean=tau_mean,
            tau_lcb=tau_lcb,
            tau_ucb=tau_ucb,
            negative_risk_mean=max(0.0, negative_risk_ucb - 0.05),
            negative_risk_ucb=negative_risk_ucb,
            low_support=False,
            behavior_probability_share=probability,
            decision_mode=mode,
            exploration_eligible=eligible,
            exploration_selected=selected,
            support_distance=support_distance,
            support_threshold=support_threshold,
        )


def _stable_unit_interval(*parts: str) -> float:
    digest = hashlib.sha256("|".join(parts).encode()).hexdigest()[:12]
    return int(digest, 16) / float(16**12 - 1)
