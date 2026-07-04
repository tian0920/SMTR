from smtr.policy.exploratory_policy import (
    ExplorationPolicyConfig,
    FrozenRiskConstrainedExplorationPolicy,
)
from smtr.policy.schemas import ContinuationPolicyManifest, with_fingerprint


def _policy() -> FrozenRiskConstrainedExplorationPolicy:
    manifest = with_fingerprint(
        ContinuationPolicyManifest(
            policy_name="FrozenRiskConstrainedExplorationPolicy",
            policy_version=FrozenRiskConstrainedExplorationPolicy.policy_version,
            policy_kind="frozen_risk_constrained_exploration",
            source_critic_checkpoint_path="checkpoints/nonexistent.joblib",
            source_critic_checkpoint_sha256="deadbeef",
            source_critic_estimand_policy_fingerprint="fp",
            feature_encoder_schema_version="1.0",
            exploration_config=ExplorationPolicyConfig().model_dump(),
        )
    )
    # The critic is unused by decide(); the exploration policy is deterministic.
    return FrozenRiskConstrainedExplorationPolicy(manifest=manifest, critic=None)


def _decisions(policy: FrozenRiskConstrainedExplorationPolicy, n: int = 300):
    modes: dict[str, list] = {}
    for i in range(n):
        decision = policy.decide(
            candidate_id=f"mem_{i}",
            candidate_position=0,
            target_index=5,
            selected_so_far=[],
            decision_context={"receiver_agent_id": "planner"},
        )
        modes.setdefault(decision.decision_mode, []).append(decision)
    return modes


def test_boundary_explore_now_fires() -> None:
    # Before the S4 fix, boundary_explore was mathematically impossible (0 occurrences).
    modes = _decisions(_policy())
    assert "boundary_explore" in modes
    assert "safe_exploit" in modes


def test_boundary_explore_respects_risk_and_tau_band() -> None:
    policy = _policy()
    config = policy.config
    modes = _decisions(policy)
    for decision in modes.get("boundary_explore", []):
        assert decision.action == "share"
        assert decision.exploration_selected is True
        # The fix does NOT loosen risk/tau gates (A-12): boundary shares stay inside band.
        assert abs(decision.tau_mean) <= config.boundary_tau_band
        assert decision.negative_risk_ucb <= config.hard_negative_risk_veto_ucb
        # Logged propensity now matches the true exploration probability (IPS correctness).
        assert decision.behavior_probability_share == config.exploration_round_probability


def test_policy_version_is_v2() -> None:
    assert FrozenRiskConstrainedExplorationPolicy.policy_version == "2"
