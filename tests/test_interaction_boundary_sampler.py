from smtr.counterfactual.interaction_boundary_sampler import (
    InteractionBoundaryConfig,
    InteractionBoundaryPrefixSampler,
)
from smtr.counterfactual.schemas import RoutingFeatureSnapshot
from smtr.router.candidate_proposer import CandidateScore


def _card(
    memory_id: str,
    *,
    required=None,
    forbidden=None,
    roles=None,
) -> RoutingFeatureSnapshot:
    return RoutingFeatureSnapshot(
        memory_id=memory_id,
        active_payload_version=1,
        goal_summary=f"goal {memory_id}",
        task_tags=["artifact"],
        required_environment_facts=required or {},
        forbidden_environment_facts=forbidden or {},
        compatible_receiver_roles=roles or ["planner"],
        compatible_receiver_capabilities=["planning"],
    )


def _scores() -> dict[str, CandidateScore]:
    return {
        "conflict": CandidateScore(memory_id="conflict", receiver_compatibility=1.0),
        "neutral": CandidateScore(memory_id="neutral", receiver_compatibility=1.0),
        "target": CandidateScore(memory_id="target", receiver_compatibility=1.0),
    }


def _cards() -> dict[str, RoutingFeatureSnapshot]:
    # target requires door=open; conflict prefix requires door=closed -> env_conflict.
    return {
        "target": _card("target", required={"door": "open"}),
        "conflict": _card("conflict", required={"door": "closed"}),
        "neutral": _card("neutral", required={"window": "open"}, roles=["executor"]),
    }


def test_boundary_sampler_prefers_conflicting_prefix() -> None:
    sampler = InteractionBoundaryPrefixSampler(
        InteractionBoundaryConfig(max_prefix_size=1),
        cards_by_id=_cards(),
    )
    selected = sampler.sample(
        candidate_order=["conflict", "neutral", "target"],
        target_index=2,
        candidate_scores=_scores(),
        seed=1,
    )
    assert selected == ["conflict"]


def test_boundary_sampler_falls_back_to_empty_without_interaction() -> None:
    unrelated_cards = {
        "target": _card("target", required={"door": "open"}, roles=["planner"]).model_copy(
            update={"task_tags": [], "compatible_receiver_capabilities": ["planning"]}
        ),
        "unrelated": _card(
            "unrelated", required={"window": "open"}, roles=["reviewer"]
        ).model_copy(
            update={"task_tags": [], "compatible_receiver_capabilities": ["reviewing"]}
        ),
    }
    sampler = InteractionBoundaryPrefixSampler(
        InteractionBoundaryConfig(max_prefix_size=1),
        cards_by_id=unrelated_cards,
    )
    selected = sampler.sample(
        candidate_order=["unrelated", "target"],
        target_index=1,
        candidate_scores={
            "unrelated": CandidateScore(memory_id="unrelated", receiver_compatibility=1.0),
            "target": CandidateScore(memory_id="target", receiver_compatibility=1.0),
        },
        seed=1,
    )
    assert selected == []
    assert sampler.fallback_count == 1


def test_boundary_sampler_uses_critic_scorer() -> None:
    sampler = InteractionBoundaryPrefixSampler(
        InteractionBoundaryConfig(max_prefix_size=1, structural_weight=0.0),
        cards_by_id=_cards(),
    )
    # Critic scorer boosts the otherwise-neutral prefix so it wins (A-07.4/5/6).
    sampler.critic_scorer = lambda target_id, prefix_id: 10.0 if prefix_id == "neutral" else 0.0
    selected = sampler.sample(
        candidate_order=["conflict", "neutral", "target"],
        target_index=2,
        candidate_scores=_scores(),
        seed=1,
    )
    assert selected == ["neutral"]
