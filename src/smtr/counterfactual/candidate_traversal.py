import random

from smtr.counterfactual.prefix_sampler import (
    ScenarioDesignatedTargetPolicy,
    StratifiedEligiblePrefixSampler,
    TargetSelectionPolicy,
)
from smtr.counterfactual.schemas import CandidateTraversalPlan, validate_selection_prefix
from smtr.router.candidate_proposer import CandidateProposal


def build_candidate_traversal_plan(
    *,
    proposal: CandidateProposal,
    traversal_seed: int,
    target_memory_id: str | None = None,
    selected_before: list[str] | None = None,
    target_selection_policy: TargetSelectionPolicy | None = None,
    prefix_sampler: StratifiedEligiblePrefixSampler | None = None,
    target_selection_seed: int | None = None,
    prefix_sampling_seed: int | None = None,
) -> CandidateTraversalPlan:
    candidate_ids = [candidate.memory_id for candidate in proposal.ranked_candidates]
    if not candidate_ids:
        raise ValueError("candidate proposal has no candidates")
    candidate_order = list(candidate_ids)
    random.Random(traversal_seed).shuffle(candidate_order)
    target_selection_seed = (
        traversal_seed + 1 if target_selection_seed is None else target_selection_seed
    )
    prefix_sampling_seed = (
        traversal_seed + 2 if prefix_sampling_seed is None else prefix_sampling_seed
    )
    target_policy = target_selection_policy
    if target_policy is None:
        target_policy = ScenarioDesignatedTargetPolicy() if target_memory_id else None
    if target_policy is None:
        target_memory_id = random.Random(target_selection_seed).choice(candidate_order)
        target_probability = 1.0 / len(candidate_order)
        target_policy_name = "legacy_uniform"
        target_policy_version = "1"
    else:
        target_memory_id, target_probability = target_policy.select(
            candidate_ids=candidate_order,
            seed=target_selection_seed,
            designated_target_id=target_memory_id,
        )
        target_policy_name = target_policy.policy_name
        target_policy_version = target_policy.policy_version
    target_index = candidate_order.index(target_memory_id)
    candidate_scores = {
        candidate.memory_id: candidate for candidate in proposal.ranked_candidates
    }
    # If selected_before is explicitly provided (forced_prefix), rearrange
    # candidate_order to ensure those memories come before the target
    if selected_before is not None:
        prefix_memories = [m for m in selected_before if m in candidate_order]
        other_memories = [
            m for m in candidate_order
            if m not in prefix_memories and m != target_memory_id
        ]
        candidate_order = prefix_memories + [target_memory_id] + other_memories
        target_index = candidate_order.index(target_memory_id)
    if selected_before is None and prefix_sampler is not None:
        selected_before = prefix_sampler.sample(
            candidate_order=candidate_order,
            target_index=target_index,
            candidate_scores=candidate_scores,
            seed=prefix_sampling_seed,
        )
        eligible_count = len(
            prefix_sampler.eligible_candidates(
                candidate_order=candidate_order,
                target_index=target_index,
                candidate_scores=candidate_scores,
            )
        )
        prefix_probability = prefix_sampler.sampling_probability(
            len(selected_before), eligible_count
        )
        prefix_policy_name = prefix_sampler.policy_name
        prefix_policy_version = prefix_sampler.policy_version
    else:
        prefix_probability = 1.0 if not selected_before else None
        prefix_policy_name = "explicit" if selected_before else "legacy_empty"
        prefix_policy_version = "1"
    selected_before = selected_before or []
    plan = CandidateTraversalPlan(
        candidate_order=candidate_order,
        target_index=target_index,
        target_memory_id=target_memory_id,
        selected_before_positions=[
            candidate_order.index(memory_id) for memory_id in selected_before
        ],
        selected_before=selected_before or [],
        traversal_seed=traversal_seed,
        target_selection_seed=target_selection_seed,
        prefix_sampling_seed=prefix_sampling_seed,
        target_selection_policy_name=target_policy_name,
        target_selection_policy_version=target_policy_version,
        prefix_sampling_policy_name=prefix_policy_name,
        prefix_sampling_policy_version=prefix_policy_version,
        target_selection_probability=target_probability,
        prefix_sampling_probability=prefix_probability,
    )
    validate_selection_prefix(plan)
    return plan
