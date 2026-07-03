from smtr.counterfactual.continuation_policy import FrozenNoShareContinuationPolicy
from smtr.counterfactual.forced_router import ForcedInterventionRouter
from smtr.counterfactual.schemas import CandidateTraversalPlan
from smtr.router.candidate_proposer import CandidateProposal, CandidateRequest, CandidateScore


def _proposal() -> CandidateProposal:
    request = CandidateRequest(
        task="task",
        task_stage="planner",
        receiver_agent_id="planner",
        receiver_role="planner",
    )
    return CandidateProposal(
        request=request,
        ranked_candidates=[
            CandidateScore(memory_id="a", total_score=1.0),
            CandidateScore(memory_id="b", total_score=0.9),
            CandidateScore(memory_id="c", total_score=0.8),
        ],
        pool_revision=1,
    )


def _plan() -> CandidateTraversalPlan:
    return CandidateTraversalPlan(
        candidate_order=["a", "b", "c"],
        target_index=1,
        target_memory_id="b",
        selected_before=["a"],
        traversal_seed=7,
    )


def test_forced_router_share_and_withhold_target_decisions() -> None:
    policy = FrozenNoShareContinuationPolicy()
    share = ForcedInterventionRouter(
        traversal_plan=_plan(),
        branch_arm="share",
        continuation_policy=policy,
        receiver_agent_id="planner",
    ).decide_from_proposal(receiver_agent_id="planner", proposal=_proposal())
    withhold = ForcedInterventionRouter(
        traversal_plan=_plan(),
        branch_arm="withhold",
        continuation_policy=policy,
        receiver_agent_id="planner",
    ).decide_from_proposal(receiver_agent_id="planner", proposal=_proposal())

    assert share.decisions[1].action == "share"
    assert share.decisions[1].reason == "forced_share_counterfactual"
    assert share.decisions[1].decision_source == "forced_intervention"
    assert withhold.decisions[1].action == "withhold"
    assert withhold.decisions[1].reason == "forced_withhold_counterfactual"
    assert withhold.decisions[1].decision_source == "forced_intervention"


def test_forced_router_prefix_and_continuation_policy() -> None:
    result = ForcedInterventionRouter(
        traversal_plan=_plan(),
        branch_arm="share",
        continuation_policy=FrozenNoShareContinuationPolicy(),
        receiver_agent_id="planner",
    ).decide_from_proposal(receiver_agent_id="planner", proposal=_proposal())

    assert result.decisions[0].action == "share"
    assert result.decisions[0].decision_source == "fixed_prefix"
    assert result.decisions[2].action == "withhold"
    assert result.decisions[2].decision_source == "frozen_continuation"
    assert result.selected_memory_ids == ["a", "b"]


def test_forced_router_branch_metadata_matches() -> None:
    policy = FrozenNoShareContinuationPolicy()
    share = ForcedInterventionRouter(
        traversal_plan=_plan(),
        branch_arm="share",
        continuation_policy=policy,
        receiver_agent_id="planner",
    )
    withhold = ForcedInterventionRouter(
        traversal_plan=_plan(),
        branch_arm="withhold",
        continuation_policy=policy,
        receiver_agent_id="planner",
    )

    assert share.traversal_plan.candidate_order == withhold.traversal_plan.candidate_order
    assert share.continuation_policy.policy_name == withhold.continuation_policy.policy_name
    assert share.continuation_policy.policy_version == withhold.continuation_policy.policy_version
