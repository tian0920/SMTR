import pytest

from smtr.counterfactual.candidate_traversal import build_candidate_traversal_plan
from smtr.counterfactual.schemas import CandidateTraversalPlan
from smtr.router.candidate_proposer import CandidateProposal, CandidateRequest, CandidateScore


def _proposal() -> CandidateProposal:
    request = CandidateRequest(
        task="task",
        task_stage="planner",
        receiver_agent_id="planner",
        receiver_role="planner",
        top_k=4,
    )
    return CandidateProposal(
        request=request,
        ranked_candidates=[
            CandidateScore(memory_id="a", total_score=1.0),
            CandidateScore(memory_id="b", total_score=0.9),
            CandidateScore(memory_id="c", total_score=0.8),
            CandidateScore(memory_id="d", total_score=0.7),
        ],
        pool_revision=1,
    )


def test_candidate_traversal_order_is_stable_for_same_seed() -> None:
    first = build_candidate_traversal_plan(proposal=_proposal(), traversal_seed=7)
    second = build_candidate_traversal_plan(proposal=_proposal(), traversal_seed=7)

    assert first == second


def test_candidate_traversal_order_can_change_with_seed() -> None:
    first = build_candidate_traversal_plan(proposal=_proposal(), traversal_seed=7)
    second = build_candidate_traversal_plan(proposal=_proposal(), traversal_seed=8)

    assert first.candidate_order != second.candidate_order


def test_selected_before_rejects_target_and_after_target_and_unknown() -> None:
    with pytest.raises(ValueError, match="target"):
        CandidateTraversalPlan(
            candidate_order=["a", "b", "c"],
            target_index=1,
            target_memory_id="b",
            selected_before=["b"],
            traversal_seed=1,
        )
    with pytest.raises(ValueError, match="before target_index"):
        CandidateTraversalPlan(
            candidate_order=["a", "b", "c"],
            target_index=1,
            target_memory_id="b",
            selected_before=["c"],
            traversal_seed=1,
        )
    with pytest.raises(ValueError, match="non-candidate"):
        CandidateTraversalPlan(
            candidate_order=["a", "b", "c"],
            target_index=1,
            target_memory_id="b",
            selected_before=["z"],
            traversal_seed=1,
        )
