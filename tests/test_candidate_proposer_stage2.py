from smtr.memory.seed_memories import build_seed_memories
from smtr.router.candidate_proposer import (
    CandidateRequest,
    DeterministicHybridCandidateProposer,
    _as_fact_observation,
)


class SpyRepository:
    def __init__(self) -> None:
        self.payload_calls = 0
        self.cards = [card for card, _ in build_seed_memories()]

    def get_routing_cards(self):
        return self.cards

    def get_payload(self, *args, **kwargs):
        self.payload_calls += 1
        raise AssertionError("payload should not be read")


def _request(role: str, observation=None) -> CandidateRequest:
    return CandidateRequest(
        task="Obtain a target artifact using the valid action sequence.",
        task_stage=role,
        receiver_agent_id=role,
        receiver_role=role,
        receiver_capabilities={
            "planner": ["planning"],
            "executor": ["execution", "tool-use"],
            "critic": ["verification"],
        }[role],
        environment_observation=_as_fact_observation(
            observation
            or {
                "tags": ["artifact", "ordered-actions", "tool-chain", "verification"],
                "resource_available": True,
                "resource_locked": False,
                "tool_version": "v1",
            }
        ),
        local_context_summary="",
        top_k=4,
        seed=7,
    )


def test_planner_role_prioritizes_planner_memory() -> None:
    cards = [card for card, _ in build_seed_memories()]
    proposal = DeterministicHybridCandidateProposer().propose_from_cards(
        request=_request("planner"),
        cards=cards,
        pool_revision=1,
    )

    assert proposal.ranked_candidates[0].memory_id == "mem_plan_artifact_sequence"


def test_executor_role_prioritizes_executor_memory() -> None:
    cards = [card for card, _ in build_seed_memories()]
    proposal = DeterministicHybridCandidateProposer().propose_from_cards(
        request=_request("executor"),
        cards=cards,
        pool_revision=1,
    )

    assert proposal.ranked_candidates[0].memory_id == "mem_execute_tool_chain"


def test_environment_conflict_ranks_below_compatible_memory() -> None:
    cards = [card for card, _ in build_seed_memories()]
    proposal = DeterministicHybridCandidateProposer().propose_from_cards(
        request=_request(
            "executor",
            observation={
                "tags": ["artifact", "tool-chain"],
                "resource_available": True,
                "resource_locked": True,
                "tool_version": "v1",
            },
        ),
        cards=cards,
        pool_revision=1,
    )

    ids = [candidate.memory_id for candidate in proposal.ranked_candidates]
    assert ids.index("mem_execute_unlocked_resource") > ids.index("mem_execute_tool_chain")
    conflict = next(
        candidate
        for candidate in proposal.ranked_candidates
        if candidate.memory_id == "mem_execute_unlocked_resource"
    )
    assert conflict.explicit_environment_conflict is True


def test_candidate_proposal_is_fully_deterministic() -> None:
    cards = [card for card, _ in build_seed_memories()]
    proposer = DeterministicHybridCandidateProposer()

    first = proposer.propose_from_cards(request=_request("critic"), cards=cards, pool_revision=3)
    second = proposer.propose_from_cards(request=_request("critic"), cards=cards, pool_revision=3)

    assert first == second


def test_candidate_proposer_does_not_call_get_payload() -> None:
    repository = SpyRepository()

    DeterministicHybridCandidateProposer().propose_from_cards(
        request=_request("planner"),
        cards=repository.get_routing_cards(),
        pool_revision=1,
    )

    assert repository.payload_calls == 0


def test_candidate_proposal_trace_contains_no_payload_steps() -> None:
    memories = build_seed_memories()
    proposal = DeterministicHybridCandidateProposer().propose_from_cards(
        request=_request("planner"),
        cards=[card for card, _ in memories],
        pool_revision=1,
    )
    payload_steps = [step for _, payload in memories for step in payload.steps]
    trace_text = repr(proposal.model_dump())

    assert all(step not in trace_text for step in payload_steps)
