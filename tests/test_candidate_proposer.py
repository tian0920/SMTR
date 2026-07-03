from smtr.memory.seed_memories import build_seed_memory_pool
from smtr.router.candidate_proposer import DeterministicHybridCandidateProposer


def test_candidate_proposer_is_deterministic_and_role_sensitive() -> None:
    pool = build_seed_memory_pool()
    proposer = DeterministicHybridCandidateProposer()
    observation = {"tags": ["artifact", "ordered-actions", "planning"]}

    first = proposer.propose(
        task="Obtain a target artifact using the valid action sequence.",
        receiver_agent="planner",
        environment_observation=observation,
        cards=pool.list_routing_cards(),
        top_k=3,
        seed=7,
    )
    second = proposer.propose(
        task="Obtain a target artifact using the valid action sequence.",
        receiver_agent="planner",
        environment_observation=observation,
        cards=pool.list_routing_cards(),
        top_k=3,
        seed=7,
    )

    assert first == second
    assert [candidate.memory_id for candidate in first]
    assert first[0].memory_id == "mem_plan_artifact_sequence"
    assert first[0].receiver_role_match == 1.0

