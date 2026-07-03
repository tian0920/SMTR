from smtr.memory.seed_memories import build_seed_memory_pool
from smtr.router.baseline_router import NoMemoryRouter
from smtr.router.candidate_proposer import DeterministicHybridCandidateProposer


def test_no_memory_router_withholds_every_candidate() -> None:
    pool = build_seed_memory_pool()
    cards = pool.list_routing_cards()
    candidates = DeterministicHybridCandidateProposer().propose(
        task="Obtain a target artifact using the valid action sequence.",
        receiver_agent="executor",
        environment_observation={"tags": ["artifact", "tool-chain", "execution"]},
        cards=cards,
        top_k=3,
        seed=7,
    )

    decisions, selected_ids = NoMemoryRouter().decide(
        task="Obtain a target artifact using the valid action sequence.",
        receiver_agent="executor",
        candidates=candidates,
        cards_by_id={card.memory_id: card for card in cards},
        seed=7,
    )

    assert selected_ids == []
    assert {decision.decision for decision in decisions} == {"withhold"}

