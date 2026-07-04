from typing import Protocol

from smtr.memory.schemas import MemoryRoutingCard
from smtr.router.baseline_router import RoutingResult
from smtr.router.candidate_proposer import CandidateProposal, CandidateRequest
from smtr.router.traces import CandidateTrace, RouterDecision


class CandidateProposer(Protocol):
    def propose_from_cards(
        self,
        *,
        request: CandidateRequest,
        cards: list[MemoryRoutingCard],
        pool_revision: int,
    ) -> CandidateProposal: ...

    def propose(
        self,
        *,
        task: str,
        receiver_agent: str,
        environment_observation: dict,
        cards: list[MemoryRoutingCard],
        top_k: int,
        seed: int,
    ) -> list[CandidateTrace]: ...


class MemoryRouter(Protocol):
    def decide_from_proposal(
        self,
        *,
        receiver_agent_id: str,
        proposal: CandidateProposal,
        cards_by_id: dict[str, MemoryRoutingCard] | None = None,
        context: object | None = None,
        traversal_seed: int | None = None,
    ) -> RoutingResult: ...

    def decide(
        self,
        *,
        task: str,
        receiver_agent: str,
        candidates: list[CandidateTrace],
        cards_by_id: dict[str, MemoryRoutingCard],
        seed: int,
    ) -> tuple[list[RouterDecision], list[str]]: ...
