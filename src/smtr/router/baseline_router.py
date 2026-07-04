from pydantic import BaseModel, ConfigDict

from smtr.memory.schemas import MemoryRoutingCard
from smtr.router.candidate_proposer import CandidateProposal
from smtr.router.traces import CandidateTrace, RouterDecision


class RoutingResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    receiver_agent_id: str
    candidate_proposal: CandidateProposal
    decisions: list[RouterDecision]
    selected_memory_ids: list[str]
    router_name: str = "NoMemoryRouter"
    router_version: str = "1"


class NoMemoryRouter:
    router_name = "NoMemoryRouter"
    router_version = "1"

    def decide_from_proposal(
        self,
        *,
        receiver_agent_id: str,
        proposal: CandidateProposal,
        cards_by_id: dict[str, MemoryRoutingCard] | None = None,
        context=None,
        traversal_seed: int | None = None,
    ) -> RoutingResult:
        del cards_by_id, context, traversal_seed
        decisions = [
            RouterDecision(
                memory_id=candidate.memory_id,
                action="withhold",
                decision="withhold",
                score=candidate.total_score,
                reason="baseline_no_memory_router",
                decision_reason="baseline_no_memory_router",
                accepted=False,
            )
            for candidate in proposal.ranked_candidates
        ]
        return RoutingResult(
            receiver_agent_id=receiver_agent_id,
            candidate_proposal=proposal,
            decisions=decisions,
            selected_memory_ids=[],
            router_name=self.router_name,
            router_version=self.router_version,
        )

    def decide(
        self,
        *,
        task: str,
        receiver_agent: str,
        candidates: list[CandidateTrace],
        cards_by_id: dict[str, MemoryRoutingCard],
        seed: int,
    ) -> tuple[list[RouterDecision], list[str]]:
        del task, receiver_agent, cards_by_id, seed
        return (
            [
                RouterDecision(
                    memory_id=candidate.memory_id,
                    action="withhold",
                    decision="withhold",
                    score=candidate.score,
                    reason="baseline_no_memory_router",
                )
                for candidate in candidates
            ],
            [],
        )
