"""B1 RelevanceTopK router — non-causal relevance baseline.

Selects the top-k candidates by proposer relevance ranking without calling
the transfer critic. Used as an ablation baseline to measure the value of
causal transfer estimation over naive relevance-based sharing.
"""

from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel, ConfigDict

from smtr.counterfactual.decision_points import canonical_digest
from smtr.memory.schemas import MemoryRoutingCard
from smtr.router.baseline_router import RoutingResult
from smtr.router.candidate_proposer import CandidateProposal
from smtr.router.traces import CandidateTrace, RouterDecision


@dataclass(frozen=True)
class RelevanceTopKRouterConfig:
    """Configuration for the RelevanceTopK router."""

    max_shares_per_invocation: int | None = None
    """Maximum number of memories to share per router invocation.
    
    If None, share all candidates. Must be >= 0 if specified.
    """

    def __post_init__(self) -> None:
        if (
            self.max_shares_per_invocation is not None
            and self.max_shares_per_invocation < 0
        ):
            raise ValueError(
                f"max_shares_per_invocation must be >= 0, got {self.max_shares_per_invocation}"
            )


class BudgetManifestConfig(BaseModel):
    """Configuration for B1-Matched budget sampling."""

    model_config = ConfigDict(frozen=True)

    count_distribution: dict[str, float]
    """P(|S|=k) for k in 0..max_shares."""
    max_shares: int
    seed: int = 0

class RelevanceTopKRouter:
    """Non-causal relevance baseline router (B1).

    Selects candidates purely by proposer relevance ranking. Does not call
    the transfer critic or compute any causal estimates (τ, η, LCB, UCB).

    This router:
    - Uses the same candidate proposer as M0 (ProductionSequentialRouter)
    - Uses the same memory store snapshot
    - Uses the same top_k
    - Selects by relevance ranking order (no random shuffle)
    - Respects max_shares_per_invocation budget
    - Only reads routing cards; payloads loaded after selection
    """

    router_name = "RelevanceTopKRouter"
    router_version = "1"

    def __init__(
        self,
        *,
        config: RelevanceTopKRouterConfig | None = None,
    ) -> None:
        self.config = config or RelevanceTopKRouterConfig()

    def decide_from_proposal(
        self,
        *,
        receiver_agent_id: str,
        proposal: CandidateProposal,
        cards_by_id: dict[str, MemoryRoutingCard] | None = None,
        context: object | None = None,
        traversal_seed: int | None = None,
    ) -> RoutingResult:
        """Select top-k candidates by relevance ranking.

        Does not access critic, cards_by_id, or context.
        """
        del cards_by_id, context

        candidates = proposal.ranked_candidates
        traversal_order = [c.memory_id for c in candidates]

        # Determine share limit
        if self.config.max_shares_per_invocation is None:
            share_limit = len(candidates)
        else:
            share_limit = min(len(candidates), self.config.max_shares_per_invocation)

        # Build decisions
        decisions: list[RouterDecision] = []
        selected_ids: list[str] = []

        for position, candidate in enumerate(candidates):
            if position < share_limit:
                action = "share"
                reason = "relevance_topk_selected"
                selected_ids.append(candidate.memory_id)
            else:
                action = "withhold"
                reason = "relevance_topk_budget_exceeded"

            decisions.append(
                RouterDecision(
                    memory_id=candidate.memory_id,
                    action=action,
                    decision=action,
                    score=candidate.total_score,
                    reason=reason,
                    decision_reason=reason,
                    candidate_position=position,
                    decision_source="relevance_topk_router",
                    accepted=(action == "share"),
                    traversal_seed=traversal_seed,
                    traversal_order=traversal_order,
                    traversal_position=position,
                    original_candidate_position=position,
                    proposal_rank=position + 1,
                    proposal_score=candidate.total_score,
                )
            )

        return RoutingResult(
            receiver_agent_id=receiver_agent_id,
            candidate_proposal=proposal,
            decisions=decisions,
            selected_memory_ids=selected_ids,
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
        """Legacy interface for compatibility with NoMemoryRouter."""
        from smtr.router.candidate_proposer import CandidateRequest, CandidateScore

        request = CandidateRequest(
            task=task,
            task_stage="legacy",
            receiver_agent_id=receiver_agent,
            receiver_role=receiver_agent,
            receiver_capabilities=[],
            environment_observation={},
            local_context_summary="",
            top_k=len(candidates),
            seed=seed,
        )

        scored_candidates = [
            CandidateScore(
                memory_id=c.memory_id,
                total_score=c.total_score,
                goal_similarity=c.goal_similarity,
                task_tag_overlap=c.task_tag_overlap,
                environment_compatibility=c.environment_compatibility,
                receiver_compatibility=c.receiver_compatibility,
                explicit_environment_conflict=c.explicit_environment_conflict,
                score_explanation=c.score_explanation,
            )
            for c in candidates
        ]

        proposal = CandidateProposal(
            request=request,
            ranked_candidates=scored_candidates,
            pool_revision=0,
        )

        result = self.decide_from_proposal(
            receiver_agent_id=receiver_agent,
            proposal=proposal,
            cards_by_id=cards_by_id,
            traversal_seed=seed,
        )

        return result.decisions, result.selected_memory_ids


class BudgetMatchedTopKRouter:
    """B1-Matched: relevance baseline with validation-calibrated budget.

    Samples per-invocation share budget from a pre-computed manifest
    (derived from M0 validation runs), then selects top-k by relevance.
    Does NOT read test-set M0 outcomes — budget is fixed before testing.
    """

    router_name = "BudgetMatchedTopKRouter"
    router_version = "1"

    def __init__(
        self,
        *,
        manifest_config: BudgetManifestConfig,
        experiment_seed: int = 0,
        base_episode_id: str = "",
        method_id: str = "B1-Matched",
    ) -> None:
        if not manifest_config.count_distribution:
            raise ValueError("B1-Matched requires a non-empty budget distribution")
        self.manifest_config = manifest_config
        self._experiment_seed = experiment_seed
        self._base_episode_id = base_episode_id
        self._method_id = method_id

    def _get_budget(self, invocation_id: str) -> int:
        """Get sampled budget for this invocation."""
        inv_seed = int(
            canonical_digest(
                {
                    "experiment_seed": self._experiment_seed,
                    "base_episode_id": self._base_episode_id,
                    "invocation_id": invocation_id,
                    "method_id": self._method_id,
                }
            )[:8],
            16,
        )
        rng = np.random.default_rng(inv_seed)

        dist = self.manifest_config.count_distribution
        keys = sorted(dist.keys(), key=int)
        probs = np.array([dist[k] for k in keys], dtype=float)
        probs = probs / probs.sum()
        values = np.array([int(k) for k in keys])
        return int(rng.choice(values, p=probs))

    def decide_from_proposal(
        self,
        *,
        receiver_agent_id: str,
        proposal: CandidateProposal,
        cards_by_id: dict[str, MemoryRoutingCard] | None = None,
        context: object | None = None,
        traversal_seed: int | None = None,
    ) -> RoutingResult:
        """Select top-budget candidates by relevance ranking."""
        del cards_by_id, context

        candidates = proposal.ranked_candidates
        traversal_order = [c.memory_id for c in candidates]

        invocation_id = canonical_digest(
            {
                "receiver_agent_id": receiver_agent_id,
                "candidate_request": proposal.request.model_dump(mode="json"),
                "traversal_seed": traversal_seed,
            }
        )[:16]
        budget = self._get_budget(invocation_id)
        share_limit = min(len(candidates), budget)

        # Build decisions
        decisions: list[RouterDecision] = []
        selected_ids: list[str] = []

        for position, candidate in enumerate(candidates):
            if position < share_limit:
                action = "share"
                reason = "relevance_topk_selected"
                selected_ids.append(candidate.memory_id)
            else:
                action = "withhold"
                reason = "relevance_topk_budget_exceeded"

            decisions.append(
                RouterDecision(
                    memory_id=candidate.memory_id,
                    action=action,
                    decision=action,
                    score=candidate.total_score,
                    reason=reason,
                    decision_reason=reason,
                    candidate_position=position,
                    decision_source="relevance_topk_router",
                    accepted=(action == "share"),
                    traversal_seed=traversal_seed,
                    traversal_order=traversal_order,
                    traversal_position=position,
                    original_candidate_position=position,
                    proposal_rank=position + 1,
                    proposal_score=candidate.total_score,
                )
            )

        return RoutingResult(
            receiver_agent_id=receiver_agent_id,
            candidate_proposal=proposal,
            decisions=decisions,
            selected_memory_ids=selected_ids,
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
        """Legacy interface."""
        from smtr.router.candidate_proposer import CandidateRequest, CandidateScore

        request = CandidateRequest(
            task=task,
            task_stage="legacy",
            receiver_agent_id=receiver_agent,
            receiver_role=receiver_agent,
            receiver_capabilities=[],
            environment_observation={},
            local_context_summary="",
            top_k=len(candidates),
            seed=seed,
        )

        scored_candidates = [
            CandidateScore(
                memory_id=c.memory_id,
                total_score=c.total_score,
                goal_similarity=c.goal_similarity,
                task_tag_overlap=c.task_tag_overlap,
                environment_compatibility=c.environment_compatibility,
                receiver_compatibility=c.receiver_compatibility,
                explicit_environment_conflict=c.explicit_environment_conflict,
                score_explanation=c.score_explanation,
            )
            for c in candidates
        ]

        proposal = CandidateProposal(
            request=request,
            ranked_candidates=scored_candidates,
            pool_revision=0,
        )

        result = self.decide_from_proposal(
            receiver_agent_id=receiver_agent,
            proposal=proposal,
            cards_by_id=cards_by_id,
            traversal_seed=seed,
        )

        return result.decisions, result.selected_memory_ids
