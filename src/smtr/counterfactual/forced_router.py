from typing import Literal

from smtr.counterfactual.continuation_policy import FrozenContinuationPolicy
from smtr.counterfactual.schemas import CandidateTraversalPlan
from smtr.router.baseline_router import RoutingResult
from smtr.router.candidate_proposer import CandidateProposal
from smtr.router.traces import RouterDecision


class ForcedInterventionRouter:
    router_name = "ForcedInterventionRouter"
    router_version = "1"

    def __init__(
        self,
        *,
        traversal_plan: CandidateTraversalPlan,
        branch_arm: Literal["share", "withhold"],
        continuation_policy: FrozenContinuationPolicy,
        receiver_agent_id: str,
    ) -> None:
        self.traversal_plan = traversal_plan
        self.branch_arm = branch_arm
        self.continuation_policy = continuation_policy
        self.receiver_agent_id = receiver_agent_id

    def decide_from_proposal(
        self,
        *,
        receiver_agent_id: str,
        proposal: CandidateProposal,
    ) -> RoutingResult:
        if receiver_agent_id != self.receiver_agent_id or not self._matches_target_proposal(
            proposal
        ):
            return self._withhold_all(receiver_agent_id=receiver_agent_id, proposal=proposal)

        decisions: list[RouterDecision] = []
        selected: list[str] = []
        score_by_id = {
            candidate.memory_id: candidate.total_score for candidate in proposal.ranked_candidates
        }
        for position, memory_id in enumerate(self.traversal_plan.candidate_order):
            if position < self.traversal_plan.target_index:
                action = "share" if memory_id in self.traversal_plan.selected_before else "withhold"
                reason = "fixed_prefix"
                source = "fixed_prefix"
            elif position == self.traversal_plan.target_index:
                action = self.branch_arm
                reason = (
                    "forced_share_counterfactual"
                    if self.branch_arm == "share"
                    else "forced_withhold_counterfactual"
                )
                source = "forced_intervention"
            else:
                continuation_decision = self.continuation_policy.decide(
                    candidate_id=memory_id,
                    candidate_position=position,
                    target_index=self.traversal_plan.target_index,
                    selected_so_far=selected,
                    decision_context={"receiver_agent_id": receiver_agent_id},
                )
                if isinstance(continuation_decision, RouterDecision):
                    if continuation_decision.action == "share":
                        selected.append(memory_id)
                    decisions.append(
                        continuation_decision.model_copy(
                            update={
                                "memory_id": memory_id,
                                "candidate_position": position,
                                "score": score_by_id.get(memory_id, 0.0),
                            }
                        )
                    )
                    continue
                action = continuation_decision
                reason = self.continuation_policy.policy_name
                source = "frozen_continuation"

            if action == "share":
                selected.append(memory_id)
            decisions.append(
                RouterDecision(
                    memory_id=memory_id,
                    action=action,
                    decision=action,
                    score=score_by_id.get(memory_id, 0.0),
                    reason=reason,
                    candidate_position=position,
                    decision_source=source,
                )
            )

        return RoutingResult(
            receiver_agent_id=receiver_agent_id,
            candidate_proposal=proposal,
            decisions=decisions,
            selected_memory_ids=selected,
            router_name=self.router_name,
            router_version=(
                f"{self.router_version}:{self.continuation_policy.policy_name}:"
                f"{self.continuation_policy.policy_version}"
            ),
        )

    def _matches_target_proposal(self, proposal: CandidateProposal) -> bool:
        proposal_ids = {candidate.memory_id for candidate in proposal.ranked_candidates}
        return set(self.traversal_plan.candidate_order).issubset(proposal_ids)

    def _withhold_all(
        self,
        *,
        receiver_agent_id: str,
        proposal: CandidateProposal,
    ) -> RoutingResult:
        decisions = []
        selected: list[str] = []
        for index, candidate in enumerate(proposal.ranked_candidates):
            decision = self.continuation_policy.decide(
                candidate_id=candidate.memory_id,
                candidate_position=index,
                target_index=-1,
                selected_so_far=selected,
                decision_context={"receiver_agent_id": receiver_agent_id},
            )
            if isinstance(decision, RouterDecision):
                decision = decision.model_copy(update={"score": candidate.total_score})
                if decision.action == "share":
                    selected.append(candidate.memory_id)
                decisions.append(decision)
            else:
                decisions.append(
                    RouterDecision(
                        memory_id=candidate.memory_id,
                        action="withhold",
                        decision="withhold",
                        score=candidate.total_score,
                        reason=self.continuation_policy.policy_name,
                        candidate_position=index,
                        decision_source="frozen_continuation",
                        decision_mode="ordinary_withhold",
                        behavior_probability_share=0.0,
                    )
                )
        return RoutingResult(
            receiver_agent_id=receiver_agent_id,
            candidate_proposal=proposal,
            decisions=decisions,
            selected_memory_ids=selected,
            router_name=self.router_name,
            router_version=(
                f"{self.router_version}:{self.continuation_policy.policy_name}:"
                f"{self.continuation_policy.policy_version}"
            ),
        )
