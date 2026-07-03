import copy
import random
from typing import Any

from smtr.counterfactual.continuation_policy import FrozenContinuationPolicy
from smtr.counterfactual.decision_points import canonical_digest
from smtr.counterfactual.forced_router import ForcedInterventionRouter
from smtr.counterfactual.schemas import (
    BranchOutcome,
    CandidateTraversalPlan,
    DecisionPoint,
    EvaluationGroupMetadata,
    PairedInterventionRecord,
    RuntimeSnapshot,
    routing_feature_snapshot_from_card,
    transfer_class_from_outcomes,
)
from smtr.counterfactual.snapshot import ReadOnlyPinnedMemoryView
from smtr.memory.execution_evidence import build_context_fingerprint
from smtr.memory.repository import SharedMemoryRepository
from smtr.runtime.environment import ToyEnvironment
from smtr.runtime.graph import WorkflowRunner


class CounterfactualIntegrityError(RuntimeError):
    pass


class PairedRolloutCollector:
    def collect(
        self,
        *,
        decision_point: DecisionPoint,
        traversal_plan: CandidateTraversalPlan,
        repository: SharedMemoryRepository,
        continuation_policy: FrozenContinuationPolicy,
        policy_round=None,
        evaluation_group_metadata: EvaluationGroupMetadata | None = None,
    ) -> PairedInterventionRecord:
        before_revision = repository.current_revision()
        runtime_snapshot = self._runtime_snapshot(decision_point)
        memory_snapshot_digest = canonical_digest(decision_point.memory_store_snapshot)
        memory_view = ReadOnlyPinnedMemoryView(
            repository=repository,
            snapshot=decision_point.memory_store_snapshot,
        )
        candidate_version = decision_point.memory_store_snapshot.get_active_version(
            traversal_plan.target_memory_id
        )
        cards_by_id = {
            card.memory_id: card for card in decision_point.memory_store_snapshot.routing_cards
        }
        candidate_card_snapshot = routing_feature_snapshot_from_card(
            cards_by_id[traversal_plan.target_memory_id]
        )
        selected_card_snapshots = [
            routing_feature_snapshot_from_card(cards_by_id[memory_id])
            for memory_id in traversal_plan.selected_before
        ]
        selected_payload_versions = {
            memory_id: decision_point.memory_store_snapshot.get_active_version(memory_id)
            for memory_id in traversal_plan.selected_before
        }

        share_router = ForcedInterventionRouter(
            traversal_plan=traversal_plan,
            branch_arm="share",
            continuation_policy=continuation_policy,
            receiver_agent_id=decision_point.receiver_agent_id,
        )
        withhold_router = ForcedInterventionRouter(
            traversal_plan=traversal_plan,
            branch_arm="withhold",
            continuation_policy=continuation_policy,
            receiver_agent_id=decision_point.receiver_agent_id,
        )

        share_state = self._run_branch(
            runtime_snapshot=runtime_snapshot,
            router=share_router,
            memory_view=memory_view,
            branch_label="share",
        )
        withhold_state = self._run_branch(
            runtime_snapshot=runtime_snapshot,
            router=withhold_router,
            memory_view=memory_view,
            branch_label="withhold",
        )
        self._assert_only_intervention_difference(
            share_state=share_state,
            withhold_state=withhold_state,
            receiver_agent_id=decision_point.receiver_agent_id,
            target_memory_id=traversal_plan.target_memory_id,
        )
        after_revision = repository.current_revision()
        if before_revision != after_revision:
            raise CounterfactualIntegrityError("paired rollout changed live memory store revision")

        y_share = int(bool(share_state["team_success"]))
        y_withhold = int(bool(withhold_state["team_success"]))
        context = build_context_fingerprint(
            task_id=decision_point.task_id,
            task_tags=decision_point.candidate_proposal.request.task.split(),
            receiver_agent_id=decision_point.receiver_agent_id,
            receiver_role=decision_point.receiver_role,
            receiver_capabilities=decision_point.candidate_proposal.request.receiver_capabilities,
            environment_observation=decision_point.environment_snapshot,
            task_stage=decision_point.task_stage,
            selected_memory_ids=traversal_plan.selected_before,
            episode_id=decision_point.episode_id,
            decision_index=traversal_plan.target_index,
        )
        return PairedInterventionRecord(
            episode_id=decision_point.episode_id,
            task_id=decision_point.task_id,
            graph_node=decision_point.graph_node,
            receiver_agent_id=decision_point.receiver_agent_id,
            receiver_role=decision_point.receiver_role,
            task_stage=decision_point.task_stage,
            collection_round_id=getattr(policy_round, "round_id", None),
            continuation_policy_fingerprint=getattr(
                getattr(policy_round, "continuation_policy", None),
                "fingerprint",
                None,
            ),
            base_memory_store_revision=getattr(
                policy_round,
                "base_memory_store_revision",
                decision_point.memory_store_snapshot.store_revision,
            ),
            base_memory_snapshot_digest=getattr(
                policy_round,
                "base_memory_snapshot_digest",
                memory_snapshot_digest,
            ),
            evaluation_group_metadata=(
                evaluation_group_metadata or EvaluationGroupMetadata()
            ),
            candidate_memory_id=traversal_plan.target_memory_id,
            candidate_payload_version=candidate_version,
            candidate_card_snapshot=candidate_card_snapshot,
            selected_before_card_snapshots=selected_card_snapshots,
            selected_before_payload_versions=selected_payload_versions,
            candidate_order=traversal_plan.candidate_order,
            target_index=traversal_plan.target_index,
            selected_before=traversal_plan.selected_before,
            prefix_size=len(traversal_plan.selected_before),
            target_selection_policy_name=traversal_plan.target_selection_policy_name,
            target_selection_policy_version=traversal_plan.target_selection_policy_version,
            prefix_sampling_policy_name=traversal_plan.prefix_sampling_policy_name,
            prefix_sampling_policy_version=traversal_plan.prefix_sampling_policy_version,
            target_selection_probability=traversal_plan.target_selection_probability,
            prefix_sampling_probability=traversal_plan.prefix_sampling_probability,
            decision_context=context,
            memory_store_revision=decision_point.memory_store_snapshot.store_revision,
            memory_snapshot_digest=memory_snapshot_digest,
            runtime_snapshot_digest=runtime_snapshot.snapshot_digest,
            continuation_policy_name=continuation_policy.policy_name,
            continuation_policy_version=continuation_policy.policy_version,
            common_seed=decision_point.run_seed,
            share_outcome=self._branch_outcome(
                state=share_state,
                receiver_agent_id=decision_point.receiver_agent_id,
                target_memory_id=traversal_plan.target_memory_id,
            ),
            withhold_outcome=self._branch_outcome(
                state=withhold_state,
                receiver_agent_id=decision_point.receiver_agent_id,
                target_memory_id=traversal_plan.target_memory_id,
            ),
            y_share=y_share,
            y_withhold=y_withhold,
            transfer_class=transfer_class_from_outcomes(y_share, y_withhold),
        )

    def _runtime_snapshot(self, decision_point: DecisionPoint) -> RuntimeSnapshot:
        random_state: dict[str, Any] = {"python_random_state": random.getstate()}
        try:
            import numpy as np  # type: ignore[import-not-found]

            random_state["numpy_random_state"] = np.random.get_state()
        except Exception:
            random_state["numpy_random_state"] = None
        payload = {
            "graph_state": decision_point.graph_state_snapshot,
            "environment_snapshot": decision_point.environment_snapshot,
            "graph_node": decision_point.graph_node,
            "receiver_agent_id": decision_point.receiver_agent_id,
            "run_seed": decision_point.run_seed,
            "memory_store_revision": decision_point.memory_store_snapshot.store_revision,
        }
        return RuntimeSnapshot(
            graph_state=copy.deepcopy(decision_point.graph_state_snapshot),
            environment_snapshot=copy.deepcopy(decision_point.environment_snapshot),
            graph_node=decision_point.graph_node,
            receiver_agent_id=decision_point.receiver_agent_id,
            run_seed=decision_point.run_seed,
            random_state=random_state,
            memory_store_revision=decision_point.memory_store_snapshot.store_revision,
            snapshot_digest=canonical_digest(payload),
        )

    def _run_branch(
        self,
        *,
        runtime_snapshot: RuntimeSnapshot,
        router: ForcedInterventionRouter,
        memory_view: ReadOnlyPinnedMemoryView,
        branch_label: str,
    ) -> dict[str, Any]:
        random.setstate(runtime_snapshot.random_state["python_random_state"])
        env = ToyEnvironment.clone_from_snapshot(
            runtime_snapshot.environment_snapshot,
            seed=runtime_snapshot.run_seed,
        )
        state = copy.deepcopy(runtime_snapshot.graph_state)
        state["environment_observation"] = copy.deepcopy(runtime_snapshot.environment_snapshot)
        return WorkflowRunner().run_from_node(
            start_node=runtime_snapshot.graph_node,
            graph_state=state,
            environment=env,
            router=router,
            memory_view=memory_view,
            run_seed=runtime_snapshot.run_seed,
            branch_label=branch_label,
        )

    def _branch_outcome(
        self,
        *,
        state: dict[str, Any],
        receiver_agent_id: str,
        target_memory_id: str,
    ) -> BranchOutcome:
        visible_payloads = state["agent_local_context"][receiver_agent_id].get(
            "visible_payloads", []
        )
        visible_ids = [payload.get("memory_id") for payload in visible_payloads]
        selected_at_target = state["selected_memory_ids_by_agent"].get(receiver_agent_id, [])
        return BranchOutcome(
            team_success=bool(state["team_success"]),
            team_reward=float(state["team_reward"] or 0.0),
            team_summary=str(state["team_summary"] or ""),
            final_environment_observation=state["environment_observation"],
            selected_memory_ids_by_agent=state["selected_memory_ids_by_agent"],
            router_trace=state["router_trace"],
            target_memory_visible_to_receiver=target_memory_id in visible_ids,
            selected_final_at_target_node=selected_at_target,
        )

    def _assert_only_intervention_difference(
        self,
        *,
        share_state: dict[str, Any],
        withhold_state: dict[str, Any],
        receiver_agent_id: str,
        target_memory_id: str,
    ) -> None:
        share_selected = set(
            share_state["selected_memory_ids_by_agent"].get(receiver_agent_id, [])
        )
        withhold_selected = set(
            withhold_state["selected_memory_ids_by_agent"].get(receiver_agent_id, [])
        )
        if target_memory_id not in share_selected:
            raise CounterfactualIntegrityError("share branch did not select target memory")
        if target_memory_id in withhold_selected:
            raise CounterfactualIntegrityError("withhold branch selected target memory")
