"""Transfer-Aware Memory Controller for RIMA-Transfer (§17-25).

Three routing modes:

* ``EXPLOIT_ONLY``   — best known LCB >= gamma  → no global retrieval
* ``EXPLOIT_EXPLORE`` — delta < best LCB < gamma → known + global
* ``EXPLORE_ONLY``   — best LCB <= delta or no valid known → global only

Key invariant: context_budget = 1 (one memory per receiver per task).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from smtr.memory.shared_memory_pool import SharedMemory, SharedMemoryPool
from smtr.rima.features import ReceiverConditionedTransferFeatures
from smtr.rima.transfer_policy import TransferPolicy, lower_confidence_bound
from smtr.rima.transfer_state import (
    ReceiverTransferState,
    ReceiverTransferStateContainer,
)
from smtr.router.official_score_transfer_critic import (
    BootstrapOfficialScoreTransferCritic,
    MatchedInterventionExample,
)

__all__ = [
    "RoutingMode",
    "TransferCandidateDecision",
    "TransferRoutingPlan",
    "EpisodeTransferDecision",
    "RoutingSemanticsLog",
    "select_episode_edge",
    "build_routing_semantics_log",
    "TransferAwareMemoryController",
]


class RoutingMode:
    EXPLOIT_ONLY = "exploit_only"
    EXPLOIT_EXPLORE = "exploit_explore"
    EXPLORE_ONLY = "explore_only"


@dataclass
class TransferCandidateDecision:
    memory_id: str
    receiver_id: str
    task_id: str

    candidate_source: str  # known | global

    mu_tau: float | None
    sigma_tau: float | None
    lcb: float | None

    eligible_for_context: bool
    selected_for_context: bool

    status: str  # positive | negative | self_transfer_excluded


@dataclass
class TransferRoutingPlan:
    receiver_id: str
    task_id: str

    routing_mode: str

    best_known_lcb: float | None

    known_candidates: list[TransferCandidateDecision] = field(default_factory=list)
    global_candidates: list[TransferCandidateDecision] = field(default_factory=list)

    selected_memory_ids: list[str] = field(default_factory=list)

    global_retrieval_triggered: bool = False


@dataclass
class EpisodeTransferDecision:
    """Episode-level single-edge transfer decision (§12).

    After evaluating all receiver plans for one task, at most one
    (receiver, memory) edge is selected for injection.
    """

    task_id: str

    selected_receiver_id: str | None
    selected_memory_id: str | None

    mu_tau: float | None
    sigma_tau: float | None
    lcb: float | None

    source: str | None
    # known | global | none


def select_episode_edge(
    receiver_plans: dict[str, TransferRoutingPlan],
    *,
    delta: float,
) -> EpisodeTransferDecision:
    """Select the globally best (receiver, memory) edge across all plans.

    Returns an ``EpisodeTransferDecision`` with at most one edge where
    ``lcb > delta``.  If no eligible edge exists, all fields are None.
    """
    task_id: str | None = None
    best_lcb: float | None = None
    best_candidate: TransferCandidateDecision | None = None
    best_source: str | None = None

    for plan in receiver_plans.values():
        if task_id is None:
            task_id = plan.task_id
        for c in plan.known_candidates + plan.global_candidates:
            if not c.eligible_for_context or c.lcb is None:
                continue
            if best_lcb is None or c.lcb > best_lcb:
                best_lcb = c.lcb
                best_candidate = c
                best_source = c.candidate_source

    if best_candidate is None or best_lcb is None or best_lcb <= delta:
        return EpisodeTransferDecision(
            task_id=task_id or "",
            selected_receiver_id=None,
            selected_memory_id=None,
            mu_tau=None,
            sigma_tau=None,
            lcb=None,
            source="none",
        )

    return EpisodeTransferDecision(
        task_id=task_id or "",
        selected_receiver_id=best_candidate.receiver_id,
        selected_memory_id=best_candidate.memory_id,
        mu_tau=best_candidate.mu_tau,
        sigma_tau=best_candidate.sigma_tau,
        lcb=best_lcb,
        source=best_source,
    )


@dataclass(frozen=True)
class RoutingSemanticsLog:
    """Immutable log of one task's full routing semantics (§17.1-17.2).

    Separates two levels of selection:

    * **Receiver-level** (``receiver_plans_generated``,
      ``candidate_receivers_considered``): how many receivers had plans
      and candidates generated.
    * **Episode-level** (``episode_selected_*``): the single
      (receiver, memory) edge chosen for injection.

    ``joint_exposure_count`` must always be 0 — the system never
    jointly exposes multiple memories to one receiver.
    """

    task_id: str

    # Episode-level (§17.2)
    episode_selected_receiver: str | None
    episode_selected_memory: str | None
    episode_selected_lcb: float | None

    # Receiver-level (§17.1)
    candidate_receivers_considered: int
    receiver_plans_generated: int

    # Invariant: must be 0
    joint_exposure_count: int = 0


def build_routing_semantics_log(
    receiver_plans: dict[str, "TransferRoutingPlan"],
    episode_decision: "EpisodeTransferDecision",
) -> RoutingSemanticsLog:
    """Build an immutable routing semantics log from plans + decision (§17)."""
    receivers_with_candidates = sum(
        1
        for plan in receiver_plans.values()
        if plan.known_candidates or plan.global_candidates
    )
    return RoutingSemanticsLog(
        task_id=episode_decision.task_id,
        episode_selected_receiver=episode_decision.selected_receiver_id,
        episode_selected_memory=episode_decision.selected_memory_id,
        episode_selected_lcb=episode_decision.lcb,
        candidate_receivers_considered=receivers_with_candidates,
        receiver_plans_generated=len(receiver_plans),
        joint_exposure_count=0,
    )


# Type alias for the feature builder callable.
# Signature: (memory, receiver_id, task, task_id) -> features
FeatureBuilder = Callable[
    [SharedMemory, str, dict[str, Any], str],
    ReceiverConditionedTransferFeatures,
]


class TransferAwareMemoryController:
    """Continual transfer-aware memory controller (§17-25).

    For each task and each receiver the controller:

    1. Recalls known candidates from the receiver transfer state.
    2. Re-predicts transfer effect (mu, sigma) with the frozen critic.
    3. Decides routing mode based on best known LCB vs delta / gamma.
    4. Optionally retrieves unseen global memories.
    5. Selects at most ``context_budget`` memories for injection.
    """

    def __init__(
        self,
        *,
        critic: BootstrapOfficialScoreTransferCritic,
        pool: SharedMemoryPool,
        transfer_states: ReceiverTransferStateContainer,
        policy: TransferPolicy,
        feature_builder: FeatureBuilder,
        known_probe_top_k: int = 20,
        global_explore_top_k: int = 5,
        context_budget: int = 1,
    ) -> None:
        self.critic = critic
        self.pool = pool
        self.transfer_states = transfer_states
        self.policy = policy
        self.feature_builder = feature_builder
        self.known_probe_top_k = known_probe_top_k
        self.global_explore_top_k = global_explore_top_k
        self.context_budget = context_budget

    # ------------------------------------------------------------------
    # Core algorithm (§18-25)
    # ------------------------------------------------------------------

    def plan_for_task(
        self,
        *,
        task: dict[str, Any],
        task_id: str,
        task_position: int,
        receiver_id: str,
    ) -> TransferRoutingPlan:
        """Execute the full routing plan for one (task, receiver) pair."""
        state = self.transfer_states.ensure(receiver_id)

        # --- Step A: recall known candidates from K_r ---
        known_mems = self._recall_known(state, task, receiver_id, task_position)

        # --- Step B: predict transfer effect for each known candidate ---
        known_decisions = self._predict_candidates(
            known_mems,
            task=task,
            task_id=task_id,
            task_position=task_position,
            receiver_id=receiver_id,
            candidate_source="known",
            state=state,
        )

        # --- Step C: determine routing mode ---
        valid_lcbs = [d.lcb for d in known_decisions if d.lcb is not None]
        best_known_lcb = max(valid_lcbs) if valid_lcbs else None
        mode = self._decide_mode(best_known_lcb)

        # --- Step D: optional global exploration ---
        global_decisions: list[TransferCandidateDecision] = []
        global_triggered = mode != RoutingMode.EXPLOIT_ONLY

        if global_triggered:
            global_mems = self.pool.retrieve_unseen(
                task,
                receiver_id,
                self.global_explore_top_k,
                current_task_position=task_position,
                exclude_memory_ids=set(state.known_memory_ids()),
            )
            # Filter self-transfer before prediction
            for mem in global_mems:
                if mem.source_agent_id == receiver_id:
                    global_decisions.append(
                        TransferCandidateDecision(
                            memory_id=mem.memory_id,
                            receiver_id=receiver_id,
                            task_id=task_id,
                            candidate_source="global",
                            mu_tau=None,
                            sigma_tau=None,
                            lcb=None,
                            eligible_for_context=False,
                            selected_for_context=False,
                            status="self_transfer_excluded",
                        )
                    )
                    continue

                # Register in K_r (even if negative)
                state.register_memory(
                    memory_id=mem.memory_id,
                    source_agent_id=mem.source_agent_id,
                    task_id=task_id,
                    task_position=task_position,
                )

            # Predict for non-self-transfer global candidates
            non_self_mems = [
                m for m in global_mems if m.source_agent_id != receiver_id
            ]
            global_predicted = self._predict_candidates(
                non_self_mems,
                task=task,
                task_id=task_id,
                task_position=task_position,
                receiver_id=receiver_id,
                candidate_source="global",
                state=state,
            )
            # Merge: self-transfer decisions first, then predicted
            self_transfer_decisions = [
                d for d in global_decisions if d.status == "self_transfer_excluded"
            ]
            global_decisions = self_transfer_decisions + global_predicted

        # --- Step E: select top eligible candidates ---
        all_decisions = known_decisions + global_decisions
        eligible = [d for d in all_decisions if d.eligible_for_context]
        eligible.sort(key=lambda d: (-( d.lcb or 0.0), -(d.mu_tau or 0.0), d.memory_id))
        selected = eligible[: self.context_budget]
        for d in selected:
            d.selected_for_context = True
            state.mark_selected(d.memory_id)

        return TransferRoutingPlan(
            receiver_id=receiver_id,
            task_id=task_id,
            routing_mode=mode,
            best_known_lcb=best_known_lcb,
            known_candidates=known_decisions,
            global_candidates=global_decisions,
            selected_memory_ids=[d.memory_id for d in selected],
            global_retrieval_triggered=global_triggered,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _recall_known(
        self,
        state: ReceiverTransferState,
        task: dict[str, Any],
        receiver_id: str,
        task_position: int,
    ) -> list[SharedMemory]:
        if len(state) == 0:
            return []
        return self.pool.rank_subset(
            memory_ids=state.known_memory_ids(),
            task=task,
            receiver_id=receiver_id,
            current_task_position=task_position,
            top_k=self.known_probe_top_k,
        )

    def _predict_candidates(
        self,
        memories: list[SharedMemory],
        *,
        task: dict[str, Any],
        task_id: str,
        task_position: int,
        receiver_id: str,
        candidate_source: str,
        state: ReceiverTransferState,
    ) -> list[TransferCandidateDecision]:
        decisions: list[TransferCandidateDecision] = []
        for mem in memories:
            features = self.feature_builder(mem, receiver_id, task, task_id)
            example = MatchedInterventionExample(
                task_id=task_id,
                memory_id=mem.memory_id,
                receiver_id=receiver_id,
                source_agent_id=mem.source_agent_id,
                official_expose_score=None,
                official_withhold_score=None,
                features=features,
            )
            dist = self.critic.predict_distribution(example)

            if dist.mu_tau is None or dist.sigma_tau is None:
                # Self-transfer or invalid
                decisions.append(
                    TransferCandidateDecision(
                        memory_id=mem.memory_id,
                        receiver_id=receiver_id,
                        task_id=task_id,
                        candidate_source=candidate_source,
                        mu_tau=None,
                        sigma_tau=None,
                        lcb=None,
                        eligible_for_context=False,
                        selected_for_context=False,
                        status="self_transfer_excluded",
                    )
                )
                continue

            lcb = lower_confidence_bound(dist.mu_tau, dist.sigma_tau, self.policy.beta)
            eligible = lcb > self.policy.delta

            status = "positive" if eligible else "negative"

            # Record prediction in transfer state
            state.record_prediction(
                memory_id=mem.memory_id,
                task_id=task_id,
                task_position=task_position,
                mu_tau=dist.mu_tau,
                sigma_tau=dist.sigma_tau,
                lcb=lcb,
                status=status,
                candidate_source=candidate_source,
            )

            decisions.append(
                TransferCandidateDecision(
                    memory_id=mem.memory_id,
                    receiver_id=receiver_id,
                    task_id=task_id,
                    candidate_source=candidate_source,
                    mu_tau=dist.mu_tau,
                    sigma_tau=dist.sigma_tau,
                    lcb=lcb,
                    eligible_for_context=eligible,
                    selected_for_context=False,
                    status=status,
                )
            )
        return decisions

    def _decide_mode(self, best_known_lcb: float | None) -> str:
        if best_known_lcb is None:
            return RoutingMode.EXPLORE_ONLY
        if best_known_lcb <= self.policy.delta:
            return RoutingMode.EXPLORE_ONLY
        if best_known_lcb >= self.policy.gamma:
            return RoutingMode.EXPLOIT_ONLY
        return RoutingMode.EXPLOIT_EXPLORE
