"""Post-task causal probing for continual transfer learning (§14).

This module implements forward-only post-task causal probing:

1. After a task's scored decision is finalized (Y_t is frozen),
   select the best global candidate for probing.
2. Run matched expose/withhold episodes to collect causal evidence.
3. Return OnlineTransferEvidence that may only affect future tasks.

Probe selection policy (§14.5):
    - Only probe when routing_mode != EXPLOIT_ONLY
    - Only probe when global_candidates is non-empty
    - Select the global candidate with highest LCB
    - At most 1 candidate edge per task

Shared control (§14.4):
    When probing multiple candidates for the same (task, receiver, seed),
    the withhold (no-memory) control is shared — not re-executed per candidate.
"""

from __future__ import annotations

import statistics
from typing import Any, Protocol

from smtr.rima.online_transfer_evidence import OnlineTransferEvidence
from smtr.rima.transfer_controller import RoutingMode, TransferCandidateDecision

__all__ = [
    "PostTaskTransferProbe",
    "ProbeSelectionPolicy",
    "select_probe_candidate",
]


class ProbeSelectionPolicy:
    """Deterministic probe candidate selection (§14.5).

    Selects the global candidate with highest LCB when:
    - routing_mode != EXPLOIT_ONLY
    - global_candidates is non-empty
    """

    @staticmethod
    def select(
        routing_mode: str,
        global_candidates: list[TransferCandidateDecision],
    ) -> TransferCandidateDecision | None:
        """Select the best global candidate for probing.

        Returns None if no probing should occur.
        """
        if routing_mode == RoutingMode.EXPLOIT_ONLY:
            return None
        if not global_candidates:
            return None

        # Filter to eligible candidates with valid LCB
        eligible = [
            c for c in global_candidates
            if c.eligible_for_context and c.lcb is not None
        ]
        if not eligible:
            return None

        # Select highest LCB
        return max(eligible, key=lambda c: c.lcb)


def select_probe_candidate(
    receiver_plans: dict[str, Any],
) -> tuple[str | None, TransferCandidateDecision | None]:
    """Select the best probe candidate across all receiver plans.

    Returns:
        (receiver_id, candidate) tuple, or (None, None) if no probe.
    """
    best_receiver: str | None = None
    best_candidate: TransferCandidateDecision | None = None
    best_lcb: float | None = None

    for rid, plan in receiver_plans.items():
        if plan.routing_mode == RoutingMode.EXPLOIT_ONLY:
            continue
        if not plan.global_candidates:
            continue

        candidate = ProbeSelectionPolicy.select(
            plan.routing_mode,
            plan.global_candidates,
        )
        if candidate is None:
            continue

        if best_lcb is None or candidate.lcb > best_lcb:
            best_lcb = candidate.lcb
            best_candidate = candidate
            best_receiver = rid

    return best_receiver, best_candidate


class EpisodeRunner(Protocol):
    """Protocol for running MARBLE episodes (for mocking in tests)."""

    def run_episode(
        self,
        *,
        task: dict[str, Any],
        receiver_id: str,
        memory_id: str | None,
        generation_seed: int,
    ) -> float:
        """Run one episode and return the official task score.

        Args:
            task: task definition.
            receiver_id: agent receiving the memory (or None for control).
            memory_id: memory to inject (None for withhold/control).
            generation_seed: MARBLE generation seed.

        Returns:
            Official task score in [0, 1].
        """
        ...


class PostTaskTransferProbe:
    """Collect matched causal evidence via post-task probing (§14).

    IMPORTANT: This class must only be invoked AFTER the scored decision
    for a task has been finalized. The evidence collected may only affect
    decisions at future tasks (forward-only invariant).

    Args:
        episode_runner: callable that runs MARBLE episodes.
        generation_seeds: seeds for matched expose/withhold runs.
    """

    def __init__(
        self,
        episode_runner: EpisodeRunner,
        generation_seeds: list[int] | None = None,
    ) -> None:
        self.episode_runner = episode_runner
        self.generation_seeds = generation_seeds or [0]

    def collect(
        self,
        *,
        task: dict[str, Any],
        task_id: str,
        task_position: int,
        receiver_id: str,
        memory_id: str,
    ) -> OnlineTransferEvidence:
        """Collect matched causal evidence for one (receiver, memory) edge.

        For each generation seed, runs:
        - expose branch: inject memory to receiver
        - withhold branch: no-memory control

        Shared control (§14.4): If probing multiple candidates for the
        same (task, receiver, seed), callers should cache withhold scores
        to avoid redundant control runs.

        Returns:
            OnlineTransferEvidence with matched tau estimate.
        """
        expose_scores: list[float] = []
        withhold_scores: list[float] = []

        for seed in self.generation_seeds:
            # Expose: inject memory to receiver
            expose_score = self.episode_runner.run_episode(
                task=task,
                receiver_id=receiver_id,
                memory_id=memory_id,
                generation_seed=seed,
            )
            expose_scores.append(expose_score)

            # Withhold: no-memory control
            withhold_score = self.episode_runner.run_episode(
                task=task,
                receiver_id=receiver_id,
                memory_id=None,
                generation_seed=seed,
            )
            withhold_scores.append(withhold_score)

        # Compute matched tau
        differences = [e - w for e, w in zip(expose_scores, withhold_scores)]
        observed_tau = statistics.mean(differences)

        if len(differences) > 1:
            tau_std = statistics.stdev(differences)
        else:
            tau_std = None

        return OnlineTransferEvidence(
            task_id=task_id,
            task_position=task_position,
            receiver_id=receiver_id,
            memory_id=memory_id,
            expose_scores=expose_scores,
            withhold_scores=withhold_scores,
            observed_tau=observed_tau,
            tau_std=tau_std,
            generation_seeds=list(self.generation_seeds),
        )

    def collect_with_shared_control(
        self,
        *,
        task: dict[str, Any],
        task_id: str,
        task_position: int,
        receiver_id: str,
        memory_id: str,
        cached_withhold_scores: dict[int, float] | None = None,
    ) -> tuple[OnlineTransferEvidence, dict[int, float]]:
        """Collect evidence reusing cached withhold scores when available.

        Args:
            cached_withhold_scores: mapping seed -> withhold score from
                a previous probe on the same (task, receiver).

        Returns:
            (evidence, updated_withhold_scores) tuple.
        """
        expose_scores: list[float] = []
        withhold_scores: list[float] = []
        updated_control = dict(cached_withhold_scores or {})

        for seed in self.generation_seeds:
            # Expose: always run
            expose_score = self.episode_runner.run_episode(
                task=task,
                receiver_id=receiver_id,
                memory_id=memory_id,
                generation_seed=seed,
            )
            expose_scores.append(expose_score)

            # Withhold: use cached if available
            if seed in updated_control:
                withhold_scores.append(updated_control[seed])
            else:
                withhold_score = self.episode_runner.run_episode(
                    task=task,
                    receiver_id=receiver_id,
                    memory_id=None,
                    generation_seed=seed,
                )
                withhold_scores.append(withhold_score)
                updated_control[seed] = withhold_score

        # Compute matched tau
        differences = [e - w for e, w in zip(expose_scores, withhold_scores)]
        observed_tau = statistics.mean(differences)

        if len(differences) > 1:
            tau_std = statistics.stdev(differences)
        else:
            tau_std = None

        evidence = OnlineTransferEvidence(
            task_id=task_id,
            task_position=task_position,
            receiver_id=receiver_id,
            memory_id=memory_id,
            expose_scores=expose_scores,
            withhold_scores=withhold_scores,
            observed_tau=observed_tau,
            tau_std=tau_std,
            generation_seeds=list(self.generation_seeds),
        )
        return evidence, updated_control
