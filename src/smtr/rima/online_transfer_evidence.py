"""Online transfer evidence from post-task causal probes (§14).

This module defines the evidence dataclass produced by forward-only
post-task causal probing. Each OnlineTransferEvidence represents a
matched expose/withhold observation collected AFTER the scored
decision for a task has been finalized.

Forward-only invariant:
    Evidence from task t may only influence decisions at t+1, t+2, ...
    The current task's score or memory selection is NEVER altered.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["OnlineTransferEvidence"]


@dataclass(frozen=True)
class OnlineTransferEvidence:
    """Matched causal observation from a post-task probe (§14).

    Attributes:
        task_id: task at which this probe was collected.
        task_position: position in the continual stream.
        receiver_id: agent that received the memory.
        memory_id: memory that was probed.
        expose_scores: official scores from expose branches (one per seed).
        withhold_scores: official scores from withhold branches (one per seed).
        observed_tau: mean(expose) - mean(withhold).
        tau_std: std of per-seed differences, or None if only one seed.
        generation_seeds: MARBLE generation seeds used for this probe.
    """

    task_id: str
    task_position: int

    receiver_id: str
    memory_id: str

    expose_scores: list[float]
    withhold_scores: list[float]

    observed_tau: float
    tau_std: float | None

    generation_seeds: list[int]
