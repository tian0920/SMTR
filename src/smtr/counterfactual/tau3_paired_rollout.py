"""τ³-bench task-start paired rollout runner.

Collects Y^(1) vs Y^(0) at task start for causal identification of τ(m|o,S).

For each held-out τ³ task:
1. Same task initial state + same user simulator config/seed
2. Branch A (share): SMTRTauAgent WITH memory injection → run full episode → Y^(1)
3. Branch B (withhold): SMTRTauAgent WITHOUT memory → run full episode → Y^(0)
4. Compute transfer label from outcome pair

No mid-dialogue snapshot/restore. Only task-start forking.

τ³-bench is an optional dependency. This module can be imported without τ³
for testing data models, but running paired rollouts requires τ³ installed.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from smtr.counterfactual.schemas import transfer_class_from_outcomes
from smtr.counterfactual.tau_eval import TauOutcome, extract_outcome

logger = logging.getLogger(__name__)

# τ³ imports — optional dependency
try:
    from tau2.orchestrator.orchestrator import Orchestrator
    from tau2.runner.build import build_environment, build_user
    from tau2.runner.simulation import run_simulation

    TAU3_AVAILABLE = True
except ImportError:
    TAU3_AVAILABLE = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Data models — always available
# ---------------------------------------------------------------------------


class Tau3BranchResult(BaseModel):
    """Result from a single branch (share or withhold) of paired rollout."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    outcome: TauOutcome
    termination_reason: str = "unknown"
    num_messages: int = 0
    messages: list[dict[str, Any]] = Field(default_factory=list)
    error: str | None = None


class Tau3PairedOutcome(BaseModel):
    """Paired rollout outcome from τ³-bench task-start forking.

    Contains Y^(1) (share) and Y^(0) (withhold) outcomes for a single task,
    plus derived transfer classification.
    """

    task_id: str
    domain: str
    seed: int

    # Branch outcomes
    share_result: Tau3BranchResult
    withhold_result: Tau3BranchResult

    # Derived from outcomes
    y_share: int = 0
    y_withhold: int = 0
    transfer_class: str = "neutral_failure"

    # Metadata
    selected_memory_ids: list[str] = Field(default_factory=list)
    routing_trace: list[dict[str, Any]] = Field(default_factory=list)
    data_source: str = "tau_bench"

    @classmethod
    def from_branch_results(
        cls,
        *,
        task_id: str,
        domain: str,
        seed: int,
        share_result: Tau3BranchResult,
        withhold_result: Tau3BranchResult,
        selected_memory_ids: list[str] | None = None,
        routing_trace: list[dict[str, Any]] | None = None,
    ) -> Tau3PairedOutcome:
        """Build paired outcome from two branch results."""
        y_share = int(share_result.outcome.success)
        y_withhold = int(withhold_result.outcome.success)
        tc = transfer_class_from_outcomes(y_share, y_withhold)

        return cls(
            task_id=task_id,
            domain=domain,
            seed=seed,
            share_result=share_result,
            withhold_result=withhold_result,
            y_share=y_share,
            y_withhold=y_withhold,
            transfer_class=tc,
            selected_memory_ids=selected_memory_ids or [],
            routing_trace=routing_trace or [],
        )


class Tau3PairedRolloutConfig(BaseModel):
    """Configuration for τ³ paired rollout."""

    domain: str = "retail"
    agent_llm: str = "gpt-4.1"
    user_llm: str = "gpt-4.1"
    agent_llm_args: dict[str, Any] = Field(default_factory=dict)
    user_llm_args: dict[str, Any] = Field(default_factory=dict)
    max_steps: int = 200
    max_errors: int = 10
    user_type: str = "user_simulator"


# ---------------------------------------------------------------------------
# Paired rollout runner — requires τ³-bench
# ---------------------------------------------------------------------------


class Tau3PairedRolloutRunner:
    """Run task-start paired rollout on τ³-bench.

    For each task, runs two branches:
    - Share: SMTRTauAgent with memory pool + selected payloads injected
    - Withhold: SMTRTauAgent without memory (or with empty pool)

    Both branches use the same task, seed, and user simulator configuration.
    The only difference is memory injection.

    Requires τ³-bench to be installed.
    """

    def __init__(self, config: Tau3PairedRolloutConfig | None = None) -> None:
        if not TAU3_AVAILABLE:
            raise ImportError(
                "τ³-bench is required for Tau3PairedRolloutRunner. "
                "Install with: cd tau2-bench && uv sync"
            )
        self.config = config or Tau3PairedRolloutConfig()

    def run_paired_episode(
        self,
        task: Any,
        *,
        memory_pool: Any | None = None,
        selected_memory_ids: list[str] | None = None,
        seed: int = 42,
    ) -> Tau3PairedOutcome:
        """Run paired rollout for a single τ³ task.

        Args:
            task: A τ³ Task object.
            memory_pool: SMTR SharedMemoryPool with routing cards and payloads.
            selected_memory_ids: Memory IDs to inject in the share branch.
                If None, uses all memories from the pool.
            seed: Random seed for both branches (must be same for fair comparison).

        Returns:
            Tau3PairedOutcome with share/withhold outcomes and transfer class.
        """
        task_id = task.id if hasattr(task, "id") else str(task)
        domain = self.config.domain
        selected_ids = selected_memory_ids or []

        logger.info(
            f"Running paired rollout: task={task_id}, "
            f"selected_memories={len(selected_ids)}, seed={seed}"
        )

        # Branch A: share (with memory injection)
        share_result = self._run_branch(
            task=task,
            memory_pool=memory_pool,
            selected_memory_ids=selected_ids,
            seed=seed,
            branch_label="share",
        )

        # Branch B: withhold (no memory injection)
        withhold_result = self._run_branch(
            task=task,
            memory_pool=None,
            selected_memory_ids=[],
            seed=seed,
            branch_label="withhold",
        )

        paired = Tau3PairedOutcome.from_branch_results(
            task_id=task_id,
            domain=domain,
            seed=seed,
            share_result=share_result,
            withhold_result=withhold_result,
            selected_memory_ids=selected_ids,
            routing_trace=share_result.outcome.metadata.get("routing_trace", []),
        )

        logger.info(
            f"Paired rollout complete: task={task_id}, "
            f"y_share={paired.y_share}, y_withhold={paired.y_withhold}, "
            f"transfer_class={paired.transfer_class}"
        )

        return paired

    def _run_branch(
        self,
        *,
        task: Any,
        memory_pool: Any | None,
        selected_memory_ids: list[str],
        seed: int,
        branch_label: str,
    ) -> Tau3BranchResult:
        """Run a single branch (share or withhold) of the paired rollout.

        Builds a τ³ orchestrator with SMTRTauAgent (with or without memory)
        and runs the simulation.
        """
        from smtr.runtime.tau3_agent import SMTRTauAgent

        # Build environment (same for both branches)
        environment = build_environment(self.config.domain)

        # Build user simulator (same for both branches)
        user = build_user(
            self.config.user_type,
            environment,
            task,
            llm=self.config.user_llm,
            llm_args=self.config.user_llm_args or None,
        )

        # Build SMTRTauAgent with or without memory
        agent = SMTRTauAgent(
            tools=environment.get_tools(),
            domain_policy=environment.get_policy(),
            llm=self.config.agent_llm,
            llm_args=self.config.agent_llm_args or None,
            memory_pool=memory_pool,
        )

        # Pre-set selected memory IDs for share branch
        if branch_label == "share" and selected_memory_ids:
            agent._preselected_memory_ids = selected_memory_ids

        # Build orchestrator
        orchestrator = Orchestrator(
            domain=self.config.domain,
            agent=agent,
            user=user,
            environment=environment,
            task=task,
            max_steps=self.config.max_steps,
            max_errors=self.config.max_errors,
            seed=seed,
        )

        # Run simulation
        try:
            simulation = run_simulation(orchestrator)
            outcome = extract_outcome(simulation, domain=self.config.domain)

            # Extract message history for diagnostics
            messages = []
            if hasattr(simulation, "messages") and simulation.messages:
                messages = [
                    {"role": getattr(m, "role", "unknown"), "content": str(m)}
                    for m in simulation.messages[:10]  # Limit for storage
                ]

            return Tau3BranchResult(
                outcome=outcome,
                termination_reason=getattr(
                    simulation, "termination_reason", "unknown"
                ),
                num_messages=len(simulation.messages)
                if hasattr(simulation, "messages")
                else 0,
                messages=messages,
            )
        except Exception as e:
            logger.error(f"Branch {branch_label} failed for task {task.id}: {e}")
            return Tau3BranchResult(
                outcome=TauOutcome(
                    success=False,
                    reward=0.0,
                    task_id=task.id if hasattr(task, "id") else "unknown",
                    domain=self.config.domain,
                    metadata={"error": str(e)},
                ),
                termination_reason="infrastructure_error",
                error=str(e),
            )


def summarize_paired_outcomes(
    outcomes: list[Tau3PairedOutcome],
) -> dict[str, Any]:
    """Summarize paired rollout outcomes for reporting.

    Args:
        outcomes: List of Tau3PairedOutcome from paired rollouts.

    Returns:
        Summary dict with transfer class distribution, success rates, etc.
    """
    if not outcomes:
        return {"count": 0}

    transfer_counts: dict[str, int] = {}
    share_successes = 0
    withhold_successes = 0

    for o in outcomes:
        tc = o.transfer_class
        transfer_counts[tc] = transfer_counts.get(tc, 0) + 1
        share_successes += o.y_share
        withhold_successes += o.y_withhold

    n = len(outcomes)
    return {
        "count": n,
        "transfer_class_distribution": transfer_counts,
        "share_success_rate": share_successes / n,
        "withhold_success_rate": withhold_successes / n,
        "positive_transfer_precision": (
            transfer_counts.get("positive", 0)
            / max(
                transfer_counts.get("positive", 0)
                + transfer_counts.get("negative", 0),
                1,
            )
        ),
        "negative_transfer_rate": (
            transfer_counts.get("negative", 0) / n
        ),
        "per_task": [
            {
                "task_id": o.task_id,
                "y_share": o.y_share,
                "y_withhold": o.y_withhold,
                "transfer_class": o.transfer_class,
            }
            for o in outcomes
        ],
    }
