"""Single-receiver set-level paired evaluation for MARBLE.

Compares S_K vs. ∅ for one receiver agent using exposure_override.
Both branches use the SAME SMTRMarbleAgent subclass via SMTRMarbleEngine.
The only difference is whether selected payload is injected.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from smtr.counterfactual.marble_eval import MarbleOutcome, extract_marble_outcome
from smtr.memory.pool import SharedMemoryPool

logger = logging.getLogger(__name__)


class MarbleBranchResult(BaseModel):
    """Result from a single branch (share or withhold) of paired rollout."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    outcome: MarbleOutcome
    num_iterations: int = 0
    error: str | None = None
    raw_summary: dict[str, Any] = Field(default_factory=dict)


class MarblePairedOutcome(BaseModel):
    """Paired rollout outcome from MARBLE multi-agent task.

    Single-receiver design: only target_receiver_agent_id gets memory.
    Both branches use SMTRMarbleAgent — only exposure_override differs.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    task_id: str
    environment_type: str
    target_receiver_agent_id: str
    seed: int
    share_result: MarbleBranchResult
    withhold_result: MarbleBranchResult
    y_share: int = 0
    y_withhold: int = 0
    transfer_class: str = "neutral_failure"
    # Metadata (topology goes here, NOT into ContextFingerprint)
    coordination_mode: str = ""
    num_agents: int = 0
    selected_memory_ids: list[str] = Field(default_factory=list)
    routing_trace: list[dict[str, Any]] = Field(default_factory=list)
    data_source: str = "marble"

    @classmethod
    def from_branch_results(
        cls,
        *,
        task_id: str,
        environment_type: str,
        target_receiver_agent_id: str,
        seed: int,
        share_result: MarbleBranchResult,
        withhold_result: MarbleBranchResult,
        selected_memory_ids: list[str],
        routing_trace: list[dict[str, Any]],
        coordination_mode: str = "",
        num_agents: int = 0,
    ) -> MarblePairedOutcome:
        """Build a MarblePairedOutcome from two branch results."""
        y_share = 1 if share_result.outcome.success else 0
        y_withhold = 1 if withhold_result.outcome.success else 0

        # Classify transfer
        transfer_class = _classify_transfer(y_share, y_withhold)

        return cls(
            task_id=task_id,
            environment_type=environment_type,
            target_receiver_agent_id=target_receiver_agent_id,
            seed=seed,
            share_result=share_result,
            withhold_result=withhold_result,
            y_share=y_share,
            y_withhold=y_withhold,
            transfer_class=transfer_class,
            coordination_mode=coordination_mode,
            num_agents=num_agents,
            selected_memory_ids=selected_memory_ids,
            routing_trace=routing_trace,
        )


def _classify_transfer(y_share: int, y_withhold: int) -> str:
    """Classify transfer type from binary outcomes.

    (Y^1, Y^0) → transfer class
    (1, 1) → neutral_success    (memory doesn't change outcome)
    (1, 0) → positive_transfer  (memory helps)
    (0, 1) → negative_transfer  (memory hurts)
    (0, 0) → neutral_failure    (memory doesn't change outcome)
    """
    if y_share == 1 and y_withhold == 1:
        return "neutral_success"
    elif y_share == 1 and y_withhold == 0:
        return "positive_transfer"
    elif y_share == 0 and y_withhold == 1:
        return "negative_transfer"
    else:
        return "neutral_failure"


class MarblePairedRolloutRunner:
    """Run task-start paired rollout on MARBLE.

    Single-receiver set-level evaluation:
    - Both branches use SMTRMarbleEngine + SMTRMarbleAgent for target receiver
    - Share branch: exposure_override=None → router selects S_K
    - Withhold branch: exposure_override=[] → forces S_K=∅
    - All other agents (PromptAwareBaseAgent), graph, model, task, seed: IDENTICAL
    """

    def __init__(self) -> None:
        pass

    def run_paired_episode(
        self,
        config: Any,
        *,
        memory_pool: SharedMemoryPool,
        selected_memory_ids: list[str],
        target_receiver_agent_id: str,
        seed: int,
    ) -> MarblePairedOutcome:
        """Run paired episode: share branch + withhold branch.

        Args:
            config: MARBLE Config object.
            memory_pool: SMTR memory pool with routing cards and payloads.
            selected_memory_ids: Memory IDs selected for injection (S_K).
            target_receiver_agent_id: Agent ID of the target receiver.
            seed: Random seed for reproducibility.
        """
        from smtr.runtime.marble_agent import SMTRMarbleEngine

        # Branch A: share (router runs normally)
        share_result = self._run_branch(
            config=config,
            memory_pool=memory_pool,
            exposure_override=None,  # router selects S_K
            target_receiver_agent_id=target_receiver_agent_id,
            seed=seed,
            branch_label="share",
        )

        # Branch B: withhold (force S_K=∅ via exposure_override=[])
        withhold_result = self._run_branch(
            config=config,
            memory_pool=memory_pool,
            exposure_override=[],  # forces empty
            target_receiver_agent_id=target_receiver_agent_id,
            seed=seed,
            branch_label="withhold",
        )

        # Determine environment type from config
        env_type = config.environment.get("type", "unknown") if hasattr(config, "environment") else "unknown"
        coordination_mode = config.coordination_mode if hasattr(config, "coordination_mode") else ""
        num_agents = len(config.agents) if hasattr(config, "agents") else 0

        return MarblePairedOutcome.from_branch_results(
            task_id=config.task.get("content", "")[:50] if hasattr(config, "task") else "",
            environment_type=env_type,
            target_receiver_agent_id=target_receiver_agent_id,
            seed=seed,
            share_result=share_result,
            withhold_result=withhold_result,
            selected_memory_ids=selected_memory_ids,
            routing_trace=share_result.raw_summary.get("routing_trace", []),
            coordination_mode=coordination_mode,
            num_agents=num_agents,
        )

    def _run_branch(
        self,
        config: Any,
        memory_pool: SharedMemoryPool,
        exposure_override: list[str] | None,
        target_receiver_agent_id: str,
        seed: int,
        branch_label: str,
    ) -> MarbleBranchResult:
        """Run one branch of the paired rollout.

        CRITICAL: Always uses SMTRMarbleEngine with SMTRMarbleAgent for target receiver.
        The withhold branch does NOT swap in BaseAgent or PromptAwareBaseAgent.
        Only difference: exposure_override=[] → render_private_guidance() returns "".
        """
        from smtr.runtime.marble_agent import SMTRMarbleEngine

        logger.info(f"Running {branch_label} branch (exposure_override={exposure_override})")

        try:
            # Create SMTRMarbleEngine with appropriate exposure_override
            engine = SMTRMarbleEngine(
                config=config,
                target_receiver_agent_id=target_receiver_agent_id,
                smtr_memory_pool=memory_pool,
                exposure_override=exposure_override,
            )

            # Run the simulation
            engine.start()

            # Extract outcome from engine results
            # MARBLE writes results to JSONL, but we can also access evaluator metrics
            task_eval = engine.evaluator.metrics.get("task_evaluation", {})
            engine_result = {"task_evaluation": task_eval}

            # Count iterations
            num_iters = engine.current_iteration

            # Build outcome
            outcome = extract_marble_outcome(
                engine_result,
                task_id=f"{branch_label}_seed{seed}",
                environment_type=config.environment.get("type", "unknown") if hasattr(config, "environment") else "unknown",
                num_agents=len(engine.agents),
                num_iterations=num_iters,
            )

            return MarbleBranchResult(
                outcome=outcome,
                num_iterations=num_iters,
                raw_summary={"task_evaluation": task_eval},
            )

        except Exception as e:
            logger.error(f"Error in {branch_label} branch: {e}")
            # Return failure outcome on error
            error_outcome = MarbleOutcome(
                success=False,
                reward=0.0,
                task_id=f"{branch_label}_seed{seed}",
                environment_type="error",
                num_agents=0,
                num_iterations=0,
                metadata={"error": str(e)},
            )
            return MarbleBranchResult(
                outcome=error_outcome,
                error=str(e),
            )
