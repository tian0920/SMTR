"""τ³-bench evaluation wrapper.

Wraps τ³'s official RewardInfo / SimulationRun into SMTR-compatible outcomes.
Does NOT replicate τ³'s evaluator — delegates to the official evaluation logic.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class TauOutcome(BaseModel):
    """SMTR-compatible outcome extracted from τ³ official evaluation."""

    success: bool
    reward: float
    task_id: str
    domain: str
    reward_info: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


def extract_outcome(simulation_run: Any, *, domain: str = "retail") -> TauOutcome:
    """Extract SMTR-compatible outcome from τ³ SimulationRun.

    Args:
        simulation_run: A τ³ SimulationRun object (or compatible dict).
        domain: The τ³ domain name.

    Returns:
        TauOutcome with success, reward, and official reward_info.
    """
    if isinstance(simulation_run, dict):
        task_id = simulation_run.get("task_id", "unknown")
        reward_info_raw = simulation_run.get("reward_info")
    else:
        task_id = getattr(simulation_run, "task_id", "unknown")
        reward_info_raw = getattr(simulation_run, "reward_info", None)

    if reward_info_raw is None:
        return TauOutcome(
            success=False,
            reward=0.0,
            task_id=task_id,
            domain=domain,
            metadata={"note": "no reward_info available"},
        )

    if isinstance(reward_info_raw, dict):
        reward = float(reward_info_raw.get("reward", 0.0))
        reward_info_dict = reward_info_raw
    else:
        reward = float(getattr(reward_info_raw, "reward", 0.0))
        reward_info_dict = (
            reward_info_raw.model_dump()
            if hasattr(reward_info_raw, "model_dump")
            else {}
        )

    return TauOutcome(
        success=reward > 0.0,
        reward=reward,
        task_id=task_id,
        domain=domain,
        reward_info=reward_info_dict,
        metadata={"domain": domain, "task_id": task_id},
    )


def summarize_outcomes(outcomes: list[TauOutcome]) -> dict[str, Any]:
    """Summarize a list of τ³ outcomes for reporting."""
    if not outcomes:
        return {"count": 0, "success_rate": 0.0, "mean_reward": 0.0}

    successes = sum(1 for o in outcomes if o.success)
    total_reward = sum(o.reward for o in outcomes)
    n = len(outcomes)

    return {
        "count": n,
        "successes": successes,
        "success_rate": successes / n,
        "mean_reward": total_reward / n,
        "total_reward": total_reward,
        "per_task": [
            {"task_id": o.task_id, "success": o.success, "reward": o.reward}
            for o in outcomes
        ],
    }
