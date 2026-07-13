"""Evaluation bridge: MARBLE DB Environment → SMTR Outcomes.

Maps MARBLE's Database Error Analysis evaluation to SMTR's binary success/failure.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


class MarbleOutcome(BaseModel):
    """MARBLE task outcome mapped to SMTR format."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    success: bool
    reward: float  # normalized to [0, 1]
    task_id: str
    environment_type: str
    num_agents: int
    num_iterations: int
    milestone_scores: dict[str, float] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


def extract_marble_outcome(
    engine_result: dict[str, Any],
    *,
    task_id: str = "",
    environment_type: str = "DB",
    num_agents: int = 0,
    num_iterations: int = 0,
) -> MarbleOutcome:
    """Extract outcome from MARBLE engine result.

    IMPORTANT: Do NOT hardcode success criteria.
    First run DB environment (Task 1c), inspect MARBLE evaluator's actual
    output structure, then define binary success from official evaluator output.

    For the DB environment, the evaluator stores:
        metrics["task_evaluation"] = {
            'root_cause': root_causes,
            'predicted': result,
        }

    Success is determined by comparing predicted root causes against labels.
    """
    # Extract task evaluation from engine result
    task_evaluation = engine_result.get("task_evaluation", {})

    if isinstance(task_evaluation, dict) and "root_cause" in task_evaluation:
        # DB environment: compare predicted vs actual root causes
        root_causes = task_evaluation.get("root_cause", [])
        predicted = task_evaluation.get("predicted", "")

        # Simple heuristic: check if any root cause label appears in the prediction
        # This will be refined after Task 1c confirms the actual output structure
        success = _check_db_success(root_causes, predicted)
        reward = 1.0 if success else 0.0
    else:
        # Unknown evaluation format — default to failure
        logger.warning(
            f"Unknown task_evaluation format: {type(task_evaluation)}. "
            "Defaulting to success=False."
        )
        success = False
        reward = 0.0

    return MarbleOutcome(
        success=success,
        reward=reward,
        task_id=task_id,
        environment_type=environment_type,
        num_agents=num_agents,
        num_iterations=num_iterations,
        metadata={"raw_evaluation": task_evaluation},
    )


def _check_db_success(root_causes: list[str], predicted: str) -> bool:
    """Check if predicted root causes match actual root causes.

    This is a simple heuristic. After Task 1c, this should be refined
    based on the actual MARBLE DB evaluator output structure.
    """
    if not root_causes or not predicted:
        return False

    predicted_lower = predicted.lower()
    # Check if any root cause label appears in the prediction
    matches = sum(1 for rc in root_causes if rc.lower() in predicted_lower)
    # Require at least one match
    return matches > 0


def extract_marble_outcome_from_jsonl(
    jsonl_entry: dict[str, Any],
    *,
    task_id: str = "",
) -> MarbleOutcome:
    """Extract outcome from a MARBLE JSONL output file entry.

    MARBLE writes results to JSONL via Engine._write_to_jsonl().
    The entry contains summary_data with task_evaluation, token_usage, etc.
    """
    environment_type = "unknown"
    task_content = jsonl_entry.get("task", "")

    # Infer environment type from task content or metadata
    if "database" in task_content.lower() or "db" in task_content.lower():
        environment_type = "DB"
    elif "werewolf" in task_content.lower():
        environment_type = "Werewolf"

    # Count agents from iteration data
    iterations = jsonl_entry.get("iterations", [])
    num_iterations = len(iterations)
    num_agents = 0
    if iterations:
        task_assignments = iterations[0].get("task_assignments", {})
        num_agents = len(task_assignments)

    return extract_marble_outcome(
        jsonl_entry,
        task_id=task_id,
        environment_type=environment_type,
        num_agents=num_agents,
        num_iterations=num_iterations,
    )
