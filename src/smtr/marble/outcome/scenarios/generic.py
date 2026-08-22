"""Generic outcome evaluator for non-database MARBLE scenarios.

Delegates to the MARBLE native evaluator when available, otherwise
falls back to parsing the engine output for team_success/score fields.

Supports: bargaining, coding, minecraft, research.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from smtr.counterfactual.decision_points import canonical_digest
from smtr.marble.outcome.protocol import MarbleOutcome


class GenericOutcomeEvaluator:
    """Evaluate MARBLE results for non-database scenarios.

    Attempts to use the MARBLE native evaluator first. If unavailable,
    falls back to parsing the raw output for ``team_success`` and ``score``.
    """

    def __init__(self, scenario: str) -> None:
        self.scenario = scenario
        self.evaluator_name = f"marble_{scenario}_generic_evaluator"

    def evaluate(self, *, task: object, run_result: object) -> MarbleOutcome:
        task_dict = task if isinstance(task, dict) else {}
        run_dict = run_result if isinstance(run_result, dict) else {}

        # Check if MARBLE native evaluator already produced results
        task_eval = run_dict.get("task_evaluation")
        native_executed = isinstance(task_eval, dict)
        native_digest = canonical_digest(task_eval) if native_executed else None

        if not native_executed:
            task_eval, native_executed, native_digest = _call_native_evaluator(
                scenario=self.scenario,
                task=task_dict,
                run_result=run_dict,
            )

        if native_executed and isinstance(task_eval, dict):
            success = bool(task_eval.get("success", task_eval.get("team_success", False)))
            score = float(task_eval.get("score", 1.0 if success else 0.0))
            return MarbleOutcome(
                success=success,
                score=score,
                failure_reason=None if success else "native_evaluator_failed",
                environment_valid=True,
                evaluator_name=self.evaluator_name,
                raw_result_digest=canonical_digest(run_result),
                native_evaluator_executed=True,
                native_evaluator_name=self.evaluator_name,
                native_evaluator_result_digest=native_digest,
            )

        # Fallback: parse raw output
        success = bool(
            run_dict.get("team_success")
            or run_dict.get("success")
        )
        score = float(run_dict.get("score", 1.0 if success else 0.0))
        real_executed = bool(run_dict.get("real_engine_executed", False))

        return MarbleOutcome(
            success=success,
            score=score,
            failure_reason=None if success else "fallback_no_success_signal",
            environment_valid=real_executed,
            evaluator_name=self.evaluator_name,
            raw_result_digest=canonical_digest(run_result),
            native_evaluator_executed=False,
            native_evaluator_name=None,
            native_evaluator_result_digest=None,
        )


def _call_native_evaluator(
    *,
    scenario: str,
    task: dict[str, Any],
    run_result: dict[str, Any],
) -> tuple[dict[str, Any] | None, bool, str | None]:
    """Attempt to call the MARBLE native evaluator for this scenario."""
    marble_root = Path(
        os.environ.get("SMTR_MARBLE_ROOT", "/home/ecs-user/MARBLE")
    )
    sys.path.insert(0, str(marble_root))
    previous_cwd = Path.cwd()
    try:
        os.chdir(marble_root / "marble")
        from marble.evaluator.evaluator import Evaluator

        evaluator = Evaluator(metrics_config=task.get("metrics", {}))
        body = task.get("task", task)
        result = (
            run_result.get("final_output")
            or run_result.get("result")
            or ""
        )

        # Try scenario-specific evaluator method
        eval_method_name = f"evaluate_task_{scenario}"
        if hasattr(evaluator, eval_method_name):
            getattr(evaluator, eval_method_name)(
                task=str(body.get("content", "")),
                result=str(result),
            )
        elif hasattr(evaluator, "evaluate_task"):
            evaluator.evaluate_task(
                task=str(body.get("content", "")),
                result=str(result),
                scenario=scenario,
            )
        else:
            return None, False, None

        task_eval = evaluator.metrics.get("task_evaluation")
        if isinstance(task_eval, dict):
            return task_eval, True, canonical_digest(task_eval)
        return None, False, None
    except Exception:
        return None, False, None
    finally:
        os.chdir(previous_cwd)
        try:
            sys.path.remove(str(marble_root))
        except ValueError:
            pass
