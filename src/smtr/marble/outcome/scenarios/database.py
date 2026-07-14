"""Database pilot outcome evaluator."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from smtr.counterfactual.decision_points import canonical_digest
from smtr.marble.outcome.protocol import MarbleOutcome


class DatabaseOutcomeEvaluator:
    """Evaluate database pilot results when explicit predictions are available.

    The evaluator only consumes MARBLE's native ``task_evaluation`` payload from
    ``Evaluator.evaluate_task_db``. It does not infer success from process exit,
    non-empty output, or tool/database side effects.
    """

    scenario = "database"
    evaluator_name = "marble_database_evaluate_task_db"

    def evaluate(self, *, task: object, run_result: object) -> MarbleOutcome:
        task_dict = task if isinstance(task, dict) else {}
        run_dict = run_result if isinstance(run_result, dict) else {}
        task_eval = run_dict.get("task_evaluation")
        native_executed = isinstance(task_eval, dict)
        native_digest = canonical_digest(task_eval) if native_executed else None
        if not native_executed:
            task_eval, native_executed, native_digest = _call_native_evaluator(
                task=task_dict,
                run_result=run_dict,
            )
        if not isinstance(task_eval, dict):
            return MarbleOutcome(
                success=False,
                score=None,
                failure_reason="native_evaluator_not_executed",
                environment_valid=False,
                evaluator_name=self.evaluator_name,
                raw_result_digest=canonical_digest(run_result),
                native_evaluator_executed=False,
                native_evaluator_name=None,
                native_evaluator_result_digest=None,
            )
        root_causes = _root_causes(task_dict)
        predictions = task_eval.get("predicted")
        if predictions is None:
            return MarbleOutcome(
                success=False,
                score=None,
                failure_reason="missing_predicted_root_causes",
                environment_valid=False,
                evaluator_name=self.evaluator_name,
                raw_result_digest=canonical_digest(run_result),
                native_evaluator_executed=native_executed,
                native_evaluator_name=self.evaluator_name if native_executed else None,
                native_evaluator_result_digest=native_digest,
            )
        predicted = {str(item) for item in _as_prediction_list(predictions)}
        expected = {str(item) for item in root_causes}
        success = bool(expected) and expected.issubset(predicted)
        return MarbleOutcome(
            success=success,
            score=1.0 if success else 0.0,
            failure_reason=None if success else "root_cause_mismatch",
            environment_valid=True,
            evaluator_name=self.evaluator_name,
            raw_result_digest=canonical_digest(run_result),
            native_evaluator_executed=native_executed,
            native_evaluator_name=self.evaluator_name if native_executed else None,
            native_evaluator_result_digest=native_digest,
        )


def _root_causes(task: dict[str, Any]) -> list[str]:
    task_body = task.get("task") if isinstance(task.get("task"), dict) else {}
    return [str(item) for item in task_body.get("root_causes", [])]


def _as_prediction_list(predictions: Any) -> list[str]:
    if isinstance(predictions, list):
        return [str(item) for item in predictions]
    text = str(predictions)
    known = [
        "INSERT_LARGE_DATA",
        "LOCK_CONTENTION",
        "VACUUM",
        "REDUNDANT_INDEX",
        "FETCH_LARGE_DATA",
        "MISSING_INDEXES",
    ]
    return [label for label in known if label in text]


def _call_native_evaluator(
    *, task: dict[str, Any], run_result: dict[str, Any]
) -> tuple[dict[str, Any] | None, bool, str | None]:
    marble_root = Path(os.environ.get("SMTR_MARBLE_ROOT", "/home/ecs-user/MARBLE"))
    sys.path.insert(0, str(marble_root))
    previous_cwd = Path.cwd()
    try:
        os.chdir(marble_root / "marble")
        from marble.evaluator.evaluator import Evaluator

        evaluator = Evaluator(metrics_config=task.get("metrics", {}))
        task_body = task.get("task") if isinstance(task.get("task"), dict) else {}
        result = run_result.get("final_output") or run_result.get("result") or ""
        evaluator.evaluate_task_db(
            task=str(task_body.get("content", "")),
            result=str(result),
            labels=[str(item) for item in task_body.get("labels", [])],
            pred_num=int(task_body.get("number_of_labels_pred", 0) or 0),
            root_causes=[str(item) for item in task_body.get("root_causes", [])],
        )
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
