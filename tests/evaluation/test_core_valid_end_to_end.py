"""Test 10 (清单 P1-3/P1-4): core-valid end-to-end outcome.

A run must never enter the team-success denominator unless all of
``real_engine_executed``, ``native_evaluator_executed``,
``environment_valid`` and ``runtime_visibility_verified`` hold; invalid
runs are reported separately and never counted as task failures.
"""

from __future__ import annotations

import pytest

from smtr.marble.end_to_end_evaluation import (
    compute_end_to_end_method_metrics,
    is_core_valid_end_to_end_run,
)


def _run(**overrides):
    base = {
        "method": "smtr",
        "task_id": "t1",
        "generation_seed": 0,
        "team_success": True,
        "score": 1.0,
        "real_engine_executed": True,
        "native_evaluator_executed": True,
        "environment_valid": True,
        "runtime_visibility_verified": True,
        "cleanup_succeeded": True,
        "invalid_reason": None,
    }
    base.update(overrides)
    return base


@pytest.mark.parametrize("field", [
    "real_engine_executed",
    "native_evaluator_executed",
    "environment_valid",
    "runtime_visibility_verified",
])
def test_each_core_flag_false_excludes_run(field):
    """Any single core flag being false must make the run core-invalid."""
    assert is_core_valid_end_to_end_run(_run()) is True
    assert is_core_valid_end_to_end_run(_run(**{field: False})) is False


def test_invalid_reason_excludes_run():
    """A run with an invalid_reason is never core-valid."""
    assert is_core_valid_end_to_end_run(_run(invalid_reason="engine timeout")) is False


def test_cleanup_failure_does_not_define_validity():
    """cleanup_succeeded is an integrity signal, not core validity (清单 P1-3)."""
    assert is_core_valid_end_to_end_run(_run(cleanup_succeeded=False)) is True


def test_core_invalid_runs_never_enter_team_success_denominator():
    """team_success_rate only counts core-valid runs (清单 P1-4)."""
    runs = [
        _run(team_success=True),
        _run(team_success=True),
        # Core-invalid run that "failed" the task: must not count as failure.
        _run(team_success=False, real_engine_executed=False),
        _run(team_success=False, native_evaluator_executed=False),
        _run(team_success=False, environment_valid=False),
        _run(team_success=False, runtime_visibility_verified=False),
    ]
    metrics = compute_end_to_end_method_metrics("smtr", runs)
    assert metrics["total_run_count"] == 6
    assert metrics["core_valid_run_count"] == 2
    assert metrics["core_invalid_run_count"] == 4
    assert metrics["core_valid_run_rate"] == round(2 / 6, 4)
    # Both core-valid runs succeeded; invalid runs did not drag the rate down.
    assert metrics["team_success_rate"] == 1.0


def test_invalid_runs_reported_not_converted_to_failures():
    """A fully invalid batch has no valid denominator, rate stays 0 not error."""
    runs = [_run(team_success=False, real_engine_executed=False)]
    metrics = compute_end_to_end_method_metrics("smtr", runs)
    assert metrics["core_valid_run_count"] == 0
    assert metrics["core_invalid_run_count"] == 1
    assert metrics["team_success_rate"] == 0.0
    assert metrics["engine_failure_rate"] == 1.0
