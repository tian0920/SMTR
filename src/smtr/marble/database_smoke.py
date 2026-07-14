"""Single-task MARBLE database smoke commands."""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

from smtr.counterfactual.decision_points import canonical_digest
from smtr.marble.artifacts import assert_marble_artifact_path
from smtr.marble.branch_runner import MarblePairedBranchRunner
from smtr.marble.engine_process import (
    DEFAULT_ENGINE_TIMEOUT_SECONDS,
    run_marble_engine_process,
    write_engine_process_result,
)
from smtr.marble.environment.database_rebuild import SequentialDatabaseRebuilder
from smtr.marble.environment.isolation import bundle_from_manifest_task
from smtr.marble.environment.scenarios.database import MarbleDatabaseEnvironment
from smtr.marble.memory_injection import MarbleMemoryInjector
from smtr.marble.outcome.factory import evaluator_for_scenario
from smtr.marble.outcome.protocol import outcome_from_failure
from smtr.marble.runtime_preflight import DEFAULT_DASHSCOPE_MODEL, run_database_runtime_preflight
from smtr.marble.task_provider import _read_jsonl_line


def run_database_b0_smoke(
    *,
    marble_root: Path,
    task_id: str,
    generation_seed: int,
    output_dir: Path,
    engine_timeout_seconds: int = DEFAULT_ENGINE_TIMEOUT_SECONDS,
    engine_timeout_source: str = "default",
) -> dict[str, Any]:
    assert_marble_artifact_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = _run_id(
        task_id=str(task_id),
        branch="b0",
        generation_seed=generation_seed,
    )
    preflight = run_database_runtime_preflight(marble_root=marble_root)
    task = _load_database_task_by_id(marble_root, task_id)
    bundle = bundle_from_manifest_task(
        {"raw_task": task, "task_id": str(task_id), "scenario": "database"},
        generation_seed=generation_seed,
    )
    env = MarbleDatabaseEnvironment(
        task=task,
        workspace=output_dir / "workspace",
        initial_state_bundle=bundle,
        agent_config={"target_receiver_agent_id": "agent1"},
        marble_root=marble_root,
    )
    base_input = env.build_agent_input(memory_payloads=())
    _, input_audit = MarbleMemoryInjector().build_agent_input(
        base_agent_input=base_input,
        memory_payloads=(),
        memory_ids=(),
    )
    config_path = output_dir / "workspace" / "marble_config.yaml"
    raw_result_path = output_dir / "workspace" / "marble_output.jsonl"
    _write_marble_config(
        task=task,
        config_path=config_path,
        raw_result_path=raw_result_path,
        generation_seed=generation_seed,
    )
    engine_result = None
    raw_result = {"task_evaluation": None}
    if preflight.ready:
        engine_result = run_marble_engine_process(
            marble_root=marble_root,
            config_path=config_path,
            raw_result_path=raw_result_path,
            output_dir=output_dir,
            run_identity={},
            timeout_seconds=engine_timeout_seconds,
            timeout_source=engine_timeout_source,
        )
        write_engine_process_result(output_dir / "engine_process.json", engine_result)
        raw_result = _load_last_jsonl(raw_result_path) or {}
    if engine_result and engine_result.real_engine_executed:
        evaluator = evaluator_for_scenario("database")
        os.environ["SMTR_MARBLE_ROOT"] = str(marble_root)
        outcome = evaluator.evaluate(task=task, run_result=raw_result)
    else:
        outcome = outcome_from_failure(
            evaluator_name="marble_database_evaluate_task_db",
            reason="real_engine_not_executed",
            raw_result=raw_result,
        )
    summary = {
        "run_id": run_id,
        "task_id": str(task_id),
        "runtime_preflight_ready": preflight.ready,
        "preflight_blocking_failures": [
            check.name for check in preflight.checks if check.blocking and not check.passed
        ],
        "real_engine_executed": bool(engine_result and engine_result.real_engine_executed),
        "engine_exit_code": engine_result.exit_code if engine_result else None,
        "engine_timed_out": engine_result.timed_out if engine_result else None,
        "engine_timeout_seconds": engine_timeout_seconds,
        "engine_timeout_source": (
            engine_result.engine_timeout_source if engine_result else engine_timeout_source
        ),
        "engine_duration_seconds": (
            engine_result.engine_duration_seconds if engine_result else None
        ),
        "started_at": engine_result.started_at if engine_result else None,
        "completed_at": engine_result.ended_at if engine_result else None,
        "engine_termination_requested": (
            engine_result.engine_termination_requested if engine_result else False
        ),
        "engine_termination_signal": (
            engine_result.engine_termination_signal if engine_result else None
        ),
        "engine_termination_grace_period_seconds": (
            engine_result.engine_termination_grace_period_seconds if engine_result else None
        ),
        "engine_kill_escalated": engine_result.engine_kill_escalated if engine_result else False,
        "last_observed_stage": (
            engine_result.last_observed_stage if engine_result else "unknown"
        ),
        "last_observed_stage_parser_version": (
            engine_result.last_observed_stage_parser_version if engine_result else "unknown"
        ),
        "engine_working_directory": engine_result.working_directory if engine_result else None,
        "engine_config_path": (
            engine_result.config_path if engine_result else str(config_path.resolve())
        ),
        "selected_python": engine_result.selected_python if engine_result else None,
        "cleanup_succeeded": bool(engine_result and engine_result.cleanup_succeeded),
        "cleanup_exit_code": engine_result.cleanup_exit_code if engine_result else None,
        "cleanup_failure_reason": (
            engine_result.cleanup_failure_reason if engine_result else "engine_not_started"
        ),
        "raw_result_exists": bool(engine_result and engine_result.raw_result_exists),
        "raw_result_nonempty": bool(engine_result and engine_result.raw_result_nonempty),
        "raw_result_fresh": bool(engine_result and engine_result.raw_result_fresh),
        "raw_result_parseable": bool(engine_result and engine_result.raw_result_parseable),
        "raw_result_identity_verified": (
            engine_result.raw_result_identity_verified if engine_result else False
        ),
        "raw_result_identity_failure_reason": (
            engine_result.raw_result_identity_failure_reason if engine_result else None
        ),
        "raw_result_path": engine_result.raw_result_path if engine_result else str(raw_result_path),
        "stdout_log_path": engine_result.stdout_log_path if engine_result else None,
        "stderr_log_path": engine_result.stderr_log_path if engine_result else None,
        "cleanup_stdout_log_path": (
            engine_result.cleanup_stdout_log_path if engine_result else None
        ),
        "cleanup_stderr_log_path": (
            engine_result.cleanup_stderr_log_path if engine_result else None
        ),
        "native_evaluator_executed": outcome.native_evaluator_executed,
        "native_evaluator_name": outcome.native_evaluator_name,
        "native_evaluator_result_digest": outcome.native_evaluator_result_digest,
        "environment_valid": (
            outcome.environment_valid
            and bool(engine_result and engine_result.cleanup_succeeded)
        ),
        "initial_state_digest": env.initial_state_digest(),
        "raw_result_digest": canonical_digest(raw_result) if raw_result else None,
        "final_state_digest": env.final_state_digest(),
        "b0_memory_absent": not input_audit.contains_memory_section,
        "outcome": outcome.__dict__,
    }
    (output_dir / "b0_smoke.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    env.close()
    return summary


def verify_database_rebuild(
    *,
    marble_root: Path,
    task_id: str,
    output_dir: Path,
) -> dict[str, Any]:
    assert_marble_artifact_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    task = _load_database_task_by_id(marble_root, task_id)
    run_id = _run_id(task_id=str(task_id), branch="rebuild", generation_seed=0)
    bundle = bundle_from_manifest_task(
        {"raw_task": task, "task_id": str(task_id), "scenario": "database"}
    )
    rebuilder = SequentialDatabaseRebuilder(marble_root=marble_root)
    fp1 = rebuilder.materialize(initial_state_bundle=bundle, branch_workspace=output_dir / "run_1")
    marker = output_dir / "run_1" / "marker.txt"
    marker.write_text("unique marker", encoding="utf-8")
    marker_present = marker.exists()
    cleanup_1 = rebuilder.destroy()
    fp2 = rebuilder.materialize(initial_state_bundle=bundle, branch_workspace=output_dir / "run_2")
    marker_leakage = (output_dir / "run_2" / "marker.txt").exists()
    cleanup_2 = rebuilder.destroy()
    cleanup_succeeded = cleanup_1.succeeded and cleanup_2.succeeded
    summary = {
        "run_id": run_id,
        "task_id": str(task_id),
        "initial_digest_run_1": fp1.combined_digest,
        "initial_digest_run_2": fp2.combined_digest,
        "initial_digests_match": fp1.combined_digest == fp2.combined_digest,
        "marker_written": marker_present,
        "marker_leakage": marker_leakage,
        "cleanup_succeeded": cleanup_succeeded,
        "cleanup_run_1": cleanup_1.to_json(),
        "cleanup_run_2": cleanup_2.to_json(),
        "fingerprint_definition": {
            "schema": "normalized sorted CREATE/ALTER/INDEX statements from init_sql",
            "content": "normalized sorted INSERT/UPDATE/DELETE/COPY statements from init_sql",
            "config": "database name, task environment config, tool config, init SQL digest",
        },
    }
    (output_dir / "database_rebuild.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def run_database_paired_smoke(
    *,
    marble_root: Path,
    task_id: str,
    memory_id: str,
    generation_seed: int,
    branch_order: str,
    output_dir: Path,
) -> dict[str, Any]:
    assert_marble_artifact_path(output_dir)
    run_id = _run_id(
        task_id=str(task_id),
        memory_id=memory_id,
        branch=branch_order,
        generation_seed=generation_seed,
    )
    task = _load_database_task_by_id(marble_root, task_id)
    memory = _memory_for_id(task_id=task_id, memory_id=memory_id)
    bundle = bundle_from_manifest_task(
        {"raw_task": task, "task_id": str(task_id), "scenario": "database"},
        generation_seed=generation_seed,
    )
    result = MarblePairedBranchRunner().run_pair(
        task=task,
        candidate_memory=memory,
        initial_state_bundle=bundle,
        agent_config={"target_receiver_agent_id": "agent1"},
        generation_seed=generation_seed,
        workspace=output_dir,
        branch_execution_order=branch_order.replace("-", "_"),
    )
    summary = {
        "run_id": run_id,
        "task_id": str(task_id),
        "candidate_memory_id": memory_id,
        "branch_order": branch_order,
        "share_real_engine_executed": result.share.real_engine_executed,
        "withhold_real_engine_executed": result.withhold.real_engine_executed,
        "share_native_evaluator_executed": result.share.outcome.native_evaluator_executed,
        "withhold_native_evaluator_executed": result.withhold.outcome.native_evaluator_executed,
        "share_cleanup_succeeded": result.share.cleanup_succeeded,
        "withhold_cleanup_succeeded": result.withhold.cleanup_succeeded,
        "share_cleanup_failure_reason": result.share.cleanup_failure_reason,
        "withhold_cleanup_failure_reason": result.withhold.cleanup_failure_reason,
        "initial_state_match": result.share.initial_digest == result.withhold.initial_digest,
        "initial_logical_digest_match": (
            result.share.initial_logical_fingerprint is not None
            and result.withhold.initial_logical_fingerprint is not None
            and result.share.initial_logical_fingerprint.get("combined_digest")
            == result.withhold.initial_logical_fingerprint.get("combined_digest")
        ),
        "memory_intervention_verified": (
            result.share.input_audit.contains_memory_section
            and not result.withhold.input_audit.contains_memory_section
        ),
        "paired_record_valid": result.paired_record_valid,
        "paired_label": result.paired_label,
        "invalid_reason": result.invalid_reason,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "paired_smoke.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _write_marble_config(
    *,
    task: dict[str, Any],
    config_path: Path,
    raw_result_path: Path,
    generation_seed: int,
) -> None:
    config = dict(task)
    config["coordinate_mode"] = "graph"
    config["llm"] = _configured_litellm_model()
    config["environment"] = dict(config.get("environment", {}))
    config["environment"]["type"] = "DB"
    config["environment"]["name"] = "DB Environment"
    config["environment"]["max_iterations"] = int(config["environment"].get("max_iterations") or 1)
    config["memory"] = {"type": "BaseMemory"}
    config["output"] = {"file_path": str(raw_result_path.resolve())}
    config["smtr_generation_seed"] = generation_seed
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _configured_litellm_model() -> str:
    model = (
        os.environ.get("MARBLE_LLM_MODEL")
        or os.environ.get("OPENAI_MODEL")
        or os.environ.get("DASHSCOPE_MODEL")
    )
    compatible_base_url_configured = bool(
        os.environ.get("DASHSCOPE_API_KEY")
        or os.environ.get("DASHSCOPE_BASE_URL")
        or os.environ.get("MARBLE_LLM_BASE_URL")
    )
    if not model and compatible_base_url_configured:
        model = DEFAULT_DASHSCOPE_MODEL
    if not model:
        return "gpt-4o-mini"
    if compatible_base_url_configured and "/" not in model:
        return f"openai/{model}"
    return model


def _load_database_task_by_id(marble_root: Path, task_id: str) -> dict[str, Any]:
    path = marble_root / "multiagentbench/database/database_main.jsonl"
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            task = json.loads(line)
            if str(task.get("task_id")) == str(task_id):
                return _read_jsonl_line(path, line_number)
    raise ValueError(f"database task_id not found: {task_id}")


def _load_last_jsonl(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return records[-1] if records else None


def _memory_for_id(*, task_id: str, memory_id: str) -> dict[str, str]:
    role = "helpful" if memory_id.endswith("_helpful") else "human"
    return {
        "memory_id": memory_id,
        "source_type": "smoke",
        "expected_role": role,
        "payload": (
            "Use pg_stat_statements, pg_locks, pg_stat_all_tables, "
            "pg_stat_user_indexes, and pg_indexes to diagnose MARBLE database root causes."
        ),
        "payload_digest": canonical_digest(memory_id),
        "task_id": str(task_id),
    }


def _run_id(
    *,
    task_id: str,
    branch: str,
    generation_seed: int,
    memory_id: str | None = None,
) -> str:
    parts = [
        f"task-{task_id}",
        f"branch-{branch}",
        f"seed-{generation_seed}",
        f"ts-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}",
        uuid.uuid4().hex[:8],
    ]
    if memory_id:
        parts.insert(1, f"memory-{memory_id}")
    return "_".join(parts)
