"""Database paired-pilot manifest and trace generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from smtr.counterfactual.decision_points import canonical_digest
from smtr.marble.artifacts import assert_marble_artifact_path
from smtr.marble.branch_runner import MarblePairedBranchRunner
from smtr.marble.environment.isolation import bundle_from_manifest_task
from smtr.marble.runtime_preflight import run_database_runtime_preflight
from smtr.marble.task_provider import MarbleTaskProvider


def create_database_pilot_manifest(
    *,
    dataset_manifest_path: Path,
    split_manifest_path: Path,
    output_path: Path,
    task_count: int = 5,
) -> dict[str, Any]:
    assert_marble_artifact_path(output_path)
    provider = MarbleTaskProvider(dataset_manifest_path=dataset_manifest_path)
    tasks = provider.iter_split(
        split_manifest_path=split_manifest_path,
        split="train",
        scenario="database",
        limit=task_count,
    )
    pilot_tasks = []
    for task in tasks:
        memories = [
            _memory(task.task_id, "helpful", "correct database diagnostic checklist"),
            _memory(task.task_id, "harmful", "misleading nonexistent table advice"),
            _memory(task.task_id, "irrelevant", "unrelated frontend debugging advice"),
        ]
        pilot_tasks.append(
            {
                "task_id": task.task_id,
                "scenario": task.scenario,
                "task_digest": task.task_digest,
                "candidate_memories": memories,
            }
        )
    manifest = {
        "scenario": "database",
        "dataset_manifest": str(dataset_manifest_path),
        "split_manifest": str(split_manifest_path),
        "task_count": len(pilot_tasks),
        "tasks": pilot_tasks,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def run_database_paired_pilot(
    *,
    pilot_manifest_path: Path,
    output_dir: Path,
    generation_seed: int = 0,
) -> dict[str, Any]:
    assert_marble_artifact_path(output_dir)
    manifest = json.loads(pilot_manifest_path.read_text(encoding="utf-8"))
    preflight = run_database_runtime_preflight(marble_root=Path("/home/ecs-user/MARBLE"))
    if not preflight.ready:
        failures = [
            check.name for check in preflight.checks if check.blocking and not check.passed
        ]
        raise RuntimeError(
            "MARBLE database paired pilot preflight failed; run runtime-preflight first. "
            f"blocking_failures={','.join(failures)}"
        )
    provider = MarbleTaskProvider(dataset_manifest_path=Path(manifest["dataset_manifest"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    label_counts: dict[str, int] = {}
    invalid_count = 0
    pair_count = 0
    for task_entry in manifest["tasks"]:
        task = provider.get_by_digest(task_entry["task_digest"])
        bundle = bundle_from_manifest_task(
            {"raw_task": task.raw_task, "task_id": task.task_id, "scenario": task.scenario},
            generation_seed=generation_seed,
        )
        for memory in task_entry["candidate_memories"]:
            pair_id = canonical_digest(
                {"task_digest": task.task_digest, "memory_id": memory["memory_id"]}
            )[:16]
            pair_dir = output_dir / pair_id
            result = MarblePairedBranchRunner().run_pair(
                task=task.raw_task,
                candidate_memory=memory,
                initial_state_bundle=bundle,
                agent_config={"target_receiver_agent_id": "agent1"},
                generation_seed=generation_seed,
                workspace=pair_dir,
            )
            summary = {
                "task_id": task.task_id,
                "candidate_memory_id": memory["memory_id"],
                "expected_role": memory["expected_role"],
                "real_engine_executed": result.real_engine_executed,
                "share_native_evaluator_executed": (
                    result.share.outcome.native_evaluator_executed
                ),
                "withhold_native_evaluator_executed": (
                    result.withhold.outcome.native_evaluator_executed
                ),
                "initial_state_match": (
                    result.share.initial_digest == result.withhold.initial_digest
                ),
                "initial_logical_digest_match": (
                    result.share.initial_logical_fingerprint is not None
                    and result.withhold.initial_logical_fingerprint is not None
                    and result.share.initial_logical_fingerprint.get("combined_digest")
                    == result.withhold.initial_logical_fingerprint.get("combined_digest")
                ),
                "agent_input_non_memory_sections_match": (
                    result.share.input_audit.system_section_digest
                    == result.withhold.input_audit.system_section_digest
                    and result.share.input_audit.task_section_digest
                    == result.withhold.input_audit.task_section_digest
                    and result.share.input_audit.tool_section_digest
                    == result.withhold.input_audit.tool_section_digest
                ),
                "memory_intervention_verified": (
                    result.share.input_audit.contains_memory_section
                    and not result.withhold.input_audit.contains_memory_section
                ),
                "share_success": result.share.outcome.success,
                "withhold_success": result.withhold.outcome.success,
                "paired_label": result.paired_label,
                "evaluator_name": result.share.outcome.evaluator_name,
                "paired_record_valid": result.paired_record_valid,
                "invalid_reason": result.invalid_reason,
            }
            _write_pair_trace(pair_dir=pair_dir, result=result, summary=summary)
            pair_count += 1
            if result.paired_record_valid and result.paired_label:
                label_counts[result.paired_label] = label_counts.get(result.paired_label, 0) + 1
            else:
                invalid_count += 1
    run_summary = {
        "pair_count": pair_count,
        "invalid_pair_count": invalid_count,
        "label_counts": label_counts,
    }
    (output_dir / "pilot_summary.json").write_text(
        json.dumps(run_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return run_summary


def _memory(task_id: str, role: str, rationale: str) -> dict[str, str]:
    payloads = {
        "helpful": (
            "Use pg_stat_statements for slow query evidence, pg_locks for lock "
            "contention, pg_stat_all_tables for vacuum/dead tuple evidence, and "
            "pg_stat_user_indexes plus pg_indexes for redundant-index evidence."
        ),
        "harmful": (
            "Assume the main evidence is in a nonexistent mysql_slow_log table and "
            "ignore pg_stat_statements and pg_locks."
        ),
        "irrelevant": "Check browser CSS layout issues and frontend event handlers first.",
    }
    payload = payloads[role]
    return {
        "memory_id": f"database_{task_id}_{role}",
        "source_type": "human_pilot_design",
        "expected_role": role,
        "payload": payload,
        "payload_digest": canonical_digest(payload),
        "human_rationale": rationale,
        "task_id": task_id,
    }


def _write_pair_trace(*, pair_dir: Path, result: Any, summary: dict[str, Any]) -> None:
    pair_dir.mkdir(parents=True, exist_ok=True)
    (pair_dir / "metadata.json").write_text(
        json.dumps(
            {
                "engine_name": result.engine_name,
                "engine_version": result.engine_version,
                "real_engine_executed": result.real_engine_executed,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (pair_dir / "share_audit.json").write_text(
        json.dumps(result.share.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (pair_dir / "withhold_audit.json").write_text(
        json.dumps(result.withhold.model_dump(mode="json"), indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    (pair_dir / "share_result.json").write_text(
        json.dumps(result.share.outcome.__dict__, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (pair_dir / "withhold_result.json").write_text(
        json.dumps(result.withhold.outcome.__dict__, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (pair_dir / "paired_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
