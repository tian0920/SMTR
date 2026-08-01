"""Candidate-level paired share/withhold intervention on MARBLE tasks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from smtr.marble.io import load_split_task_ids


def generate_candidate_level_pairs(
    *,
    marble_root: Path,
    dataset_manifest_path: Path,
    split_manifest_path: Path,
    split: str,
    candidate_manifest_path: Path,
    memory_pool_path: Path,
    generation_seeds: list[int],
    limit_pairs: int | None = None,
    output_dir: Path,
    branch_execution_order: str = "share_then_withhold",
    engine_timeout_seconds: int = 1800,
) -> dict[str, Any]:
    """Generate candidate-level paired records via MarblePairedBranchRunner.run_pair.

    Each pair holds constant: MARBLE task, receiver agent, seed, environment snapshot,
    non-memory input. The only difference is whether the candidate payload is injected.
    """
    from smtr.marble.branch_runner import MarblePairedBranchRunner
    from smtr.marble.paired_context import build_pair_execution_context

    dataset = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    candidates_manifest = json.loads(candidate_manifest_path.read_text(encoding="utf-8"))

    tasks = {str(t["task_id"]): t for t in dataset.get("tasks", [])}
    split_task_ids = load_split_task_ids(split_manifest_path, split)

    memory_pool: dict[str, dict] = {}
    for line in memory_pool_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            mem = json.loads(line)
            memory_pool[mem["memory_id"]] = mem

    # Build intervention edges from candidate manifest
    edges: list[dict[str, Any]] = []
    for entry in candidates_manifest.get("candidates", []):
        for rec in entry.get("candidate_records", []):
            edges.append({
                "task_id": entry["task_id"],
                "receiver_agent_id": entry.get("receiver_agent_id", ""),
                "receiver_role": entry.get("receiver_role", "unknown"),
                "receiver_capabilities": entry.get("receiver_capabilities", []),
                "task_instruction": entry.get("task_instruction", ""),
                "environment_signature": entry.get("environment_signature", []),
                "candidate_memory_id": rec["memory_id"],
                "writer_agent_id": rec.get("writer_agent_id", ""),
                "writer_role": rec.get("writer_role", "unknown"),
                "writer_capabilities": rec.get("writer_capabilities", []),
                "candidate_rank": rec.get("rank", 0),
                "candidate_score": rec.get("score", 0.0),
            })

    if limit_pairs:
        edges = edges[:limit_pairs]

    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    runner = MarblePairedBranchRunner()

    for edge in edges:
        # Validate edge task belongs to requested split
        if edge["task_id"] not in split_task_ids:
            continue

        mem_entry = memory_pool.get(edge["candidate_memory_id"])
        if mem_entry is None:
            continue
        task_entry = tasks.get(str(edge["task_id"]))
        if task_entry is None:
            continue

        for seed in generation_seeds:
            pair_workspace = output_dir / "pairs" / f"{edge['task_id']}_{edge['receiver_agent_id']}_{edge['candidate_memory_id']}_{seed}"

            context = build_pair_execution_context(
                marble_root=marble_root,
                task_entry=task_entry,
                receiver_agent_id=edge["receiver_agent_id"],
                workspace=pair_workspace,
            )

            pair_result = runner.run_pair(
                task=context.task,
                candidate_memory=mem_entry,
                initial_state_bundle=context.initial_state_bundle,
                agent_config=context.agent_config,
                generation_seed=seed,
                workspace=pair_workspace,
                branch_execution_order=branch_execution_order,
                engine_timeout_seconds=engine_timeout_seconds,
            )

            record = paired_result_to_record(
                pair_result=pair_result,
                edge=edge,
                seed=seed,
            )
            records.append(record)

    out_path = output_dir / "paired_records.jsonl"
    out_path.write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in records),
        encoding="utf-8",
    )
    return {
        "attempted": len(records),
        "valid": sum(r["valid"] for r in records),
        "invalid": sum(not r["valid"] for r in records),
        "output": str(out_path),
    }


def paired_result_to_record(
    *,
    pair_result: Any,
    edge: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    """Convert a PairedBranchResult into a serializable paired record.

    All audit fields come from the real PairedBranchResult, not fabricated.
    """
    return {
        "record_type": "marble_candidate_level_pair",
        "schema_version": "v2",
        "scenario": pair_result.scenario,

        "task_id": pair_result.task_id,
        "generation_seed": seed,

        "receiver_agent_id": edge["receiver_agent_id"],
        "receiver_role": edge["receiver_role"],
        "receiver_capabilities": edge["receiver_capabilities"],

        "candidate_memory_id": pair_result.candidate_memory_id,
        "writer_agent_id": edge["writer_agent_id"],
        "writer_role": edge["writer_role"],
        "writer_capabilities": edge["writer_capabilities"],

        "selected_prefix_memory_ids": [],
        "candidate_rank": edge["candidate_rank"],
        "candidate_score": edge["candidate_score"],

        "task_instruction": edge.get("task_instruction", ""),
        "environment_signature": edge.get("environment_signature", []),
        "local_context_summary": edge.get("local_context_summary", ""),
        "team_context_summary": edge.get("team_context_summary", ""),

        "share": {
            "team_success": pair_result.share.outcome.success,
            "local_success": None,
            "environment_valid": pair_result.share.outcome.environment_valid,
            "native_evaluator_executed":
                pair_result.share.outcome.native_evaluator_executed,
            "real_engine_executed":
                pair_result.share.real_engine_executed,
            "runtime_visibility_verified":
                pair_result.share.runtime_visibility_verified,
            "cleanup_succeeded":
                pair_result.share.cleanup_succeeded,
        },

        "withhold": {
            "team_success": pair_result.withhold.outcome.success,
            "local_success": None,
            "environment_valid": pair_result.withhold.outcome.environment_valid,
            "native_evaluator_executed":
                pair_result.withhold.outcome.native_evaluator_executed,
            "real_engine_executed":
                pair_result.withhold.real_engine_executed,
            "runtime_visibility_verified":
                pair_result.withhold.runtime_visibility_verified,
            "cleanup_succeeded":
                pair_result.withhold.cleanup_succeeded,
        },

        "label": pair_result.paired_label,
        "valid": pair_result.paired_record_valid,
        "invalid_reason": pair_result.invalid_reason,

        "branch_execution_order": pair_result.branch_execution_order,

        "digests": {
            "share_initial_digest":
                pair_result.share.initial_digest,
            "withhold_initial_digest":
                pair_result.withhold.initial_digest,
            "share_initial_logical_digest":
                (
                    pair_result.share.initial_logical_fingerprint or {}
                ).get("combined_digest"),
            "withhold_initial_logical_digest":
                (
                    pair_result.withhold.initial_logical_fingerprint or {}
                ).get("combined_digest"),
            "share_agent_config_digest":
                pair_result.share.agent_config_digest,
            "withhold_agent_config_digest":
                pair_result.withhold.agent_config_digest,
            "share_task_digest":
                pair_result.share.task_digest,
            "withhold_task_digest":
                pair_result.withhold.task_digest,
            "share_tool_config_digest":
                pair_result.share.tool_config_digest,
            "withhold_tool_config_digest":
                pair_result.withhold.tool_config_digest,
        },
    }
