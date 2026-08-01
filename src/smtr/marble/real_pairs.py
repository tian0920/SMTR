"""Candidate-level paired share/withhold intervention on MARBLE tasks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from smtr.core.types import PairedTransferOutcome


def _digest(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _four_outcome_label(y_share: bool, y_withhold: bool) -> str:
    if y_share and not y_withhold:
        return "positive_transfer"
    if not y_share and y_withhold:
        return "negative_transfer"
    if y_share and y_withhold:
        return "neutral_success"
    return "neutral_failure"


def generate_candidate_level_pairs(
    *,
    dataset_manifest_path: Path,
    split_manifest_path: Path,
    candidate_manifest_path: Path,
    memory_pool_path: Path,
    generation_seeds: list[int],
    limit_pairs: int | None = None,
    output_dir: Path,
) -> dict[str, Any]:
    """Generate candidate-level paired records: share one candidate vs withhold it.

    Each pair holds constant: MARBLE task, receiver agent, seed, environment snapshot,
    non-memory input. The only difference is whether the candidate payload is injected.
    """
    dataset = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    candidates_manifest = json.loads(candidate_manifest_path.read_text(encoding="utf-8"))
    split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))

    tasks = {str(t["task_id"]): t for t in dataset.get("tasks", [])}
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

    for edge in edges:
        mem_entry = memory_pool.get(edge["candidate_memory_id"])
        if mem_entry is None:
            continue
        task_entry = tasks.get(str(edge["task_id"]))
        if task_entry is None:
            continue

        for seed in generation_seeds:
            task_digest = _digest({"task_id": edge["task_id"], "seed": seed})
            env_digest = _digest(edge["environment_signature"])
            tool_digest = _digest({"scenario": "database"})

            # In a real run, branch_runner executes share/withhold on MARBLE.
            # Here we produce the record schema; actual execution is wired via branch_runner.
            share_result = _run_branch(
                task_entry=task_entry,
                memory_entry=mem_entry,
                seed=seed,
                inject=True,
                output_dir=output_dir,
            )
            withhold_result = _run_branch(
                task_entry=task_entry,
                memory_entry=mem_entry,
                seed=seed,
                inject=False,
                output_dir=output_dir,
            )

            y_share = share_result["team_success"]
            y_withhold = withhold_result["team_success"]
            label = _four_outcome_label(y_share, y_withhold)

            record = {
                "record_type": "marble_candidate_level_pair",
                "schema_version": "v1",
                "scenario": "database",
                "task_id": edge["task_id"],
                "receiver_agent_id": edge["receiver_agent_id"],
                "receiver_role": edge["receiver_role"],
                "receiver_capabilities": edge["receiver_capabilities"],
                "candidate_memory_id": edge["candidate_memory_id"],
                "writer_agent_id": edge["writer_agent_id"],
                "writer_role": edge["writer_role"],
                "writer_capabilities": edge["writer_capabilities"],
                "selected_prefix_memory_ids": [],
                "candidate_rank": edge["candidate_rank"],
                "candidate_score": edge["candidate_score"],
                "share": {
                    "team_success": y_share,
                    "local_success": None,
                    "environment_valid": share_result["environment_valid"],
                    "native_evaluator_executed": share_result["native_evaluator_executed"],
                },
                "withhold": {
                    "team_success": y_withhold,
                    "local_success": None,
                    "environment_valid": withhold_result["environment_valid"],
                    "native_evaluator_executed": withhold_result["native_evaluator_executed"],
                },
                "label": label,
                "valid": share_result["valid"] and withhold_result["valid"],
                "invalid_reason": share_result.get("invalid_reason") or withhold_result.get("invalid_reason"),
                "digests": {
                    "share_initial_digest": share_result["initial_digest"],
                    "withhold_initial_digest": withhold_result["initial_digest"],
                    "task_digest": task_digest,
                    "tool_config_digest": tool_digest,
                },
            }
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


def _run_branch(
    *,
    task_entry: dict[str, Any],
    memory_entry: dict[str, Any],
    seed: int,
    inject: bool,
    output_dir: Path,
) -> dict[str, Any]:
    """Execute a single branch (share or withhold) via MARBLE branch runner.

    Returns outcome dict. In production this calls MarblePairedBranchRunner;
    the interface is stable for wiring.
    """
    try:
        from smtr.marble.branch_runner import MarblePairedBranchRunner

        runner = MarblePairedBranchRunner()
        result = runner.run_single_branch(
            task_entry=task_entry,
            memory_entry=memory_entry if inject else None,
            seed=seed,
            workspace=output_dir / "branches",
        )
        return {
            "team_success": result.get("team_success", False),
            "environment_valid": result.get("environment_valid", True),
            "native_evaluator_executed": result.get("native_evaluator_executed", True),
            "initial_digest": result.get("initial_digest", ""),
            "valid": result.get("valid", True),
            "invalid_reason": result.get("invalid_reason"),
        }
    except Exception as exc:
        return {
            "team_success": False,
            "environment_valid": False,
            "native_evaluator_executed": False,
            "initial_digest": "",
            "valid": False,
            "invalid_reason": str(exc),
        }
