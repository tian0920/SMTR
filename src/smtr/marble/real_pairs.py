"""Candidate-level paired share/withhold intervention on MARBLE tasks.

A treatment *edge* is the triple (target_task_id, receiver_agent_id,
candidate_memory_id). Each edge is executed once per generation seed
(replicate), with deterministic branch-order counterbalancing, and
aggregate empirical outcome probabilities are computed per edge.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from smtr.marble.io import load_split_task_ids
from smtr.marble.paired_outcomes import get_paired_outcomes, paired_record_label

TREATMENT_DEFINITION_VERSION = "v1"

BranchOrder = Literal["share_then_withhold", "withhold_then_share"]


def stable_hash(*parts: object) -> int:
    """Deterministic 64-bit hash over stringified parts (order-sensitive)."""
    hasher = hashlib.sha256()
    for part in parts:
        hasher.update(str(part).encode("utf-8"))
        hasher.update(b"\x1f")
    return int.from_bytes(hasher.digest()[:8], "big")


def compute_edge_id(
    target_task_id: str,
    receiver_agent_id: str,
    candidate_memory_id: str,
) -> str:
    """Stable identity for a treatment edge (task + receiver + memory)."""
    return f"edge_{stable_hash(target_task_id, receiver_agent_id, candidate_memory_id):016x}"


def branch_order_for_edge(edge_id: str, generation_seed: int) -> BranchOrder:
    """Deterministic counterbalanced branch order for one replicate."""
    if stable_hash(edge_id, generation_seed) % 2 == 0:
        return "share_then_withhold"
    return "withhold_then_share"


def aggregate_edge_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate replicate-level paired records into per-edge empirical stats.

    Only valid records contribute to the empirical probabilities. For each
    edge: q_ab = count(Y_share=a, Y_withhold=b) / n, tau = q10 - q01,
    eta = q01.
    """
    groups: dict[str, list[dict[str, Any]]] = {}
    attempted: dict[str, int] = {}
    for rec in records:
        edge_id = rec.get("edge_id")
        if not edge_id:
            continue
        attempted[edge_id] = attempted.get(edge_id, 0) + 1
        if rec.get("valid"):
            groups.setdefault(edge_id, []).append(rec)

    aggregates: list[dict[str, Any]] = []
    for edge_id, valid_records in sorted(groups.items()):
        n = len(valid_records)
        counts = {(0, 0): 0, (0, 1): 0, (1, 0): 0, (1, 1): 0}
        for rec in valid_records:
            y_share, y_withhold = get_paired_outcomes(rec)
            counts[(y_share, y_withhold)] += 1
        first = valid_records[0]
        q00 = counts[(0, 0)] / n
        q01 = counts[(0, 1)] / n
        q10 = counts[(1, 0)] / n
        q11 = counts[(1, 1)] / n
        aggregates.append({
            "edge_id": edge_id,
            "target_task_id": first.get("task_id"),
            "receiver_agent_id": first.get("receiver_agent_id"),
            "candidate_memory_id": first.get("candidate_memory_id"),
            "treatment_definition_version":
                first.get("treatment_definition_version", TREATMENT_DEFINITION_VERSION),
            "n_replicates": n,
            "n_attempted": attempted[edge_id],
            "q00_empirical": q00,
            "q01_empirical": q01,
            "q10_empirical": q10,
            "q11_empirical": q11,
            "tau_empirical": q10 - q01,
            "eta_empirical": q01,
        })
    return aggregates


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
    branch_execution_order: str = "counterbalanced",
    engine_timeout_seconds: int = 1800,
) -> dict[str, Any]:
    """Generate candidate-level paired records via MarblePairedBranchRunner.run_pair.

    Each pair holds constant: MARBLE task, receiver agent, seed, environment snapshot,
    non-memory input. The only difference is whether the candidate payload is injected.

    Every candidate defines a treatment edge (task + receiver + memory) which is
    executed once per generation seed. Branch order is deterministic-counterbalanced
    per (edge, seed) unless an explicit order is requested.
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
                "edge_id": compute_edge_id(
                    entry["task_id"],
                    entry.get("receiver_agent_id", ""),
                    rec["memory_id"],
                ),
                "task_id": entry["task_id"],
                "receiver_agent_id": entry.get("receiver_agent_id", ""),
                "receiver_role": entry.get("receiver_role", "unknown"),
                "receiver_capabilities": entry.get("receiver_capabilities", []),
                "receiver_tool_names": entry.get("receiver_tool_names", []),
                "receiver_model_name": entry.get("receiver_model_name"),
                "task_instruction": entry.get("task_instruction", ""),
                "environment_signature": entry.get("environment_signature", []),
                "local_context_summary": entry.get("local_context_summary", ""),
                "team_context_summary": entry.get("team_context_summary", ""),
                "candidate_memory_id": rec["memory_id"],
                "writer_agent_id": rec.get("writer_agent_id", ""),
                "writer_role": rec.get("writer_role", "unknown"),
                "writer_capabilities": rec.get("writer_capabilities", []),
                "writer_tool_names": rec.get("writer_tool_names", []),
                "writer_model_name": rec.get("writer_model_name"),
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

        for replicate_index, seed in enumerate(generation_seeds):
            if branch_execution_order in ("share_then_withhold", "withhold_then_share"):
                replicate_branch_order: BranchOrder = branch_execution_order  # type: ignore[assignment]
            else:
                replicate_branch_order = branch_order_for_edge(edge["edge_id"], seed)
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
                branch_execution_order=replicate_branch_order,
                engine_timeout_seconds=engine_timeout_seconds,
            )

            record = paired_result_to_record(
                pair_result=pair_result,
                edge=edge,
                seed=seed,
                replicate_id=f"{edge['edge_id']}:r{replicate_index}",
            )
            records.append(record)

    out_path = output_dir / "paired_records.jsonl"
    out_path.write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in records),
        encoding="utf-8",
    )

    edge_aggregates = aggregate_edge_records(records)
    aggregate_path = output_dir / "edge_aggregates.jsonl"
    aggregate_path.write_text(
        "".join(json.dumps(a, sort_keys=True) + "\n" for a in edge_aggregates),
        encoding="utf-8",
    )

    return {
        "attempted": len(records),
        "valid": sum(r["valid"] for r in records),
        "invalid": sum(not r["valid"] for r in records),
        "n_edges": len(edge_aggregates),
        "output": str(out_path),
        "edge_aggregates": str(aggregate_path),
    }


def paired_result_to_record(
    *,
    pair_result: Any,
    edge: dict[str, Any],
    seed: int,
    replicate_id: str | None = None,
    treatment_definition_version: str = TREATMENT_DEFINITION_VERSION,
) -> dict[str, Any]:
    """Convert a PairedBranchResult into a serializable paired record.

    All audit fields come from the real PairedBranchResult, not fabricated.
    """
    edge_id = edge.get("edge_id") or compute_edge_id(
        pair_result.task_id,
        edge["receiver_agent_id"],
        pair_result.candidate_memory_id,
    )
    record = {
        "record_type": "marble_candidate_level_pair",
        "schema_version": "v2",
        "scenario": pair_result.scenario,

        "edge_id": edge_id,
        "replicate_id": replicate_id if replicate_id is not None else f"{edge_id}:r0",
        "treatment_definition_version": treatment_definition_version,

        "task_id": pair_result.task_id,
        "generation_seed": seed,

        "receiver_agent_id": edge["receiver_agent_id"],
        "receiver_role": edge["receiver_role"],
        "receiver_capabilities": edge["receiver_capabilities"],
        "receiver_tool_names": edge.get("receiver_tool_names", []),
        "receiver_model_name": edge.get("receiver_model_name"),

        "candidate_memory_id": pair_result.candidate_memory_id,
        "writer_agent_id": edge["writer_agent_id"],
        "writer_role": edge["writer_role"],
        "writer_capabilities": edge["writer_capabilities"],
        "writer_tool_names": edge.get("writer_tool_names", []),
        "writer_model_name": edge.get("writer_model_name"),

        "selected_prefix_memory_ids": [],
        "candidate_rank": edge["candidate_rank"],
        "candidate_score": edge["candidate_score"],

        "task_instruction": edge.get("task_instruction", ""),
        "environment_signature": edge.get("environment_signature", []),
        "subtask": edge.get("subtask"),
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
        "branch_order_assignment": pair_result.branch_execution_order,

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
    # Label is always derived from the canonical nested outcomes, never
    # trusted from the upstream pair runner alone.
    record["label"] = paired_record_label(record)
    return record
