"""Generate real cross-task share/withhold pairs from frozen candidates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from smtr.counterfactual.decision_points import canonical_digest
from smtr.marble.branch_runner import MarblePairedBranchRunner
from smtr.marble.environment.isolation import bundle_from_manifest_task
from smtr.marble.real_data import RealPairedRecord, RealProceduralMemory
from smtr.marble.task_provider import _read_jsonl_line


def generate_real_database_pairs(
    *,
    dataset_manifest_path: Path,
    split_manifest_path: Path,
    candidate_manifest_path: Path,
    memory_pool_path: Path,
    output_dir: Path,
    generation_seeds: list[int],
    limit_pairs: int | None = None,
) -> dict[str, Any]:
    dataset = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    candidates = json.loads(candidate_manifest_path.read_text(encoding="utf-8"))
    tasks = {str(task["task_id"]): task for task in dataset["tasks"]}
    memories = {
        memory.memory_id: memory
        for line in memory_pool_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for memory in [RealProceduralMemory.model_validate_json(line)]
    }
    edges: list[dict[str, Any]] = []
    for candidate_set in candidates["candidates"]:
        for memory_id, score in zip(
            candidate_set["candidate_memory_ids"],
            candidate_set["retrieval_scores"],
            strict=True,
        ):
            edges.append(
                {
                    "recipient_task_id": candidate_set["recipient_task_id"],
                    "memory_id": memory_id,
                    "retrieval_score": score,
                }
            )
    edges = edges[:limit_pairs] if limit_pairs else edges
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    runner = MarblePairedBranchRunner()
    for edge in edges:
        memory = memories[edge["memory_id"]]
        task_entry = tasks[str(edge["recipient_task_id"])]
        task = _read_jsonl_line(Path(task_entry["source_path"]), int(task_entry["source_line"]))
        bundle = bundle_from_manifest_task(
            {"raw_task": task, "task_id": edge["recipient_task_id"], "scenario": "database"}
        )
        for seed in generation_seeds:
            pair_id = canonical_digest({"edge": edge, "generation_seed": seed})[:24]
            result = runner.run_pair(
                task=task,
                candidate_memory={
                    "memory_id": memory.memory_id,
                    "payload": memory.procedure_payload.model_dump_json(),
                },
                initial_state_bundle=bundle,
                agent_config={"target_receiver_agent_id": "agent1"},
                generation_seed=seed,
                workspace=output_dir / "pairs" / pair_id,
            )
            try:
                if not result.paired_record_valid:
                    raise ValueError(result.invalid_reason or "invalid paired branches")
                share_logical = (result.share.initial_logical_fingerprint or {}).get(
                    "combined_digest"
                )
                withhold_logical = (result.withhold.initial_logical_fingerprint or {}).get(
                    "combined_digest"
                )
                record = RealPairedRecord(
                    pair_id=pair_id,
                    recipient_task_id=str(edge["recipient_task_id"]),
                    memory_id=memory.memory_id,
                    source_task_id=memory.source_task_id,
                    generation_seed=seed,
                    Y_withhold=int(result.withhold.outcome.success),
                    Y_share=int(result.share.outcome.success),
                    tau=int(result.share.outcome.success) - int(result.withhold.outcome.success),
                    withhold_score=float(result.withhold.outcome.score or 0.0),
                    share_score=float(result.share.outcome.score or 0.0),
                    initial_state_match=result.share.initial_digest
                    == result.withhold.initial_digest,
                    memory_intervention_verified=(
                        result.share.input_audit.contains_memory_section
                        and not result.withhold.input_audit.contains_memory_section
                    ),
                    valid=(
                        result.share.initial_digest == result.withhold.initial_digest
                        and share_logical == withhold_logical
                        and result.share.input_audit.contains_memory_section
                        and not result.withhold.input_audit.contains_memory_section
                    ),
                    failure_reason=None,
                )
                payload = record.model_dump(mode="json")
            except Exception as exc:
                payload = {
                    "pair_id": pair_id,
                    "recipient_task_id": edge["recipient_task_id"],
                    "memory_id": memory.memory_id,
                    "source_task_id": memory.source_task_id,
                    "generation_seed": seed,
                    "Y_withhold": None,
                    "Y_share": None,
                    "tau": None,
                    "withhold_score": None,
                    "share_score": None,
                    "initial_state_match": False,
                    "memory_intervention_verified": False,
                    "valid": False,
                    "failure_reason": str(exc),
                }
            records.append(payload)
    path = output_dir / "paired_records.jsonl"
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    return {
        "attempted": len(records),
        "valid": sum(bool(record.get("valid")) for record in records),
        "invalid": sum(not bool(record.get("valid")) for record in records),
        "output": str(path),
    }
