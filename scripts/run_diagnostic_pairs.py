"""Diagnostic 64-pair runner.

Reads pair_manifest.jsonl + memory_manifest.jsonl and executes
paired share/withhold experiments using the enhanced pilot_runner.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from smtr.marble.branch_runner import MarblePairedBranchRunner
from smtr.marble.environment.isolation import bundle_from_manifest_task
from smtr.marble.pilot_runner import (
    _compute_treatment_effect,
    _extract_branch_metrics,
    _pair_result_to_dict,
)
from smtr.marble.task_provider import _read_jsonl_line


def _load_memories(memory_manifest_path: Path) -> dict[str, dict]:
    """Load memory manifest into a dict keyed by memory_id."""
    memories: dict[str, dict] = {}
    with memory_manifest_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            m = json.loads(line)
            memories[m["memory_id"]] = m
    return memories


def _load_pairs(pair_manifest_path: Path) -> tuple[dict, list[dict]]:
    """Load pair manifest, returning (header, pairs)."""
    lines = pair_manifest_path.read_text(encoding="utf-8").strip().splitlines()
    header = json.loads(lines[0])
    pairs = [json.loads(line) for line in lines[1:] if line.strip()]
    return header, pairs


def run_diagnostic_pairs(
    *,
    pair_manifest_path: Path,
    memory_manifest_path: Path,
    output_dir: Path,
    marble_root: Path,
    max_pairs: int | None = None,
    resume: bool = False,
    retry_invalid: bool = False,
    dry_run: bool = False,
    engine_timeout_seconds: int = 900,
) -> dict[str, Any]:
    """Execute diagnostic paired experiments."""
    header, pairs = _load_pairs(pair_manifest_path)
    memories = _load_memories(memory_manifest_path)

    if max_pairs is not None:
        pairs = pairs[:max_pairs]

    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "diagnostic_results.jsonl"
    status_path = output_dir / "diagnostic_status.json"

    # Load existing results if resuming
    existing_results: dict[str, dict[str, Any]] = {}
    if resume and results_path.exists():
        for line in results_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            existing_results[record.get("pair_key", "")] = record

    valid_complete = 0
    invalid_complete = 0
    failed_count = 0
    pending = 0
    effects: list[dict[str, Any]] = []

    runner = MarblePairedBranchRunner()

    for pair_spec in pairs:
        pair_key = pair_spec["pair_key"]
        task_id = pair_spec["task_id"]
        memory_id = pair_spec["memory_id"]
        memory_type = pair_spec["memory_type"]
        seed = pair_spec.get("seed", 41)
        branch_order = pair_spec["branch_order"]
        receiver = pair_spec.get("receiver_agent_id", "agent1")

        # Skip if already completed
        if resume and pair_key in existing_results:
            prev = existing_results[pair_key]
            if prev.get("status") == "valid_complete" and not retry_invalid:
                valid_complete += 1
                continue
            elif prev.get("status") == "invalid_complete" and not retry_invalid:
                invalid_complete += 1
                continue

        if dry_run:
            print(f"[dry-run] Would execute pair: {pair_key} ({branch_order})")
            pending += 1
            continue

        # Load task from MARBLE dataset
        try:
            task_path = marble_root / "multiagentbench/database/database_main.jsonl"
            task = _read_jsonl_line(task_path, int(task_id))
        except Exception as exc:
            print(f"[error] Cannot load task {task_id}: {exc}")
            failed_count += 1
            continue

        # Look up memory
        memory = memories.get(memory_id)
        if not memory:
            print(f"[error] Cannot find memory {memory_id}")
            failed_count += 1
            continue

        # Build initial state bundle
        bundle = bundle_from_manifest_task(
            {"raw_task": task, "task_id": task_id, "scenario": "database"},
            generation_seed=seed,
        )

        candidate_dict = {
            "memory_id": memory_id,
            "payload": memory["payload"],
        }

        pair_output = output_dir / pair_key
        pair_output.mkdir(parents=True, exist_ok=True)

        print(f"[run] {pair_key} task={task_id} mem={memory_type} seed={seed} order={branch_order}")
        start_time = time.time()

        try:
            result = runner.run_pair(
                task=task,
                candidate_memory=candidate_dict,
                initial_state_bundle=bundle,
                agent_config={"target_receiver_agent_id": receiver},
                generation_seed=seed,
                workspace=pair_output,
                branch_execution_order=branch_order,
                engine_timeout_seconds=engine_timeout_seconds,
            )

            elapsed = time.time() - start_time
            result_dict = _pair_result_to_dict(result)
            result_dict["pair_key"] = pair_key
            result_dict["pair_id"] = pair_spec.get("pair_id", "")
            result_dict["memory_id"] = memory_id
            result_dict["memory_type"] = memory_type
            result_dict["seed"] = seed
            result_dict["receiver_agent_id"] = receiver
            result_dict["elapsed_seconds"] = round(elapsed, 2)

            # Extract branch-level metrics
            share_metrics = _extract_branch_metrics(pair_output / "share")
            withhold_metrics = _extract_branch_metrics(pair_output / "withhold")
            result_dict["share_metrics"] = share_metrics
            result_dict["withhold_metrics"] = withhold_metrics

            # Determine diagnostic validity:
            # Core requirement = real engine + evaluator executed.
            # Visibility verification is recorded but not blocking
            # (audit files are cleaned up by engine subprocess).
            engine_ok = result.real_engine_executed
            eval_ok = (
                result.share.outcome.native_evaluator_executed
                and result.withhold.outcome.native_evaluator_executed
            )
            fingerprint_ok = (
                result.share.initial_digest == result.withhold.initial_digest
            )
            input_ok = (
                result.share.input_audit.system_section_digest
                == result.withhold.input_audit.system_section_digest
                and result.share.input_audit.task_section_digest
                == result.withhold.input_audit.task_section_digest
            )
            memory_ok = (
                result.share.input_audit.contains_memory_section
                and not result.withhold.input_audit.contains_memory_section
            )
            diagnostic_valid = engine_ok and eval_ok and fingerprint_ok and input_ok and memory_ok

            # Track visibility as warning, not blocking
            vis_warnings: list[str] = []
            if not result.share_runtime_visibility_verified:
                vis_warnings.append("share_visibility_not_verified")
            if not result.withhold_runtime_visibility_verified:
                vis_warnings.append("withhold_visibility_not_verified")

            if diagnostic_valid:
                result_dict["status"] = "valid_complete"
                result_dict["first_attempt_valid"] = True
                valid_complete += 1
            else:
                result_dict["status"] = "invalid_complete"
                reasons = []
                if not engine_ok:
                    reasons.append("real_marble_engine_not_executed")
                if not eval_ok:
                    reasons.append("native_evaluator_not_executed")
                if not fingerprint_ok:
                    reasons.append("initial_fingerprint_mismatch")
                if not input_ok:
                    reasons.append("input_sections_mismatch")
                if not memory_ok:
                    reasons.append("memory_injection_mismatch")
                result_dict["invalid_reason"] = ",".join(reasons)
                result_dict["first_attempt_valid"] = False
                invalid_complete += 1
            if vis_warnings:
                result_dict["visibility_warnings"] = vis_warnings

            # Compute treatment effect
            share_outcome = result.share.outcome
            withhold_outcome = result.withhold.outcome
            share_effect_input = {
                "score": share_outcome.score or (1.0 if share_outcome.success else 0.0),
                "success": share_outcome.success,
                "tokens": share_metrics.get("tokens"),
                "latency_seconds": share_metrics.get("latency_seconds"),
                "rounds": share_metrics.get("rounds"),
            }
            withhold_effect_input = {
                "score": withhold_outcome.score or (1.0 if withhold_outcome.success else 0.0),
                "success": withhold_outcome.success,
                "tokens": withhold_metrics.get("tokens"),
                "latency_seconds": withhold_metrics.get("latency_seconds"),
                "rounds": withhold_metrics.get("rounds"),
            }
            effect = _compute_treatment_effect(share_effect_input, withhold_effect_input)
            effect["pair_key"] = pair_key
            effect["paired_label"] = result.paired_label
            effect["memory_type"] = memory_type
            effects.append(effect)
            result_dict["treatment_effect"] = effect

            print(f"  -> {result_dict['status']} label={result.paired_label} "
                  f"score_delta={effect['score_delta']:+.1f}")

        except Exception as exc:
            elapsed = time.time() - start_time
            result_dict = {
                "pair_key": pair_key,
                "pair_id": pair_spec.get("pair_id", ""),
                "memory_id": memory_id,
                "memory_type": memory_type,
                "seed": seed,
                "status": "failed",
                "error": str(exc),
                "elapsed_seconds": round(elapsed, 2),
                "first_attempt_valid": False,
            }
            failed_count += 1
            print(f"  -> FAILED: {exc}")

        # Append result
        with results_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(result_dict, sort_keys=True, default=str) + "\n")

    # Write status summary
    status = {
        "pair_manifest": str(pair_manifest_path),
        "memory_manifest": str(memory_manifest_path),
        "total_pairs": len(pairs),
        "valid_complete": valid_complete,
        "invalid_complete": invalid_complete,
        "failed": failed_count,
        "pending": pending,
        "validity_rate": valid_complete / max(len(pairs), 1),
    }
    status_path.write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return status


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Run 64-pair diagnostic experiment")
    parser.add_argument(
        "--pair-manifest",
        default="artifacts/paper_experiments/diagnostic_64/pair_manifest.jsonl",
    )
    parser.add_argument(
        "--memory-manifest",
        default="artifacts/paper_experiments/diagnostic_64/memory_manifest.jsonl",
    )
    parser.add_argument(
        "--output",
        default="artifacts/paper_experiments/diagnostic_64/run_output",
    )
    parser.add_argument("--marble-root", default="/home/ecs-user/MARBLE")
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-invalid", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--engine-timeout", type=int, default=900)
    args = parser.parse_args()

    result = run_diagnostic_pairs(
        pair_manifest_path=Path(args.pair_manifest),
        memory_manifest_path=Path(args.memory_manifest),
        output_dir=Path(args.output),
        marble_root=Path(args.marble_root),
        max_pairs=args.max_pairs,
        resume=args.resume,
        retry_invalid=args.retry_invalid,
        dry_run=args.dry_run,
        engine_timeout_seconds=args.engine_timeout,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
