"""Resumable paired causal pilot runner.

Executes paired share/withhold pilot experiments from a manifest,
with support for resume, retry, dry-run, and effect computation.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from smtr.marble.branch_runner import MarblePairedBranchRunner, PairedBranchResult
from smtr.marble.environment.isolation import bundle_from_manifest_task
from smtr.marble.real_pairs import compute_control_family_id, compute_edge_id
from smtr.marble.pilot_manifest import (
    read_paired_pilot_manifest,
    write_paired_pilot_manifest,
)
from smtr.marble.task_provider import _read_jsonl_line


def _extract_branch_metrics(branch_workspace: Path) -> dict[str, Any]:
    """Extract token/latency/round metrics from branch workspace output."""
    metrics: dict[str, Any] = {
        "tokens": None,
        "latency_seconds": None,
        "rounds": None,
        "iterations": None,
    }
    output_path = branch_workspace / "marble_output.jsonl"
    if not output_path.exists():
        return metrics
    try:
        with output_path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                raw = json.loads(line)
                if "token_usage" in raw:
                    try:
                        metrics["tokens"] = int(raw["token_usage"])
                    except (ValueError, TypeError):
                        pass
                if "iterations" in raw:
                    iters = raw["iterations"]
                    if isinstance(iters, list):
                        metrics["rounds"] = len(iters)
                        metrics["iterations"] = len(iters)
                    elif isinstance(iters, (int, str)):
                        try:
                            metrics["rounds"] = int(iters)
                            metrics["iterations"] = int(iters)
                        except (ValueError, TypeError):
                            pass
                if "planning_scores" in raw:
                    metrics["planning_scores"] = raw["planning_scores"]
                if "total_milestones" in raw:
                    try:
                        metrics["total_milestones"] = int(raw["total_milestones"])
                    except (ValueError, TypeError):
                        pass
    except Exception:
        pass
    return metrics


def _pair_result_to_dict(result: PairedBranchResult) -> dict[str, Any]:
    """Convert a PairedBranchResult to a serialisable dict."""
    return {
        "task_id": result.task_id,
        "candidate_memory_id": result.candidate_memory_id,
        "scenario": result.scenario,
        "real_engine_executed": result.real_engine_executed,
        "paired_record_valid": result.paired_record_valid,
        "invalid_reason": result.invalid_reason,
        "paired_label": result.paired_label,
        "branch_execution_order": result.branch_execution_order,
        "share_runtime_visibility_verified": result.share_runtime_visibility_verified,
        "withhold_runtime_visibility_verified": result.withhold_runtime_visibility_verified,
        "share_success": result.share.outcome.success,
        "withhold_success": result.withhold.outcome.success,
        "share_score": result.share.outcome.score,
        "withhold_score": result.withhold.outcome.score,
        "share_native_evaluator_executed": result.share.outcome.native_evaluator_executed,
        "withhold_native_evaluator_executed": result.withhold.outcome.native_evaluator_executed,
        "share_initial_digest": result.share.initial_digest,
        "withhold_initial_digest": result.withhold.initial_digest,
        "initial_fingerprint_match": (
            result.share.initial_digest == result.withhold.initial_digest
        ),
        "share_fingerprint": result.share.initial_logical_fingerprint,
        "withhold_fingerprint": result.withhold.initial_logical_fingerprint,
    }


def _compute_treatment_effect(
    share_result: dict[str, Any],
    withhold_result: dict[str, Any],
) -> dict[str, Any]:
    """Compute continuous treatment effect between share and withhold."""
    share_score = share_result.get("score", 0.0) or 0.0
    withhold_score = withhold_result.get("score", 0.0) or 0.0
    share_success = 1.0 if share_result.get("success") else 0.0
    withhold_success = 1.0 if withhold_result.get("success") else 0.0
    effect: dict[str, Any] = {
        "score_delta": share_score - withhold_score,
        "success_delta": share_success - withhold_success,
        "share_score": share_score,
        "withhold_score": withhold_score,
        "share_success": bool(share_result.get("success")),
        "withhold_success": bool(withhold_result.get("success")),
    }
    # Optional metrics
    for key in ("tokens", "latency_seconds", "rounds"):
        s_val = share_result.get(key)
        w_val = withhold_result.get(key)
        if s_val is not None and w_val is not None:
            effect[f"{key}_delta"] = s_val - w_val
            effect[f"share_{key}"] = s_val
            effect[f"withhold_{key}"] = w_val
    return effect


def run_paired_pilot(
    *,
    manifest_path: Path,
    output_dir: Path,
    marble_root: Path,
    max_pairs: int | None = None,
    resume: bool = False,
    retry_invalid: bool = False,
    dry_run: bool = False,
    order_seed: int = 0,
    engine_timeout_seconds: int = 900,
) -> dict[str, Any]:
    """Execute paired pilot experiments from a manifest.

    Parameters
    ----------
    manifest_path:
        Path to the paired pilot manifest JSONL.
    output_dir:
        Directory for pilot run outputs.
    marble_root:
        Path to MARBLE installation.
    max_pairs:
        Maximum number of pairs to execute (None = all).
    resume:
        If True, skip pairs that already have valid results.
    retry_invalid:
        If True, re-run pairs that were previously invalid.
    dry_run:
        If True, only print what would be executed.
    order_seed:
        Seed for execution ordering.
    engine_timeout_seconds:
        Timeout for each engine run.

    Returns
    -------
    Summary dict with pair_count, valid/invalid/failed counts, effects.
    """
    manifest = read_paired_pilot_manifest(manifest_path)
    pairs = manifest.get("pairs", [])
    if max_pairs is not None:
        pairs = pairs[:max_pairs]

    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "pilot_results.jsonl"
    status_path = output_dir / "pilot_status.json"

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
    failed = 0
    pending = 0
    effects: list[dict[str, Any]] = []

    runner = MarblePairedBranchRunner()

    for pair_spec in pairs:
        pair_key = pair_spec["pair_key"]
        task_id = pair_spec["task_id"]
        candidate_memory = pair_spec["candidate_memory"]
        branch_order = pair_spec["branch_order"]

        # Skip if already completed and not retrying
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
            failed += 1
            continue

        # Build initial state bundle
        bundle = bundle_from_manifest_task(
            {"raw_task": task, "task_id": task_id, "scenario": pair_spec["scenario"]},
            generation_seed=order_seed,
        )

        # Build candidate memory dict for branch runner
        candidate_dict = {
            "memory_id": candidate_memory["memory_id"],
            "payload": candidate_memory["payload"],
        }

        pair_output = output_dir / pair_key
        pair_output.mkdir(parents=True, exist_ok=True)

        try:
            receiver_agent_id = "agent1"
            task_id_str = str(task.get("task_id", pair_spec.get("task_id", "")))
            memory_id = str(candidate_dict.get("memory_id", "unknown"))
            edge_id = compute_edge_id(task_id_str, receiver_agent_id, memory_id)
            control_group_id = compute_control_family_id(
                task_id_str, receiver_agent_id
            )
            control = runner.run_no_memory_control(
                control_group_id=f"{control_group_id}_{order_seed}",
                task=task,
                initial_state_bundle=bundle,
                agent_config={"target_receiver_agent_id": receiver_agent_id},
                generation_seed=order_seed,
                workspace=pair_output,
                forbidden_memory_ids=(memory_id,),
                engine_timeout_seconds=engine_timeout_seconds,
            )
            share = runner.run_candidate_share(
                edge_id=edge_id,
                task=task,
                candidate_memory=candidate_dict,
                initial_state_bundle=bundle,
                agent_config={"target_receiver_agent_id": receiver_agent_id},
                generation_seed=order_seed,
                workspace=pair_output,
                engine_timeout_seconds=engine_timeout_seconds,
            )
            result = runner.assemble_shared_control_pair(
                control=control,
                share=share,
                candidate_memory_id=memory_id,
                branch_execution_order=branch_order,
            )

            result_dict = _pair_result_to_dict(result)
            result_dict["pair_key"] = pair_key
            result_dict["pair_id"] = pair_spec.get("pair_id", "")
            result_dict["memory_type"] = pair_spec.get("candidate_memory", {}).get("category", "")
            result_dict["seed"] = pair_spec.get("seed", order_seed)

            # Extract branch-level metrics from workspace outputs
            share_metrics = _extract_branch_metrics(pair_output / "share")
            withhold_metrics = _extract_branch_metrics(pair_output / "withhold")
            result_dict["share_metrics"] = share_metrics
            result_dict["withhold_metrics"] = withhold_metrics

            if result.paired_record_valid:
                result_dict["status"] = "valid_complete"
                result_dict["first_attempt_valid"] = True
                valid_complete += 1
            else:
                result_dict["status"] = "invalid_complete"
                result_dict["invalid_reason"] = result.invalid_reason
                result_dict["first_attempt_valid"] = False
                invalid_complete += 1

            # Compute treatment effect with full metrics
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
            effects.append(effect)
            result_dict["treatment_effect"] = effect

        except Exception as exc:
            result_dict = {
                "pair_key": pair_key,
                "status": "failed",
                "error": str(exc),
            }
            failed += 1

        # Append result
        with results_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(result_dict, sort_keys=True) + "\n")

    # Write status summary
    status = {
        "manifest_path": str(manifest_path),
        "total_pairs": len(pairs),
        "valid_complete": valid_complete,
        "invalid_complete": invalid_complete,
        "failed": failed,
        "pending": pending,
        "effects": effects,
    }
    status_path.write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return status


def main() -> None:
    """CLI entry point for run-paired-pilot."""
    import argparse
    parser = argparse.ArgumentParser(description="Run paired causal pilot")
    parser.add_argument("--manifest", required=True, help="Pilot manifest JSONL")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--marble-root", default="/home/ecs-user/MARBLE")
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-invalid", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--order-seed", type=int, default=0)
    parser.add_argument("--engine-timeout", type=int, default=900)
    args = parser.parse_args()

    result = run_paired_pilot(
        manifest_path=Path(args.manifest),
        output_dir=Path(args.output),
        marble_root=Path(args.marble_root),
        max_pairs=args.max_pairs,
        resume=args.resume,
        retry_invalid=args.retry_invalid,
        dry_run=args.dry_run,
        order_seed=args.order_seed,
        engine_timeout_seconds=args.engine_timeout,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
