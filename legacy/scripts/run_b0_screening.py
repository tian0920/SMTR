"""B0 screening runner: 20 tasks x 3 seeds = 60 no-memory baseline runs.

Runs single-branch B0 (no memory) experiments to determine which tasks
fall in the sweet spot (neither ceiling nor floor) for the main experiment.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from smtr.marble.environment.isolation import bundle_from_manifest_task
from smtr.marble.environment.scenarios.database import MarbleDatabaseEnvironment
from smtr.marble.memory_injection import MarbleMemoryInjector
from smtr.marble.outcome.factory import evaluator_for_scenario
from smtr.marble.outcome.protocol import outcome_from_failure
from smtr.marble.task_provider import _read_jsonl_line


def run_b0_single(
    *,
    task: dict[str, Any],
    task_id: int,
    seed: int,
    workspace: Path,
    marble_root: Path = Path("/home/ecs-user/MARBLE"),
    engine_timeout_seconds: int = 900,
) -> dict[str, Any]:
    """Run a single B0 (no memory) branch and return detailed results."""
    bundle = bundle_from_manifest_task(
        {"raw_task": task, "task_id": str(task_id), "scenario": "database"},
        generation_seed=seed,
    )
    agent_config = {"target_receiver_agent_id": "agent1"}
    env = MarbleDatabaseEnvironment(
        task=task,
        workspace=workspace,
        initial_state_bundle=bundle,
        agent_config=agent_config,
        marble_root=marble_root,
    )
    base_input = env.build_agent_input(memory_payloads=())
    run_metadata = {
        "run_id": f"b0_screening_task{task_id}_seed{seed}",
        "task_id": str(task_id),
        "scenario": "database",
        "method": "b0_no_memory",
        "branch": "b0",
    }
    try:
        raw_result = env.run(
            agent_input=base_input,
            generation_seed=seed,
            memory_injection=None,
            engine_timeout_seconds=engine_timeout_seconds,
            run_metadata=run_metadata,
        )
        evaluator = evaluator_for_scenario("database")
        outcome = evaluator.evaluate(task=task, run_result=raw_result)
        real_engine_executed = bool(raw_result.get("real_engine_executed"))
    except Exception as exc:
        raw_result = {"error": str(exc)}
        outcome = outcome_from_failure(
            evaluator_name="marble_database_engine",
            reason=str(exc),
            raw_result=raw_result,
        )
        real_engine_executed = False
    finally:
        env.close()

    # Extract fine-grained metrics
    fg = outcome.fine_grained or {}
    task_eval = raw_result.get("task_evaluation") or {}

    return {
        "task_id": task_id,
        "seed": seed,
        "success": outcome.success,
        "score": outcome.score,
        "native_evaluator_executed": outcome.native_evaluator_executed,
        "real_engine_executed": real_engine_executed,
        "failure_reason": outcome.failure_reason,
        "tp": fg.get("tp", 0),
        "fp": fg.get("fp", 0),
        "recall": fg.get("recall", 0.0),
        "precision": fg.get("precision", 0.0),
        "f1": fg.get("f1", 0.0),
        "predicted_labels": fg.get("predicted_labels", []),
        "expected_labels": fg.get("expected_labels", []),
        "initial_digest": env.initial_state_digest(),
    }


def run_b0_screening(
    *,
    task_ids: list[int],
    seeds: list[int],
    output_dir: Path,
    marble_root: Path = Path("/home/ecs-user/MARBLE"),
    resume: bool = False,
    dry_run: bool = False,
    engine_timeout_seconds: int = 900,
) -> dict[str, Any]:
    """Execute B0 screening runs for all task x seed combinations."""
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "b0_screening_results.jsonl"
    status_path = output_dir / "b0_screening_status.json"

    # Load existing results if resuming
    existing: dict[str, dict] = {}
    if resume and results_path.exists():
        for line in results_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            key = f"{r['task_id']}_{r['seed']}"
            existing[key] = r

    total = len(task_ids) * len(seeds)
    completed = 0
    failed = 0
    pending = 0

    for task_id in task_ids:
        for seed in seeds:
            key = f"{task_id}_{seed}"
            if resume and key in existing:
                completed += 1
                continue

            if dry_run:
                print(f"[dry-run] Would run B0: task={task_id} seed={seed}")
                pending += 1
                continue

            # Load task from MARBLE dataset
            try:
                task_path = marble_root / "multiagentbench/database/database_main.jsonl"
                task = _read_jsonl_line(task_path, task_id)
            except Exception as exc:
                print(f"[error] Cannot load task {task_id}: {exc}")
                failed += 1
                continue

            workspace = output_dir / f"b0_task{task_id}_seed{seed}"
            workspace.mkdir(parents=True, exist_ok=True)

            print(f"[run] B0 task={task_id} seed={seed}")
            start_time = time.time()

            try:
                result = run_b0_single(
                    task=task,
                    task_id=task_id,
                    seed=seed,
                    workspace=workspace,
                    marble_root=marble_root,
                    engine_timeout_seconds=engine_timeout_seconds,
                )
                elapsed = time.time() - start_time
                result["elapsed_seconds"] = round(elapsed, 2)
                result["status"] = "complete"

                if result["success"]:
                    print(f"  -> SUCCESS f1={result['f1']:.2f} "
                          f"predicted={result['predicted_labels']}")
                else:
                    print(f"  -> FAIL f1={result['f1']:.2f} "
                          f"predicted={result['predicted_labels']} "
                          f"expected={result['expected_labels']}")
                completed += 1

            except Exception as exc:
                elapsed = time.time() - start_time
                result = {
                    "task_id": task_id,
                    "seed": seed,
                    "status": "failed",
                    "error": str(exc),
                    "elapsed_seconds": round(elapsed, 2),
                }
                failed += 1
                print(f"  -> FAILED: {exc}")

            # Append result
            with results_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(result, sort_keys=True, default=str) + "\n")

    # Write status
    status = {
        "total_runs": total,
        "completed": completed,
        "failed": failed,
        "pending": pending,
        "task_ids": task_ids,
        "seeds": seeds,
    }
    status_path.write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return status


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Run B0 screening experiment")
    parser.add_argument(
        "--task-selection",
        default="artifacts/paper_experiments/memory_intervention/b0_screening/task_selection.json",
    )
    parser.add_argument(
        "--seeds", default="41,42,43",
        help="Comma-separated seeds",
    )
    parser.add_argument(
        "--output",
        default="artifacts/marble/outputs/b0_screening",
    )
    parser.add_argument("--marble-root", default="/home/ecs-user/MARBLE")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--engine-timeout", type=int, default=900)
    parser.add_argument(
        "--task-ids", default=None,
        help="Override: comma-separated task IDs (ignores task-selection file)",
    )
    args = parser.parse_args()

    if args.task_ids:
        task_ids = [int(x.strip()) for x in args.task_ids.split(",")]
    else:
        selection = json.loads(Path(args.task_selection).read_text(encoding="utf-8"))
        task_ids = [t["task_id"] for t in selection]

    seeds = [int(x.strip()) for x in args.seeds.split(",")]

    result = run_b0_screening(
        task_ids=task_ids,
        seeds=seeds,
        output_dir=Path(args.output),
        marble_root=Path(args.marble_root),
        resume=args.resume,
        dry_run=args.dry_run,
        engine_timeout_seconds=args.engine_timeout,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
