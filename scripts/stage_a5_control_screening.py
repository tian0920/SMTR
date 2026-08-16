"""Stage A.5-Fast: Control-only screening for train tasks.

For each train task NOT already in train_base, runs one no-memory
control (seed=0) to identify positive-transfer opportunity (Y0=0 → Group H).

Cost: ~60 engine runs × 5 min/run ≈ 5 hours.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from smtr.marble.environment.database_rebuild import SequentialDatabaseRebuilder
from smtr.marble.environment.isolation import bundle_from_manifest_task
from smtr.marble.environment.scenarios.database import MarbleDatabaseEnvironment
from smtr.marble.engine_process import DEFAULT_ENGINE_TIMEOUT_SECONDS
from smtr.marble.memory_injection import MarbleMemoryInjector
from smtr.marble.outcome.factory import evaluator_for_scenario
from smtr.marble.outcome.protocol import outcome_from_failure
from smtr.marble.io import load_split_task_ids
from smtr.marble.task_provider import load_database_task_by_id


def run_single_control(
    *,
    task_id: str,
    receiver_agent_id: str,
    seed: int,
    marble_root: Path,
    workspace_root: Path,
    engine_timeout: int,
) -> dict[str, Any]:
    """Run one no-memory control for a task and return result dict."""

    # Load full task from MARBLE source JSONL
    full_task = load_database_task_by_id(marble_root, task_id)

    # Build agent config targeting the specific receiver
    scenario = full_task.get("scenario", "database")
    agent_config: dict[str, Any] = {
        "target_receiver_agent_id": receiver_agent_id,
        "scenario": scenario,
        "task_id": task_id,
        "agents": full_task.get("agents", []),
    }

    # Build initial state bundle WITH workspace_template
    task_entry = {"task_id": task_id, "scenario": scenario, "raw_task": full_task}
    bundle = bundle_from_manifest_task(
        task_entry,
        environment_seed=0,
        generation_seed=seed,
    )

    group_workspace = workspace_root / f"task{task_id}_s{seed}_screen"
    group_workspace.mkdir(parents=True, exist_ok=True)

    rebuilder = SequentialDatabaseRebuilder()
    env = MarbleDatabaseEnvironment(
        task=full_task,
        workspace=group_workspace / "env_work",
        initial_state_bundle=bundle,
        agent_config=agent_config,
    )

    # Build agent input with NO memory
    base_input = env.build_agent_input(memory_payloads=())
    env.close()

    injector = MarbleMemoryInjector()
    control_input, _ = injector.build_agent_input(
        base_agent_input=base_input,
        memory_payloads=(),
        memory_ids=(),
    )

    run_metadata = {
        "run_id": f"screening_task{task_id}_s{seed}",
        "task_id": task_id,
        "scenario": scenario,
        "method": "control_only_screening",
        "branch": "withhold",
    }

    evaluator = evaluator_for_scenario(scenario)
    result: dict[str, Any] = {
        "task_id": task_id,
        "receiver_agent_id": receiver_agent_id,
        "seed": seed,
        "success": None,
        "native_score": None,
        "runtime_valid": False,
        "error": None,
    }

    try:
        rebuilder.materialize(
            initial_state_bundle=bundle,
            branch_workspace=group_workspace / "branch",
        )
        try:
            run_result = env.run(
                agent_input=control_input,
                generation_seed=seed,
                memory_injection=None,
                engine_timeout_seconds=engine_timeout,
                run_metadata=run_metadata,
            )
            outcome = evaluator.evaluate(task=full_task, run_result=run_result)
            result["success"] = bool(outcome.success)
            result["native_score"] = float(getattr(outcome, "native_score", None) or 0.0)
            result["runtime_valid"] = True
        except Exception as exc:
            outcome = outcome_from_failure(
                evaluator_name="marble_database_engine",
                reason=str(exc),
            )
            result["success"] = bool(outcome.success)
            result["error"] = f"run_error: {type(exc).__name__}: {exc}"
        finally:
            rebuilder.destroy(remove_workspace=False)
    except Exception as exc:
        result["error"] = f"setup_error: {type(exc).__name__}: {exc}"

    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--marble-root", required=True, type=Path)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--split", default="train")
    parser.add_argument("--candidate-manifest", required=True, type=Path,
                        help="Path to candidate manifest (JSON with 'candidates' key)")
    parser.add_argument("--train-base-tasks", required=True, type=str,
                        help="Comma-separated task_ids already in train_base (skip these)")
    parser.add_argument("--generation-seed", type=int, default=0)
    parser.add_argument("--engine-timeout", type=int, default=DEFAULT_ENGINE_TIMEOUT_SECONDS)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    marble_root = args.marble_root

    # Load task-receiver mapping from candidate manifest
    with open(args.candidate_manifest) as f:
        cand_data = json.load(f)
    task_receivers: dict[str, str] = {}
    for c in cand_data.get("candidates", []):
        tid = str(c["task_id"])
        if tid not in task_receivers:
            task_receivers[tid] = c.get("receiver_agent_id", "agent1")

    # Determine tasks to screen
    split_task_ids = load_split_task_ids(args.split_manifest, args.split)
    base_task_ids = set(args.train_base_tasks.split(","))
    screen_task_ids = sorted(split_task_ids - base_task_ids, key=int)

    print(f"Total split tasks: {len(split_task_ids)}", flush=True)
    print(f"Train base tasks (skip): {sorted(base_task_ids, key=int)}", flush=True)
    print(f"Tasks to screen: {len(screen_task_ids)}", flush=True)
    print(f"Seed: {args.generation_seed}", flush=True)
    print(f"Engine timeout: {args.engine_timeout}s", flush=True)
    print("", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    y0_fail = 0
    y0_success = 0
    errors = 0

    t0 = time.monotonic()
    for i, task_id in enumerate(screen_task_ids, 1):
        receiver = task_receivers.get(task_id, "agent1")
        print(f"[{i}/{len(screen_task_ids)}] Task {task_id} (receiver={receiver})...",
              end=" ", flush=True)

        r = run_single_control(
            task_id=task_id,
            receiver_agent_id=receiver,
            seed=args.generation_seed,
            marble_root=marble_root,
            workspace_root=args.output.parent / "screening_workspaces",
            engine_timeout=args.engine_timeout,
        )

        if r["error"]:
            status = f"ERROR: {r['error'][:80]}"
            errors += 1
        elif r["success"]:
            status = "Y0=1 (success, no H opportunity)"
            y0_success += 1
        else:
            status = "Y0=0 (FAILURE -> H candidate!)"
            y0_fail += 1

        print(status, flush=True)

        elapsed = time.monotonic() - t0
        rate = i / elapsed if elapsed > 0 else 0
        remaining = len(screen_task_ids) - i
        eta_h = remaining / rate / 3600 if rate > 0 else 0
        print(f"  Progress: {i}/{len(screen_task_ids)} "
              f"Rate: {rate*3600:.1f}/hr ETA: {eta_h:.1f}h", flush=True)
        print("", flush=True)

        results.append(r)

    # Write results
    args.output.write_text(
        "\n".join(json.dumps(r) for r in results) + "\n",
        encoding="utf-8",
    )

    # Summary
    print("=" * 60, flush=True)
    print(f"Screening complete: {len(results)} tasks", flush=True)
    print(f"  Y0=1 (success -> Group S, harmful-transfer opportunity): {y0_success}", flush=True)
    print(f"  Y0=0 (failure -> Group H, positive-transfer opportunity): {y0_fail}", flush=True)
    print(f"  Errors: {errors}", flush=True)
    print(f"Output: {args.output}", flush=True)

    if y0_fail == 0:
        print("\n*** WARNING: No H tasks found! All controls succeeded.", flush=True)
        print("*** BENCHMARK PROBLEM: no positive-transfer opportunity in train split.", flush=True)
    elif y0_fail < 4:
        print(f"\n*** WARNING: Only {y0_fail} H tasks found (need >=4 for 4-6 target).", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
