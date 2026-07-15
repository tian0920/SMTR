"""CLI for real MARBLE dataset and pilot-isolation workflows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from smtr.marble.capabilities import write_capability_manifest
from smtr.marble.database_pilot import (
    create_database_pilot_manifest,
    run_database_paired_pilot,
)
from smtr.marble.database_smoke import (
    run_database_b0_smoke,
    run_database_paired_smoke,
    verify_database_rebuild,
)
from smtr.marble.dataset import DEFAULT_MARBLE_ROOT, write_marble_dataset_manifest
from smtr.marble.engine_audit import audit_database_engine
from smtr.marble.engine_process import DEFAULT_ENGINE_TIMEOUT_SECONDS
from smtr.marble.environment.isolation import bundle_from_manifest_task
from smtr.marble.environment.scenarios.database import MarbleDatabaseEnvironment
from smtr.marble.evaluation import MarbleExperimentRunner
from smtr.marble.integrity import audit_marble_pilot, audit_marble_pilot_run
from smtr.marble.paired_records import MarblePairedRecordGenerator
from smtr.marble.real_data import (
    RealDatabaseTrajectory,
    RealProceduralMemory,
    build_cross_task_candidates,
    extract_procedural_memories,
    file_sha256,
)
from smtr.marble.real_pairs import generate_real_database_pairs
from smtr.marble.real_workflows import collect_database_trajectories
from smtr.marble.runtime_preflight import write_runtime_preflight
from smtr.marble.splits import write_split_manifest
from smtr.marble.task_provider import _read_jsonl_line
from smtr.marble.training import MarbleTrainingPipeline


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m smtr.marble.cli")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect-dataset")
    inspect_parser.add_argument("--marble-root", default=str(DEFAULT_MARBLE_ROOT))
    inspect_parser.add_argument("--output", required=True)

    splits_parser = subparsers.add_parser("create-splits")
    splits_parser.add_argument("--dataset-manifest", required=True)
    splits_parser.add_argument("--output", required=True)
    splits_parser.add_argument("--seed", type=int, default=0)

    capabilities_parser = subparsers.add_parser("inspect-capabilities")
    capabilities_parser.add_argument("--marble-root", default=str(DEFAULT_MARBLE_ROOT))
    capabilities_parser.add_argument("--output", required=True)

    engine_audit_parser = subparsers.add_parser("audit-database-engine")
    engine_audit_parser.add_argument("--marble-root", default=str(DEFAULT_MARBLE_ROOT))
    engine_audit_parser.add_argument("--output", required=True)

    preflight_parser = subparsers.add_parser("runtime-preflight")
    preflight_parser.add_argument("--marble-root", default=str(DEFAULT_MARBLE_ROOT))
    preflight_parser.add_argument("--output", required=True)

    b0_smoke_parser = subparsers.add_parser("run-database-b0-smoke")
    b0_smoke_parser.add_argument("--marble-root", default=str(DEFAULT_MARBLE_ROOT))
    b0_smoke_parser.add_argument("--task-id", required=True)
    b0_smoke_parser.add_argument("--generation-seed", type=int, default=0)
    b0_smoke_parser.add_argument("--engine-timeout-seconds", type=int)
    b0_smoke_parser.add_argument("--output", required=True)

    rebuild_parser = subparsers.add_parser("verify-database-rebuild")
    rebuild_parser.add_argument("--marble-root", default=str(DEFAULT_MARBLE_ROOT))
    rebuild_parser.add_argument("--task-id", required=True)
    rebuild_parser.add_argument("--output", required=True)

    paired_smoke_parser = subparsers.add_parser("run-database-paired-smoke")
    paired_smoke_parser.add_argument("--marble-root", default=str(DEFAULT_MARBLE_ROOT))
    paired_smoke_parser.add_argument("--task-id", required=True)
    paired_smoke_parser.add_argument("--memory-id", required=True)
    paired_smoke_parser.add_argument("--generation-seed", type=int, default=0)
    paired_smoke_parser.add_argument(
        "--branch-order",
        choices=["share-then-withhold", "withhold-then-share"],
        default="share-then-withhold",
    )
    paired_smoke_parser.add_argument("--output", required=True)

    engine_smoke_parser = subparsers.add_parser("run-database-engine-smoke")
    engine_smoke_parser.add_argument("--marble-root", default=str(DEFAULT_MARBLE_ROOT))
    engine_smoke_parser.add_argument("--task-id", required=True)
    engine_smoke_parser.add_argument("--output", required=True)

    pilot_manifest_parser = subparsers.add_parser("create-database-pilot-manifest")
    pilot_manifest_parser.add_argument("--dataset-manifest", required=True)
    pilot_manifest_parser.add_argument("--split-manifest", required=True)
    pilot_manifest_parser.add_argument("--output", required=True)
    pilot_manifest_parser.add_argument("--task-count", type=int, default=5)

    pilot_parser = subparsers.add_parser("run-database-paired-pilot")
    pilot_parser.add_argument("--marble-root", default=str(DEFAULT_MARBLE_ROOT))
    pilot_parser.add_argument("--pilot-manifest", required=True)
    pilot_parser.add_argument("--output", required=True)
    pilot_parser.add_argument("--generation-seed", type=int, default=0)

    records_parser = subparsers.add_parser("generate-paired-records")
    records_parser.add_argument("--marble-root", default=str(DEFAULT_MARBLE_ROOT))
    records_parser.add_argument("--dataset-manifest", required=True)
    records_parser.add_argument("--split-manifest", required=True)
    records_parser.add_argument("--split", choices=["train", "validation", "test"], required=True)
    records_parser.add_argument("--scenario", required=True)
    records_parser.add_argument("--output", required=True)
    records_parser.add_argument("--generation-seeds", type=int, nargs="+", default=[0])
    records_parser.add_argument("--limit-tasks", type=int)

    trajectory_parser = subparsers.add_parser("collect-database-trajectories")
    trajectory_parser.add_argument("--marble-root", default=str(DEFAULT_MARBLE_ROOT))
    trajectory_parser.add_argument("--dataset-manifest", required=True)
    trajectory_parser.add_argument("--split-manifest", required=True)
    trajectory_parser.add_argument("--split", choices=["train"], required=True)
    trajectory_parser.add_argument("--task-ids", nargs="+")
    trajectory_parser.add_argument("--task-count", type=int)
    trajectory_parser.add_argument("--generation-seeds", type=int, nargs="+", default=[0])
    trajectory_parser.add_argument("--engine-timeout-seconds", type=int)
    trajectory_parser.add_argument("--output", required=True)
    trajectory_parser.add_argument("--resume", action="store_true")

    memory_parser = subparsers.add_parser("extract-database-memories")
    memory_parser.add_argument("--trajectory-index", required=True)
    memory_parser.add_argument("--split-manifest", required=True)
    memory_parser.add_argument("--output", required=True)

    candidate_parser = subparsers.add_parser("build-database-candidates")
    candidate_parser.add_argument("--dataset-manifest", required=True)
    candidate_parser.add_argument("--split-manifest", required=True)
    candidate_parser.add_argument("--memory-pool", required=True)
    candidate_parser.add_argument("--output", required=True)
    candidate_parser.add_argument("--top-k", type=int, default=4)

    real_pair_parser = subparsers.add_parser("generate-database-paired-records")
    real_pair_parser.add_argument("--dataset-manifest", required=True)
    real_pair_parser.add_argument("--split-manifest", required=True)
    real_pair_parser.add_argument("--candidate-manifest", required=True)
    real_pair_parser.add_argument("--memory-pool", required=True)
    real_pair_parser.add_argument("--generation-seeds", type=int, nargs="+", default=[0])
    real_pair_parser.add_argument("--limit-pairs", type=int)
    real_pair_parser.add_argument("--output", required=True)

    train_parser = subparsers.add_parser("train-critic")
    train_parser.add_argument("--train-records", required=True)
    train_parser.add_argument("--validation-records", required=True)
    train_parser.add_argument("--memory-pool", required=True)
    train_parser.add_argument("--seed", type=int, default=7)
    train_parser.add_argument("--n-bootstrap", type=int, default=31)
    train_parser.add_argument("--n-features", type=int, default=512)
    train_parser.add_argument("--feature-block", default="full")
    train_parser.add_argument("--output", required=True)

    eval_parser = subparsers.add_parser("run-evaluation")
    eval_parser.add_argument("--marble-root", default=str(DEFAULT_MARBLE_ROOT))
    eval_parser.add_argument("--dataset-manifest", required=True)
    eval_parser.add_argument("--split-manifest", required=True)
    eval_parser.add_argument("--split", choices=["test"], required=True)
    eval_parser.add_argument("--scenario", required=True)
    eval_parser.add_argument("--memory-pool", required=True)
    eval_parser.add_argument("--checkpoint", required=True)
    eval_parser.add_argument(
        "--methods",
        nargs="+",
        default=["smtr", "b0_no_memory", "all_share"],
    )
    eval_parser.add_argument(
        "--negative-risk-budget", type=float, default=0.2,
    )
    eval_parser.add_argument("--output", required=True)

    # --- New commands (Milestone 5) ---
    preflight_alias = subparsers.add_parser("preflight")
    preflight_alias.add_argument("--marble-root", default=str(DEFAULT_MARBLE_ROOT))
    preflight_alias.add_argument("--output", default="artifacts/marble/outputs/preflight.json")

    run_cmd = subparsers.add_parser("run")
    run_cmd.add_argument("--marble-root", default=str(DEFAULT_MARBLE_ROOT))
    run_cmd.add_argument("--method", choices=["b0", "allshare", "smtr"], required=True)
    run_cmd.add_argument("--scenario", default="database")
    run_cmd.add_argument("--task-id", required=True)
    run_cmd.add_argument("--seed", type=int, default=0)
    run_cmd.add_argument("--engine-timeout-seconds", type=int)
    run_cmd.add_argument("--memory-pool")
    run_cmd.add_argument("--checkpoint")
    run_cmd.add_argument("--output", required=True)

    run_pair_cmd = subparsers.add_parser("run-pair")
    run_pair_cmd.add_argument("--marble-root", default=str(DEFAULT_MARBLE_ROOT))
    run_pair_cmd.add_argument("--scenario", default="database")
    run_pair_cmd.add_argument("--task-id", required=True)
    run_pair_cmd.add_argument("--candidate-memory-id", required=True)
    run_pair_cmd.add_argument("--seed", type=int, default=0)
    run_pair_cmd.add_argument(
        "--branch-order",
        choices=["share-then-withhold", "withhold-then-share"],
        default="share-then-withhold",
    )
    run_pair_cmd.add_argument("--output", required=True)

    eval_methods = subparsers.add_parser("evaluate")
    eval_methods.add_argument("--marble-root", default=str(DEFAULT_MARBLE_ROOT))
    eval_methods.add_argument("--methods", nargs="+", default=["b0", "allshare", "smtr"])
    eval_methods.add_argument("--scenario", default="database")
    eval_methods.add_argument("--task-id", required=True)
    eval_methods.add_argument("--seed", type=int, default=0)
    eval_methods.add_argument("--memory-pool")
    eval_methods.add_argument("--checkpoint")
    eval_methods.add_argument("--engine-timeout-seconds", type=int)
    eval_methods.add_argument("--output", required=True)

    audit_parser = subparsers.add_parser("integrity-audit")
    audit_parser.add_argument("--run-dir")
    audit_parser.add_argument("--split-manifest")
    audit_parser.add_argument("--paired-records")
    audit_parser.add_argument("--output")

    args = parser.parse_args()
    if args.command == "inspect-dataset":
        manifest = write_marble_dataset_manifest(
            marble_root=Path(args.marble_root),
            output_path=Path(args.output),
            scenarios={"database"},
        )
        _print_counts(manifest.total_tasks, manifest.scenario_counts)
    elif args.command == "create-splits":
        manifest = write_split_manifest(
            dataset_manifest_path=Path(args.dataset_manifest),
            output_path=Path(args.output),
            seed=args.seed,
        )
        _print_counts(len(manifest.records), manifest.split_counts)
    elif args.command == "inspect-capabilities":
        manifest = write_capability_manifest(
            marble_root=Path(args.marble_root),
            output_path=Path(args.output),
        )
        print(f"pilot_scenario={manifest.pilot_scenario}")
        for scenario, capability in manifest.scenarios.items():
            print(f"scenario.{scenario}.pilot_supported={capability.pilot_supported}")
    elif args.command == "audit-database-engine":
        summary = audit_database_engine(
            marble_root=Path(args.marble_root),
            output_path=Path(args.output),
        )
        print(
            f"real_engine_execution_safe={summary['real_engine_execution_safe_for_paired_isolation']}"
        )
    elif args.command == "runtime-preflight":
        result = write_runtime_preflight(
            marble_root=Path(args.marble_root),
            output_path=Path(args.output),
        )
        print(f"runtime_preflight.ready={result.ready}")
        for check in result.checks:
            if check.blocking and not check.passed:
                print(f"blocking_failure={check.name}: {check.detail}")
    elif args.command == "run-database-b0-smoke":
        if args.engine_timeout_seconds is not None and args.engine_timeout_seconds <= 0:
            raise SystemExit("--engine-timeout-seconds must be positive")
        engine_timeout_seconds = (
            args.engine_timeout_seconds
            if args.engine_timeout_seconds is not None
            else DEFAULT_ENGINE_TIMEOUT_SECONDS
        )
        summary = run_database_b0_smoke(
            marble_root=Path(args.marble_root),
            task_id=str(args.task_id),
            generation_seed=args.generation_seed,
            output_dir=Path(args.output),
            engine_timeout_seconds=engine_timeout_seconds,
        )
        print(f"real_engine_executed={summary['real_engine_executed']}")
        print(f"native_evaluator_executed={summary['native_evaluator_executed']}")
        print(f"environment_valid={summary['environment_valid']}")
        print(f"timeout_seconds={summary['timeout_seconds']}")
        print(f"timed_out={summary['timed_out']}")
    elif args.command == "verify-database-rebuild":
        summary = verify_database_rebuild(
            marble_root=Path(args.marble_root),
            task_id=str(args.task_id),
            output_dir=Path(args.output),
        )
        print(f"initial_digests_match={summary['initial_digests_match']}")
        print(f"marker_leakage={summary['marker_leakage']}")
    elif args.command == "run-database-paired-smoke":
        summary = run_database_paired_smoke(
            marble_root=Path(args.marble_root),
            task_id=str(args.task_id),
            memory_id=str(args.memory_id),
            generation_seed=args.generation_seed,
            branch_order=args.branch_order,
            output_dir=Path(args.output),
        )
        print(f"paired_record_valid={summary['paired_record_valid']}")
        print(f"paired_label={summary['paired_label']}")
    elif args.command == "run-database-engine-smoke":
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        task = _load_database_task_by_id(Path(args.marble_root), str(args.task_id))
        bundle = bundle_from_manifest_task(
            {"raw_task": task, "task_id": str(args.task_id), "scenario": "database"}
        )
        env = MarbleDatabaseEnvironment(
            task=task,
            workspace=output / "workspace",
            initial_state_bundle=bundle,
            agent_config={"target_receiver_agent_id": "agent1"},
            marble_root=Path(args.marble_root),
        )
        try:
            try:
                env.run(agent_input=env.build_agent_input(memory_payloads=()), generation_seed=0)
                real_engine_executed = True
                error = None
            except Exception as exc:
                real_engine_executed = False
                error = str(exc)
            smoke = {
                "task_id": str(args.task_id),
                "real_engine_executed": real_engine_executed,
                "error": error,
                "initial_state_digest": env.initial_state_digest(),
                "final_state_digest": env.final_state_digest(),
            }
            (output / "engine_smoke.json").write_text(
                json.dumps(smoke, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(f"real_engine_executed={real_engine_executed}")
            if error:
                print(f"error={error}")
        finally:
            env.close()
    elif args.command == "create-database-pilot-manifest":
        manifest = create_database_pilot_manifest(
            dataset_manifest_path=Path(args.dataset_manifest),
            split_manifest_path=Path(args.split_manifest),
            output_path=Path(args.output),
            task_count=args.task_count,
        )
        print(f"task_count={manifest['task_count']}")
    elif args.command == "run-database-paired-pilot":
        summary = run_database_paired_pilot(
            pilot_manifest_path=Path(args.pilot_manifest),
            output_dir=Path(args.output),
            generation_seed=args.generation_seed,
        )
        print(f"pair_count={summary['pair_count']}")
        print(f"invalid_pair_count={summary['invalid_pair_count']}")
        print(f"label_counts={json.dumps(summary['label_counts'], sort_keys=True)}")
    elif args.command == "generate-paired-records":
        summary = MarblePairedRecordGenerator().generate(
            dataset_manifest_path=Path(args.dataset_manifest),
            split_manifest_path=Path(args.split_manifest),
            split=args.split,
            scenario=args.scenario,
            output_dir=Path(args.output),
            generation_seeds=args.generation_seeds,
            limit_tasks=args.limit_tasks,
        )
        print(f"record_count={summary.record_count}")
        print(f"valid_count={summary.valid_count}")
        print(f"invalid_count={summary.invalid_count}")
        print(f"label_counts={json.dumps(summary.label_counts, sort_keys=True)}")
    elif args.command == "collect-database-trajectories":
        if args.engine_timeout_seconds is not None and args.engine_timeout_seconds <= 0:
            raise SystemExit("--engine-timeout-seconds must be positive")
        engine_timeout_seconds = (
            args.engine_timeout_seconds
            if args.engine_timeout_seconds is not None
            else DEFAULT_ENGINE_TIMEOUT_SECONDS
        )
        summary = collect_database_trajectories(
            marble_root=Path(args.marble_root),
            dataset_manifest_path=Path(args.dataset_manifest),
            split_manifest_path=Path(args.split_manifest),
            split=args.split,
            task_ids=args.task_ids,
            task_count=args.task_count,
            generation_seeds=args.generation_seeds,
            output_dir=Path(args.output),
            resume=args.resume,
            engine_timeout_seconds=engine_timeout_seconds,
        )
        print(json.dumps(summary, sort_keys=True))
    elif args.command == "extract-database-memories":
        split_payload = json.loads(Path(args.split_manifest).read_text(encoding="utf-8"))
        group_by_task = {
            str(record["task_id"]): str(record["group_id"]) for record in split_payload["records"]
        }
        trajectories = []
        for line in Path(args.trajectory_index).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            index_record = json.loads(line)
            if not index_record.get("valid"):
                continue
            trajectory_path = Path(index_record["path"])
            trajectories.append(
                RealDatabaseTrajectory.model_validate_json(
                    trajectory_path.read_text(encoding="utf-8")
                )
            )
        memories = extract_procedural_memories(
            trajectories,
            group_by_task=group_by_task,
            created_at=str(split_payload.get("created_at") or "unknown"),
        )
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            "".join(memory.model_dump_json() + "\n" for memory in memories),
            encoding="utf-8",
        )
        print(f"memory_count={len(memories)}")
    elif args.command == "build-database-candidates":
        dataset_path = Path(args.dataset_manifest)
        split_path = Path(args.split_manifest)
        dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
        splits = json.loads(split_path.read_text(encoding="utf-8"))
        tasks = {str(task["task_id"]): task for task in dataset["tasks"]}
        validation_records = [
            record for record in splits["records"] if record["split"] == "validation"
        ]
        recipients = [
            {
                "task_id": str(record["task_id"]),
                "group_id": str(record["group_id"]),
                "instruction": str(
                    _read_jsonl_line(
                        Path(tasks[str(record["task_id"])]["source_path"]),
                        int(tasks[str(record["task_id"])]["source_line"]),
                    )["task"]["content"]
                ),
            }
            for record in validation_records
        ]
        memories = [
            RealProceduralMemory.model_validate_json(line)
            for line in Path(args.memory_pool).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        manifest = build_cross_task_candidates(
            memories=memories,
            recipients=recipients,
            group_by_task={
                str(record["task_id"]): str(record["group_id"])
                for record in splits["records"]
            },
            dataset_manifest_sha256=file_sha256(dataset_path),
            split_manifest_sha256=file_sha256(split_path),
            created_at=str(splits.get("created_at") or "unknown"),
            top_k=args.top_k,
        )
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
        print(f"candidate_set_count={len(manifest.candidates)}")
    elif args.command == "generate-database-paired-records":
        summary = generate_real_database_pairs(
            dataset_manifest_path=Path(args.dataset_manifest),
            split_manifest_path=Path(args.split_manifest),
            candidate_manifest_path=Path(args.candidate_manifest),
            memory_pool_path=Path(args.memory_pool),
            generation_seeds=args.generation_seeds,
            limit_pairs=args.limit_pairs,
            output_dir=Path(args.output),
        )
        print(json.dumps(summary, sort_keys=True))
    elif args.command == "train-critic":
        summary = MarbleTrainingPipeline().train(
            train_records=Path(args.train_records),
            validation_records=Path(args.validation_records),
            memory_pool=Path(args.memory_pool),
            output=Path(args.output),
            seed=args.seed,
            n_bootstrap=args.n_bootstrap,
            n_features=args.n_features,
            feature_block=args.feature_block,
        )
        print(f"train_records_bridged={summary['train_record_count_bridged']}")
        print(f"validation_records_bridged={summary['validation_record_count_bridged']}")
        print(f"checkpoint_sha256={summary['checkpoint_sha256'][:16]}")
        metrics = summary.get('validation_metrics', {})
        if metrics.get('accuracy') is not None:
            print(f"validation_accuracy={metrics['accuracy']:.4f}")
        if metrics.get('log_loss') is not None:
            print(f"validation_log_loss={metrics['log_loss']:.4f}")
    elif args.command == "run-evaluation":
        summary = MarbleExperimentRunner().run(
            dataset_manifest=Path(args.dataset_manifest),
            split_manifest=Path(args.split_manifest),
            split=args.split,
            scenario=args.scenario,
            checkpoint=Path(args.checkpoint),
            memory_pool=Path(args.memory_pool),
            output=Path(args.output),
            methods=args.methods,
            negative_risk_budget=args.negative_risk_budget,
        )
        print(f"task_count={summary['task_count']}")
        for method, agg in summary.get('aggregate', {}).items():
            print(
                f"{method}: share={agg['share_count']} "
                f"withhold={agg['withhold_count']} "
                f"rate={agg['share_rate']:.3f}"
            )
    elif args.command == "preflight":
        result = write_runtime_preflight(
            marble_root=Path(args.marble_root),
            output_path=Path(args.output),
        )
        print(f"runtime_preflight.ready={result.ready}")
        for check in result.checks:
            if check.blocking and not check.passed:
                print(f"blocking_failure={check.name}: {check.detail}")
    elif args.command == "run":
        from smtr.marble.marble_environment_evaluation import MarbleEnvironmentEvaluator
        method_map = {"b0": "b0_no_memory", "allshare": "all_share", "smtr": "smtr"}
        method = method_map[args.method]
        marble_root = Path(args.marble_root)
        task = _load_database_task_by_id(marble_root, str(args.task_id))
        engine_timeout = args.engine_timeout_seconds or DEFAULT_ENGINE_TIMEOUT_SECONDS
        evaluator = MarbleEnvironmentEvaluator()
        result = evaluator.evaluate_method(
            method=method,
            task=task,
            task_id=str(args.task_id),
            scenario=args.scenario,
            marble_root=marble_root,
            output_dir=Path(args.output),
            generation_seed=args.seed,
            engine_timeout_seconds=engine_timeout,
        )
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "run_result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8",
        )
        print(f"real_engine_executed={result['real_engine_executed']}")
        print(f"native_evaluator_executed={result['native_evaluator_executed']}")
        print(f"method={result['method']}")
    elif args.command == "run-pair":
        marble_root = Path(args.marble_root)
        task = _load_database_task_by_id(marble_root, str(args.task_id))
        summary = run_database_paired_smoke(
            marble_root=marble_root,
            task_id=str(args.task_id),
            memory_id=str(args.candidate_memory_id),
            generation_seed=args.seed,
            branch_order=args.branch_order,
            output_dir=Path(args.output),
        )
        print(f"paired_record_valid={summary['paired_record_valid']}")
        print(f"paired_label={summary['paired_label']}")
    elif args.command == "evaluate":
        from smtr.marble.marble_environment_evaluation import MarbleEnvironmentEvaluator
        marble_root = Path(args.marble_root)
        task = _load_database_task_by_id(marble_root, str(args.task_id))
        engine_timeout = args.engine_timeout_seconds or DEFAULT_ENGINE_TIMEOUT_SECONDS
        method_map = {"b0": "b0_no_memory", "allshare": "all_share", "smtr": "smtr"}
        evaluator = MarbleEnvironmentEvaluator()
        results: list[dict] = []
        for m in args.methods:
            method_key = method_map.get(m, m)
            result = evaluator.evaluate_method(
                method=method_key,
                task=task,
                task_id=str(args.task_id),
                scenario=args.scenario,
                marble_root=marble_root,
                output_dir=Path(args.output) / method_key,
                generation_seed=args.seed,
                engine_timeout_seconds=engine_timeout,
            )
            results.append(result)
            print(f"{method_key}: real_engine={result['real_engine_executed']} "
                  f"evaluator={result['native_evaluator_executed']}")
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "evaluate_results.json").write_text(
            json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8",
        )
    elif args.command == "integrity-audit":
        if args.run_dir:
            summary = audit_marble_pilot_run(run_dir=Path(args.run_dir))
            output = Path(args.output or (Path(args.run_dir) / "integrity_summary.json"))
        else:
            if not args.split_manifest or not args.paired_records:
                raise SystemExit(
                    "integrity-audit requires --run-dir or old --split-manifest/--paired-records"
                )
            summary = audit_marble_pilot(
                split_manifest_path=Path(args.split_manifest),
                paired_records_path=Path(args.paired_records),
            )
            output = Path(args.output or "artifacts/marble/outputs/integrity_summary.json")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(json.dumps(summary, sort_keys=True))


def _print_counts(total: int, counts: dict[str, int]) -> None:
    print(f"total_tasks={total}")
    for key, value in counts.items():
        print(f"{key}={value}")


def _load_database_task_by_id(marble_root: Path, task_id: str) -> dict:
    path = marble_root / "multiagentbench/database/database_main.jsonl"
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            task = json.loads(line)
            if str(task.get("task_id")) == task_id:
                return _read_jsonl_line(path, line_number)
    raise ValueError(f"database task_id not found: {task_id}")


if __name__ == "__main__":
    main()
