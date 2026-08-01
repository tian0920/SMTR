"""CLI for MARBLE cross-agent shared memory pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from smtr.marble.engine_process import DEFAULT_ENGINE_TIMEOUT_SECONDS


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m smtr.marble.cli", description="SMTR MARBLE pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- Main pipeline commands ---

    p = subparsers.add_parser("inspect-dataset", help="Inspect MARBLE database tasks")
    p.add_argument("--marble-root", required=True)
    p.add_argument("--output", required=True)

    p = subparsers.add_parser("create-splits", help="Create train/validation/test splits")
    p.add_argument("--dataset-manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=0)

    p = subparsers.add_parser("collect-database-trajectories", help="Collect training trajectories")
    p.add_argument("--marble-root", required=True)
    p.add_argument("--dataset-manifest", required=True)
    p.add_argument("--split-manifest", required=True)
    p.add_argument("--split", default="train")
    p.add_argument("--task-ids", nargs="+", default=None)
    p.add_argument("--task-count", type=int, default=20)
    p.add_argument("--generation-seeds", type=int, nargs="+", default=[0])
    p.add_argument("--engine-timeout-seconds", type=int, default=DEFAULT_ENGINE_TIMEOUT_SECONDS)
    p.add_argument("--output", required=True)
    p.add_argument("--resume", action="store_true")

    p = subparsers.add_parser("extract-database-memories", help="Extract writer-agent procedural memories")
    p.add_argument("--trajectory-index", required=True)
    p.add_argument("--split-manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--min-actions", type=int, default=2)

    p = subparsers.add_parser("build-database-candidates", help="Build receiver-conditioned candidates")
    p.add_argument("--dataset-manifest", required=True)
    p.add_argument("--split-manifest", required=True)
    p.add_argument("--split", required=True, choices=["train", "validation", "test"])
    p.add_argument("--memory-pool", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--top-k", type=int, default=4)

    p = subparsers.add_parser("generate-database-paired-records", help="Generate candidate-level paired records")
    p.add_argument("--marble-root", required=True)
    p.add_argument("--dataset-manifest", required=True)
    p.add_argument("--split-manifest", required=True)
    p.add_argument("--split", required=True, choices=["train", "validation", "test"])
    p.add_argument("--candidate-manifest", required=True)
    p.add_argument("--memory-pool", required=True)
    p.add_argument("--generation-seeds", type=int, nargs="+", default=[0])
    p.add_argument("--limit-pairs", type=int, default=None)
    p.add_argument("--branch-order", choices=["share_then_withhold", "withhold_then_share"], default="share_then_withhold")
    p.add_argument("--engine-timeout-seconds", type=int, default=1800)
    p.add_argument("--output", required=True)

    p = subparsers.add_parser("train-critic", help="Train four-outcome transfer critic")
    p.add_argument("--train-records", required=True)
    p.add_argument("--validation-records", default=None)
    p.add_argument("--memory-pool", required=True)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--n-bootstrap", type=int, default=31)
    p.add_argument("--n-features", type=int, default=512)
    p.add_argument("--feature-block", default="full", choices=["full", "no_writer_receiver"])
    p.add_argument("--output", required=True)

    p = subparsers.add_parser("run-paired-decision-evaluation", help="Paired decision evaluation on test pairs")
    p.add_argument("--candidate-manifest", required=True)
    p.add_argument("--paired-records", required=True)
    p.add_argument("--memory-pool", required=True)
    p.add_argument("--checkpoint-full", required=True)
    p.add_argument("--checkpoint-no-writer-receiver", required=True)
    p.add_argument("--methods", nargs="+", default=["b0_no_memory", "top1_relevance", "all_share", "factual_success", "smtr", "smtr_no_risk", "smtr_no_writer_receiver"])
    p.add_argument("--negative-risk-budget", type=float, default=0.2)
    p.add_argument("--output", required=True)

    p = subparsers.add_parser("run-marble-evaluation", help="End-to-end MARBLE evaluation")
    p.add_argument("--marble-root", required=True)
    p.add_argument("--dataset-manifest", required=True)
    p.add_argument("--split-manifest", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--candidate-manifest", required=True)
    p.add_argument("--memory-pool", required=True)
    p.add_argument("--checkpoint-full", required=True)
    p.add_argument("--checkpoint-no-writer-receiver", required=True)
    p.add_argument("--methods", nargs="+", default=["b0_no_memory", "top1_relevance", "all_share", "factual_success", "smtr", "smtr_no_risk", "smtr_no_writer_receiver"])
    p.add_argument("--generation-seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--negative-risk-budget", type=float, default=0.2)
    p.add_argument("--output", required=True)

    p = subparsers.add_parser("integrity-audit", help="Run integrity audit")
    p.add_argument("--candidate-manifest", required=True)
    p.add_argument("--paired-records", required=True)
    p.add_argument("--memory-pool", required=True)
    p.add_argument("--paired-eval-dir", default=None)
    p.add_argument("--end-to-end-eval-dir", default=None)
    p.add_argument("--feature-audit", default=None)
    p.add_argument("--output", required=True)

    # --- Deprecated ---
    p = subparsers.add_parser("run-evaluation", help="[deprecated] Use run-paired-decision-evaluation or run-marble-evaluation")

    # --- Dev-only commands (prefixed with dev-) ---
    p = subparsers.add_parser("dev-runtime-preflight", help="[dev] Runtime preflight check")
    p.add_argument("--marble-root", required=True)
    p.add_argument("--output", required=True)

    args = parser.parse_args()
    _dispatch(args)


def _dispatch(args: argparse.Namespace) -> None:
    cmd = args.command

    if cmd == "inspect-dataset":
        from smtr.marble.dataset import build_marble_dataset_manifest, write_marble_dataset_manifest
        manifest = build_marble_dataset_manifest(marble_root=Path(args.marble_root))
        write_marble_dataset_manifest(manifest, Path(args.output))
        print(f"Dataset manifest written to {args.output}")

    elif cmd == "create-splits":
        from smtr.marble.splits import write_split_manifest
        write_split_manifest(
            dataset_manifest_path=Path(args.dataset_manifest),
            output_path=Path(args.output),
            seed=args.seed,
        )
        print(f"Splits written to {args.output}")

    elif cmd == "collect-database-trajectories":
        from smtr.marble.real_workflows import collect_database_trajectories
        result = collect_database_trajectories(
            marble_root=Path(args.marble_root),
            dataset_manifest_path=Path(args.dataset_manifest),
            split_manifest_path=Path(args.split_manifest),
            split=args.split,
            task_ids=args.task_ids,
            task_count=args.task_count,
            generation_seeds=args.generation_seeds,
            engine_timeout_seconds=args.engine_timeout_seconds,
            output_dir=Path(args.output),
            resume=args.resume,
        )
        print(json.dumps(result, indent=2))

    elif cmd == "extract-database-memories":
        from smtr.marble.real_data import (
            load_trajectories_from_index,
            extract_procedural_memories,
            write_memory_pool,
        )

        trajectories = load_trajectories_from_index(
            trajectory_index_path=Path(args.trajectory_index),
            split_manifest_path=Path(args.split_manifest),
            required_split="train",
        )

        memories = extract_procedural_memories(
            trajectories,
            min_actions=args.min_actions,
        )

        result = write_memory_pool(
            memories=memories,
            output_path=Path(args.output),
        )

        print(json.dumps(result, indent=2))

    elif cmd == "build-database-candidates":
        from smtr.marble.real_data import (
            load_memory_pool,
            load_receiver_entries,
            build_cross_task_candidates,
            write_candidate_manifest,
        )

        memories = load_memory_pool(Path(args.memory_pool))

        recipients = load_receiver_entries(
            dataset_manifest_path=Path(args.dataset_manifest),
            split_manifest_path=Path(args.split_manifest),
            split=args.split,
        )

        manifest = build_cross_task_candidates(
            memories=memories,
            recipients=recipients,
            top_k=args.top_k,
            target_split=args.split,
        )

        result = write_candidate_manifest(
            manifest=manifest,
            output_path=Path(args.output),
        )

        print(json.dumps(result, indent=2))

    elif cmd == "generate-database-paired-records":
        from smtr.marble.real_pairs import generate_candidate_level_pairs
        result = generate_candidate_level_pairs(
            marble_root=Path(args.marble_root),
            dataset_manifest_path=Path(args.dataset_manifest),
            split_manifest_path=Path(args.split_manifest),
            split=args.split,
            candidate_manifest_path=Path(args.candidate_manifest),
            memory_pool_path=Path(args.memory_pool),
            generation_seeds=args.generation_seeds,
            limit_pairs=args.limit_pairs,
            output_dir=Path(args.output),
            branch_execution_order=args.branch_order,
            engine_timeout_seconds=args.engine_timeout_seconds,
        )
        print(json.dumps(result, indent=2))

    elif cmd == "train-critic":
        from smtr.marble.training import train_critic
        result = train_critic(
            train_records_path=Path(args.train_records),
            validation_records_path=Path(args.validation_records) if args.validation_records else None,
            memory_pool_path=Path(args.memory_pool),
            output_path=Path(args.output),
            seed=args.seed,
            n_bootstrap=args.n_bootstrap,
            n_features=args.n_features,
            feature_block=args.feature_block,
        )
        print(json.dumps(result, indent=2))

    elif cmd == "run-paired-decision-evaluation":
        from smtr.marble.paired_evaluation import run_paired_decision_evaluation
        result = run_paired_decision_evaluation(
            candidate_manifest_path=Path(args.candidate_manifest),
            paired_records_path=Path(args.paired_records),
            memory_pool_path=Path(args.memory_pool),
            checkpoint_full=Path(args.checkpoint_full),
            checkpoint_no_writer_receiver=Path(args.checkpoint_no_writer_receiver),
            methods=args.methods,
            negative_risk_budget=args.negative_risk_budget,
            output=Path(args.output),
        )
        print(json.dumps(result, indent=2))

    elif cmd == "run-marble-evaluation":
        from smtr.marble.end_to_end_evaluation import run_end_to_end_evaluation
        result = run_end_to_end_evaluation(
            marble_root=Path(args.marble_root),
            dataset_manifest_path=Path(args.dataset_manifest),
            split_manifest_path=Path(args.split_manifest),
            split=args.split,
            candidate_manifest_path=Path(args.candidate_manifest),
            memory_pool_path=Path(args.memory_pool),
            checkpoint_full=Path(args.checkpoint_full),
            checkpoint_no_writer_receiver=Path(args.checkpoint_no_writer_receiver),
            methods=args.methods,
            generation_seeds=args.generation_seeds,
            negative_risk_budget=args.negative_risk_budget,
            output=Path(args.output),
        )
        print(json.dumps(result, indent=2))

    elif cmd == "integrity-audit":
        from smtr.marble.integrity import run_integrity_audit
        result = run_integrity_audit(
            candidate_manifest_path=Path(args.candidate_manifest),
            paired_records_path=Path(args.paired_records),
            memory_pool_path=Path(args.memory_pool),
            paired_eval_dir=Path(args.paired_eval_dir) if args.paired_eval_dir else None,
            end_to_end_eval_dir=Path(args.end_to_end_eval_dir) if args.end_to_end_eval_dir else None,
            feature_audit_path=Path(args.feature_audit) if args.feature_audit else None,
        )
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))

    elif cmd == "run-evaluation":
        print("Deprecated: use run-paired-decision-evaluation or run-marble-evaluation.")

    elif cmd == "dev-runtime-preflight":
        print("dev-runtime-preflight: not implemented in mainline")

    else:
        print(f"Unknown command: {cmd}")


if __name__ == "__main__":
    main()
