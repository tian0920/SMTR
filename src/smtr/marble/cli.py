"""CLI for MARBLE cross-agent shared memory pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


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
    p.add_argument("--task-count", type=int, default=20)
    p.add_argument("--generation-seeds", type=int, nargs="+", default=[0])
    p.add_argument("--output", required=True)
    p.add_argument("--resume", action="store_true")

    p = subparsers.add_parser("extract-database-memories", help="Extract writer-agent procedural memories")
    p.add_argument("--trajectory-index", required=True)
    p.add_argument("--split-manifest", required=True)
    p.add_argument("--output", required=True)

    p = subparsers.add_parser("build-database-candidates", help="Build receiver-conditioned candidates")
    p.add_argument("--dataset-manifest", required=True)
    p.add_argument("--split-manifest", required=True)
    p.add_argument("--memory-pool", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--top-k", type=int, default=4)

    p = subparsers.add_parser("generate-database-paired-records", help="Generate candidate-level paired records")
    p.add_argument("--dataset-manifest", required=True)
    p.add_argument("--split-manifest", required=True)
    p.add_argument("--candidate-manifest", required=True)
    p.add_argument("--memory-pool", required=True)
    p.add_argument("--generation-seeds", type=int, nargs="+", default=[0])
    p.add_argument("--limit-pairs", type=int, default=None)
    p.add_argument("--output", required=True)

    p = subparsers.add_parser("train-critic", help="Train four-outcome transfer critic")
    p.add_argument("--train-records", required=True)
    p.add_argument("--validation-records", default=None)
    p.add_argument("--memory-pool", required=True)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--n-bootstrap", type=int, default=31)
    p.add_argument("--n-features", type=int, default=512)
    p.add_argument("--feature-block", default="full", choices=["full", "no_writer_receiver", "no_risk"])
    p.add_argument("--output", required=True)

    p = subparsers.add_parser("run-evaluation", help="Run evaluation on test split")
    p.add_argument("--marble-root", required=True)
    p.add_argument("--dataset-manifest", required=True)
    p.add_argument("--split-manifest", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--scenario", default="database")
    p.add_argument("--memory-pool", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--methods", nargs="+", default=["b0_no_memory", "top1_relevance", "all_share", "factual_success", "smtr", "smtr_no_risk", "smtr_no_writer_receiver"])
    p.add_argument("--negative-risk-budget", type=float, default=0.2)
    p.add_argument("--output", required=True)

    p = subparsers.add_parser("integrity-audit", help="Run integrity audit")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--output", required=True)

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
            task_count=args.task_count,
            generation_seeds=args.generation_seeds,
            output_dir=Path(args.output),
            resume=args.resume,
        )
        print(json.dumps(result, indent=2))

    elif cmd == "extract-database-memories":
        print("extract-database-memories: use real_data.extract_procedural_memories")

    elif cmd == "build-database-candidates":
        print("build-database-candidates: use real_data.build_cross_task_candidates")

    elif cmd == "generate-database-paired-records":
        from smtr.marble.real_pairs import generate_candidate_level_pairs
        result = generate_candidate_level_pairs(
            dataset_manifest_path=Path(args.dataset_manifest),
            split_manifest_path=Path(args.split_manifest),
            candidate_manifest_path=Path(args.candidate_manifest),
            memory_pool_path=Path(args.memory_pool),
            generation_seeds=args.generation_seeds,
            limit_pairs=args.limit_pairs,
            output_dir=Path(args.output),
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

    elif cmd == "run-evaluation":
        from smtr.marble.evaluation import run_evaluation
        result = run_evaluation(
            dataset_manifest=Path(args.dataset_manifest),
            split_manifest=Path(args.split_manifest),
            split=args.split,
            scenario=args.scenario,
            memory_pool=Path(args.memory_pool),
            checkpoint=Path(args.checkpoint),
            methods=args.methods,
            negative_risk_budget=args.negative_risk_budget,
            output=Path(args.output),
        )
        print(json.dumps(result, indent=2))

    elif cmd == "integrity-audit":
        from smtr.marble.integrity import run_integrity_audit
        result = run_integrity_audit(run_dir=Path(args.run_dir))
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))

    elif cmd == "dev-runtime-preflight":
        print("dev-runtime-preflight: not implemented in mainline")

    else:
        print(f"Unknown command: {cmd}")


if __name__ == "__main__":
    main()
