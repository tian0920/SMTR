"""CLI for MARBLE cross-agent shared memory pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from smtr.marble.engine_process import DEFAULT_ENGINE_TIMEOUT_SECONDS
from smtr.router.baselines import FORMAL_METHOD_NAMES


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
    p.add_argument("--cohort-quotas", default="",
                   help="JSON object of cohort quotas, e.g. "
                        "'{\"semantic_top\":2,\"receiver_compatible\":2,\"receiver_incompatible_hard_negative\":2,\"cross_receiver_anchor\":2}'")
    p.add_argument("--min-task-relevance", type=float, default=None)
    p.add_argument("--experiment-mode", choices=["pilot", "formal"], default="pilot")

    p = subparsers.add_parser("generate-database-paired-records", help="Generate candidate-level paired records")
    p.add_argument("--marble-root", required=True)
    p.add_argument("--dataset-manifest", required=True)
    p.add_argument("--split-manifest", required=True)
    p.add_argument("--split", required=True, choices=["train", "validation", "test"])
    p.add_argument("--candidate-manifest", required=True)
    p.add_argument("--memory-pool", required=True)
    p.add_argument("--generation-seeds", type=int, nargs="+", default=[0])
    p.add_argument("--limit-pairs", type=int, default=None)
    p.add_argument("--engine-timeout-seconds", type=int, default=1800)
    p.add_argument("--experiment-mode", choices=["pilot", "formal"], default="pilot")
    p.add_argument("--output", required=True)

    p = subparsers.add_parser("audit-splits", help="Audit train/validation/test split isolation (清单 P0-15)")
    p.add_argument("--train-paired-records", required=True)
    p.add_argument("--validation-paired-records", required=True)
    p.add_argument("--test-paired-records", required=True)
    p.add_argument("--memory-pool", required=True)
    # 清单 P0-2: per-role checkpoint binding; legacy --checkpoint removed.
    p.add_argument("--checkpoint-full", required=False, default=None)
    p.add_argument("--checkpoint-global-transfer", required=False, default=None)
    p.add_argument("--checkpoint-no-compatibility-interaction", required=False, default=None)
    # 清单 P0-1: bind the test candidate manifest into the audit artifact.
    p.add_argument("--test-candidate-manifest", required=False, default=None)
    p.add_argument("--methods", nargs="+", default=None)
    p.add_argument("--experiment-mode", choices=["pilot", "formal"], default="pilot")
    # 清单 R6 P1-5: bind the manifests into the audit artifact by digest.
    p.add_argument("--dataset-manifest", required=False, default=None)
    p.add_argument("--split-manifest", required=False, default=None)
    # 清单 §8: budget manifest binding for split audit v4.
    p.add_argument("--train-budget-candidate-manifest", required=False, default=None)
    p.add_argument("--output", required=True)

    p = subparsers.add_parser("train-critic", help="Train four-outcome transfer critic")
    p.add_argument("--train-records", required=True)
    p.add_argument("--validation-records", default=None)
    p.add_argument("--test-records", default=None)
    p.add_argument("--memory-pool", required=True)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--n-bootstrap", type=int, default=31)
    p.add_argument("--n-features", type=int, default=512)
    p.add_argument("--feature-block", default="full", choices=[
        "full", "global_transfer", "no_compatibility_interaction",
    ])
    p.add_argument("--coverage-mode", default="formal", choices=["formal", "pilot"])
    p.add_argument("--risk-delta", type=float, default=0.10)
    # 清单 Shared-Control 第16章: budget checkpoints bind the budgeted train
    # candidate manifest so provenance records requested/realized fractions.
    p.add_argument("--budget-candidate-manifest", default=None)
    # 清单 Fixed-Budget 第12章: mode B trains on pre-filtered records; the
    # edge set is still validated against the budget manifest.
    p.add_argument("--train-records-already-budgeted", action="store_true")
    p.add_argument("--experiment-mode", choices=["pilot", "formal"], default=None)
    p.add_argument("--output", required=True)

    p = subparsers.add_parser("run-paired-decision-evaluation", help="Paired decision evaluation on test pairs")
    p.add_argument("--candidate-manifest", required=True)
    p.add_argument("--paired-records", required=True)
    # 清单 P0-17: formal evaluations additionally require all three split
    # paired-record files so the split audit can run before evaluation.
    p.add_argument("--train-paired-records", default=None)
    p.add_argument("--validation-paired-records", default=None)
    p.add_argument("--test-paired-records", default=None)
    p.add_argument("--memory-pool", required=True)
    p.add_argument("--checkpoint-full", required=True)
    # 清单最终闭环 §24: unified checkpoint flag names across all commands.
    p.add_argument("--checkpoint-global-transfer", default=None)
    p.add_argument("--checkpoint-no-compatibility-interaction", default=None)
    # 清单最终闭环 §22: the formal method set comes from the single registry.
    p.add_argument("--methods", nargs="+", default=list(FORMAL_METHOD_NAMES))
    # Formal evaluations must read epsilon_star from the checkpoint; an
    # explicit budget is a debug-only override, never a silent 0.2 default.
    p.add_argument("--negative-risk-budget", type=float, default=None)
    p.add_argument("--allow-risk-budget-override", action="store_true")
    # 清单最终闭环 P0-3: the train budget manifest binds formal evaluations
    # to the same effective training support the critics saw.
    p.add_argument(
        "--train-budget-candidate-manifest",
        required=False,
        default=None,
        help=(
            "Frozen train budget candidate manifest. "
            "Required in formal mode, including B=1.0."
        ),
    )
    p.add_argument("--experiment-mode", choices=["pilot", "formal"], default=None)
    p.add_argument("--output", required=True)

    p = subparsers.add_parser("run-marble-evaluation", help="End-to-end MARBLE evaluation")
    p.add_argument("--marble-root", required=True)
    p.add_argument("--dataset-manifest", required=True)
    p.add_argument("--split-manifest", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--candidate-manifest", required=True)
    p.add_argument("--memory-pool", required=True)
    p.add_argument("--checkpoint-full", required=True)
    # 清单最终闭环 §24: unified checkpoint flag names across all commands.
    p.add_argument("--checkpoint-global-transfer", default=None)
    p.add_argument("--checkpoint-no-compatibility-interaction", default=None)
    # 清单最终闭环 §22: the formal method set comes from the single registry.
    p.add_argument("--methods", nargs="+", default=list(FORMAL_METHOD_NAMES))
    # 清单 R6 P1-3: no default seeds; users must supply them explicitly so
    # nobody mistakes a default for the formal seed protocol.
    p.add_argument("--generation-seeds", type=int, nargs="+", required=True)
    # Same rule as run-paired-decision-evaluation: no silent 0.2 fallback.
    p.add_argument("--negative-risk-budget", type=float, default=None)
    p.add_argument("--allow-risk-budget-override", action="store_true")
    # 清单最终闭环 P0-3: same train budget manifest binding as paired eval.
    p.add_argument(
        "--train-budget-candidate-manifest",
        required=False,
        default=None,
        help=(
            "Frozen train budget candidate manifest. "
            "Required in formal mode, including B=1.0."
        ),
    )
    p.add_argument("--experiment-mode", choices=["pilot", "formal"], default="pilot")
    # 清单 R6 P1-6: formal runs must bind a verified split-audit artifact.
    p.add_argument("--split-audit", required=False, default=None)
    p.add_argument("--output", required=True)

    p = subparsers.add_parser("integrity-audit", help="Run integrity audit")
    p.add_argument("--candidate-manifest", required=True)
    p.add_argument("--paired-records", required=True)
    p.add_argument("--memory-pool", required=True)
    p.add_argument("--paired-eval-dir", default=None)
    p.add_argument("--end-to-end-eval-dir", default=None)
    p.add_argument("--feature-audit", default=None)
    # 清单 P0-18: supply the three split files to run the real split audit.
    p.add_argument("--train-paired-records", default=None)
    p.add_argument("--validation-paired-records", default=None)
    p.add_argument("--test-paired-records", default=None)
    p.add_argument("--checkpoint-full", default=None)
    p.add_argument("--output", required=True)

    p = subparsers.add_parser(
        "build-budgeted-candidates",
        help=(
            "Build a fixed stratified train "
            "candidate subset for budget analysis"
        ),
    )
    p.add_argument("--candidate-manifest", required=True)
    p.add_argument(
        "--budget-fraction",
        type=float,
        required=True,
        choices=[0.25, 0.50, 0.75, 1.00],
    )
    p.add_argument("--output", required=True)

    # 清单 Fixed-Budget 第11章: materialize the whole-edge filtered training
    # records plus a provenance summary for independent inspection.
    p = subparsers.add_parser(
        "materialize-budgeted-records",
        help="Write budget-filtered train paired records and a provenance summary",
    )
    p.add_argument("--source-train-records", required=True)
    p.add_argument("--budget-candidate-manifest", required=True)
    p.add_argument(
        "--experiment-mode",
        choices=["pilot", "formal"],
        required=True,
    )
    p.add_argument("--output-records", required=True)
    p.add_argument("--output-summary", required=True)

    # --- Dev-only commands (prefixed with dev-) ---
    p = subparsers.add_parser("dev-runtime-preflight", help="[dev] Runtime preflight check")
    p.add_argument("--marble-root", required=True)
    p.add_argument("--output", required=True)

    args = parser.parse_args()
    # 清单 R6 P1-6: a formal end-to-end evaluation must bind a split-audit
    # artifact; the CLI rejects the invocation before any dispatch.
    if (
        args.command == "run-marble-evaluation"
        and args.experiment_mode == "formal"
        and not args.split_audit
    ):
        parser.error("--split-audit is required in formal mode")
    # 清单最终闭环 P0-3: formal evaluations fail closed before dispatch when
    # the train budget manifest is missing; pilots keep None -> full support.
    if (
        args.command
        in ("run-paired-decision-evaluation", "run-marble-evaluation")
        and args.experiment_mode == "formal"
        and not args.train_budget_candidate_manifest
    ):
        parser.error(
            "--train-budget-candidate-manifest is required in formal mode"
        )
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
            validate_receiver_effect_coverage,
            require_receiver_effect_coverage,
            CandidateCohortQuotas,
        )

        memories = load_memory_pool(Path(args.memory_pool))

        recipients = load_receiver_entries(
            dataset_manifest_path=Path(args.dataset_manifest),
            split_manifest_path=Path(args.split_manifest),
            split=args.split,
        )

        cohort_quotas = None
        if getattr(args, "cohort_quotas", ""):
            cohort_quotas = CandidateCohortQuotas(**json.loads(args.cohort_quotas))

        manifest = build_cross_task_candidates(
            memories=memories,
            recipients=recipients,
            top_k=args.top_k,
            target_split=args.split,
            cohort_quotas=cohort_quotas,
            min_task_relevance=getattr(args, "min_task_relevance", None),
            experiment_mode=args.experiment_mode,
        )

        result = write_candidate_manifest(
            manifest=manifest,
            output_path=Path(args.output),
        )

        coverage = validate_receiver_effect_coverage(manifest)
        coverage_path = Path(args.output).with_suffix(".coverage.json")
        coverage_path.write_text(json.dumps(coverage, indent=2) + "\n", encoding="utf-8")
        result["receiver_effect_coverage"] = coverage["statistics"]
        result["coverage_checks"] = coverage["checks"]
        result["coverage_ok"] = coverage["ok"]
        result["coverage_audit"] = str(coverage_path)
        if args.experiment_mode == "formal":
            # Formal data generation fails fast instead of only warning.
            require_receiver_effect_coverage(coverage)

        print(json.dumps(result, indent=2))

    elif cmd == "generate-database-paired-records":
        # 清单 Formal Protocol §1: exact seed protocol enforcement at
        # generation; arbitrary seeds or wrong counts fail closed.
        from smtr.evaluation.experiment_protocol import (
            validate_generation_seed_protocol,
        )

        validate_generation_seed_protocol(
            generation_seeds=args.generation_seeds,
            experiment_mode=args.experiment_mode,
        )
        # 清单 Fixed-Budget 第12章: --limit-pairs truncates the treatment-edge
        # population, so formal budget experiments must never use it.
        if args.experiment_mode == "formal" and args.limit_pairs:
            raise ValueError(
                "--limit-pairs is debug-only and cannot be used for "
                "formal budget experiments"
            )
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
            engine_timeout_seconds=args.engine_timeout_seconds,
            experiment_mode=args.experiment_mode,
        )
        print(json.dumps(result, indent=2))

    elif cmd == "audit-splits":
        from smtr.evaluation.split_audit import audit_split_files
        # 清单 P0-2: formal audits must bind the full checkpoint at minimum;
        # pilot audits keep the legacy optional single-checkpoint behaviour.
        checkpoint_paths: dict[str, str] = {}
        if args.checkpoint_full:
            checkpoint_paths["full"] = args.checkpoint_full
        if args.checkpoint_global_transfer:
            checkpoint_paths["global_transfer"] = args.checkpoint_global_transfer
        if args.checkpoint_no_compatibility_interaction:
            checkpoint_paths["no_compatibility_interaction"] = (
                args.checkpoint_no_compatibility_interaction
            )
        if args.experiment_mode == "formal" and "full" not in checkpoint_paths:
            raise SystemExit(
                "audit-splits --checkpoint-full is required in formal mode"
            )
        summary = audit_split_files(
            train_records_path=Path(args.train_paired_records),
            validation_records_path=Path(args.validation_paired_records),
            test_records_path=Path(args.test_paired_records),
            memory_pool_path=Path(args.memory_pool),
            test_candidate_manifest_path=(
                Path(args.test_candidate_manifest)
                if args.test_candidate_manifest
                else None
            ),
            checkpoint_paths=(
                {role: Path(path) for role, path in checkpoint_paths.items()}
                or None
            ),
            methods=args.methods,
            dataset_manifest_path=Path(args.dataset_manifest) if args.dataset_manifest else None,
            split_manifest_path=Path(args.split_manifest) if args.split_manifest else None,
            train_budget_candidate_manifest_path=(
                Path(args.train_budget_candidate_manifest)
                if args.train_budget_candidate_manifest
                else None
            ),
            strict_candidate_support=True,
            experiment_mode=args.experiment_mode,
        )
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps(summary, indent=2, sort_keys=True))
        if not summary["split_integrity_passed"]:
            raise SystemExit(2)

    elif cmd == "train-critic":
        from smtr.marble.training import train_critic
        result = train_critic(
            train_records_path=Path(args.train_records),
            validation_records_path=Path(args.validation_records) if args.validation_records else None,
            test_records_path=Path(args.test_records) if args.test_records else None,
            memory_pool_path=Path(args.memory_pool),
            output_path=Path(args.output),
            seed=args.seed,
            n_bootstrap=args.n_bootstrap,
            n_features=args.n_features,
            feature_block=args.feature_block,
            coverage_mode=args.coverage_mode,
            risk_delta=args.risk_delta,
            budget_candidate_manifest_path=(
                Path(args.budget_candidate_manifest)
                if getattr(args, "budget_candidate_manifest", None)
                else None
            ),
            train_records_already_budgeted=getattr(
                args, "train_records_already_budgeted", False
            ),
            experiment_mode=getattr(args, "experiment_mode", None),
        )
        print(json.dumps(result, indent=2))

    elif cmd == "run-paired-decision-evaluation":
        from smtr.marble.paired_evaluation import run_paired_decision_evaluation
        result = run_paired_decision_evaluation(
            candidate_manifest_path=Path(args.candidate_manifest),
            paired_records_path=Path(args.paired_records),
            train_paired_records_path=Path(args.train_paired_records) if args.train_paired_records else None,
            validation_paired_records_path=Path(args.validation_paired_records) if args.validation_paired_records else None,
            test_paired_records_path=Path(args.test_paired_records) if args.test_paired_records else None,
            memory_pool_path=Path(args.memory_pool),
            checkpoint_full=Path(args.checkpoint_full),
            checkpoint_global_transfer_critic=Path(args.checkpoint_global_transfer) if args.checkpoint_global_transfer else None,
            checkpoint_smtr_no_compatibility_interaction=(
                Path(args.checkpoint_no_compatibility_interaction)
                if args.checkpoint_no_compatibility_interaction else None
            ),
            methods=args.methods,
            train_budget_candidate_manifest_path=(
                Path(args.train_budget_candidate_manifest)
                if args.train_budget_candidate_manifest
                else None
            ),
            negative_risk_budget=args.negative_risk_budget,
            allow_risk_budget_override=args.allow_risk_budget_override,
            experiment_mode=args.experiment_mode,
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
            checkpoint_global_transfer_critic=Path(args.checkpoint_global_transfer) if args.checkpoint_global_transfer else None,
            checkpoint_smtr_no_compatibility_interaction=(
                Path(args.checkpoint_no_compatibility_interaction)
                if args.checkpoint_no_compatibility_interaction else None
            ),
            methods=args.methods,
            generation_seeds=args.generation_seeds,
            negative_risk_budget=args.negative_risk_budget,
            allow_risk_budget_override=args.allow_risk_budget_override,
            experiment_mode=args.experiment_mode,
            split_audit_path=Path(args.split_audit) if args.split_audit else None,
            train_budget_candidate_manifest_path=(
                Path(args.train_budget_candidate_manifest)
                if args.train_budget_candidate_manifest
                else None
            ),
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
            train_paired_records_path=Path(args.train_paired_records) if args.train_paired_records else None,
            validation_paired_records_path=Path(args.validation_paired_records) if args.validation_paired_records else None,
            test_paired_records_path=Path(args.test_paired_records) if args.test_paired_records else None,
            checkpoint_path=Path(args.checkpoint_full) if args.checkpoint_full else None,
        )
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))

    elif cmd == "build-budgeted-candidates":
        from smtr.marble.budget_sampling import (
            build_budgeted_candidate_manifest,
        )
        from smtr.marble.real_data import (
            DatabaseCandidateManifest,
            write_candidate_manifest,
        )

        parent = DatabaseCandidateManifest.model_validate_json(
            Path(args.candidate_manifest).read_text(encoding="utf-8")
        )
        manifest = build_budgeted_candidate_manifest(
            parent_manifest=parent,
            budget_fraction=args.budget_fraction,
        )
        result = write_candidate_manifest(
            manifest=manifest,
            output_path=Path(args.output),
        )
        if manifest.budget_metadata is not None:
            result["requested_fraction"] = (
                manifest.budget_metadata.requested_fraction
            )
            result["realized_edge_fraction"] = (
                manifest.budget_metadata.realized_edge_fraction
            )
            result["selected_edge_count"] = (
                manifest.budget_metadata.selected_edge_count
            )
        print(json.dumps(result, indent=2))

    elif cmd == "materialize-budgeted-records":
        from smtr.marble.budget_sampling import (
            write_budgeted_paired_records,
        )
        summary = write_budgeted_paired_records(
            source_records_path=Path(args.source_train_records),
            budget_manifest_path=Path(args.budget_candidate_manifest),
            output_records_path=Path(args.output_records),
            output_summary_path=Path(args.output_summary),
            experiment_mode=args.experiment_mode,
        )
        print(json.dumps(summary, indent=2))

    elif cmd == "dev-runtime-preflight":
        print("dev-runtime-preflight: not implemented in mainline")

    else:
        print(f"Unknown command: {cmd}")


if __name__ == "__main__":
    main()
