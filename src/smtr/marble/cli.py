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
    p.add_argument("--parallel", type=int, default=1,
                   help="Parallelism degree: number of concurrent Docker slots (default 1 = sequential)")
    p.add_argument("--api-keys", nargs="+", default=None,
                   help="One or more LLM API keys for round-robin assignment across slots")
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

    p = subparsers.add_parser("train-critic", help="Train flat or opportunity-factorized transfer critic")
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
    # Counterfactual Opportunity v1: critic mode selection.
    p.add_argument("--critic-mode", default="flat", choices=[
        "flat", "opportunity_factorized",
    ])
    # 清单 Shared-Control 第16章: budget checkpoints bind the budgeted train
    # candidate manifest so provenance records requested/realized fractions.
    p.add_argument("--budget-candidate-manifest", default=None)
    # 清单 Fixed-Budget 第12章: mode B trains on pre-filtered records; the
    # edge set is still validated against the budget manifest.
    p.add_argument("--train-records-already-budgeted", action="store_true")
    p.add_argument("--experiment-mode", choices=["pilot", "formal"], default=None)
    # P1-A: MARBLE source path for task instruction injection.
    p.add_argument("--marble-source", default=None,
                   help="Path to MARBLE database_main.jsonl for task instruction back-fill")
    # TCI Distillation (Task 6): optional intervention-contrast supervision.
    # When provided, TCI pairs are appended as soft-labeled examples with
    # total weight tci-alpha (observational+tci training mode).
    p.add_argument("--tci-contrasts", default=None,
                   help="Path to intervention_contrasts.jsonl for TCI distillation")
    p.add_argument("--tci-perturbations-manifest", default=None,
                   help="Path to perturbations.json for TCI distillation")
    p.add_argument("--tci-paired-records", default=None,
                   help="Path to paired_records.jsonl used for TCI receiver context")
    p.add_argument("--tci-alpha", type=float, default=1.0,
                   help="Total weight of TCI supervision block (default 1.0)")
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
    # P1-A: MARBLE source path for task instruction injection in evaluation.
    p.add_argument("--marble-source", default=None,
                   help="Path to MARBLE database_main.jsonl for task instruction back-fill")
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

    # --- P2: Transfer-critical counterfactual intervention ---
    p = subparsers.add_parser(
        "build-transfer-perturbations",
        help="Build perturbation specs for transfer-critical edges (P2-A)",
    )
    p.add_argument("--paired-records", required=True)
    p.add_argument("--memory-pool", required=True)
    p.add_argument("--candidate-manifest", default=None)
    p.add_argument("--perturbation-budget", type=int, default=100)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--output", required=True)

    p = subparsers.add_parser(
        "run-transfer-perturbations",
        help="Execute perturbed branches in MARBLE (P2-A)",
    )
    p.add_argument("--marble-root", default=None)
    p.add_argument("--perturbation-manifest", required=True)
    p.add_argument("--perturbations", required=True)
    p.add_argument("--paired-records", required=True)
    p.add_argument("--memory-pool", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--resume", action="store_true", default=False)
    p.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Skip MARBLE execution (structural validation only)",
    )

    p = subparsers.add_parser(
        "analyze-transfer-perturbations",
        help="Compute P2-B intervention metrics from outcomes",
    )
    p.add_argument("--outcomes", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--original-damage-positives", type=int, default=0)
    p.add_argument(
        "--allow-dry-run",
        action="store_true",
        default=False,
        help="Allow dry-run outcomes (results are not causal evidence)",
    )

    # --- P2 Intervention Contrast Layer ---
    p = subparsers.add_parser(
        "build-intervention-contrasts",
        help="Build pairwise intervention contrasts from outcomes (P2-B)",
    )
    p.add_argument("--outcomes", required=True)
    p.add_argument("--output", required=True)

    p = subparsers.add_parser(
        "train-tci-ranker",
        help="Train TCI pairwise ranker offline (P2-C)",
    )
    p.add_argument("--contrasts", required=True,
                   help="Path to intervention_contrasts.jsonl")
    p.add_argument("--output", required=True,
                   help="Path to save TCI ranker checkpoint")
    p.add_argument("--feature-dim", type=int, default=512)
    p.add_argument("--learning-rate", type=float, default=0.01)
    p.add_argument("--n-epochs", type=int, default=50)
    p.add_argument("--seed", type=int, default=7)

    p = subparsers.add_parser(
        "evaluate-tci-ranker",
        help="Evaluate TCI ranker on offline contrasts (P2-C)",
    )
    p.add_argument("--contrasts", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)

    p = subparsers.add_parser(
        "evaluate-tci-baselines",
        help="Evaluate TCI ranker vs random pair baseline (P2-C)",
    )
    p.add_argument("--contrasts", required=True,
                   help="Path to intervention_contrasts.jsonl")
    p.add_argument("--checkpoint", required=True,
                   help="Path to trained TCI ranker checkpoint")
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=42)

    p = subparsers.add_parser(
        "evaluate-tci-generalization",
        help="Full TCI held-out generalization evaluation (P2-C)",
    )
    p.add_argument("--contrasts", required=True,
                   help="Path to intervention_contrasts.jsonl")
    p.add_argument("--checkpoint", default=None,
                   help="Path to trained TCI ranker checkpoint "
                        "(if None, trains from scratch on train split)")
    p.add_argument("--output", required=True)
    p.add_argument("--feature-dim", type=int, default=128)
    p.add_argument("--learning-rate", type=float, default=0.01)
    p.add_argument("--n-epochs", type=int, default=100)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--perturbations-manifest", default=None,
                   help="Path to perturbations.json with perturbed_card data")
    p.add_argument("--memory-pool", default=None,
                   help="Path to memory_pool.jsonl with original cards")
    p.add_argument("--paired-records", default=None,
                   help="Path to paired_records.jsonl with receiver/task context")

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
        from smtr.marble.dataset import write_marble_dataset_manifest
        manifest = write_marble_dataset_manifest(
            output_path=Path(args.output),
            marble_root=Path(args.marble_root),
        )
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
            parallel=args.parallel,
            api_keys=args.api_keys,
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
            marble_source_path=(
                Path(args.marble_source)
                if getattr(args, "marble_source", None)
                else None
            ),
            critic_mode=getattr(args, "critic_mode", "flat"),
            tci_contrasts_path=(
                Path(args.tci_contrasts)
                if getattr(args, "tci_contrasts", None)
                else None
            ),
            tci_perturbations_manifest_path=(
                Path(args.tci_perturbations_manifest)
                if getattr(args, "tci_perturbations_manifest", None)
                else None
            ),
            tci_paired_records_path=(
                Path(args.tci_paired_records)
                if getattr(args, "tci_paired_records", None)
                else None
            ),
            tci_alpha=getattr(args, "tci_alpha", 1.0),
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
            experiment_mode=args.experiment_mode,
            output=Path(args.output),
            marble_source_path=(
                Path(args.marble_source)
                if getattr(args, "marble_source", None)
                else None
            ),
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
            checkpoint_paths={"full": Path(args.checkpoint_full)} if args.checkpoint_full else None,
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

    elif cmd == "build-transfer-perturbations":
        import hashlib

        from smtr.intervention.perturbation_runner import (
            load_paired_records,
            load_memory_pool,
        )
        from smtr.intervention.perturbation_selector import (
            select_perturbation_edges,
        )

        records = load_paired_records(args.paired_records)
        pool = load_memory_pool(args.memory_pool)
        selections = select_perturbation_edges(
            paired_records=records,
            memory_pool=pool,
            perturbation_budget=args.perturbation_budget,
            seed=args.seed,
        )

        # Compute provenance digests.
        def _file_sha256(path: str) -> str:
            h = hashlib.sha256()
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    h.update(chunk)
            return h.hexdigest()

        source_digest = _file_sha256(args.paired_records)
        pool_digest = _file_sha256(args.memory_pool)
        cand_digest = (
            _file_sha256(args.candidate_manifest)
            if args.candidate_manifest
            else None
        )

        # Compute operator counts from selections.
        operator_counts: dict[str, int] = {}
        for s in selections:
            op = s.spec.perturbation_type
            operator_counts[op] = operator_counts.get(op, 0) + 1

        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        manifest = {
            "n_perturbations": len(selections),
            "seed": args.seed,
            "budget": args.perturbation_budget,
            "operator_counts": operator_counts,
            "provenance": {
                "source_records_digest": source_digest,
                "memory_pool_digest": pool_digest,
                "candidate_manifest_digest": cand_digest,
                "selection_seed": args.seed,
            },
            "perturbations": [
                {
                    "spec": s.spec.to_dict(),
                    "edge_id": s.edge_id,
                    "perturbed_card": s.perturbed_card.model_dump(),
                }
                for s in selections
            ],
        }
        out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(json.dumps(
            {
                "n_perturbations": len(selections),
                "operator_counts": operator_counts,
            },
            indent=2,
        ))

    elif cmd == "run-transfer-perturbations":
        from smtr.intervention.perturbation_runner import (
            load_paired_records,
            load_memory_pool,
            find_original_paired_record,
            run_perturbed_exposure_branch,
        )
        from smtr.intervention.perturbation_schema import (
            PerturbationSpec,
            PerturbationOutcomeRecord,
        )
        from smtr.core.types import MemoryRoutingCard

        records = load_paired_records(args.paired_records)
        pool_raw = load_memory_pool(args.memory_pool)

        # Load perturbation manifest.
        manifest_data = json.loads(
            Path(args.perturbation_manifest).read_text(encoding="utf-8")
        )
        pert_list = manifest_data.get("perturbations", [])

        # Optional resume.
        existing_outcomes: list[dict] = []
        done_ids: set[str] = set()
        out_path = Path(args.output)
        if args.resume and out_path.exists():
            for line in out_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    existing_outcomes.append(rec)
                    done_ids.add(rec.get("spec", {}).get("perturbation_id", ""))

        with open(out_path, "a" if args.resume else "w", encoding="utf-8") as fout:
            for entry in pert_list:
                spec = PerturbationSpec.from_dict(entry["spec"])
                if spec.perturbation_id in done_ids:
                    continue
                perturbed_card = MemoryRoutingCard.model_validate(
                    entry["perturbed_card"]
                )
                # Find original record and card.
                orig_rec = find_original_paired_record(
                    records, spec.task_id,
                    spec.receiver_agent_id,
                    spec.candidate_memory_id,
                    spec.generation_seed,
                )
                if orig_rec is None:
                    continue
                orig_card_data = pool_raw.get(
                    spec.candidate_memory_id, {}
                ).get("routing_card", {})
                orig_card_data["memory_id"] = spec.candidate_memory_id
                original_card = MemoryRoutingCard.model_validate(orig_card_data)

                outcome = run_perturbed_exposure_branch(
                    original_paired_record=orig_rec,
                    perturbation_spec=spec,
                    perturbed_memory_card=perturbed_card,
                    original_memory_card=original_card,
                    dry_run=args.dry_run,
                )
                fout.write(json.dumps(outcome.to_dict()) + "\n")

        execution_mode = "dry_run" if args.dry_run else "real"
        print(json.dumps(
            {
                "status": "complete",
                "execution_mode": execution_mode,
                "dry_run": args.dry_run,
            },
            indent=2,
        ))

    elif cmd == "analyze-transfer-perturbations":
        from smtr.intervention.perturbation_schema import (
            PerturbationOutcomeRecord,
        )
        from smtr.intervention.perturbation_analysis import (
            compute_perturbation_metrics,
            compute_operator_level_metrics,
            compute_baseline_conditioned_flips,
            compute_support_gain,
            compute_triple_counts,
            compute_operator_distribution,
            compute_pilot_gate,
            validate_real_execution_records,
            format_results_table,
            audit_perturbation_leakage,
        )

        outcomes: list[PerturbationOutcomeRecord] = []
        for line in Path(args.outcomes).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                outcomes.append(
                    PerturbationOutcomeRecord.from_dict(json.loads(line))
                )

        # Validate execution mode.
        allow_dry_run = getattr(args, "allow_dry_run", False)
        is_dry_run = any(
            r.runtime_metadata.get("dry_run") is True for r in outcomes
        )
        analysis_valid = True
        execution_mode = "real"

        if is_dry_run and not allow_dry_run:
            validate_real_execution_records(outcomes)

        if is_dry_run:
            analysis_valid = False
            execution_mode = "dry_run"

        overall = compute_perturbation_metrics(outcomes)
        by_op = compute_operator_level_metrics(outcomes)
        baseline = compute_baseline_conditioned_flips(outcomes)
        support = compute_support_gain(
            original_damage_positives=args.original_damage_positives,
            outcomes=outcomes,
        )
        triples = compute_triple_counts(outcomes)
        op_dist = compute_operator_distribution(outcomes)
        gate = compute_pilot_gate(triples, op_dist)
        table = format_results_table(overall, by_op, support)

        result = {
            "analysis_valid": analysis_valid,
            "execution_mode": execution_mode,
            "overall": overall.to_dict(),
            "by_operator": {k: v.to_dict() for k, v in by_op.items()},
            "baseline_conditioned": baseline.to_dict(),
            "support_gain": support.to_dict(),
            "triple_analysis": triples.to_dict(),
            "operator_distribution": op_dist,
            "pilot_gate": gate.to_dict(),
            "results_table": table,
        }
        if not analysis_valid:
            result["reason"] = "dry_run_only"

        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(table)
        print(f"\nGate: {gate.gate} — {', '.join(gate.reasons)}")

    elif cmd == "build-intervention-contrasts":
        from smtr.intervention.perturbation_schema import (
            PerturbationOutcomeRecord,
            INTERVENTION_CONTRAST_SCHEMA_VERSION,
        )
        from smtr.intervention.contrast_builder import (
            build_intervention_contrasts,
        )
        from smtr.intervention.contrast_types import classify_contrast
        from smtr.intervention.perturbation_analysis import (
            compute_contrast_summary,
            compute_operator_contrast,
        )

        outcomes: list[PerturbationOutcomeRecord] = []
        for line in Path(args.outcomes).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                outcomes.append(
                    PerturbationOutcomeRecord.from_dict(json.loads(line))
                )

        contrasts = build_intervention_contrasts(outcomes)
        summary = compute_contrast_summary(contrasts)
        op_contrast = compute_operator_contrast(contrasts)

        # Write contrasts as JSONL.
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as fout:
            for c in contrasts:
                rec = c.to_dict()
                rec["schema_version"] = INTERVENTION_CONTRAST_SCHEMA_VERSION
                fout.write(json.dumps(rec) + "\n")

        result = {
            "n_outcomes": len(outcomes),
            "n_contrasts": len(contrasts),
            "contrast_summary": summary.to_dict(),
            "operator_contrast": {
                k: v.to_dict() for k, v in op_contrast.items()
            },
        }
        print(json.dumps(result, indent=2))

    elif cmd == "train-tci-ranker":
        import numpy as np
        from smtr.intervention.intervention_contrast import (
            InterventionContrast,
        )
        from smtr.router.tci_ranker import TCIRanker, TCIRankerConfig
        from smtr.router.tci_metrics import evaluate_tci_ranker

        contrasts: list[InterventionContrast] = []
        for line in Path(args.contrasts).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                contrasts.append(
                    InterventionContrast.from_dict(json.loads(line))
                )

        # Filter direction != 0.
        valid = [c for c in contrasts if c.contrast_direction != 0]
        if not valid:
            print(json.dumps({"status": "no_valid_pairs", "n_contrasts": 0}))
            return

        # Build feature matrices (placeholder: random features for now).
        # In production, encode via HashingTransferFeatureEncoder.
        rng = np.random.RandomState(args.seed)
        n = len(valid)
        feat_dim = args.feature_dim
        feat_orig = rng.randn(n, feat_dim)
        feat_pert = rng.randn(n, feat_dim)
        dirs = np.array([c.contrast_direction for c in valid], dtype=float)

        config = TCIRankerConfig(
            feature_dim=feat_dim,
            learning_rate=args.learning_rate,
            n_epochs=args.n_epochs,
            seed=args.seed,
        )
        ranker = TCIRanker(config)
        train_result = ranker.train(feat_orig, feat_pert, dirs)

        # Evaluate on training set.
        s_orig = ranker.score(feat_orig)
        s_pert = ranker.score(feat_pert)
        ops = [c.perturbation_type for c in valid]
        metrics = evaluate_tci_ranker(s_orig, s_pert, dirs, ops)

        # Save checkpoint.
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        ranker.to_checkpoint().save(out)

        result = {
            "status": "complete",
            "n_pairs": n,
            "training": {"final_loss": train_result["final_loss"]},
            "evaluation": metrics.to_dict(),
        }
        print(json.dumps(result, indent=2))

    elif cmd == "evaluate-tci-ranker":
        import numpy as np
        from smtr.intervention.intervention_contrast import (
            InterventionContrast,
        )
        from smtr.router.tci_ranker import TCIRankerCheckpoint, TCIRanker
        from smtr.router.tci_metrics import (
            evaluate_tci_ranker,
            compute_regret,
        )

        contrasts: list[InterventionContrast] = []
        for line in Path(args.contrasts).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                contrasts.append(
                    InterventionContrast.from_dict(json.loads(line))
                )

        valid = [c for c in contrasts if c.contrast_direction != 0]
        ckpt = TCIRankerCheckpoint.load(args.checkpoint)
        ranker = TCIRanker.from_checkpoint(ckpt)

        rng = np.random.RandomState(7)
        n = len(valid)
        feat_orig = rng.randn(n, ckpt.feature_dim)
        feat_pert = rng.randn(n, ckpt.feature_dim)
        dirs = np.array([c.contrast_direction for c in valid], dtype=float)

        s_orig = ranker.score(feat_orig)
        s_pert = ranker.score(feat_pert)
        ops = [c.perturbation_type for c in valid]
        metrics = evaluate_tci_ranker(s_orig, s_pert, dirs, ops)
        regret = compute_regret(s_orig, s_pert, dirs)

        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        result = {
            "evaluation": metrics.to_dict(),
            "regret": regret,
        }
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))

    elif cmd == "evaluate-tci-baselines":
        import numpy as np
        from smtr.intervention.intervention_contrast import (
            InterventionContrast,
        )
        from smtr.intervention.contrast_builder import (
            build_intervention_contrasts,
        )
        from smtr.router.tci_dataset import build_tci_pairs
        from smtr.router.tci_ranker import TCIRankerCheckpoint, TCIRanker
        from smtr.router.tci_baselines import evaluate_random_baseline
        from smtr.router.tci_metrics import evaluate_tci_ranker, _score_pairs
        from smtr.router.tci_metrics import compute_regret

        contrasts: list[InterventionContrast] = []
        for line in Path(args.contrasts).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                contrasts.append(
                    InterventionContrast.from_dict(json.loads(line))
                )
        pairs = build_tci_pairs(contrasts)

        ckpt = TCIRankerCheckpoint.load(args.checkpoint)
        ranker = TCIRanker.from_checkpoint(ckpt)

        # TCI accuracy on real pairs.
        s_orig, s_pert, dirs = _score_pairs(ranker, pairs, None)
        ops = [p.perturbation_type for p in pairs]
        tci_metrics = evaluate_tci_ranker(s_orig, s_pert, dirs, ops)
        tci_regret = compute_regret(s_orig, s_pert, dirs)

        # Random baseline.
        random_bl = evaluate_random_baseline(
            ranker, pairs, seed=args.seed
        )

        result = {
            "TCI": {
                "n_pairs": tci_metrics.n_pairs,
                "accuracy": tci_metrics.pairwise_accuracy,
                "margin": tci_metrics.pairwise_margin,
                "regret": tci_regret,
            },
            "RandomPair": random_bl.to_dict(),
        }
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))

    elif cmd == "evaluate-tci-generalization":
        import numpy as np
        from smtr.intervention.intervention_contrast import (
            InterventionContrast,
        )
        from smtr.router.tci_dataset import (
            build_tci_pairs,
            build_tci_pairs_with_features,
        )
        from smtr.router.tci_split import split_tci_pairs
        from smtr.router.tci_ranker import (
            TCIRanker, TCIRankerCheckpoint, TCIRankerConfig,
        )
        from smtr.router.tci_metrics import (
            evaluate_split_metrics,
            evaluate_by_factor,
            compute_margin_accuracy_curve,
            compute_feature_shift,
            _score_pairs,
        )
        from smtr.router.tci_baselines import evaluate_random_baseline
        from smtr.router.tci_features import TCIFeatureEncoder

        # 1. Load contrasts.
        contrasts: list[InterventionContrast] = []
        for line in Path(args.contrasts).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                contrasts.append(
                    InterventionContrast.from_dict(json.loads(line))
                )

        # 2. Load context data (optional).
        use_structural = False
        original_cards: dict = {}
        perturbed_cards: dict = {}
        receiver_contexts: dict = {}
        task_contexts: dict = {}

        if args.memory_pool:
            use_structural = True
            for line in Path(args.memory_pool).read_text(
                encoding="utf-8"
            ).splitlines():
                line = line.strip()
                if line:
                    mem = json.loads(line)
                    mid = mem.get("memory_id", "")
                    original_cards[mid] = mem.get("routing_card", {})

        if args.perturbations_manifest:
            use_structural = True
            pman = json.loads(
                Path(args.perturbations_manifest).read_text(encoding="utf-8")
            )
            for pert in pman.get("perturbations", []):
                pid = pert.get("spec", {}).get("perturbation_id", "")
                perturbed_cards[pid] = pert.get("perturbed_card", {})

        if args.paired_records:
            use_structural = True
            for line in Path(args.paired_records).read_text(
                encoding="utf-8"
            ).splitlines():
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                rid = rec.get("receiver_agent_id", "")
                tid = rec.get("task_id", "")
                if rid and rid not in receiver_contexts:
                    receiver_contexts[rid] = {
                        "receiver_role": rec.get("receiver_role", ""),
                        "receiver_capabilities": rec.get(
                            "receiver_capabilities", []
                        ),
                        "receiver_tool_names": rec.get(
                            "receiver_tool_names", []
                        ),
                        "environment_signature": rec.get(
                            "environment_signature", []
                        ),
                    }
                if tid and tid not in task_contexts:
                    task_contexts[str(tid)] = {
                        "task_instruction": rec.get("task_instruction", ""),
                        "scenario": rec.get("scenario", ""),
                        "task_tags": rec.get("target_task_group", []),
                    }

        # 3. Build pairs (with or without structural features).
        if use_structural:
            feature_encoder = TCIFeatureEncoder(
                feature_dim=args.feature_dim, seed=args.seed,
            )
            pairs = build_tci_pairs_with_features(
                contrasts,
                feature_encoder=feature_encoder,
                original_cards=original_cards,
                perturbed_cards=perturbed_cards,
                receiver_contexts=receiver_contexts,
                task_contexts=task_contexts,
            )
            feature_mode = "structural"
        else:
            pairs = build_tci_pairs(contrasts)
            feature_mode = "hash"

        # 4. Split.
        tci_split = split_tci_pairs(pairs, seed=args.split_seed)

        # 5. Build or load ranker.
        if args.checkpoint:
            ckpt = TCIRankerCheckpoint.load(args.checkpoint)
            ranker = TCIRanker.from_checkpoint(ckpt)
        else:
            config = TCIRankerConfig(
                feature_dim=args.feature_dim,
                learning_rate=args.learning_rate,
                n_epochs=args.n_epochs,
                seed=args.seed,
            )
            ranker = TCIRanker(config)
            if tci_split.n_train > 0:
                # Extract features via _score_pairs (uses structural
                # if available, else hash fallback).
                fd = args.feature_dim

                def _extract_features(pair_list):
                    s_o, s_p, d = _score_pairs(ranker, pair_list, None)
                    n = len(pair_list)
                    fo = np.zeros((n, fd))
                    fp = np.zeros((n, fd))
                    use_sf = (
                        pair_list
                        and hasattr(pair_list[0], "has_structural_features")
                        and pair_list[0].has_structural_features
                    )
                    if use_sf:
                        for i, p in enumerate(pair_list):
                            of = list(p.original_features)
                            pf = list(p.perturbed_features)
                            fo[i] = (of + [0.0] * fd)[:fd]
                            fp[i] = (pf + [0.0] * fd)[:fd]
                    else:
                        import hashlib
                        for i, p in enumerate(pair_list):
                            h_o = hashlib.sha256(
                                f"orig:{p.perturbation_id}".encode()
                            ).digest()
                            h_p = hashlib.sha256(
                                f"pert:{p.perturbation_id}".encode()
                            ).digest()
                            # SHA-256 = 32 bytes; tile to fill fd.
                            raw_o = (
                                np.frombuffer(h_o, dtype=np.uint8)
                                .astype(float)
                            )
                            raw_p = (
                                np.frombuffer(h_p, dtype=np.uint8)
                                .astype(float)
                            )
                            reps_o = (fd // len(raw_o)) + 1
                            reps_p = (fd // len(raw_p)) + 1
                            fo[i] = np.tile(raw_o, reps_o)[:fd]
                            fp[i] = np.tile(raw_p, reps_p)[:fd]
                        fo = fo / 255.0
                        fp = fp / 255.0
                    dirs = np.array(
                        [p.direction for p in pair_list], dtype=float
                    )
                    return fo, fp, dirs

                tr_fo, tr_fp, tr_dirs = _extract_features(
                    tci_split.train_pairs
                )
                if tci_split.n_valid > 0:
                    va_fo, va_fp, va_dirs = _extract_features(
                        tci_split.valid_pairs
                    )
                else:
                    va_fo = np.zeros((1, fd))
                    va_fp = np.zeros((1, fd))
                    va_dirs = np.array([0.0])

                ranker.train_with_validation(
                    tr_fo, tr_fp, tr_dirs,
                    va_fo, va_fp, va_dirs,
                    epochs=args.n_epochs,
                )

        # 6. Evaluate splits.
        split_metrics = evaluate_split_metrics(
            ranker,
            tci_split.train_pairs,
            tci_split.valid_pairs,
            tci_split.test_pairs,
        )

        # 7. Random baseline on test pairs.
        random_bl = evaluate_random_baseline(
            ranker, tci_split.test_pairs, seed=args.seed
        )

        # 8. Factor breakdown on test pairs.
        factor_analysis = evaluate_by_factor(
            tci_split.test_pairs, ranker
        )

        # 9. Margin calibration on test pairs.
        margin_curve = {}
        if tci_split.test_pairs:
            s_o, s_p, d = _score_pairs(
                ranker, tci_split.test_pairs, None
            )
            margin_curve = compute_margin_accuracy_curve(s_o, s_p, d)

        # 10. Feature shift analysis.
        feature_shift = compute_feature_shift(tci_split.test_pairs)

        # 11. Gate judgment.
        test_acc = split_metrics.get("test", {}).get("accuracy", 0.0)
        test_margin = split_metrics.get("test", {}).get("margin", 0.0)
        rand_acc = random_bl.pairwise_accuracy
        advantage = test_acc - rand_acc

        if test_acc >= 0.70 and test_margin > 0 and advantage >= 0.10:
            gate = "PASS"
        elif 0.60 <= test_acc < 0.70:
            gate = "BORDERLINE"
        else:
            gate = "FAIL"

        result = {
            "feature_mode": feature_mode,
            "split": tci_split.to_dict(),
            "splits": split_metrics,
            "TCI": {
                "test_accuracy": test_acc,
                "test_margin": test_margin,
                "train_accuracy": split_metrics.get("train", {}).get("accuracy", 0.0),
                "valid_accuracy": split_metrics.get("valid", {}).get("accuracy", 0.0),
                "train_margin": split_metrics.get("train", {}).get("margin", 0.0),
                "valid_margin": split_metrics.get("valid", {}).get("margin", 0.0),
            },
            "RandomPair": random_bl.to_dict(),
            "advantage_over_random": advantage,
            "factor_analysis": factor_analysis,
            "margin_calibration": margin_curve,
            "feature_shift": feature_shift,
            "gate": gate,
        }
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))

    elif cmd == "dev-runtime-preflight":
        print("dev-runtime-preflight: not implemented in mainline")

    else:
        print(f"Unknown command: {cmd}")


if __name__ == "__main__":
    main()
