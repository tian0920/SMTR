"""CLI for explicitly requested Robust-SMTR smoke runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from smtr.config import RuntimeConfig
from smtr.counterfactual.snapshot import ReadOnlyPinnedMemoryView
from smtr.counterfactual.task_provider import CounterfactualToyTaskProvider
from smtr.memory.seed_memories import seed_repository
from smtr.memory.store import SQLiteSharedMemoryRepository
from smtr.robust.factory import build_robust_smtr_router
from smtr.router.candidate_proposer import DeterministicHybridCandidateProposer
from smtr.runtime.graph import build_graph
from smtr.runtime.state import initial_state


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m smtr.robust.cli")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run-experiment")
    run_parser.add_argument("--db", required=True)
    run_parser.add_argument("--critic-checkpoint", required=True)
    run_parser.add_argument("--method", choices=["Robust-SMTR"], default="Robust-SMTR")
    run_parser.add_argument("--negative-risk-budget", type=float, default=0.2)
    run_parser.add_argument("--confidence-level", type=float, default=0.9)
    run_parser.add_argument("--task-seed", type=int, default=0)
    run_parser.add_argument("--generation-seed", type=int, default=0)
    run_parser.add_argument("--traversal-seed", type=int, default=0)
    run_parser.add_argument("--top-k", type=int, default=4)
    run_parser.add_argument("--max-shares-per-invocation", type=int, default=3)
    run_parser.add_argument("--scenario", default="positive")
    run_parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    if args.command == "run-experiment":
        _run_experiment(args)


def _run_experiment(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    repository = SQLiteSharedMemoryRepository(args.db)
    seed_repository(repository)
    task_provider = CounterfactualToyTaskProvider()
    task_provider.ensure_memories(repository)
    task_spec = task_provider.generate(scenario=args.scenario, seed=args.task_seed)
    snapshot = repository.create_read_snapshot()
    router = build_robust_smtr_router(
        critic_checkpoint=args.critic_checkpoint,
        negative_risk_budget=args.negative_risk_budget,
        confidence_level=args.confidence_level,
        max_shares_per_invocation=args.max_shares_per_invocation,
        seed=args.traversal_seed,
    )
    app = build_graph(
        memory_pool=ReadOnlyPinnedMemoryView(repository=repository, snapshot=snapshot),
        proposer=DeterministicHybridCandidateProposer(),
        router=router,
        config=RuntimeConfig(seed=args.traversal_seed, top_k=args.top_k),
    )
    result = app.invoke(
        initial_state(
            task=task_spec.task,
            environment_observation=task_spec.environment_observation,
            run_seed=args.generation_seed,
            episode_id="robust_smoke",
            task_id="robust_smoke",
            top_k=args.top_k,
        )
    )
    trace = result.get("router_trace", [])
    robust_decisions = [
        decision
        for invocation in trace
        for decision in invocation.get("decisions", [])
        if decision.get("robust_diagnostics") is not None
    ]
    (output_dir / "robust_smoke.json").write_text(
        json.dumps(
            {
                "method": args.method,
                "team_success": bool(result.get("team_success", False)),
                "invocation_count": len(trace),
                "robust_decision_count": len(robust_decisions),
                "confidence_level": args.confidence_level,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"method={args.method}")
    print(f"team_success={bool(result.get('team_success', False))}")
    print(f"invocation_count={len(trace)}")
    print(f"robust_decision_count={len(robust_decisions)}")


if __name__ == "__main__":
    main()
