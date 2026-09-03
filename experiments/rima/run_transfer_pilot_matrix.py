"""Pilot matrix orchestrator (Phase 23-24).

Runs the full pilot matrix:
    scenario × stream_seed × execution_seed × method

Two phases:
    Phase 1 — 4 core methods (rima_receiver, frozen, adaptive, positive_stop)
    Phase 2 — +2 ablation/baseline (static_same_probe_budget, no_uncertainty)

Usage::

    python experiments/rima/run_transfer_pilot_matrix.py \\
        --config configs/rima/pilot_mechanism.yaml \\
        --output-dir results/rima_transfer/pilot/runs

Resume-safe: DONE → skip, FAILED → retry, missing → run.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import random
import sys
import traceback
from pathlib import Path
from typing import Any

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))

from smtr.marble.experience_extractor import ExperienceExtractor  # noqa: E402
from smtr.marble.task_loader import MarbleTask, MarbleTaskLoader  # noqa: E402
from smtr.marble.trajectory_collector import TrajectoryCollector  # noqa: E402
from smtr.rima.experiment_config import (  # noqa: E402
    ALL_METHOD_VARIANTS,
    get_method_variant,
)
from smtr.router.official_score_transfer_critic import (  # noqa: E402
    BootstrapOfficialScoreTransferCritic,
    MatchedInterventionExample,
)

from experiments.rima.run_continual_transfer import (  # noqa: E402
    TransferContinualProtocol,
    _build_run_id,
    _git_info,
    build_base_examples,
    load_transfer_policy,
)

logger = logging.getLogger("rima.pilot_matrix")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def load_config(path: str | Path) -> dict[str, Any]:
    """Load pilot configuration from YAML."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return {
        "scenarios": cfg.get("scenarios", ["bargaining"]),
        "stream_seeds": cfg.get("stream_seeds", [0]),
        "execution_seeds": cfg.get("execution_seeds", [0]),
        "probe_seeds": cfg.get("probe_seeds", [0]),
        "n_tasks_per_stream": cfg.get("n_tasks_per_stream", 30),
        "methods_phase1": cfg.get("methods_phase1", []),
        "methods_phase2": cfg.get("methods_phase2", []),
        "critic_checkpoint": cfg.get("critic_checkpoint", ""),
        "transfer_policy": cfg.get("transfer_policy", ""),
        "intervention_records": cfg.get("intervention_records", ""),
        "source_agents": cfg.get("source_agents", ""),
    }


# ---------------------------------------------------------------------------
# Task stream generation
# ---------------------------------------------------------------------------


def generate_task_stream(
    tasks: list[MarbleTask],
    stream_seed: int,
    n_tasks: int,
) -> list[MarbleTask]:
    """Create a deterministic task ordering for a given stream seed."""
    rng = random.Random(stream_seed)
    indices = list(range(len(tasks)))
    rng.shuffle(indices)
    return [tasks[i] for i in indices[:n_tasks]]


def save_stream_manifest(
    stream_dir: Path,
    scenario: str,
    stream_seed: int,
    task_ids: list[str],
) -> None:
    """Save a task stream manifest for reproducibility."""
    stream_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "scenario": scenario,
        "stream_seed": stream_seed,
        "n_tasks": len(task_ids),
        "task_ids": task_ids,
    }
    fname = f"{scenario}__stream{stream_seed}.json"
    with open(stream_dir / fname, "w") as f:
        json.dump(manifest, f, indent=2)


# ---------------------------------------------------------------------------
# Single run executor
# ---------------------------------------------------------------------------


def execute_run(
    *,
    run_id: str,
    scenario: str,
    stream_seed: int,
    exec_seed: int,
    method_id: str,
    tasks: list[MarbleTask],
    collector: TrajectoryCollector,
    extractor: ExperienceExtractor,
    critic: BootstrapOfficialScoreTransferCritic | None,
    transfer_policy: Any,
    base_examples: list[MatchedInterventionExample],
    source_agent_ids: dict[str, str],
    probe_seeds: list[int],
    receiver_count: int,
    output_dir: Path,
    log_dir: Path,
) -> dict[str, Any] | None:
    """Execute a single pilot run with resume-safe logic.

    Returns summary dict on success, None on skip or failure.
    """
    run_dir = output_dir / run_id
    done_marker = run_dir / "DONE"
    failed_marker = run_dir / "FAILED"

    if done_marker.exists():
        logger.info("SKIP %s (DONE)", run_id)
        return None

    if failed_marker.exists():
        logger.info("RETRY %s (was FAILED)", run_id)
        failed_marker.unlink()

    run_dir.mkdir(parents=True, exist_ok=True)

    # Set up per-run logging.
    log_path = log_dir / f"{run_id}.log"
    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(logging.Formatter("%(asctime)s %(name)s %(message)s"))
    logging.getLogger().addHandler(fh)

    try:
        variant = get_method_variant(method_id)
        proto = TransferContinualProtocol(
            scenario=scenario,
            seed=exec_seed,
            method=method_id,
            tasks=tasks,
            collector=collector,
            extractor=extractor,
            receiver_count=receiver_count,
            critic_receiver=critic,
            transfer_policy=transfer_policy,
            method_variant=variant,
            stream_seed=stream_seed,
            execution_seed=exec_seed,
            probe_seeds=probe_seeds,
            base_examples=base_examples,
            source_agent_ids=source_agent_ids,
        )
        proto._init_run_dir(output_dir)
        summary = proto.run()
        logger.info("DONE %s", run_id)
        return summary

    except Exception as exc:
        logger.error("FAILED %s: %s", run_id, exc)
        # Write FAILED marker (remove DONE if partial).
        if done_marker.exists():
            done_marker.unlink()
        with open(failed_marker, "w") as f:
            f.write(traceback.format_exc())
        return None

    finally:
        logging.getLogger().removeHandler(fh)
        fh.close()


# ---------------------------------------------------------------------------
# Phase runner
# ---------------------------------------------------------------------------


def run_phase(
    *,
    phase_name: str,
    methods: list[str],
    scenarios: list[str],
    stream_seeds: list[int],
    exec_seeds: list[int],
    streams: dict[tuple[str, int], list[MarbleTask]],
    collector: TrajectoryCollector,
    extractor: ExperienceExtractor,
    critic: BootstrapOfficialScoreTransferCritic | None,
    transfer_policy: Any,
    base_examples: list[MatchedInterventionExample],
    source_agent_ids: dict[str, str],
    probe_seeds: list[int],
    receiver_count: int,
    output_dir: Path,
    log_dir: Path,
) -> list[dict[str, Any]]:
    """Run all combinations for one phase, returning summaries."""
    summaries: list[dict[str, Any]] = []
    n_total = len(scenarios) * len(stream_seeds) * len(exec_seeds) * len(methods)
    n_done = 0
    n_skip = 0
    n_fail = 0

    logger.info(
        "%s: %d runs (%d scenarios × %d streams × %d execs × %d methods)",
        phase_name, n_total,
        len(scenarios), len(stream_seeds), len(exec_seeds), len(methods),
    )

    for scenario in scenarios:
        for ss in stream_seeds:
            tasks = streams.get((scenario, ss))
            if not tasks:
                logger.warning(
                    "No stream for %s/stream%d — skipping", scenario, ss,
                )
                continue
            for es in exec_seeds:
                for method_id in methods:
                    run_id = _build_run_id(scenario, ss, es, method_id)
                    result = execute_run(
                        run_id=run_id,
                        scenario=scenario,
                        stream_seed=ss,
                        exec_seed=es,
                        method_id=method_id,
                        tasks=tasks,
                        collector=collector,
                        extractor=extractor,
                        critic=critic,
                        transfer_policy=transfer_policy,
                        base_examples=base_examples,
                        source_agent_ids=source_agent_ids,
                        probe_seeds=probe_seeds,
                        receiver_count=receiver_count,
                        output_dir=output_dir,
                        log_dir=log_dir,
                    )
                    if result is not None:
                        summaries.append(result)
                        n_done += 1
                    elif (output_dir / run_id / "DONE").exists():
                        n_skip += 1
                    else:
                        n_fail += 1

    logger.info(
        "%s complete: %d new, %d skipped, %d failed",
        phase_name, n_done, n_skip, n_fail,
    )
    return summaries


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Pilot matrix orchestrator (Phase 23-24).",
    )
    parser.add_argument(
        "--config", default="configs/rima/pilot_mechanism.yaml",
    )
    parser.add_argument(
        "--output-dir",
        default="results/rima_transfer/pilot/runs",
    )
    parser.add_argument(
        "--phase", choices=["1", "2", "all"], default="all",
        help="Run only one phase or both.",
    )
    parser.add_argument(
        "--method", type=str, default=None,
        help="Run only this method (overrides config).",
    )
    parser.add_argument(
        "--scenario", type=str, default=None,
        help="Run only this scenario (overrides config).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print run matrix without executing.",
    )
    parser.add_argument(
        "--receiver-count", type=int, default=3,
    )
    parser.add_argument("--engine-timeout", type=int, default=600)
    parser.add_argument(
        "--n-tasks-override", type=int, default=None,
        help="Override n_tasks_per_stream from config.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    # --- Load config ---
    cfg = load_config(args.config)
    project_root = _PROJECT_ROOT

    scenarios = [args.scenario] if args.scenario else cfg["scenarios"]
    stream_seeds = cfg["stream_seeds"]
    exec_seeds = cfg["execution_seeds"]
    probe_seeds = cfg["probe_seeds"]
    n_tasks = args.n_tasks_override or cfg["n_tasks_per_stream"]

    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # --- Load artifacts ---
    critic: BootstrapOfficialScoreTransferCritic | None = None
    if cfg["critic_checkpoint"]:
        cp = project_root / cfg["critic_checkpoint"]
        if cp.exists():
            critic = BootstrapOfficialScoreTransferCritic.load(str(cp))
            logger.info("Loaded critic from %s", cp)
        else:
            logger.warning("Critic checkpoint not found: %s", cp)

    transfer_policy = None
    if cfg["transfer_policy"]:
        pp = project_root / cfg["transfer_policy"]
        if pp.exists():
            transfer_policy = load_transfer_policy(str(pp))
            logger.info("Loaded transfer policy from %s", pp)
        else:
            logger.warning("Transfer policy not found: %s", pp)

    base_examples: list[MatchedInterventionExample] = []
    source_agent_ids: dict[str, str] = {}
    if cfg["intervention_records"]:
        ir = project_root / cfg["intervention_records"]
        sa_path = (
            (project_root / cfg["source_agents"])
            if cfg.get("source_agents") else None
        )
        if ir.exists():
            base_examples, source_agent_ids = build_base_examples(
                str(ir),
                str(sa_path) if sa_path and sa_path.exists() else None,
            )
            logger.info(
                "Loaded %d base examples from %s",
                len(base_examples), ir,
            )
        else:
            logger.warning("Intervention records not found: %s", ir)

    # --- Load tasks ---
    loader = MarbleTaskLoader()
    collector = TrajectoryCollector(engine_timeout=args.engine_timeout)
    extractor = ExperienceExtractor()

    # --- Generate fixed task streams ---
    streams_dir = output_dir.parent / "streams"
    streams: dict[tuple[str, int], list[MarbleTask]] = {}

    for scenario in scenarios:
        all_tasks = loader.load_scenario(scenario, limit=None)
        logger.info("Loaded %d tasks for %s", len(all_tasks), scenario)
        for ss in stream_seeds:
            stream = generate_task_stream(all_tasks, ss, n_tasks)
            streams[(scenario, ss)] = stream
            save_stream_manifest(
                streams_dir, scenario, ss,
                [t.task_id for t in stream],
            )

    # --- Determine methods ---
    methods_p1 = (
        [args.method] if args.method else cfg["methods_phase1"]
    )
    methods_p2 = (
        [args.method] if args.method else cfg["methods_phase2"]
    )

    if args.dry_run:
        print("=== DRY RUN ===")
        print(f"Scenarios: {scenarios}")
        print(f"Stream seeds: {stream_seeds}")
        print(f"Execution seeds: {exec_seeds}")
        print(f"Probe seeds: {probe_seeds}")
        print(f"Tasks per stream: {n_tasks}")
        print()
        if args.phase in ("1", "all"):
            n1 = (
                len(scenarios) * len(stream_seeds)
                * len(exec_seeds) * len(methods_p1)
            )
            print(f"Phase 1: {n1} runs — {methods_p1}")
        if args.phase in ("2", "all"):
            n2 = (
                len(scenarios) * len(stream_seeds)
                * len(exec_seeds) * len(methods_p2)
            )
            print(f"Phase 2: {n2} runs — {methods_p2}")
        return 0

    # --- Phase 1 ---
    if args.phase in ("1", "all"):
        run_phase(
            phase_name="Phase 1",
            methods=methods_p1,
            scenarios=scenarios,
            stream_seeds=stream_seeds,
            exec_seeds=exec_seeds,
            streams=streams,
            collector=collector,
            extractor=extractor,
            critic=critic,
            transfer_policy=transfer_policy,
            base_examples=base_examples,
            source_agent_ids=source_agent_ids,
            probe_seeds=probe_seeds,
            receiver_count=args.receiver_count,
            output_dir=output_dir,
            log_dir=log_dir,
        )

    # --- Phase 2 ---
    if args.phase in ("2", "all"):
        run_phase(
            phase_name="Phase 2",
            methods=methods_p2,
            scenarios=scenarios,
            stream_seeds=stream_seeds,
            exec_seeds=exec_seeds,
            streams=streams,
            collector=collector,
            extractor=extractor,
            critic=critic,
            transfer_policy=transfer_policy,
            base_examples=base_examples,
            source_agent_ids=source_agent_ids,
            probe_seeds=probe_seeds,
            receiver_count=args.receiver_count,
            output_dir=output_dir,
            log_dir=log_dir,
        )

    logger.info("Pilot matrix complete. Results in %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
