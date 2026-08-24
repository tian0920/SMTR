"""Official metric profiling: re-evaluate qwen3-30b-a3b with official Task Score.

This script runs no_memory baseline for the current backbone (qwen3-30b-a3b)
and records the official MultiAgentBench Task Score instead of binary team_success.

Purpose: Determine if the current backbone is truly saturated under the
official continuous metric (not the binary heuristic).

Settings:
    - Model: qwen3-30b-a3b (current backbone)
    - Method: no_memory only
    - Scenarios: database, research, minecraft, coding, bargaining
    - Tasks per scenario: 20
    - Seeds: [0, 1]
    - Total: 200 episodes

Output:
    results/marble/official_metric_profile/
    ├── episode_scores.csv
    ├── task_summary.csv
    ├── scenario_summary.csv
    └── evaluator_failures.csv

Usage::

    export DASHSCOPE_API_KEY="sk-..."
    export OPENAI_API_KEY="$DASHSCOPE_API_KEY"
    export DASHSCOPE_BASE_URL="https://llm-jhxtd03gjg0gd2o2.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1"
    export OPENAI_BASE_URL="$DASHSCOPE_BASE_URL"
    export MARBLE_LLM_MODEL="openai/qwen3-30b-a3b"
    export SMTR_LLM_ENABLE_THINKING="false"

    python experiments/marble_receiver3/run_official_metric_profile.py
"""

from __future__ import annotations

import csv
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from smtr.marble.task_loader import MarbleTaskLoader, ALL_SCENARIOS
from smtr.marble.trajectory_collector import Trajectory, TrajectoryCollector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Configuration
SCENARIOS = list(ALL_SCENARIOS)
TASKS_PER_SCENARIO = 20
SEEDS = [0, 1]
OUTPUT_DIR = Path("results/marble/official_metric_profile")


def run_episode(
    *,
    collector: TrajectoryCollector,
    scenario: str,
    task_id: str,
    seed: int,
) -> dict[str, Any]:
    """Run one no_memory episode and extract official metric."""
    start_time = time.time()
    loader = MarbleTaskLoader()
    task = loader.get_task(scenario, task_id)

    try:
        traj = collector.collect(
            task,
            seed=seed,
            method="no_memory",
        )
        duration = time.time() - start_time

        return {
            "scenario": scenario,
            "task_id": task_id,
            "seed": seed,
            "raw_task_score": traj.official_metric_raw,
            "normalized_task_score": traj.official_metric_normalized,
            "metric_name": traj.official_metric_name,
            "metric_valid": traj.official_metric_valid,
            "metric_error": traj.official_metric_error,
            "team_success": traj.team_success,
            "coordination_score": None,  # Not available in graph mode
            "runtime": duration,
            "evaluator_status": "valid" if traj.official_metric_valid else "failure",
            "engine_status": "ok" if traj.real_engine_executed else "failed",
            "exit_code": traj.exit_code,
        }
    except Exception as e:
        duration = time.time() - start_time
        logger.exception(f"Episode failed: {scenario}/{task_id} seed={seed}")
        return {
            "scenario": scenario,
            "task_id": task_id,
            "seed": seed,
            "raw_task_score": None,
            "normalized_task_score": None,
            "metric_name": "unknown",
            "metric_valid": False,
            "metric_error": f"exception: {e}",
            "team_success": False,
            "coordination_score": None,
            "runtime": duration,
            "evaluator_status": "exception",
            "engine_status": "failed",
            "exit_code": -1,
        }


def _preflight_env_check() -> None:
    """Fail fast if the LLM environment is not configured.

    Prevents silently burning hours of API time producing invalid episodes
    when DASHSCOPE/OPENAI env vars are missing (see 2026-08-24 bad run).
    """
    model = os.environ.get("MARBLE_LLM_MODEL", "")
    api_key = os.environ.get("DASHSCOPE_API_KEY", "") or os.environ.get("OPENAI_API_KEY", "")
    missing = []
    if not model:
        missing.append("MARBLE_LLM_MODEL")
    if not api_key:
        missing.append("DASHSCOPE_API_KEY/OPENAI_API_KEY")
    if missing:
        logger.error(
            "Missing required env vars: %s. "
            "Source scripts/env_dashscope.sh before running.",
            ", ".join(missing),
        )
        sys.exit(2)


def main() -> None:
    _preflight_env_check()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    collector = TrajectoryCollector(
        marble_root=Path("/home/ecs-user/MARBLE"),
        workspace_root=OUTPUT_DIR / "workspaces",
    )

    total_episodes = len(SCENARIOS) * TASKS_PER_SCENARIO * len(SEEDS)
    logger.info(f"Official metric profiling: {total_episodes} episodes")
    logger.info(f"Model: {os.environ.get('MARBLE_LLM_MODEL', 'not set')}")
    logger.info(f"Scenarios: {SCENARIOS}")
    logger.info(f"Tasks per scenario: {TASKS_PER_SCENARIO}")
    logger.info(f"Seeds: {SEEDS}")

    # Episode scores — write header now, append each row so a crash/kill
    # never loses completed episodes.
    episode_path = OUTPUT_DIR / "episode_scores.csv"
    fieldnames = [
        "scenario", "task_id", "seed", "raw_task_score",
        "normalized_task_score", "metric_name", "metric_valid",
        "metric_error", "team_success", "coordination_score",
        "runtime", "evaluator_status", "engine_status", "exit_code",
    ]
    with episode_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    episode_rows: list[dict[str, Any]] = []

    for scenario in SCENARIOS:
        logger.info(f"\n{'='*60}\nScenario: {scenario}\n{'='*60}")
        loader = MarbleTaskLoader()
        tasks = loader.load_scenario(scenario, limit=TASKS_PER_SCENARIO)

        for task in tasks:
            for seed in SEEDS:
                n = len(episode_rows) + 1
                logger.info(f"[{n}/{total_episodes}] {scenario}/{task.task_id} seed={seed}")

                row = run_episode(
                    collector=collector,
                    scenario=scenario,
                    task_id=task.task_id,
                    seed=seed,
                )
                episode_rows.append(row)

                with episode_path.open("a", newline="", encoding="utf-8") as f:
                    csv.DictWriter(f, fieldnames=fieldnames).writerow(row)

                status = "✅" if row["metric_valid"] else "❌"
                score_str = (
                    f"{row['normalized_task_score']:.3f}"
                    if row["normalized_task_score"] is not None
                    else "None"
                )
                logger.info(
                    f"  {status} score={score_str} "
                    f"team_success={row['team_success']} "
                    f"runtime={row['runtime']:.1f}s"
                )

    logger.info(f"\nEpisode rows appended incrementally to {episode_path}")

    # Write task_summary.csv
    task_summary_path = OUTPUT_DIR / "task_summary.csv"
    task_groups: dict[str, list[dict]] = {}
    for row in episode_rows:
        key = f"{row['scenario']}/{row['task_id']}"
        task_groups.setdefault(key, []).append(row)

    task_rows = []
    for key, rows in sorted(task_groups.items()):
        valid_scores = [
            r["normalized_task_score"]
            for r in rows
            if r["normalized_task_score"] is not None
        ]
        task_rows.append({
            "scenario": rows[0]["scenario"],
            "task_id": rows[0]["task_id"],
            "n_seeds": len(rows),
            "n_valid": len(valid_scores),
            "mean_score": sum(valid_scores) / len(valid_scores) if valid_scores else None,
            "team_success_rate": sum(1 for r in rows if r["team_success"]) / len(rows),
        })

    with task_summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "scenario", "task_id", "n_seeds", "n_valid", "mean_score",
            "team_success_rate",
        ])
        writer.writeheader()
        writer.writerows(task_rows)
    logger.info(f"Wrote {task_summary_path}")

    # Write scenario_summary.csv
    scenario_path = OUTPUT_DIR / "scenario_summary.csv"
    scenario_rows = []
    for scenario in SCENARIOS:
        s_rows = [r for r in episode_rows if r["scenario"] == scenario]
        valid_scores = [
            r["normalized_task_score"]
            for r in s_rows
            if r["normalized_task_score"] is not None
        ]
        n_valid = len(valid_scores)
        n_total = len(s_rows)
        scenario_rows.append({
            "scenario": scenario,
            "n_episodes": n_total,
            "n_valid_evaluator": n_valid,
            "valid_evaluator_rate": n_valid / n_total if n_total else 0,
            "mean_score": sum(valid_scores) / n_valid if n_valid else None,
            "team_success_rate": (
                sum(1 for r in s_rows if r["team_success"]) / n_total
                if n_total else 0
            ),
        })

    with scenario_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "scenario", "n_episodes", "n_valid_evaluator",
            "valid_evaluator_rate", "mean_score", "team_success_rate",
        ])
        writer.writeheader()
        writer.writerows(scenario_rows)
    logger.info(f"Wrote {scenario_path}")

    # Write evaluator_failures.csv
    failure_path = OUTPUT_DIR / "evaluator_failures.csv"
    failure_rows = [
        {
            "scenario": r["scenario"],
            "task_id": r["task_id"],
            "seed": r["seed"],
            "metric_name": r["metric_name"],
            "metric_error": r["metric_error"],
            "evaluator_status": r["evaluator_status"],
            "engine_status": r["engine_status"],
        }
        for r in episode_rows
        if not r["metric_valid"]
    ]

    with failure_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "scenario", "task_id", "seed", "metric_name",
            "metric_error", "evaluator_status", "engine_status",
        ])
        writer.writeheader()
        writer.writerows(failure_rows)
    logger.info(f"Wrote {failure_path} ({len(failure_rows)} failures)")

    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("PROFILING COMPLETE")
    logger.info(f"{'='*60}")
    for row in scenario_rows:
        score_str = (
            f"{row['mean_score']:.3f}"
            if row["mean_score"] is not None
            else "N/A"
        )
        logger.info(
            f"  {row['scenario']:12s}: "
            f"valid={row['n_valid_evaluator']}/{row['n_episodes']} "
            f"mean_score={score_str} "
            f"team_success_rate={row['team_success_rate']:.1%}"
        )


if __name__ == "__main__":
    main()
