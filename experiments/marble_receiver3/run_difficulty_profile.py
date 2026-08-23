"""MARBLE Task Difficulty Profiling.

Runs only the no_memory baseline across all MARBLE domains to measure
each task's intrinsic difficulty.  No memory retrieval, no TCI, no
persistent memory — just raw agent capability on each task.

Output: ``results/marble/difficulty_profile/``

CSV files:
  - difficulty_episode.csv  — per-episode results
  - difficulty_summary.csv  — per-task aggregate
"""

from __future__ import annotations

import csv
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))

from smtr.marble.task_loader import MarbleTask, MarbleTaskLoader
from smtr.marble.trajectory_collector import Trajectory, TrajectoryCollector

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ALL_SCENARIOS = ("bargaining", "coding", "database", "minecraft", "research")
DEFAULT_TASKS_PER_DOMAIN = 20
DEFAULT_SEEDS = [0, 1]
RECEIVER_IDS = ["agent1", "agent2", "agent3"]


# ---------------------------------------------------------------------------
# Profiling runner
# ---------------------------------------------------------------------------

def run_difficulty_profile(
    *,
    tasks: list[MarbleTask],
    seeds: list[int] = DEFAULT_SEEDS,
    receiver_ids: list[str] = RECEIVER_IDS,
) -> list[dict[str, Any]]:
    """Run no_memory baseline on each (task, seed) and record rewards.

    Returns a list of episode dicts, one per (task, seed).
    """
    collector = TrajectoryCollector()

    episode_rows: list[dict[str, Any]] = []
    total = len(tasks) * len(seeds)
    progress = 0

    for task in tasks:
        for seed in seeds:
            progress += 1
            t0 = time.time()
            print(
                f"[{progress}/{total}] task={task.scenario}/{task.task_id} "
                f"seed={seed}",
                end="",
                flush=True,
            )

            # Run no_memory episode
            traj = collector.collect(
                task, seed=seed, method="no_memory"
            )

            duration = time.time() - t0

            # Extract per-receiver rewards (approximated from team reward)
            # In MARBLE graph mode all agents share team success
            receiver_rewards = {
                rid: traj.team_reward
                for rid in receiver_ids
            }

            row: dict[str, Any] = {
                "domain": task.scenario,
                "task_id": task.task_id,
                "seed": seed,
                "episode_id": traj.trajectory_id,
                "team_reward": traj.team_reward,
                "team_success": traj.team_success,
                "receiver_rewards": json.dumps(receiver_rewards),
                "success": traj.team_success,
                "real_engine_executed": traj.real_engine_executed,
                "engine_duration_seconds": round(traj.engine_duration_seconds, 2),
                "wall_seconds": round(duration, 2),
                "exit_code": traj.exit_code,
                "score": traj.score,
            }
            episode_rows.append(row)

            reward_str = f"reward={traj.team_reward:.2f}"
            eng_str = f"eng={traj.engine_duration_seconds:.0f}s"
            print(f"  ({duration:.0f}s, {reward_str}, {eng_str})")

    return episode_rows


# ---------------------------------------------------------------------------
# Summary aggregation
# ---------------------------------------------------------------------------

def _aggregate_summary(episode_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate episode rows into per-task summary."""
    # Group by (domain, task_id)
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in episode_rows:
        key = (row["domain"], row["task_id"])
        groups.setdefault(key, []).append(row)

    summaries: list[dict[str, Any]] = []
    for (domain, task_id), rows in sorted(groups.items()):
        rewards = [r["team_reward"] for r in rows]
        successes = [r["success"] for r in rows]
        mean_r = sum(rewards) / len(rewards) if rewards else 0.0
        std_r = (
            (sum((r - mean_r) ** 2 for r in rewards) / len(rewards)) ** 0.5
            if len(rewards) > 1
            else 0.0
        )
        succ_rate = sum(successes) / len(successes) if successes else 0.0

        summaries.append({
            "task_id": task_id,
            "domain": domain,
            "mean_reward": round(mean_r, 4),
            "std_reward": round(std_r, 4),
            "success_rate": round(succ_rate, 4),
            "failure_rate": round(1.0 - succ_rate, 4),
            "n_episodes": len(rows),
            "n_real_engine": sum(1 for r in rows if r["real_engine_executed"]),
        })

    return summaries


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def _write_episode_csv(
    rows: list[dict[str, Any]], output_dir: Path
) -> Path:
    path = output_dir / "difficulty_episode.csv"
    if not rows:
        return path
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_summary_csv(
    rows: list[dict[str, Any]], output_dir: Path
) -> Path:
    path = output_dir / "difficulty_summary.csv"
    if not rows:
        return path
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="MARBLE Task Difficulty Profiling (no_memory baseline only)",
    )
    parser.add_argument(
        "--scenarios", nargs="+", default=list(ALL_SCENARIOS),
        help="Scenarios to profile (default: all 5)",
    )
    parser.add_argument(
        "--tasks-per-domain", type=int, default=DEFAULT_TASKS_PER_DOMAIN,
        help="Max tasks to sample per domain (default: 20)",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=DEFAULT_SEEDS,
        help="Generation seeds (default: 0 1)",
    )
    parser.add_argument(
        "--receivers", nargs="+", default=RECEIVER_IDS,
        help="Receiver agent IDs",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Override output directory",
    )
    parser.add_argument(
        "--random-seed", type=int, default=42,
        help="Seed for task sampling RNG",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )

    # Load and sample tasks
    loader = MarbleTaskLoader()
    rng = random.Random(args.random_seed)
    all_tasks: list[MarbleTask] = []

    for scenario in sorted(args.scenarios):
        scenario_tasks = loader.load_scenario(scenario)
        if len(scenario_tasks) > args.tasks_per_domain:
            scenario_tasks = rng.sample(scenario_tasks, args.tasks_per_domain)
        all_tasks.extend(scenario_tasks)
        print(f"  {scenario}: {len(scenario_tasks)} tasks sampled")

    print("=== MARBLE Task Difficulty Profiling ===")
    print(f"  scenarios: {args.scenarios}")
    print(f"  tasks: {len(all_tasks)} total ({args.tasks_per_domain}/domain)")
    print(f"  seeds: {args.seeds}")
    print(f"  receivers: {args.receivers}")
    print(f"  episodes: {len(all_tasks) * len(args.seeds)}")
    print()

    if not all_tasks:
        print("No tasks loaded. Exiting.")
        return

    # Run profiling
    episode_rows = run_difficulty_profile(
        tasks=all_tasks,
        seeds=args.seeds,
        receiver_ids=args.receivers,
    )

    # Aggregate summary
    summary_rows = _aggregate_summary(episode_rows)

    # Write output
    output_dir = (
        Path(args.output_dir) if args.output_dir
        else _PROJECT_ROOT / "results" / "marble" / "difficulty_profile"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    ep_path = _write_episode_csv(episode_rows, output_dir)
    print(f"\nWritten: {ep_path} ({len(episode_rows)} rows)")

    sum_path = _write_summary_csv(summary_rows, output_dir)
    print(f"Written: {sum_path} ({len(summary_rows)} rows)")

    # Quick inline summary
    print("\n=== Difficulty Summary ===")
    for domain in sorted(args.scenarios):
        domain_rows = [r for r in summary_rows if r["domain"] == domain]
        if not domain_rows:
            continue
        easy = sum(1 for r in domain_rows if r["mean_reward"] > 0.9)
        medium = sum(1 for r in domain_rows if 0.5 < r["mean_reward"] <= 0.9)
        hard = sum(1 for r in domain_rows if r["mean_reward"] <= 0.5)
        mean_all = sum(r["mean_reward"] for r in domain_rows) / len(domain_rows)
        print(
            f"  {domain:15s}  n={len(domain_rows):2d}  "
            f"mean={mean_all:.2f}  easy={easy} med={medium} hard={hard}"
        )


if __name__ == "__main__":
    main()
