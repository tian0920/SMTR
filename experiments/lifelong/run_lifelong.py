"""Long-term memory lifecycle runner (Task 3).

Episode loop:

    for episode in episodes:
        sample task -> retrieve persistent memory -> execute agent
        -> collect experience -> extract candidate memory
        -> run TCI validation -> consolidate memory

Artifacts written under --output (default results/lifelong):
    trajectory.jsonl      per-episode execution records
    memory_history.jsonl  per-candidate lifecycle events
    performance.csv       per-episode reward / cumulative / bank size

Seeds 0-4 supported. Pure synthetic environment — no engine / LLM cost,
fully reproducible given (experiment, seed).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import zlib
from collections.abc import Callable
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.lifelong.lifelong_env import (
    LifelongEnvironment,
    StoredMemory,
    TaskSample,
)
from experiments.lifelong.methods import METHODS, LifelongPolicy
from experiments.lifelong.baseline_policies import BASELINE_METHODS

# Merge baseline methods into the global registry so --methods can
# reference them without extra plumbing.
METHODS = {**METHODS, **BASELINE_METHODS}

ALL_TOPICS = tuple(range(10))
TOPICS_A = tuple(range(5))
TOPICS_B = tuple(range(5, 10))


def run_episode_sequence(
    *,
    policy: LifelongPolicy,
    env: LifelongEnvironment,
    episodes: int,
    topics: tuple[int, ...],
    topics_after_change: tuple[int, ...] | None,
    contamination_ratio: float,
    seed: int,
    trajectory_rows: list[dict],
    history_rows: list[dict],
    performance_rows: list[dict],
    after_episode: Callable[[int, TaskSample, LifelongPolicy, LifelongEnvironment], None] | None = None,
) -> None:
    cumulative = 0.0
    for episode in range(episodes):
        active_topics = topics
        if topics_after_change is not None and env.change_episode is not None \
                and episode >= env.change_episode:
            active_topics = topics_after_change
        task = env.sample_task(episode, active_topics)

        # retrieve + execute
        injected = policy.select_memories(task)
        success, reward = env.execute(task, injected)
        cumulative += reward

        # collect experience + extract candidate
        candidate = env.extract_candidate(task, contamination_ratio)

        # optional contamination injection (e.g. outdated memories)
        if after_episode is not None:
            after_episode(episode, task, policy, env)

        # TCI validation + consolidation (policy-specific)
        policy.process_candidate(task, candidate)
        try:
            entry = policy.bank.get(candidate.memory_id)
        except KeyError:
            entry = None  # discarded (no_memory) or evicted

        history_rows.append({
            "episode": episode,
            "method": policy.name,
            "seed": seed,
            "memory_id": candidate.memory_id,
            "topic": candidate.topic,
            "contamination": candidate.contamination,
            "status": entry.status if entry else "discarded",
            "tci_effect": entry.tci_effect if entry else None,
        })
        trajectory_rows.append({
            "episode": episode,
            "topic": task.topic,
            "distribution": task.distribution,
            "method": policy.name,
            "seed": seed,
            "success": success,
            "reward": reward,
            "n_injected": len(injected),
            "injected_ids": [m.memory_id for m in injected],
        })
        stats = policy.bank.get_statistics()
        performance_rows.append({
            "episode": episode,
            "method": policy.name,
            "seed": seed,
            "reward": reward,
            "cumulative_reward": cumulative,
            "n_stored": stats["total"],
            "n_validated": stats["validated"],
            "n_rejected": stats["rejected"],
        })


def run_experiment(
    *,
    experiment: str,
    output_dir: Path,
    episodes: int,
    seeds: list[int],
    methods: list[str],
    contamination_ratio: float,
    change_episode: int | None,
    changed_topics: tuple[int, ...],
    topics: tuple[int, ...],
    topics_after_change: tuple[int, ...] | None,
    capacity: int | None,
    after_episode: Callable[[int, TaskSample, LifelongPolicy, LifelongEnvironment], None] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectory_rows: list[dict] = []
    history_rows: list[dict] = []
    performance_rows: list[dict] = []

    for seed in seeds:
        for method_name in methods:
            env = LifelongEnvironment(
                seed=seed,
                method_seed=zlib.crc32(method_name.encode()) % 100000,
                change_episode=change_episode,
                changed_topics=changed_topics,
            )
            policy_cls = METHODS[method_name]
            policy = policy_cls(env, capacity=capacity)
            run_episode_sequence(
                policy=policy,
                env=env,
                episodes=episodes,
                topics=topics,
                topics_after_change=topics_after_change,
                contamination_ratio=contamination_ratio,
                seed=seed,
                trajectory_rows=trajectory_rows,
                history_rows=history_rows,
                performance_rows=performance_rows,
                after_episode=after_episode,
            )

    _write_jsonl(output_dir / "trajectory.jsonl", trajectory_rows)
    _write_jsonl(output_dir / "memory_history.jsonl", history_rows)
    _write_csv(output_dir / "performance.csv", performance_rows)

    # Export memory audit for SMTR-TCI policies (P0-3)
    audit_rows: list[dict] = []
    for seed in seeds:
        for method_name in methods:
            if method_name != "smtr_tci":
                continue
            # Re-run to get the policy bank (lightweight: env is fast)
            env = LifelongEnvironment(
                seed=seed,
                method_seed=zlib.crc32(method_name.encode()) % 100000,
                change_episode=change_episode,
                changed_topics=changed_topics,
            )
            policy = METHODS[method_name](env, capacity=capacity)
            run_episode_sequence(
                policy=policy, env=env, episodes=episodes,
                topics=topics, topics_after_change=topics_after_change,
                contamination_ratio=contamination_ratio, seed=seed,
                trajectory_rows=[], history_rows=[], performance_rows=[],
                after_episode=after_episode,
            )
            for entry in policy.bank.export_memory_audit():
                entry["seed"] = seed
                audit_rows.append(entry)
    if audit_rows:
        (output_dir / "memory_audit.json").write_text(
            json.dumps(audit_rows, indent=2, default=str)
        )

    config = {
        "experiment": experiment,
        "episodes": episodes,
        "seeds": seeds,
        "methods": methods,
        "contamination_ratio": contamination_ratio,
        "change_episode": change_episode,
        "changed_topics": list(changed_topics),
        "topics": list(topics),
        "topics_after_change": list(topics_after_change) if topics_after_change else None,
        "capacity": capacity,
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(f"Saved experiment artifacts to {output_dir}")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Long-term memory lifecycle runner")
    parser.add_argument("--experiment", default="formation",
                        choices=["formation", "contamination", "transfer",
                                 "multi_agent", "budget"])
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--methods", nargs="+",
                        default=["no_memory", "full_memory", "retrieval", "smtr_tci",
                                 "reflexion", "agile", "heuristic", "agemem"])
    parser.add_argument("--contamination-ratio", type=float, default=0.2)
    parser.add_argument("--change-episode", type=int, default=None)
    parser.add_argument("--changed-topics", type=int, nargs="*", default=[])
    parser.add_argument("--transfer", action="store_true",
                        help="episodes 0..N/2 from distribution A (topics 0-4), "
                             "rest from distribution B (topics 5-9)")
    parser.add_argument("--capacity", type=int, default=None)
    parser.add_argument("--output", default="results/lifelong")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    topics: tuple[int, ...] = ALL_TOPICS
    topics_after_change: tuple[int, ...] | None = None
    change_episode = args.change_episode
    if args.transfer:
        topics = TOPICS_A
        topics_after_change = TOPICS_B
        change_episode = change_episode if change_episode is not None else args.episodes // 2
    run_experiment(
        experiment=args.experiment,
        output_dir=Path(args.output) / args.experiment,
        episodes=args.episodes,
        seeds=args.seeds,
        methods=args.methods,
        contamination_ratio=args.contamination_ratio,
        change_episode=change_episode,
        changed_topics=tuple(args.changed_topics),
        topics=topics,
        topics_after_change=topics_after_change,
        capacity=args.capacity,
    )


if __name__ == "__main__":
    main()
