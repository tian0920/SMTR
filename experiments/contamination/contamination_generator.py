"""Task 5: Memory contamination benchmark.

Three contamination types (all modeled with ground truth known):

  1. false      — plausible but wrong procedure (harms future tasks)
  2. spurious   — one-off success that does not generalize
  3. outdated   — knowledge invalidated by an environment change

Parameter grid: contamination_ratio in {0.1, 0.2, 0.3} only.

Methods compared: no_memory / full_memory / retrieval / smtr_tci.

Output: results/contamination/contamination_results.csv with columns
  final_reward, performance_drop, recovery_episodes,
  harmful_memory_retention_rate.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.lifelong.run_lifelong import ALL_TOPICS, run_experiment

RATIOS = [0.1, 0.2, 0.3]
EPISODES = 100
SEEDS = [0, 1, 2, 3, 4]
METHODS = ["no_memory", "full_memory", "retrieval", "smtr_tci"]
# Outdated variant: environment change at episode 60 affects topics 0-2.
OUTDATED_CHANGE_EPISODE = 60
OUTDATED_CHANGED_TOPICS = (0, 1, 2)
OUTDATED_RATIO = 0.2


def generate(output_root: Path) -> None:
    for ratio in RATIOS:
        variant = f"false_spurious_r{ratio}"
        run_experiment(
            experiment=variant,
            output_dir=output_root / variant,
            episodes=EPISODES,
            seeds=SEEDS,
            methods=METHODS,
            contamination_ratio=ratio,
            change_episode=None,
            changed_topics=(),
            topics=ALL_TOPICS,
            topics_after_change=None,
            capacity=None,
        )
    run_experiment(
        experiment="outdated",
        output_dir=output_root / "outdated",
        episodes=EPISODES,
        seeds=SEEDS,
        methods=METHODS,
        contamination_ratio=OUTDATED_RATIO,
        change_episode=OUTDATED_CHANGE_EPISODE,
        changed_topics=OUTDATED_CHANGED_TOPICS,
        topics=ALL_TOPICS,
        topics_after_change=None,  # same topics, environment drifted
        capacity=None,
    )


# ----------------------------------------------------------------------
# Analysis
# ----------------------------------------------------------------------
def _load_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _reward_by_method_seed(perf_rows: list[dict]) -> dict[tuple[str, int], list[float]]:
    grouped: dict[tuple[str, int], list[tuple[int, float]]] = defaultdict(list)
    for row in perf_rows:
        grouped[(row["method"], int(row["seed"]))].append(
            (int(row["episode"]), float(row["reward"]))
        )
    return {
        key: [r for _, r in sorted(vals)]
        for key, vals in grouped.items()
    }


def _recovery_episodes(rewards: list[float], change_episode: int) -> int:
    """Episodes after the change until rolling mean returns to baseline."""
    baseline = float(np.mean(rewards[max(0, change_episode - 20):change_episode]))
    window = 10
    after = rewards[change_episode:]
    for i in range(window, len(after)):
        if float(np.mean(after[i - window:i])) >= baseline - 0.05:
            return i
    return len(after)  # censored: never recovered


def _retention_rate(history_rows: list[dict], method: str) -> float:
    """Fraction of contaminated memories retained (still usable) by the method."""
    contaminated = [
        r for r in history_rows
        if r["method"] == method and r["contamination"] != "none"
    ]
    if not contaminated:
        return 0.0
    retained = [r for r in contaminated if r["status"] in ("validated", "candidate")]
    return len(retained) / len(contaminated)


def analyze(output_root: Path) -> None:
    results: list[dict] = []
    for variant_dir in sorted(output_root.iterdir()):
        perf_path = variant_dir / "performance.csv"
        history_path = variant_dir / "memory_history.jsonl"
        if not perf_path.exists():
            continue
        perf_rows = _load_rows(perf_path)
        history_rows = [
            json.loads(line)
            for line in history_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        rewards = _reward_by_method_seed(perf_rows)
        is_outdated = variant_dir.name == "outdated"
        for method in METHODS:
            finals, drops, recoveries = [], [], []
            for (m, _seed), curve in rewards.items():
                if m != method:
                    continue
                n = len(curve)
                finals.append(float(np.mean(curve[-max(1, n // 10):])))
                if is_outdated:
                    pre = float(np.mean(curve[:OUTDATED_CHANGE_EPISODE]))
                    post = float(np.mean(curve[OUTDATED_CHANGE_EPISODE:]))
                    drops.append(pre - post)
                    recoveries.append(_recovery_episodes(curve, OUTDATED_CHANGE_EPISODE))
            results.append({
                "variant": variant_dir.name,
                "method": method,
                "final_reward_mean": float(np.mean(finals)),
                "final_reward_std": float(np.std(finals)),
                "performance_drop_mean": float(np.mean(drops)) if drops else None,
                "recovery_episodes_mean": float(np.mean(recoveries)) if recoveries else None,
                "harmful_memory_retention_rate": _retention_rate(history_rows, method),
            })

    out_path = output_root / "contamination_results.csv"
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved: {out_path}")
    for row in results:
        drop = "" if row["performance_drop_mean"] is None else \
            f" drop={row['performance_drop_mean']:.3f}"
        rec = "" if row["recovery_episodes_mean"] is None else \
            f" recover={row['recovery_episodes_mean']:.1f}ep"
        print(
            f"  {row['variant']:<24} {row['method']:<12}"
            f" final={row['final_reward_mean']:.3f}{drop}{rec}"
            f" retention={row['harmful_memory_retention_rate']:.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/contamination")
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    output_root = Path(args.output)
    if not args.analyze_only:
        generate(output_root)
    analyze(output_root)


if __name__ == "__main__":
    main()
