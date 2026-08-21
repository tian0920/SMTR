"""P0-2: Retention Rule Ablation.

Compares three memory retention strategies under identical conditions:

  Rule-1 (single_negative):  δ ≤ 0 once → reject immediately
  Rule-2 (double_negative):  δ ≤ 0 twice consecutively → reject (current default)
  Rule-3 (always_retain):    once validated, never rejected

All other conditions identical (same TCI gate δ > 0 for admission,
same probe protocol, same environment).

Experiments:
  - lifelong (formation, contamination_ratio=0.2)
  - contamination (false/spurious/outdated @ ratio 0.1/0.2/0.3)

Output:
  results/ablation/retention_ablation.csv
  figures/retention_ablation.png
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import zlib
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.lifelong.lifelong_env import (
    LifelongEnvironment,
    StoredMemory,
    TaskSample,
    topic_affinity,
)
from experiments.lifelong.methods import LifelongPolicy, SMTRTCIPolicy
from experiments.lifelong.run_lifelong import (
    ALL_TOPICS,
    run_episode_sequence,
    _write_csv,
    _write_jsonl,
)

EPISODES = 100
SEEDS = [0, 1, 2, 3, 4]
CONTAMINATION_RATIO = 0.2


# ──────────────────────────────────────────────────────────────────────
# Retention rule variants
# ──────────────────────────────────────────────────────────────────────
class SingleNegativePolicy(SMTRTCIPolicy):
    """Rule-1: reject on first non-positive probe (no suspect grace)."""

    name = "rule1_single_negative"

    def _revalidate_topic(self, task: TaskSample) -> None:
        for entry in self.bank.retrieve_validated():
            memory = self._meta.get(entry.memory_id)
            if memory is None or topic_affinity(memory.topic, task.topic) == 0:
                continue
            if memory.source_episode == task.episode:
                continue
            delta = self._env.tci_probe_delta(memory, episode=task.episode)
            # No suspect streak: immediate admit (reject if δ ≤ 0)
            self.admission.admit(
                memory.memory_id,
                reward_expose=delta,
                reward_withhold=0.0,
                episode_id=task.episode,
            )


class DoubleNegativePolicy(SMTRTCIPolicy):
    """Rule-2: reject on two consecutive non-positive probes (default)."""

    name = "rule2_double_negative"
    # Inherits _revalidate_topic from SMTRTCIPolicy unchanged


class AlwaysRetainPolicy(SMTRTCIPolicy):
    """Rule-3: once validated, never reject (only probe for audit)."""

    name = "rule3_always_retain"

    def _revalidate_topic(self, task: TaskSample) -> None:
        for entry in self.bank.retrieve_validated():
            memory = self._meta.get(entry.memory_id)
            if memory is None or topic_affinity(memory.topic, task.topic) == 0:
                continue
            if memory.source_episode == task.episode:
                continue
            delta = self._env.tci_probe_delta(memory, episode=task.episode)
            # Record probe but never reject
            self.bank.validate_memory(
                memory.memory_id, delta,
                episode_id=task.episode,
                expose_reward=delta,
                withhold_reward=0.0,
                decision="retained",
            )


RETENTION_METHODS = {
    "rule1_single_negative": SingleNegativePolicy,
    "rule2_double_negative": DoubleNegativePolicy,
    "rule3_always_retain": AlwaysRetainPolicy,
}


# ──────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────
def compute_metrics(
    perf_rows: list[dict],
    hist_rows: list[dict],
    policy: LifelongPolicy,
    method: str,
    seed: int,
) -> dict:
    """Compute ablation metrics for one (method, seed) run."""
    # Final reward = mean of last 20 episodes
    seed_perf = [r for r in perf_rows if r["method"] == method and r["seed"] == seed]
    final_rewards = [r["reward"] for r in seed_perf[-20:]]
    final_reward = float(np.mean(final_rewards)) if final_rewards else 0.0
    cumulative = seed_perf[-1]["cumulative_reward"] if seed_perf else 0.0

    # Memory size (total entries)
    stats = policy.bank.get_statistics()
    memory_size = stats["total"]
    n_validated = stats["validated"]
    n_rejected = stats["rejected"]

    # Contamination rate: fraction of contaminated memories that are validated
    seed_hist = [r for r in hist_rows if r["method"] == method and r["seed"] == seed]
    contaminated = [r for r in seed_hist if r["contamination"] != "none"]
    contaminated_validated = [
        r for r in contaminated if r["status"] in ("validated", "candidate")
    ]
    contamination_rate = (
        len(contaminated_validated) / len(contaminated) if contaminated else 0.0
    )

    # Harmful memory retention: fraction of false memories still validated
    false_mems = [r for r in seed_hist if r["contamination"] == "false"]
    false_retained = [r for r in false_mems if r["status"] in ("validated", "candidate")]
    harmful_retention = (
        len(false_retained) / len(false_mems) if false_mems else 0.0
    )

    return {
        "method": method,
        "seed": seed,
        "final_reward": round(final_reward, 4),
        "cumulative_reward": round(cumulative, 2),
        "memory_size": memory_size,
        "n_validated": n_validated,
        "n_rejected": n_rejected,
        "contamination_rate": round(contamination_rate, 4),
        "harmful_retention": round(harmful_retention, 4),
    }


# ──────────────────────────────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────────────────────────────
def run_ablation(
    *,
    scenario: str,
    episodes: int,
    seeds: list[int],
    contamination_ratio: float,
    change_episode: int | None,
    changed_topics: tuple[int, ...],
    output_dir: Path,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Run all 3 retention rules on one scenario."""
    all_perf: list[dict] = []
    all_hist: list[dict] = []
    all_metrics: list[dict] = []

    for seed in seeds:
        for method_name, policy_cls in RETENTION_METHODS.items():
            env = LifelongEnvironment(
                seed=seed,
                method_seed=zlib.crc32(method_name.encode()) % 100000,
                change_episode=change_episode,
                changed_topics=changed_topics,
            )
            policy = policy_cls(env)
            perf_rows: list[dict] = []
            hist_rows: list[dict] = []
            traj_rows: list[dict] = []

            run_episode_sequence(
                policy=policy,
                env=env,
                episodes=episodes,
                topics=ALL_TOPICS,
                topics_after_change=None,
                contamination_ratio=contamination_ratio,
                seed=seed,
                trajectory_rows=traj_rows,
                history_rows=hist_rows,
                performance_rows=perf_rows,
            )
            all_perf.extend(perf_rows)
            all_hist.extend(hist_rows)
            metrics = compute_metrics(perf_rows, hist_rows, policy, method_name, seed)
            metrics["scenario"] = scenario
            all_metrics.append(metrics)

    return all_perf, all_hist, all_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=EPISODES)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--output", default="results/ablation/retention_rule")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_metrics: list[dict] = []

    # Scenario 1: Lifelong formation (contamination_ratio=0.2)
    print("=== Scenario: lifelong (contamination=0.2) ===")
    perf, hist, metrics = run_ablation(
        scenario="lifelong",
        episodes=args.episodes,
        seeds=args.seeds,
        contamination_ratio=CONTAMINATION_RATIO,
        change_episode=None,
        changed_topics=(),
        output_dir=output_dir,
    )
    all_metrics.extend(metrics)

    # Scenario 2: Contamination @ ratio=0.1
    for ratio in [0.1, 0.2, 0.3]:
        print(f"\n=== Scenario: contamination ratio={ratio} ===")
        _, _, metrics = run_ablation(
            scenario=f"contamination_{ratio}",
            episodes=args.episodes,
            seeds=args.seeds,
            contamination_ratio=ratio,
            change_episode=None,
            changed_topics=(),
            output_dir=output_dir,
        )
        all_metrics.extend(metrics)

    # Scenario 3: Outdated (environment change)
    print("\n=== Scenario: outdated (change at ep 60) ===")
    _, _, metrics = run_ablation(
        scenario="outdated",
        episodes=args.episodes,
        seeds=args.seeds,
        contamination_ratio=0.0,
        change_episode=60,
        changed_topics=(0, 1, 2),
        output_dir=output_dir,
    )
    all_metrics.extend(metrics)

    # Write CSV
    _write_csv(output_dir / "retention_ablation.csv", all_metrics)

    # Aggregate summary
    print("\n── Aggregate (mean ± std across seeds) ──")
    summary_rows: list[dict] = []
    for scenario in sorted(set(m["scenario"] for m in all_metrics)):
        for method in RETENTION_METHODS:
            rows = [m for m in all_metrics
                    if m["scenario"] == scenario and m["method"] == method]
            if not rows:
                continue
            agg = {
                "scenario": scenario,
                "method": method,
                "final_reward": f"{np.mean([r['final_reward'] for r in rows]):.3f}"
                               f"±{np.std([r['final_reward'] for r in rows]):.3f}",
                "contamination_rate": f"{np.mean([r['contamination_rate'] for r in rows]):.3f}",
                "memory_size": f"{np.mean([r['memory_size'] for r in rows]):.0f}",
                "harmful_retention": f"{np.mean([r['harmful_retention'] for r in rows]):.3f}",
            }
            summary_rows.append(agg)
            print(f"  {scenario:<20} {method:<26} "
                  f"reward={agg['final_reward']:<12} "
                  f"contam={agg['contamination_rate']:<6} "
                  f"size={agg['memory_size']:<5} "
                  f"harmful={agg['harmful_retention']}")

    # Save summary
    (output_dir / "summary.json").write_text(
        json.dumps(summary_rows, indent=2)
    )

    # Generate figure
    _generate_figure(all_metrics, output_dir)


def _generate_figure(all_metrics: list[dict], output_dir: Path) -> None:
    """Generate retention ablation comparison figure."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping figure generation")
        return

    scenarios = sorted(set(m["scenario"] for m in all_metrics))
    methods = list(RETENTION_METHODS.keys())
    method_labels = {
        "rule1_single_negative": "Rule-1: Single Negative",
        "rule2_double_negative": "Rule-2: Double Negative (default)",
        "rule3_always_retain": "Rule-3: Always Retain",
    }
    colors = {
        "rule1_single_negative": "#e74c3c",
        "rule2_double_negative": "#2ecc71",
        "rule3_always_retain": "#3498db",
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Retention Rule Ablation (P0-2)", fontsize=14, fontweight="bold")

    metric_panels = [
        ("final_reward", "Final Reward (last 20 ep)", axes[0, 0]),
        ("contamination_rate", "Contamination Rate", axes[0, 1]),
        ("memory_size", "Memory Size", axes[1, 0]),
        ("harmful_retention", "Harmful Memory Retention", axes[1, 1]),
    ]

    for metric_key, ylabel, ax in metric_panels:
        x = np.arange(len(scenarios))
        width = 0.25
        for i, method in enumerate(methods):
            vals = []
            for sc in scenarios:
                rows = [m for m in all_metrics
                        if m["scenario"] == sc and m["method"] == method]
                vals.append(np.mean([r[metric_key] for r in rows]) if rows else 0)
            ax.bar(x + i * width, vals, width, label=method_labels[method],
                   color=colors[method], alpha=0.85)
        ax.set_xticks(x + width)
        ax.set_xticklabels(scenarios, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=7, loc="best")

    plt.tight_layout()

    # Save both formats
    fig_dir = output_dir.parent.parent.parent / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / "retention_ablation.png", dpi=150, bbox_inches="tight")
    fig.savefig(fig_dir / "retention_ablation.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figures to {fig_dir}/retention_ablation.{{png,pdf}}")


if __name__ == "__main__":
    main()
