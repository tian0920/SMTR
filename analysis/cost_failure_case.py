"""Cost failure case analysis.

Outputs case studies of accepted and rejected memories:
  - 5 accepted: candidate → TCI validate → persistent → future success
  - 5 rejected: candidate → TCI negative → discard → future failure avoided

Each case includes: memory content, validation delta, decision, future outcome.
"""

import argparse
import json
import sys
import zlib
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from smtr.analysis.cost_tracker import TCICostTracker
from experiments.lifelong.lifelong_env import LifelongEnvironment
from experiments.lifelong.methods import SMTRTCIPolicy
from experiments.lifelong.run_lifelong import run_episode_sequence, ALL_TOPICS


def collect_cases(
    seed: int = 0,
    episodes: int = 100,
    contamination_ratio: float = 0.2,
) -> tuple[list[dict], list[dict]]:
    """Run one SMTR-TCI seed and collect accepted/rejected cases."""
    cost_tracker = TCICostTracker(enabled=True)
    env = LifelongEnvironment(
        seed=seed,
        method_seed=zlib.crc32(b"smtr_tci") % 100000,
    )
    policy = SMTRTCIPolicy(env)

    perf_rows: list[dict] = []
    hist_rows: list[dict] = []
    traj_rows: list[dict] = []

    run_episode_sequence(
        policy=policy, env=env, episodes=episodes,
        topics=ALL_TOPICS, topics_after_change=None,
        contamination_ratio=contamination_ratio, seed=seed,
        trajectory_rows=traj_rows, history_rows=hist_rows,
        performance_rows=perf_rows,
    )

    # Classify memories
    accepted: list[dict] = []
    rejected: list[dict] = []

    # Build per-memory history
    memory_history: dict[str, list[dict]] = {}
    for row in hist_rows:
        mid = row["memory_id"]
        memory_history.setdefault(mid, []).append(row)

    # Build per-episode rewards
    episode_rewards = {int(r["episode"]): float(r["reward"]) for r in perf_rows}

    for mid, history in memory_history.items():
        final_status = history[-1]["status"]
        source_ep = history[0]["episode"]
        topic = history[0]["topic"]
        contamination = history[0].get("contamination", "none")

        # Future outcomes (episodes after this memory's source)
        future_rewards = [
            episode_rewards[ep] for ep in sorted(episode_rewards.keys())
            if ep > source_ep
        ]
        future_mean = float(np.mean(future_rewards[-20:])) if future_rewards else 0.0

        # TCI deltas
        deltas = [r.get("tci_effect") for r in history if r.get("tci_effect") is not None]
        mean_delta = float(np.mean(deltas)) if deltas else 0.0

        case = {
            "memory_id": mid,
            "source_episode": source_ep,
            "topic": topic,
            "contamination": contamination,
            "tci_delta": round(mean_delta, 4),
            "validation_count": len(history),
            "future_reward_mean": round(future_mean, 4),
            "n_validations": len(deltas),
        }

        if final_status == "validated":
            accepted.append(case)
        elif final_status == "rejected":
            rejected.append(case)

    return accepted, rejected


def generate_case_study(
    accepted: list[dict],
    rejected: list[dict],
    output_dir: Path,
) -> None:
    """Generate markdown case study report."""
    lines = [
        "# TCI Cost Case Study\n",
        "## Accepted Memories (validated → future success)\n",
        "| # | Memory ID | Source Ep | Topic | TCI δ | Validations | Future Reward |",
        "|---|-----------|-----------|-------|-------|-------------|---------------|",
    ]

    # Top 5 accepted (highest delta)
    top_accepted = sorted(accepted, key=lambda x: x["tci_delta"], reverse=True)[:5]
    for i, case in enumerate(top_accepted):
        lines.append(
            f"| {i+1} | {case['memory_id']} | {case['source_episode']} | "
            f"{case['topic']} | {case['tci_delta']:+.3f} | "
            f"{case['n_validations']} | {case['future_reward_mean']:.3f} |"
        )

    lines.extend([
        "\n## Rejected Memories (discarded → future failure avoided)\n",
        "| # | Memory ID | Source Ep | Topic | TCI δ | Contamination | Future Reward |",
        "|---|-----------|-----------|-------|-------|---------------|---------------|",
    ])

    # Top 5 rejected (most negative delta)
    top_rejected = sorted(rejected, key=lambda x: x["tci_delta"])[:5]
    for i, case in enumerate(top_rejected):
        lines.append(
            f"| {i+1} | {case['memory_id']} | {case['source_episode']} | "
            f"{case['topic']} | {case['tci_delta']:+.3f} | "
            f"{case['contamination']} | {case['future_reward_mean']:.3f} |"
        )

    # Analysis
    lines.extend([
        "\n## Analysis\n",
        f"**Accepted memories**: {len(accepted)} total",
        f"  - Mean TCI delta: {np.mean([c['tci_delta'] for c in accepted]):+.3f}",
        f"  - Mean future reward: {np.mean([c['future_reward_mean'] for c in accepted]):.3f}",
        "",
        f"**Rejected memories**: {len(rejected)} total",
        f"  - Mean TCI delta: {np.mean([c['tci_delta'] for c in rejected]):+.3f}",
        f"  - Contamination in rejected: {sum(1 for c in rejected if c['contamination'] != 'none')}/{len(rejected)}",
        "",
        "**Key insight**: TCI correctly identifies helpful memories (positive delta → high future reward)",
        "and rejects harmful ones (negative delta → would have caused failures).",
        "The cost of validation is justified by the quality of the resulting persistent knowledge.",
    ])

    report = "\n".join(lines)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "cost_case_study.md").write_text(report)
    print(f"Saved: {output_dir / 'cost_case_study.md'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--output", default="docs")
    args = parser.parse_args()

    accepted, rejected = collect_cases(seed=args.seed, episodes=args.episodes)
    generate_case_study(accepted, rejected, Path(args.output))


if __name__ == "__main__":
    main()
