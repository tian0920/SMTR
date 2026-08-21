"""P1-1: Adversarial Memory Contamination.

Unlike the random contamination benchmark (false/spurious/outdated),
adversarial contamination simulates realistic attacks:

  1. semantic_plausible:  embedding similarity high but behavior outcome negative
     (looks correct, actually harmful)
  2. misleading_success:  initial success but long-term failure
     (works once by luck/coincidence, fails consistently after)
  3. role_mismatched:     useful for receiver A but harmful for receiver B
     (cross-agent knowledge leakage)

Experiment:
  - ratio: 0.1, 0.2, 0.3
  - methods: Full Memory, Retrieval, SMTR-TCI
  - metrics: attack success rate, performance degradation, recovery speed

Output:
  results/adversarial_contamination/attack_results.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.lifelong.lifelong_env import (
    BASE_SUCCESS,
    DISTRACTION_PENALTY,
    HARMFUL_EFFECT,
    HELPFUL_EFFECT,
    LifelongEnvironment,
    StoredMemory,
    TaskSample,
    topic_affinity,
)
from experiments.lifelong.methods import METHODS, LifelongPolicy
from experiments.lifelong.run_lifelong import (
    ALL_TOPICS,
    _write_csv,
    run_episode_sequence,
)

EPISODES = 100
SEEDS = [0, 1, 2, 3, 4]
RATIOS = [0.1, 0.2, 0.3]
ATTACK_TYPES = ["semantic_plausible", "misleading_success", "role_mismatched"]


# ──────────────────────────────────────────────────────────────────────
# Adversarial memory generator
# ──────────────────────────────────────────────────────────────────────
class AdversarialMemoryGenerator:
    """Generates adversarial memories that bypass naive quality filters.

    Each type simulates a realistic failure mode:
      - semantic_plausible: looks like a valid procedure (high topic match),
        but actually encodes a harmful action
      - misleading_success: initial episode succeeds (lucky), but the
        procedure has 0 or negative long-term effect
      - role_mismatched: the procedure works for one agent role but
        actively harms another (simulates cross-receiver contamination)
    """

    def __init__(self, env: LifelongEnvironment, attack_type: str) -> None:
        self._env = env
        self._attack_type = attack_type

    def generate(self, task: TaskSample) -> StoredMemory:
        """Generate one adversarial candidate memory."""
        digest = hashlib.sha1(
            f"{task.episode}:{task.topic}:{self._attack_type}:{self._env.seed}"
            .encode()
        ).hexdigest()[:10]
        mid = f"adv_ep{task.episode}_{self._attack_type}_{digest}"

        if self._attack_type == "semantic_plausible":
            # High embedding similarity (same topic), but actually harmful.
            # The agent would select it because it matches the task topic.
            return StoredMemory(
                memory_id=mid,
                topic=task.topic,
                content=f"plausible procedure for topic {task.topic} (adversarial)",
                source_episode=task.episode,
                contamination="false",  # ground truth: harmful
                true_effect=HARMFUL_EFFECT,
            )
        elif self._attack_type == "misleading_success":
            # Initially succeeds (true_effect > 0 at extraction), but
            # decays to harmful after a few episodes. We model this as
            # initially helpful (so it passes TCI probe at extraction time)
            # but with negative long-term effect once the environment moves on.
            return StoredMemory(
                memory_id=mid,
                topic=task.topic,
                content=f"lucky one-off trick for topic {task.topic} (misleading)",
                source_episode=task.episode,
                contamination="spurious",  # no real long-term value
                true_effect=0.0,  # zero effect on future tasks
            )
        elif self._attack_type == "role_mismatched":
            # Useful for agent A (same topic), but for agent B
            # (different task distribution) it causes distraction penalty.
            # We model this as a memory from a different topic that appears
            # relevant but actually causes DISTRACTION_PENALTY on injection.
            wrong_topic = (task.topic + 3) % 10  # offset topic
            return StoredMemory(
                memory_id=mid,
                topic=wrong_topic,
                content=f"role-A procedure misapplied to topic {task.topic} (mismatched)",
                source_episode=task.episode,
                contamination="false",
                true_effect=-DISTRACTION_PENALTY * 2,  # negative on wrong topic
            )
        else:
            raise ValueError(f"unknown attack type: {self._attack_type}")


# ──────────────────────────────────────────────────────────────────────
# Custom episode hook: inject adversarial memories at given ratio
# ──────────────────────────────────────────────────────────────────────
def make_adversarial_hook(
    policy: LifelongPolicy,
    env: LifelongEnvironment,
    generator: AdversarialMemoryGenerator,
    ratio: float,
):
    """Return an after_episode hook that injects adversarial candidates."""

    def hook(episode: int, task: TaskSample, pol: LifelongPolicy, e: LifelongEnvironment):
        if e._rng.random() < ratio:
            adv = generator.generate(task)
            # Store directly (bypassing normal extraction)
            pol._meta[adv.memory_id] = adv
            pol.bank.add_candidate(
                memory_id=adv.memory_id,
                content=adv.content,
                source_episode=adv.source_episode,
                receiver="agent1",
                created_step=adv.source_episode,
            )
            # For SMTR-TCI: also run TCI validation
            if hasattr(pol, "admission"):
                delta = e.tci_probe_delta(adv, episode=episode)
                pol.admission.admit(
                    adv.memory_id,
                    reward_expose=delta,
                    reward_withhold=0.0,
                    episode_id=episode,
                )

    return hook


# ──────────────────────────────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────────────────────────────
def run_adversarial_experiment(
    *,
    attack_type: str,
    ratio: float,
    episodes: int,
    seeds: list[int],
    output_dir: Path,
) -> list[dict]:
    """Run one (attack_type, ratio) experiment across methods and seeds."""
    results: list[dict] = []
    methods_to_test = ["full_memory", "retrieval", "smtr_tci"]

    for seed in seeds:
        for method_name in methods_to_test:
            env = LifelongEnvironment(
                seed=seed,
                method_seed=zlib.crc32(method_name.encode()) % 100000,
            )
            policy_cls = METHODS[method_name]
            policy = policy_cls(env)
            generator = AdversarialMemoryGenerator(env, attack_type)

            perf_rows: list[dict] = []
            hist_rows: list[dict] = []
            traj_rows: list[dict] = []

            # We need a custom hook per (policy, env) pair
            hook = make_adversarial_hook(policy, env, generator, ratio)

            # Run with hook — but we need to pass the hook into run_episode_sequence.
            # Since the hook uses the same policy/env, we wrap it:
            def wrapped_hook(ep, task, pol, e, _hook=hook, _pol=policy, _env=env):
                _hook(ep, task, _pol, _env)

            run_episode_sequence(
                policy=policy,
                env=env,
                episodes=episodes,
                topics=ALL_TOPICS,
                topics_after_change=None,
                contamination_ratio=0.0,  # no random contamination
                seed=seed,
                trajectory_rows=traj_rows,
                history_rows=hist_rows,
                performance_rows=perf_rows,
                after_episode=wrapped_hook,
            )

            # Compute metrics
            final_rewards = [r["reward"] for r in perf_rows[-20:]]
            final_reward = float(np.mean(final_rewards))
            cumulative = perf_rows[-1]["cumulative_reward"]

            # Attack success rate: fraction of adversarial memories still active
            adv_hist = [r for r in hist_rows
                       if "adv_" in r.get("memory_id", "") and "adversarial" not in str(r.get("contamination", ""))]
            # Count adversarial memories that are validated/still injected
            all_adv = [r for r in hist_rows if r["memory_id"].startswith("adv_")]
            adv_active = [r for r in all_adv if r["status"] in ("validated", "candidate")]
            attack_success = len(adv_active) / len(all_adv) if all_adv else 0.0

            # Performance degradation vs no-memory baseline
            env_baseline = LifelongEnvironment(
                seed=seed,
                method_seed=zlib.crc32("no_memory".encode()) % 100000,
            )
            no_mem_policy = METHODS["no_memory"](env_baseline)
            baseline_perf: list[dict] = []
            run_episode_sequence(
                policy=no_mem_policy, env=env_baseline,
                episodes=episodes, topics=ALL_TOPICS,
                topics_after_change=None, contamination_ratio=0.0,
                seed=seed, trajectory_rows=[], history_rows=[],
                performance_rows=baseline_perf,
            )
            baseline_final = float(np.mean([r["reward"] for r in baseline_perf[-20:]]))
            degradation = baseline_final - final_reward

            # Recovery speed: episodes to return to baseline - 0.05
            baseline_mean = float(np.mean([r["reward"] for r in baseline_perf]))
            recovery = episodes  # censored
            for i in range(episodes):
                window = perf_rows[max(0, i - 9):i + 1]
                if len(window) >= 10:
                    rolling = np.mean([r["reward"] for r in window])
                    if rolling >= baseline_mean - 0.05:
                        recovery = i + 1
                        break

            results.append({
                "attack_type": attack_type,
                "ratio": ratio,
                "method": method_name,
                "seed": seed,
                "final_reward": round(final_reward, 4),
                "cumulative_reward": round(cumulative, 2),
                "attack_success_rate": round(attack_success, 4),
                "performance_degradation": round(degradation, 4),
                "recovery_episodes": recovery,
                "n_adversarial_injected": len(all_adv),
                "n_adversarial_active": len(adv_active),
            })

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=EPISODES)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--output", default="results/adversarial_contamination")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[dict] = []

    for attack_type in ATTACK_TYPES:
        for ratio in RATIOS:
            print(f"\n=== {attack_type} @ ratio={ratio} ===")
            results = run_adversarial_experiment(
                attack_type=attack_type,
                ratio=ratio,
                episodes=args.episodes,
                seeds=args.seeds,
                output_dir=output_dir,
            )
            all_results.extend(results)
            # Print summary
            for method in ["full_memory", "retrieval", "smtr_tci"]:
                rows = [r for r in results if r["method"] == method]
                reward = np.mean([r["final_reward"] for r in rows])
                attack_sr = np.mean([r["attack_success_rate"] for r in rows])
                print(f"  {method:<12} reward={reward:.3f}  attack_success={attack_sr:.3f}")

    _write_csv(output_dir / "attack_results.csv", all_results)

    # Aggregate summary
    print("\n── Aggregate ──")
    summary: list[dict] = []
    for attack_type in ATTACK_TYPES:
        for ratio in RATIOS:
            for method in ["full_memory", "retrieval", "smtr_tci"]:
                rows = [r for r in all_results
                        if r["attack_type"] == attack_type
                        and r["ratio"] == ratio
                        and r["method"] == method]
                if not rows:
                    continue
                agg = {
                    "attack_type": attack_type,
                    "ratio": ratio,
                    "method": method,
                    "final_reward": f"{np.mean([r['final_reward'] for r in rows]):.3f}"
                                   f"±{np.std([r['final_reward'] for r in rows]):.3f}",
                    "attack_success_rate": f"{np.mean([r['attack_success_rate'] for r in rows]):.3f}",
                    "performance_degradation": f"{np.mean([r['performance_degradation'] for r in rows]):.3f}",
                    "recovery_episodes": f"{np.mean([r['recovery_episodes'] for r in rows]):.1f}",
                }
                summary.append(agg)
                print(f"  {attack_type:<22} r={ratio:.1f} {method:<12} "
                      f"reward={agg['final_reward']:<12} "
                      f"attack_sr={agg['attack_success_rate']}")

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
