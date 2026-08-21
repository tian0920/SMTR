"""TCI noise robustness experiment (Task B2).

Tests whether TCI's validation decisions degrade gracefully when reward
observations are noisy (answering the "perfect oracle" reviewer concern).

Setup:
  - Uses existing MARBLE paired records
  - Adds Gaussian noise to share/withhold outcomes
  - SMTR-TCI: inject if noisy_tau > 0
  - Random Validation: inject with 50% probability regardless of tau

Noise levels (sigma): 0.0, 0.1, 0.2, 0.3

Metrics:
  1. final reward (mean method reward under noise)
  2. late-stage reward (last seed)
  3. harmful retention (fraction of negative_transfer still injected)
  4. validation precision (fraction of injected that are positive_transfer)
  5. false rejection rate (fraction of positive_transfer rejected)

Output: results/noise_robustness/
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

from smtr.memory.noisy_validation import NOISE_LEVELS, apply_noise_to_paired_outcome

SEEDS = [0, 1, 2, 3, 4]
N_TASKS = 50


def load_paired_records(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def run_noise_experiment(
    records: list[dict],
    noise_sigma: float,
    seeds: list[int],
    n_tasks: int | None = None,
    n_noise_trials: int = 10,
) -> list[dict]:
    """Run noise robustness experiment for one noise level."""
    valid = [r for r in records if r.get("valid")]

    # Group by (task_id, receiver_agent_id, generation_seed)
    groups: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    for r in valid:
        key = (r["task_id"], r["receiver_agent_id"], r["generation_seed"])
        groups[key].append(r)

    if n_tasks is not None:
        task_ids = sorted(set(k[0] for k in groups))[:n_tasks]
        groups = {k: v for k, v in groups.items() if k[0] in task_ids}

    rows: list[dict] = []

    for method in ("smtr_tci", "random_validation"):
        for noise_trial in range(n_noise_trials):
            # Fresh RNG per (noise_sigma, method, trial) for reproducibility
            master_seed = hash((noise_sigma, method, noise_trial)) % (2**31)
            rng = np.random.RandomState(master_seed)

            for (task_id, receiver_id, gen_seed), group_records in sorted(groups.items()):
                if gen_seed not in seeds:
                    continue

                n_injected = 0
                n_positive_injected = 0
                n_harmful_injected = 0
                n_positive_total = 0
                total_tau_true = 0.0
                total_tau_observed = 0.0
                withhold_rewards: list[float] = []

                for r in group_records:
                    # True outcome
                    share_true = bool(r.get("share", {}).get("team_success", False))
                    withhold_true = bool(r.get("withhold", {}).get("team_success", False))
                    true_tau = int(share_true) - int(withhold_true)
                    withhold_rewards.append(float(withhold_true))

                    # Noisy observation
                    share_obs, withhold_obs = apply_noise_to_paired_outcome(
                        r, noise_sigma, rng
                    )
                    observed_tau = int(share_obs) - int(withhold_obs)

                    label = r.get("label", "")
                    is_positive = label == "positive_transfer"
                    is_negative = label == "negative_transfer"
                    if is_positive:
                        n_positive_total += 1

                    # Method decision
                    if method == "smtr_tci":
                        inject = observed_tau > 0
                    else:  # random_validation
                        inject = rng.random() < 0.5

                    if inject:
                        n_injected += 1
                        total_tau_true += true_tau
                        total_tau_observed += observed_tau
                        if is_positive:
                            n_positive_injected += 1
                        if is_negative:
                            n_harmful_injected += 1

                withhold_mean = float(np.mean(withhold_rewards)) if withhold_rewards else 0.0

                # Metrics
                method_reward = withhold_mean + total_tau_true
                validation_precision = n_positive_injected / max(n_injected, 1)
                false_rejection = (
                    (n_positive_total - n_positive_injected) / max(n_positive_total, 1)
                )

                rows.append({
                    "method": method,
                    "noise_sigma": noise_sigma,
                    "noise_trial": noise_trial,
                    "task_id": task_id,
                    "receiver_id": receiver_id,
                    "seed": gen_seed,
                    "n_injected": n_injected,
                    "n_positive_injected": n_positive_injected,
                    "n_harmful_injected": n_harmful_injected,
                    "n_positive_total": n_positive_total,
                    "method_reward": method_reward,
                    "validation_precision": validation_precision,
                    "false_rejection": false_rejection,
                    "total_tau_true": total_tau_true,
                    "total_tau_observed": total_tau_observed,
                })

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--n-tasks", type=int, default=N_TASKS)
    args = parser.parse_args()

    paired_path = _PROJECT_ROOT / "artifacts" / "marble" / "paired" / "train" / "paired_records.jsonl"
    records = load_paired_records(paired_path)

    output_dir = Path(args.output) if args.output else _PROJECT_ROOT / "results" / "noise_robustness"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    for sigma in NOISE_LEVELS:
        print(f"\n=== Noise sigma = {sigma} ===")
        rows = run_noise_experiment(
            records, noise_sigma=sigma, seeds=args.seeds,
            n_tasks=args.n_tasks, n_noise_trials=5,
        )
        all_rows.extend(rows)

        # Quick summary
        for method in ("smtr_tci", "random_validation"):
            m_rows = [r for r in rows if r["method"] == method]
            if not m_rows:
                continue
            mean_reward = float(np.mean([r["method_reward"] for r in m_rows]))
            mean_precision = float(np.mean([r["validation_precision"] for r in m_rows]))
            mean_false_rej = float(np.mean([r["false_rejection"] for r in m_rows]))
            mean_harmful = float(np.mean([r["n_harmful_injected"] for r in m_rows]))
            print(f"  {method:<22} reward={mean_reward:.4f}  precision={mean_precision:.4f}  "
                  f"false_rej={mean_false_rej:.4f}  harmful={mean_harmful:.2f}")

    # Write CSV
    if all_rows:
        fieldnames = list(all_rows[0].keys())
        csv_path = output_dir / "noise_robustness_results.csv"
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\nWritten: {csv_path}")

    # Aggregate summary
    summary: list[dict] = []
    for sigma in NOISE_LEVELS:
        for method in ("smtr_tci", "random_validation"):
            m_rows = [r for r in all_rows if r["method"] == method and r["noise_sigma"] == sigma]
            if not m_rows:
                continue
            rewards = [r["method_reward"] for r in m_rows]
            precisions = [r["validation_precision"] for r in m_rows]
            false_rejs = [r["false_rejection"] for r in m_rows]
            harmfuls = [r["n_harmful_injected"] for r in m_rows]
            summary.append({
                "method": method,
                "noise_sigma": sigma,
                "n_groups": len(m_rows),
                "mean_reward": float(np.mean(rewards)),
                "std_reward": float(np.std(rewards)),
                "mean_precision": float(np.mean(precisions)),
                "mean_false_rejection": float(np.mean(false_rejs)),
                "mean_harmful": float(np.mean(harmfuls)),
            })

    summary_path = output_dir / "noise_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Written: {summary_path}")


if __name__ == "__main__":
    main()
