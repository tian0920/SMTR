"""Statistical significance of receiver conditioning (Phase 3, Task 2).

Compares SMTR-uniform vs SMTR-receiver-conditioned TCI with proper
statistical testing:

  - per-seed mean / std
  - paired t-test (paired by task_id within each seed)
  - bootstrap 95% CI on the paired difference
  - effect size (Cohen's d on paired differences)

Metrics tested:
  - team reward
  - negative transfer (total negative injections per episode)
  - receiver disagreement (std of injected counts)
  - contamination rate (from contamination experiment)

Output: results/receiver3/significance.csv
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = _PROJECT_ROOT / "results" / "marble" / "receiver3"
OUTPUT_DIR = _PROJECT_ROOT / "results" / "receiver3"


def load_episodes() -> list[dict]:
    path = RESULTS_DIR / "main" / "main_episodes.csv"
    with path.open() as f:
        return list(csv.DictReader(f))


def load_contamination() -> list[dict]:
    path = RESULTS_DIR / "contamination" / "contamination_results.csv"
    with path.open() as f:
        return list(csv.DictReader(f))


def paired_bootstrap_ci(
    a: np.ndarray, b: np.ndarray, n_boot: int = 10000, alpha: float = 0.05, seed: int = 42
) -> tuple[float, float]:
    """Bootstrap CI on mean(b - a) with paired resampling."""
    diff = b - a
    rng = np.random.RandomState(seed)
    n = len(diff)
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.randint(0, n, size=n)
        boot_means[i] = diff[idx].mean()
    lo = float(np.percentile(boot_means, 100 * alpha / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return lo, hi


def paired_test(
    uniform_vals: np.ndarray,
    receiver_vals: np.ndarray,
    metric: str,
    paired_keys: list[str],
) -> dict:
    """Run paired t-test + bootstrap on one metric."""
    diff = receiver_vals - uniform_vals
    t_stat, p_value = stats.ttest_rel(receiver_vals, uniform_vals)
    ci_lo, ci_hi = paired_bootstrap_ci(uniform_vals, receiver_vals)
    cohens_d = float(diff.mean() / diff.std()) if diff.std() > 0 else float("inf")

    return {
        "metric": metric,
        "paired_by": "task_id (within seed)",
        "n_pairs": len(diff),
        "uniform_mean": float(uniform_vals.mean()),
        "receiver_mean": float(receiver_vals.mean()),
        "diff_mean": float(diff.mean()),
        "diff_std": float(diff.std()),
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "significant_005": p_value < 0.05,
        "bootstrap_ci_lo": ci_lo,
        "bootstrap_ci_hi": ci_hi,
        "cohens_d": cohens_d,
        "paired_keys": ";".join(paired_keys),
    }


def align_by_task_seed(
    episodes: list[dict], metric_fn
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Align uniform/receiver rows by (task_id, seed); return paired arrays."""
    u_map: dict[tuple[str, str], float] = {}
    r_map: dict[tuple[str, str], float] = {}
    for e in episodes:
        key = (e["task_id"], e["seed"])
        val = metric_fn(e)
        if e["method"] == "smtr_uniform":
            u_map[key] = val
        elif e["method"] == "smtr_receiver":
            r_map[key] = val

    common = sorted(set(u_map) & set(r_map))
    uniform_vals = np.array([u_map[k] for k in common])
    receiver_vals = np.array([r_map[k] for k in common])
    keys = [f"{k[0]}|s{k[1]}" for k in common]
    return uniform_vals, receiver_vals, keys


def main() -> None:
    episodes = load_episodes()
    contamination = load_contamination()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []

    # 1. Team reward
    u, r, keys = align_by_task_seed(episodes, lambda e: float(e["team_reward"]))
    results.append(paired_test(u, r, "team_reward", keys[:3]))

    # 2. Negative transfer (total negative injections)
    u, r, keys = align_by_task_seed(
        episodes,
        lambda e: float(e["receiver_1_negative"])
        + float(e["receiver_2_negative"])
        + float(e["receiver_3_negative"]),
    )
    results.append(paired_test(u, r, "negative_transfer_count", keys[:3]))

    # 3. Positive transfer captured
    u, r, keys = align_by_task_seed(
        episodes,
        lambda e: float(e["receiver_1_positive"])
        + float(e["receiver_2_positive"])
        + float(e["receiver_3_positive"]),
    )
    results.append(paired_test(u, r, "positive_transfer_count", keys[:3]))

    # 4. Contamination rate (ratio=0.3, averaged over types)
    def contam_metric(rows, method):
        vals = [float(x["contamination_rate"]) for x in rows
                if x["method"] == method and float(x["ratio"]) == 0.3]
        return vals

    c_u = contam_metric(contamination, "smtr_uniform")
    c_r = contam_metric(contamination, "smtr_receiver")
    n_min = min(len(c_u), len(c_r))
    if n_min > 1:
        u_arr, r_arr = np.array(c_u[:n_min]), np.array(c_r[:n_min])
        results.append(paired_test(u_arr, r_arr, "contamination_rate_ratio0.3", ["episode_index"]))

    # 5. Per-seed breakdown
    seeds = sorted(set(int(e["seed"]) for e in episodes))
    seed_rows: list[dict] = []
    for seed in seeds:
        u_s = [float(e["team_reward"]) for e in episodes
               if e["method"] == "smtr_uniform" and int(e["seed"]) == seed]
        r_s = [float(e["team_reward"]) for e in episodes
               if e["method"] == "smtr_receiver" and int(e["seed"]) == seed]
        if u_s and r_s:
            t_stat, p_val = stats.ttest_rel(
                np.array(r_s), np.array(u_s)
            ) if len(u_s) == len(r_s) else (np.nan, np.nan)
            seed_rows.append({
                "seed": seed,
                "uniform_mean": float(np.mean(u_s)),
                "uniform_std": float(np.std(u_s)),
                "receiver_mean": float(np.mean(r_s)),
                "receiver_std": float(np.std(r_s)),
                "gain": float(np.mean(r_s) - np.mean(u_s)),
                "paired_p_value": float(p_val) if p_val == p_val else np.nan,
            })

    # Write significance CSV
    csv_path = OUTPUT_DIR / "significance.csv"
    fieldnames = list(results[0].keys())
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    print(f"Written: {csv_path}")

    # Per-seed CSV
    seed_path = OUTPUT_DIR / "significance_per_seed.csv"
    with seed_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(seed_rows[0].keys()))
        writer.writeheader()
        writer.writerows(seed_rows)
    print(f"Written: {seed_path}")

    # JSON summary
    summary = {
        "paired_tests": results,
        "per_seed": seed_rows,
    }
    json_path = OUTPUT_DIR / "significance_summary.json"
    json_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"Written: {json_path}")

    # Print report
    print()
    print("=" * 100)
    print("Receiver Conditioning Significance Tests (paired by task_id within seed)")
    print("=" * 100)
    for row in results:
        sig = "***" if row["p_value"] < 0.001 else "**" if row["p_value"] < 0.01 else "*" if row["p_value"] < 0.05 else "ns"
        print(
            f"{row['metric']:<30} n={row['n_pairs']:>4}  "
            f"uniform={row['uniform_mean']:.4f}  receiver={row['receiver_mean']:.4f}  "
            f"Δ={row['diff_mean']:+.4f}  p={row['p_value']:.2e} {sig}  "
            f"CI95=[{row['bootstrap_ci_lo']:+.4f}, {row['bootstrap_ci_hi']:+.4f}]  "
            f"d={row['cohens_d']:.3f}"
        )
    print()
    print("Per-seed team reward:")
    for s in seed_rows:
        print(
            f"  seed {s['seed']}: uniform={s['uniform_mean']:.4f}±{s['uniform_std']:.4f}  "
            f"receiver={s['receiver_mean']:.4f}±{s['receiver_std']:.4f}  "
            f"gain={s['gain']:+.4f}  p={s['paired_p_value']:.2e}"
        )


if __name__ == "__main__":
    main()
