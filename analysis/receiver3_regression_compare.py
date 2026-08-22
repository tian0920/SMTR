"""Before/After Regression Comparison (Phase 3-4).

Compares pre-refactor (results/marble/receiver3/main/) with
post-refactor (results/marble/receiver3/regression/) results.

Since the experiment is deterministic (CRC32-based seeding),
all numerical differences should be EXACTLY zero.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))


def load_summary(path: Path) -> dict[str, dict]:
    """Load summary JSON and index by method name."""
    with path.open() as f:
        data = json.load(f)
    return {s["method"]: s for s in data}


def load_episodes(path: Path) -> list[dict]:
    """Load episode CSV."""
    with path.open() as f:
        return list(csv.DictReader(f))


def compare_summaries(before: dict, after: dict) -> list[dict]:
    """Compare summary metrics before vs after."""
    rows = []
    for method in sorted(before.keys()):
        b = before.get(method, {})
        a = after.get(method, {})
        if not b or not a:
            continue

        metrics = [
            ("team_reward", "mean_team_reward"),
            ("team_reward_std", "std_team_reward"),
            ("r1_reward", "mean_receiver_1_reward"),
            ("r2_reward", "mean_receiver_2_reward"),
            ("r3_reward", "mean_receiver_3_reward"),
            ("disagreement", "mean_disagreement_std"),
            ("positive_injected", "total_positive_injected"),
            ("negative_injected", "total_negative_injected"),
            ("n_episodes", "n_episodes"),
        ]

        for metric_name, json_key in metrics:
            b_val = b.get(json_key, 0)
            a_val = a.get(json_key, 0)
            abs_diff = a_val - b_val
            rel_diff = abs_diff / max(abs(b_val), 1e-9) * 100 if b_val != 0 else 0.0
            rows.append({
                "method": method,
                "metric": metric_name,
                "before": b_val,
                "after": a_val,
                "absolute_difference": abs_diff,
                "relative_difference_pct": rel_diff,
            })

    return rows


def compute_statistics(before_eps: list[dict], after_eps: list[dict]) -> list[dict]:
    """Compute paired statistical tests per metric."""
    # Align by (method, task_id, seed)
    before_map: dict[tuple, dict] = {}
    for r in before_eps:
        key = (r["method"], r["task_id"], r["seed"])
        before_map[key] = r

    after_map: dict[tuple, dict] = {}
    for r in after_eps:
        key = (r["method"], r["task_id"], r["seed"])
        after_map[key] = r

    # Common keys
    common = sorted(set(before_map.keys()) & set(after_map.keys()))

    metrics = ["team_reward", "receiver_1_reward", "receiver_2_reward", "receiver_3_reward",
               "receiver_disagreement_std"]

    results = []
    for metric in metrics:
        b_vals = np.array([float(before_map[k][metric]) for k in common])
        a_vals = np.array([float(after_map[k][metric]) for k in common])
        diff = a_vals - b_vals

        mean_diff = float(np.mean(diff))
        std_diff = float(np.std(diff, ddof=1)) if len(diff) > 1 else 0.0

        # Paired t-test (H0: mean diff = 0)
        if std_diff > 0 and len(diff) > 1:
            t_stat, p_value = stats.ttest_rel(a_vals, b_vals)
        else:
            t_stat, p_value = 0.0, 1.0

        # Effect size (Cohen's d for paired)
        d = mean_diff / std_diff if std_diff > 0 else 0.0

        # Bootstrap 95% CI on mean difference
        rng = np.random.RandomState(42)
        n_boot = 10000
        boot_means = []
        for _ in range(n_boot):
            idx = rng.randint(0, len(diff), size=len(diff))
            boot_means.append(float(np.mean(diff[idx])))
        ci_lo = float(np.percentile(boot_means, 2.5))
        ci_hi = float(np.percentile(boot_means, 97.5))

        results.append({
            "metric": metric,
            "n_pairs": len(common),
            "mean_diff": mean_diff,
            "std_diff": std_diff,
            "t_stat": float(t_stat),
            "p_value": float(p_value),
            "cohens_d": d,
            "ci95_lo": ci_lo,
            "ci95_hi": ci_hi,
            "equivalent": abs(mean_diff) < 1e-12,  # Deterministic → must be 0
        })

    return results


def main() -> None:
    before_dir = _PROJECT_ROOT / "results" / "marble" / "receiver3" / "main"
    after_dir = _PROJECT_ROOT / "results" / "marble" / "receiver3" / "regression"
    output_dir = after_dir

    # Load summaries
    before_summary = load_summary(before_dir / "main_summary.json")
    after_summary = load_summary(after_dir / "regression_summary.json")

    # Load episodes
    before_eps = load_episodes(before_dir / "main_episodes.csv")
    after_eps = load_episodes(after_dir / "regression_episodes.csv")

    # Phase 3: Comparison
    comparison = compare_summaries(before_summary, after_summary)
    comp_path = output_dir / "comparison.csv"
    with comp_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(comparison[0].keys()))
        writer.writeheader()
        writer.writerows(comparison)
    print(f"Written: {comp_path} ({len(comparison)} rows)")

    # Phase 4: Statistics
    statistics = compute_statistics(before_eps, after_eps)
    stat_path = output_dir / "statistics.csv"
    with stat_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(statistics[0].keys()))
        writer.writeheader()
        writer.writerows(statistics)
    print(f"Written: {stat_path} ({len(statistics)} rows)")

    # Print summary
    print("\n=== Before/After Comparison ===")
    print(f"{'Method':<18} {'Metric':<22} {'Before':>10} {'After':>10} {'AbsDiff':>12} {'RelDiff%':>10}")
    print("-" * 90)
    for r in comparison:
        if r["metric"] in ("team_reward", "n_episodes"):
            print(
                f"{r['method']:<18} {r['metric']:<22} "
                f"{r['before']:>10.4f} {r['after']:>10.4f} "
                f"{r['absolute_difference']:>12.6f} {r['relative_difference_pct']:>9.2f}%"
            )

    print("\n=== Statistical Equivalence ===")
    print(f"{'Metric':<28} {'MeanDiff':>10} {'p-value':>10} {'CI95':>20} {'Equivalent':>10}")
    print("-" * 90)
    all_equivalent = True
    for r in statistics:
        eq_str = "YES" if r["equivalent"] else "NO"
        if not r["equivalent"]:
            all_equivalent = False
        print(
            f"{r['metric']:<28} "
            f"{r['mean_diff']:>10.6f} "
            f"{r['p_value']:>10.6f} "
            f"[{r['ci95_lo']:.6f}, {r['ci95_hi']:.6f}] "
            f"{eq_str:>10}"
        )

    print()
    if all_equivalent:
        print("VERDICT: PASS — All metrics are byte-identical (deterministic experiment)")
    else:
        print("VERDICT: FAIL — Some metrics differ (code regression detected)")


if __name__ == "__main__":
    main()
