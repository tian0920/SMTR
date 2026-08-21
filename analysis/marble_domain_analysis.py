"""MARBLE domain-wise analysis (Task A1).

Groups MARBLE tasks by agent count (complexity) to demonstrate that
SMTR's improvement is stable across different multi-agent configurations.

Domains (by agent count):
  solo:       1-2 agents
  small:      3 agents
  medium:     4-5 agents
  large:      6 agents
  complex:    7+ agents

Output: results/marble/domain_analysis/domain_wise_results.csv
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))

DOMAIN_DEFS: list[tuple[str, int, int]] = [
    ("solo", 1, 2),
    ("small", 3, 3),
    ("medium", 4, 5),
    ("large", 6, 6),
    ("complex", 7, 99),
]


def agent_count_to_domain(agent_count: int) -> str:
    for name, lo, hi in DOMAIN_DEFS:
        if lo <= agent_count <= hi:
            return name
    return "complex"


def load_task_info() -> dict[str, dict]:
    """Load task manifest for agent_count mapping."""
    manifest_path = _PROJECT_ROOT / "artifacts" / "marble" / "manifests" / "dataset.json"
    manifest = json.loads(manifest_path.read_text())
    return {
        str(t["task_id"]): {
            "agent_count": t.get("agent_count", 0),
            "scenario": t.get("scenario", ""),
            "domain": agent_count_to_domain(t.get("agent_count", 0)),
        }
        for t in manifest.get("tasks", [])
    }


def load_baseline_results(path: Path) -> list[dict]:
    """Load baseline_results.csv."""
    rows: list[dict] = []
    with path.open() as f:
        for row in csv.DictReader(f):
            row["method_reward"] = float(row["method_reward"])
            row["withhold_reward"] = float(row["withhold_reward"])
            row["n_injected"] = float(row["n_injected"])
            row["n_positive"] = float(row["n_positive"])
            row["n_harmful"] = float(row["n_harmful"])
            row["total_tau"] = float(row["total_tau"])
            row["seed"] = int(row["seed"])
            rows.append(row)
    return rows


def domain_analysis(
    results: list[dict],
    task_info: dict[str, dict],
) -> list[dict]:
    """Compute per-domain, per-method statistics."""
    # Enrich results with domain
    enriched: list[dict] = []
    for r in results:
        tid = r["task_id"]
        info = task_info.get(tid, {})
        domain = info.get("domain", "unknown")
        enriched.append({**r, "domain": domain, "agent_count": info.get("agent_count", 0)})

    rows: list[dict] = []
    for domain, _, _ in DOMAIN_DEFS:
        for method in sorted(set(r["method"] for r in enriched)):
            d_rows = [r for r in enriched if r["domain"] == domain and r["method"] == method]
            if not d_rows:
                continue

            rewards = [r["method_reward"] for r in d_rows]
            injected = [r["n_injected"] for r in d_rows]
            positive = [r["n_positive"] for r in d_rows]
            harmful = [r["n_harmful"] for r in d_rows]

            # Per-seed aggregation for late-stage proxy
            seed_rewards: dict[int, list[float]] = defaultdict(list)
            for r in d_rows:
                seed_rewards[r["seed"]].append(r["method_reward"])
            per_seed_mean = [float(np.mean(v)) for v in seed_rewards.values()]

            # Late-stage proxy: last seed's performance (seeds are chronological)
            seeds_sorted = sorted(seed_rewards.keys())
            late_reward = float(np.mean(seed_rewards.get(seeds_sorted[-1], [0]))) if seeds_sorted else 0.0
            early_reward = float(np.mean(seed_rewards.get(seeds_sorted[0], [0]))) if seeds_sorted else 0.0

            # Contamination rate: fraction of harmful injections
            total_inj = sum(injected)
            total_harm = sum(harmful)
            contamination_rate = total_harm / max(total_inj, 1)

            # Memory efficiency: reward per injected memory
            mean_inj = float(np.mean(injected))
            mem_efficiency = float(np.mean(rewards)) / max(mean_inj, 0.01)

            rows.append({
                "domain": domain,
                "method": method,
                "n_groups": len(d_rows),
                "mean_reward": float(np.mean(rewards)),
                "std": float(np.std(rewards)),
                "early_reward": early_reward,
                "late_reward": late_reward,
                "memory_count": float(np.mean(injected)),
                "positive_rate": float(np.mean(positive)),
                "harmful_rate": float(np.mean(harmful)),
                "contamination": contamination_rate,
                "memory_efficiency": mem_efficiency,
                "mean_tau": float(np.mean([r["total_tau"] for r in d_rows])),
            })

    return rows


def main() -> None:
    results_path = _PROJECT_ROOT / "results" / "marble" / "main" / "baseline_results.csv"
    if not results_path.exists():
        print(f"ERROR: {results_path} not found")
        return

    task_info = load_task_info()
    results = load_baseline_results(results_path)

    rows = domain_analysis(results, task_info)

    # Write output
    output_dir = _PROJECT_ROOT / "results" / "marble" / "domain_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "domain_wise_results.csv"
    if rows:
        fieldnames = list(rows[0].keys())
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    # Print summary
    print("=== MARBLE Domain-wise Analysis ===")
    print(f"{'Domain':<10} {'Method':<15} {'N':>5} {'Reward':>8} {'Std':>8} {'Late':>8} {'MemEff':>8} {'Contam':>8}")
    print("-" * 80)
    for domain, _, _ in DOMAIN_DEFS:
        d_rows = [r for r in rows if r["domain"] == domain]
        for r in d_rows:
            print(f"{r['domain']:<10} {r['method']:<15} {r['n_groups']:>5} "
                  f"{r['mean_reward']:>8.4f} {r['std']:>8.4f} "
                  f"{r['late_reward']:>8.4f} {r['memory_efficiency']:>8.4f} "
                  f"{r['contamination']:>8.4f}")
        print()

    # SMTR win count per domain
    print("=== SMTR Win Count ===")
    for domain, _, _ in DOMAIN_DEFS:
        d_rows = [r for r in rows if r["domain"] == domain]
        if not d_rows:
            continue
        smtr = next((r for r in d_rows if r["method"] == "smtr_tci"), None)
        baselines = [r for r in d_rows if r["method"] != "smtr_tci"]
        if smtr and baselines:
            best_bl = max(baselines, key=lambda r: r["mean_reward"])
            delta = smtr["mean_reward"] - best_bl["mean_reward"]
            win = "YES" if delta > 0 else "NO"
            print(f"  {domain}: SMTR={smtr['mean_reward']:.4f} vs "
                  f"best({best_bl['method']})={best_bl['mean_reward']:.4f} "
                  f"delta={delta:+.4f} [{win}]")

    print(f"\nWritten: {csv_path}")


if __name__ == "__main__":
    main()
