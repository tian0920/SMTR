"""Generate baseline performance table and ranking CSV.

Reads results from ``results/baseline_comparison/formation/`` (produced
by ``run_lifelong.py`` with all baselines) and generates:

  - ``paper/tables/table_baseline_performance.tex``
  - ``results/baseline_ranking.csv``

Columns:
  Method | Final Reward | Average Reward | Late-stage Reward | Memory Size

Rows:
  Full Memory | Retrieval | Reflexion | AGILE | Heuristic | AgeMem | SMTR
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))

METHOD_LABELS: dict[str, str] = {
    "full_memory": "Full Memory",
    "retrieval": "Retrieval",
    "reflexion": "Reflexion",
    "agile": "AGILE-inspired",
    "heuristic": "Heuristic",
    "agemem": "AgeMem-inspired",
    "smtr_tci": "SMTR-TCI",
}

ROW_ORDER = [
    "full_memory", "retrieval", "reflexion", "agile",
    "heuristic", "agemem", "smtr_tci",
]


def load_performance(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def compute_method_metrics(rows: list[dict]) -> dict[str, dict]:
    """Aggregate per-method metrics across seeds."""
    grouped: dict[str, dict[str, list]] = defaultdict(
        lambda: {"rewards": [], "episodes": [], "n_stored": []}
    )
    for row in rows:
        m = row["method"]
        grouped[m]["rewards"].append(float(row["reward"]))
        grouped[m]["episodes"].append(int(row["episode"]))
        grouped[m]["n_stored"].append(int(row["n_stored"]))

    metrics: dict[str, dict] = {}
    for method, data in grouped.items():
        rewards = np.array(data["rewards"])
        episodes = np.array(data["episodes"])
        n_stored = np.array(data["n_stored"])

        avg_reward = float(np.mean(rewards))
        final_reward = avg_reward  # mean over all episodes × seeds

        # Late-stage: last 20% of unique episodes
        unique_eps = sorted(set(episodes))
        cutoff_idx = max(1, int(len(unique_eps) * 0.8))
        cutoff_ep = unique_eps[cutoff_idx - 1]
        late_mask = episodes >= cutoff_ep
        late_reward = float(np.mean(rewards[late_mask])) if late_mask.any() else avg_reward

        # Memory size: max n_stored across episodes (final bank size)
        memory_size = int(np.max(n_stored)) if len(n_stored) > 0 else 0

        metrics[method] = {
            "final_reward": final_reward,
            "average_reward": avg_reward,
            "late_reward": late_reward,
            "memory_size": memory_size,
        }
    return metrics


def generate_latex_table(metrics: dict[str, dict]) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\caption{Baseline memory controller performance comparison.}",
        r"\label{tab:baseline_performance}",
        r"\centering",
        r"\begin{tabular}{l c c c c}",
        r"\toprule",
        r"Method & Final Reward & Average Reward & Late-stage Reward & Memory Size \\",
        r"\midrule",
    ]
    for method_key in ROW_ORDER:
        label = METHOD_LABELS.get(method_key, method_key)
        m = metrics.get(method_key, {})
        fr = f"{m['final_reward']:.3f}" if "final_reward" in m else "---"
        ar = f"{m['average_reward']:.3f}" if "average_reward" in m else "---"
        lr = f"{m['late_reward']:.3f}" if "late_reward" in m else "---"
        ms = str(m.get("memory_size", "---"))
        lines.append(f"{label} & {fr} & {ar} & {lr} & {ms} \\\\")
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines) + "\n"


def generate_ranking_csv(metrics: dict[str, dict]) -> list[dict]:
    """Sort methods by final_reward descending."""
    rows = []
    for method_key in ROW_ORDER:
        m = metrics.get(method_key, {})
        rows.append({
            "rank": 0,
            "method": METHOD_LABELS.get(method_key, method_key),
            "method_key": method_key,
            "final_reward": m.get("final_reward", 0.0),
            "average_reward": m.get("average_reward", 0.0),
            "late_reward": m.get("late_reward", 0.0),
            "memory_size": m.get("memory_size", 0),
        })
    rows.sort(key=lambda r: r["final_reward"], reverse=True)
    for i, row in enumerate(rows, 1):
        row["rank"] = i
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results", default="results/baseline_comparison/formation",
        help="Path to formation experiment directory.",
    )
    parser.add_argument(
        "--output-table", default="paper/tables/table_baseline_performance.tex",
    )
    parser.add_argument(
        "--output-ranking", default="results/baseline_ranking.csv",
    )
    args = parser.parse_args()

    results_dir = Path(args.results)
    perf_path = results_dir / "performance.csv"
    if not perf_path.exists():
        print(f"ERROR: {perf_path} not found. Run the benchmark first.")
        sys.exit(1)

    rows = load_performance(perf_path)
    metrics = compute_method_metrics(rows)

    # LaTeX table
    table_path = Path(args.output_table)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.write_text(generate_latex_table(metrics))
    print(f"Wrote {table_path}")

    # Ranking CSV
    ranking_path = Path(args.output_ranking)
    ranking_path.parent.mkdir(parents=True, exist_ok=True)
    ranking_rows = generate_ranking_csv(metrics)
    with ranking_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(ranking_rows[0].keys()))
        writer.writeheader()
        writer.writerows(ranking_rows)
    print(f"Wrote {ranking_path}")

    # Print summary
    print("\nPerformance Ranking:")
    for row in ranking_rows:
        print(f"  #{row['rank']} {row['method']:<20} "
              f"final={row['final_reward']:.3f}  "
              f"late={row['late_reward']:.3f}  "
              f"mem_size={row['memory_size']}")


if __name__ == "__main__":
    main()
