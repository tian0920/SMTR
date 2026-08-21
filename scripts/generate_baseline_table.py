"""Generate paper-ready baseline comparison LaTeX table.

Reads results from ``results/baselines/`` (one sub-directory per
``--memory_controller`` run) and produces
``paper/tables/table_baselines.tex``.

Columns:
  Method | Memory Type | Selection Principle | Final Reward | Late Reward | Transfer | Contamination

Rows:
  Full Memory | Retrieval | Reflexion | AGILE | Heuristic | AgeMem | SMTR

Usage::

    python scripts/generate_baseline_table.py
    python scripts/generate_baseline_table.py --results-dir results/baselines
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Method metadata (static — does not depend on run results)
# ---------------------------------------------------------------------------
METHOD_INFO: dict[str, dict[str, str]] = {
    "full_memory": {
        "label": "Full Memory",
        "memory_type": "Verbatim experience",
        "selection": "All stored memories",
    },
    "retrieval": {
        "label": "Retrieval",
        "memory_type": "Verbatim experience",
        "selection": "Top-k topic match",
    },
    "reflexion": {
        "label": "Reflexion",
        "memory_type": "Verbal reflection",
        "selection": "All reflections (topic filter)",
    },
    "agile": {
        "label": "AGILE",
        "memory_type": "Consolidated experience",
        "selection": "Experience-score ranked",
    },
    "heuristic": {
        "label": "Heuristic",
        "memory_type": "Importance-scored",
        "selection": "Recency + usage + retrieval",
    },
    "agemem": {
        "label": "AgeMem",
        "memory_type": "Managed (ADD/DELETE/COMPRESS)",
        "selection": "Frozen learned-style policy",
    },
    "smtr_tci": {
        "label": "SMTR",
        "memory_type": "TCI-validated knowledge",
        "selection": "Causal-utility gated",
    },
}


# ---------------------------------------------------------------------------
# Result aggregation
# ---------------------------------------------------------------------------
def _load_performance(results_dir: Path, controller: str) -> dict[str, dict]:
    """Load performance.csv from a baseline run directory."""
    perf_path = results_dir / controller / "formation" / "performance.csv"
    if not perf_path.exists():
        return {}
    rows: list[dict] = []
    with perf_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    # Aggregate by method
    methods: dict[str, dict] = {}
    for row in rows:
        method = row["method"]
        methods.setdefault(method, {"rewards": [], "episodes": []})
        methods[method]["rewards"].append(float(row["reward"]))
        methods[method]["episodes"].append(int(row["episode"]))
    return methods


def _compute_metrics(method_data: dict) -> dict[str, str]:
    """Compute Final Reward, Late Reward from raw episode data."""
    rewards = method_data["rewards"]
    episodes = method_data["episodes"]
    if not rewards:
        return {"final_reward": "---", "late_reward": "---"}
    # Final reward = mean over all episodes
    final = sum(rewards) / len(rewards)
    # Late reward = mean over last 20% of episodes
    n = len(rewards)
    cutoff = max(1, int(n * 0.8))
    late_rewards = [r for r, e in zip(rewards, episodes) if e >= cutoff]
    late = sum(late_rewards) / len(late_rewards) if late_rewards else final
    return {
        "final_reward": f"{final:.3f}",
        "late_reward": f"{late:.3f}",
    }


# ---------------------------------------------------------------------------
# Table generation
# ---------------------------------------------------------------------------
def generate_table(results_dir: Path, output_path: Path) -> None:
    """Build the LaTeX table from all available baseline results."""
    # Collect results per controller directory
    all_results: dict[str, dict[str, dict]] = {}
    if results_dir.exists():
        for controller_dir in sorted(results_dir.iterdir()):
            if controller_dir.is_dir():
                all_results[controller_dir.name] = _load_performance(
                    results_dir, controller_dir.name
                )

    # Ordered rows
    row_order = [
        "full_memory", "retrieval", "reflexion", "agile",
        "heuristic", "agemem", "smtr_tci",
    ]

    lines = [
        r"\begin{table}[t]",
        r"\caption{Baseline memory controller comparison.}",
        r"\label{tab:baselines}",
        r"\centering",
        r"\begin{tabular}{l l l c c c c}",
        r"\toprule",
        r"Method & Memory Type & Selection Principle & Final Reward & Late Reward & Transfer & Contamination \\",
        r"\midrule",
    ]

    for method_key in row_order:
        info = METHOD_INFO[method_key]
        # Find this method in any controller's results
        metrics = {"final_reward": "---", "late_reward": "---"}
        for controller_results in all_results.values():
            if method_key in controller_results:
                metrics = _compute_metrics(controller_results[method_key])
                break

        line = (
            f"{info['label']} & {info['memory_type']} & {info['selection']} "
            f"& {metrics['final_reward']} & {metrics['late_reward']} "
            f"& --- & --- \\\\"
        )
        lines.append(line)

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        default="results/baselines",
        help="Directory containing baseline run results.",
    )
    parser.add_argument(
        "--output",
        default="paper/tables/table_baselines.tex",
        help="Output LaTeX file path.",
    )
    args = parser.parse_args()
    generate_table(Path(args.results_dir), Path(args.output))


if __name__ == "__main__":
    main()
