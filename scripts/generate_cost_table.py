"""Generate LaTeX table for TCI cost-benefit analysis.

Output: paper/tables/table_cost_analysis.tex
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results/cost_analysis")
    parser.add_argument("--output", default="paper/tables")
    args = parser.parse_args()

    results_dir = Path(args.results)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = results_dir / "cost_results.csv"
    with csv_path.open() as f:
        results = list(csv.DictReader(f))

    method_labels = {
        "full_memory": "Full Memory",
        "retrieval": "Retrieval",
        "random_validation": "Random Validation",
        "smtr_tci": "\\textbf{SMTR-TCI}",
    }

    rows_tex: list[str] = []
    for method in ["full_memory", "retrieval", "random_validation", "smtr_tci"]:
        rows = [r for r in results if r["method"] == method]
        if not rows:
            continue
        reward = np.mean([float(r["final_reward"]) for r in rows])
        reward_std = np.std([float(r["final_reward"]) for r in rows])
        interventions = np.mean([int(r["intervention_count"]) for r in rows])
        mem_size = np.mean([int(r["total_memories"]) for r in rows])
        contam = np.mean([float(r["contamination_rate"]) for r in rows])

        rows_tex.append(
            f"{method_labels[method]} & {interventions:.0f} & "
            f"{reward:.3f} $\\pm$ {reward_std:.3f} & "
            f"{mem_size:.0f} & {contam:.3f} \\\\"
        )

    tex = (
        "\\begin{table}[h]\n"
        "\\centering\n"
        "\\caption{Cost-Benefit Analysis of Causal Validation}\n"
        "\\label{tab:cost_analysis}\n"
        "\\begin{tabular}{lcccc}\n"
        "\\toprule\n"
        "Method & Interventions & Final Reward & Memory Size & Contam. Rate \\\\\n"
        "\\midrule\n"
        + "\n".join(rows_tex) + "\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )

    out_path = output_dir / "table_cost_analysis.tex"
    out_path.write_text(tex)
    print(f"Saved: {out_path}")
    print(tex)


if __name__ == "__main__":
    main()
