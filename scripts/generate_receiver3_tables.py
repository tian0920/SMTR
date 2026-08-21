"""Generate Receiver=3 Paper Tables (Phase 10).

Produces LaTeX tables for the paper:
  - Table 1: MARBLE receiver=3 main results
  - Table 2: Receiver-conditioned transfer analysis
  - Table 3: Contamination propagation
  - Table 4: Cost analysis

Output: paper/tables/receiver3/
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))

OUTPUT_DIR = _PROJECT_ROOT / "paper" / "tables" / "receiver3"
RESULTS_DIR = _PROJECT_ROOT / "results" / "marble" / "receiver3"


def generate_table1_main_results() -> str:
    """Table 1: MARBLE receiver=3 main results."""
    summary_path = RESULTS_DIR / "main" / "main_summary.json"
    with summary_path.open() as f:
        summary = json.load(f)

    lines = [
        r"\begin{table}[t]",
        r"\caption{MARBLE Receiver=3 Main Results}",
        r"\label{tab:receiver3_main}",
        r"\centering",
        r"\begin{tabular}{l c c c c c c}",
        r"\toprule",
        r"Method & Episodes & Team Reward & R1 & R2 & R3 & Neg. Inj. \\",
        r"\midrule",
    ]

    for s in summary:
        name = s["method"].replace("_", r"\_")
        if s["n_episodes"] == 0:
            continue
        lines.append(
            f"{name} & {s['n_episodes']} & "
            f"{s['mean_team_reward']:.4f} $\\pm$ {s['std_team_reward']:.4f} & "
            f"{s['mean_receiver_1_reward']:.4f} & "
            f"{s['mean_receiver_2_reward']:.4f} & "
            f"{s['mean_receiver_3_reward']:.4f} & "
            f"{s['total_negative_injected']} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def generate_table2_receiver_conditioned() -> str:
    """Table 2: Receiver-conditioned transfer analysis."""
    summary_path = RESULTS_DIR / "main" / "receiver_conditioned_summary.json"
    with summary_path.open() as f:
        summary = json.load(f)

    sel = summary["selective_transfer"]
    total = summary["total_memories"]

    lines = [
        r"\begin{table}[t]",
        r"\caption{Receiver-Conditioned Transfer Analysis}",
        r"\label{tab:receiver_conditioned}",
        r"\centering",
        r"\begin{tabular}{l c}",
        r"\toprule",
        r"Metric & Value \\",
        r"\midrule",
        f"Total memories analyzed & {total} \\\\",
        f"Memories with disagreement & {summary['disagreement_count']} ({summary['disagreement_count']/max(total,1)*100:.1f}\\%) \\\\",
        f"Mean disagreement rate & {summary['disagreement_rate']:.4f} \\\\",
        f"Mean $\\Delta$ variance & {summary['mean_delta_variance']:.6f} \\\\",
        r"\midrule",
        r"\multicolumn{2}{l}{\textit{Selective transfer (k-of-3 receivers useful)}} \\",
        f"  k=0 (none) & {sel.get('0', 0)} ({sel.get('0', 0)/max(total,1)*100:.1f}\\%) \\\\",
        f"  k=1 & {sel.get('1', 0)} ({sel.get('1', 0)/max(total,1)*100:.1f}\\%) \\\\",
        f"  k=2 & {sel.get('2', 0)} ({sel.get('2', 0)/max(total,1)*100:.1f}\\%) \\\\",
        f"  k=3 (all) & {sel.get('3', 0)} ({sel.get('3', 0)/max(total,1)*100:.1f}\\%) \\\\",
        r"\midrule",
        r"\multicolumn{2}{l}{\textit{Decision accuracy}} \\",
        f"  Uniform TCI & {summary['uniform_accuracy']*100:.1f}\\% \\\\",
        f"  Receiver-conditioned TCI & {summary['receiver_accuracy']*100:.1f}\\% \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def generate_table3_contamination() -> str:
    """Table 3: Contamination propagation results."""
    summary_path = RESULTS_DIR / "contamination" / "contamination_summary.json"
    with summary_path.open() as f:
        summary = json.load(f)

    lines = [
        r"\begin{table}[t]",
        r"\caption{Contamination Propagation (Receiver=3)}",
        r"\label{tab:receiver3_contamination}",
        r"\centering",
        r"\begin{tabular}{l c c c c}",
        r"\toprule",
        r"Method & Ratio & Reward & Contam. Rate & Prop. Depth \\",
        r"\midrule",
    ]

    # Group by ratio for readability
    for ratio in [0.1, 0.2, 0.3]:
        for method in ["full_memory", "retrieval", "smtr_uniform", "smtr_receiver"]:
            entries = [
                s for s in summary
                if s["ratio"] == ratio and s["method"] == method
            ]
            if not entries:
                continue
            # Average across contamination types
            mean_reward = float(np.mean([s["mean_team_reward"] for s in entries]))
            mean_contam = float(np.mean([s["mean_contamination_rate"] for s in entries]))
            mean_prop = float(np.mean([s["mean_propagation_depth"] for s in entries]))
            name = method.replace("_", r"\_")
            lines.append(
                f"{name} & {ratio:.1f} & {mean_reward:.4f} & "
                f"{mean_contam:.4f} & {mean_prop:.4f} \\\\"
            )
        if ratio < 0.3:
            lines.append(r"\midrule")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def generate_table4_cost() -> str:
    """Table 4: Cost analysis."""
    summary_path = RESULTS_DIR / "cost" / "cost_summary.json"
    with summary_path.open() as f:
        costs = json.load(f)

    lines = [
        r"\begin{table}[t]",
        r"\caption{Validation Cost Analysis}",
        r"\label{tab:receiver3_cost}",
        r"\centering",
        r"\begin{tabular}{l c c}",
        r"\toprule",
        r"Metric & SMTR-Uniform & SMTR-Receiver \\",
        r"\midrule",
    ]

    u = costs["smtr_uniform"]
    r = costs["smtr_receiver"]

    metrics = [
        ("Unique memories validated", "n_unique_memories", "d"),
        ("Receivers per validation", "n_receivers_validated", "d"),
        ("Total validations", "n_validations", "d"),
        ("Expose rollouts", "n_expose_rollouts", "d"),
        ("Withhold rollouts", "n_withhold_rollouts", "d"),
        ("Total rollouts", "total_rollouts", "d"),
        ("Mean team reward", "mean_team_reward", ".4f"),
        ("Reward / validation", "reward_per_validation", ".6f"),
        ("Knowledge quality", "knowledge_quality", ".4f"),
        ("Quality / validation", "quality_per_validation", ".6f"),
    ]

    for label, key, fmt in metrics:
        u_val = u[key]
        r_val = r[key]
        if fmt == "d":
            lines.append(f"{label} & {u_val} & {r_val} \\\\")
        else:
            lines.append(f"{label} & {u_val:{fmt}} & {r_val:{fmt}} \\\\")

    cost_ratio = r["n_validations"] / max(u["n_validations"], 1)
    lines.append(r"\midrule")
    lines.append(f"Cost multiplier & 1$\\times$ & {cost_ratio:.1f}$\\times$ \\\\")

    reward_gain = (r["mean_team_reward"] - u["mean_team_reward"]) / max(abs(u["mean_team_reward"]), 1e-9) * 100
    lines.append(f"Reward gain & -- & {reward_gain:+.1f}\\% \\\\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tables = {
        "table1_main": generate_table1_main_results(),
        "table2_receiver_conditioned": generate_table2_receiver_conditioned(),
        "table3_contamination": generate_table3_contamination(),
        "table4_cost": generate_table4_cost(),
    }

    for name, content in tables.items():
        path = OUTPUT_DIR / f"{name}.tex"
        path.write_text(content)
        print(f"Written: {path}")

    # Also write combined file
    combined = "\n\n".join([
        r"% Receiver=3 Paper Tables",
        r"% Auto-generated by scripts/generate_receiver3_tables.py",
        r"",
    ] + list(tables.values()))
    combined_path = OUTPUT_DIR / "all_receiver3_tables.tex"
    combined_path.write_text(combined)
    print(f"Written: {combined_path}")


if __name__ == "__main__":
    main()
