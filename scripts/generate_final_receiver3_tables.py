"""Generate Final Receiver=3 Paper Tables (5 Domains, Clean Run).

Reads from results/marble/receiver3/ and results/receiver3/ (permutation).
Outputs to paper/tables/final_receiver3/.

Tables:
  1. Five-domain main performance
  2. Per-domain breakdown
  3. Contamination propagation
  4. Permutation test
  5. Cost-benefit analysis

All tables note: receiver=3, 5 domains, 5 seeds, clean run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))

OUTPUT_DIR = _PROJECT_ROOT / "paper" / "tables" / "final_receiver3"
RESULTS_DIR = _PROJECT_ROOT / "results" / "marble" / "receiver3"
PERM_DIR = _PROJECT_ROOT / "results" / "receiver3"

FOOTNOTE = (
    r"% receiver=3, 5 domains (bargaining, coding, database, minecraft, research),"
    r" 5 seeds, clean run"
)


def _table_header(caption: str, label: str, ncols: int) -> list[str]:
    return [
        r"\begin{table}[t]",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\centering",
        r"\begin{tabular}{l " + " ".join(["c"] * (ncols - 1)) + "}",
        r"\toprule",
    ]


def _table_footer() -> list[str]:
    return [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]


def generate_table1_main() -> str:
    """Table 1: Five-domain main performance."""
    summary_path = RESULTS_DIR / "main" / "main_summary.json"
    with summary_path.open() as f:
        data = json.load(f)
    summary = data["global"]

    lines = _table_header(
        "MARBLE Receiver=3 Main Results (5 Domains, Synthetic Data, Clean Run)",
        "tab:final_receiver3_main",
        8,
    )
    lines.append(
        r"Method & Episodes & Team Reward & R1 & R2 & R3 & Pos. Inj. & Neg. Inj. \\"
    )
    lines.append(r"\midrule")

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
            f"{s['total_positive_injected']} & "
            f"{s['total_negative_injected']} \\\\"
        )

    lines.extend(_table_footer())
    return "\n".join(lines)


def generate_table2_per_domain() -> str:
    """Table 2: Per-domain team reward breakdown."""
    summary_path = RESULTS_DIR / "main" / "main_summary.json"
    with summary_path.open() as f:
        data = json.load(f)
    per_scenario = data["per_scenario"]

    scenarios = sorted(per_scenario.keys())
    methods_order = [
        "no_memory", "full_memory", "retrieval",
        "smtr_uniform", "smtr_receiver",
    ]

    lines = _table_header(
        "Per-Domain Team Reward (5 MARBLE Domains, Clean Run)",
        "tab:final_receiver3_per_domain",
        len(scenarios) + 2,
    )
    header = "Method & " + " & ".join(s.capitalize() for s in scenarios) + r" & Avg \\"
    lines.append(header)
    lines.append(r"\midrule")

    for method in methods_order:
        name = method.replace("_", r"\_")
        vals = []
        for sc in scenarios:
            entry = next(
                (s for s in per_scenario[sc] if s["method"] == method), None
            )
            if entry and entry["n_episodes"] > 0:
                vals.append(entry["mean_team_reward"])
            else:
                vals.append(0.0)
        avg = float(np.mean(vals))
        cells = " & ".join(f"{v:.4f}" for v in vals)
        lines.append(f"{name} & {cells} & {avg:.4f} \\\\")

    lines.extend(_table_footer())
    return "\n".join(lines)


def generate_table3_contamination() -> str:
    """Table 3: Contamination propagation."""
    summary_path = RESULTS_DIR / "contamination" / "contamination_summary.json"
    if not summary_path.exists():
        return "% Contamination table: data not available"

    with summary_path.open() as f:
        summary = json.load(f)

    lines = _table_header(
        "Contamination Propagation (Receiver=3, 5 Domains, Clean Run)",
        "tab:final_receiver3_contamination",
        5,
    )
    lines.append(r"Method & Ratio & Reward & Contam. Rate & Prop. Depth \\")
    lines.append(r"\midrule")

    for ratio in [0.1, 0.2, 0.3]:
        for method in [
            "full_memory", "retrieval", "smtr_uniform", "smtr_receiver"
        ]:
            entries = [
                s for s in summary
                if s["ratio"] == ratio and s["method"] == method
            ]
            if not entries:
                continue
            mean_reward = float(
                np.mean([s["mean_team_reward"] for s in entries])
            )
            mean_contam = float(
                np.mean([s["mean_contamination_rate"] for s in entries])
            )
            mean_prop = float(
                np.mean([s["mean_propagation_depth"] for s in entries])
            )
            name = method.replace("_", r"\_")
            lines.append(
                f"{name} & {ratio:.1f} & {mean_reward:.4f} & "
                f"{mean_contam:.4f} & {mean_prop:.4f} \\\\"
            )
        if ratio < 0.3:
            lines.append(r"\midrule")

    lines.extend(_table_footer())
    return "\n".join(lines)


def generate_table4_permutation() -> str:
    """Table 4: Permutation test results."""
    summary_path = PERM_DIR / "permutation_summary.json"
    if not summary_path.exists():
        return "% Permutation table: data not available"

    with summary_path.open() as f:
        results = json.load(f)

    true_agg = results["smtr_receiver"]
    perm_mean = results["smtr_permuted_mean"]
    perm_std = results["smtr_permuted_std"]
    p_value_str = results.get(
        "reward_drop_p_value_str",
        f"{results['reward_drop_p_value']:.2e}",
    )

    lines = _table_header(
        "Receiver Permutation Test (5 Domains, Clean Run): "
        "receiver identity is causally necessary",
        "tab:final_receiver_permutation",
        5,
    )
    lines.append(
        r"Condition & Team Reward & Pos. Inj. & Neg. Inj. & Decision Align. \\"
    )
    lines.append(r"\midrule")
    lines.append(
        f"SMTR-receiver (true identity) & {true_agg['team_reward']:.4f} & "
        f"{true_agg['n_positive']} & {true_agg['n_negative']} & "
        f"{true_agg['decision_alignment']:.4f} \\\\"
    )
    lines.append(
        f"SMTR-permuted ($n$={results['n_permutations']}, mean) & "
        f"{perm_mean['team_reward']:.4f} $\\pm$ {perm_std['team_reward']:.4f} & "
        f"-- & {perm_mean.get('n_negative', 0):.1f} & "
        f"{perm_mean['decision_alignment']:.4f} $\\pm$ "
        f"{perm_std['decision_alignment']:.4f} \\\\"
    )
    lines.append(r"\midrule")
    lines.append(
        f"Reward drop & \\multicolumn{{4}}{{c}}{{{results['reward_drop']:+.4f} "
        f"($p = {p_value_str}$, paired permutation test)}} \\\\"
    )

    lines.extend(_table_footer())
    return "\n".join(lines)


def generate_table5_cost() -> str:
    """Table 5: Cost-benefit analysis."""
    summary_path = RESULTS_DIR / "cost" / "cost_summary.json"
    if not summary_path.exists():
        return "% Cost table: data not available"

    with summary_path.open() as f:
        costs = json.load(f)

    u = costs["smtr_uniform"]
    r = costs["smtr_receiver"]

    lines = _table_header(
        "Validation Cost-Benefit Analysis (5 Domains, Clean Run)",
        "tab:final_receiver3_cost",
        3,
    )
    lines.append(r"Metric & SMTR-Uniform & SMTR-Receiver \\")
    lines.append(r"\midrule")

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
    lines.append(
        f"Cost multiplier & 1$\\times$ & {cost_ratio:.1f}$\\times$ \\\\"
    )

    reward_gain = (
        (r["mean_team_reward"] - u["mean_team_reward"])
        / max(abs(u["mean_team_reward"]), 1e-9)
        * 100
    )
    lines.append(f"Reward gain & -- & {reward_gain:+.1f}\\% \\\\")

    lines.extend(_table_footer())
    return "\n".join(lines)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tables = {
        "table1_main": generate_table1_main(),
        "table2_per_domain": generate_table2_per_domain(),
        "table3_contamination": generate_table3_contamination(),
        "table4_permutation": generate_table4_permutation(),
        "table5_cost": generate_table5_cost(),
    }

    for name, content in tables.items():
        path = OUTPUT_DIR / f"{name}.tex"
        path.write_text(content + "\n")
        print(f"Written: {path}")

    # Combined file
    combined_lines = [
        FOOTNOTE,
        r"% Auto-generated by scripts/generate_final_receiver3_tables.py",
        r"% receiver=3, 5 domains, 5 seeds, clean run",
        r"",
    ]
    combined_lines.extend(tables.values())
    combined_path = OUTPUT_DIR / "all_final_receiver3_tables.tex"
    combined_path.write_text("\n\n".join(combined_lines) + "\n")
    print(f"Written: {combined_path}")

    # Print summary
    print()
    print("=== Final Receiver=3 Tables Summary ===")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Tables: {len(tables)}")
    print(f"  Config: receiver=3, 5 domains, 5 seeds, clean run")
    print(f"  Data source: synthetic paired records")


if __name__ == "__main__":
    main()
