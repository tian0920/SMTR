"""P2-3: Automated Paper Table Generator.

Reads experiment results and generates LaTeX tables for the paper:

  Table 1: Long-term formation (lifelong benchmark)
  Table 2: Contamination resistance
  Table 3: Cross-task transfer
  Table 4: Multi-agent propagation
  Table 5: Retention rule ablation

Format: LaTeX booktabs style.
Output: paper/tables/table_*.tex
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────
def read_csv(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def agg(rows: list[dict], key: str) -> tuple[float, float]:
    vals = [float(r[key]) for r in rows if key in r and r[key] is not None]
    return float(np.mean(vals)), float(np.std(vals))


def fmt(mean: float, std: float, decimals: int = 3) -> str:
    return f"{mean:.{decimals}f} $\\pm$ {std:.{decimals}f}"


def write_tex(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    print(f"  → {path}")


# ──────────────────────────────────────────────────────────────────────
# Table 1: Long-term formation
# ──────────────────────────────────────────────────────────────────────
def table_formation(results_dir: Path, output_dir: Path) -> None:
    perf = read_csv(results_dir / "performance.csv")
    methods = ["no_memory", "full_memory", "retrieval", "smtr_tci"]
    method_labels = {
        "no_memory": "No Memory",
        "full_memory": "Full Memory",
        "retrieval": "Retrieval",
        "smtr_tci": "\\textbf{SMTR-TCI}",
    }

    rows_tex: list[str] = []
    for method in methods:
        method_rows = [r for r in perf if r["method"] == method]
        # Final reward (last 20%)
        seeds = sorted(set(int(r["seed"]) for r in method_rows))
        final_rewards = []
        cumulative_rewards = []
        late20_rewards = []
        for seed in seeds:
            seed_rows = sorted(
                [r for r in method_rows if int(r["seed"]) == seed],
                key=lambda r: int(r["episode"]),
            )
            n = len(seed_rows)
            final_rewards.append(float(seed_rows[-1]["cumulative_reward"]))
            cumulative_rewards.append(float(seed_rows[-1]["cumulative_reward"]))
            late = [float(r["reward"]) for r in seed_rows[int(n * 0.8):]]
            late20_rewards.append(float(np.mean(late)) if late else 0.0)

        fr_mean, fr_std = float(np.mean(late20_rewards)), float(np.std(late20_rewards))
        cr_mean, cr_std = float(np.mean(cumulative_rewards)), float(np.std(cumulative_rewards))
        rows_tex.append(
            f"{method_labels[method]} & {fmt(fr_mean, fr_std)} & "
            f"{cr_mean:.1f} $\\pm$ {cr_std:.1f} \\\\"
        )

    tex = (
        "\\begin{table}[h]\n"
        "\\centering\n"
        "\\caption{Long-term Knowledge Formation (100 episodes, 5 seeds)}\n"
        "\\label{tab:formation}\n"
        "\\begin{tabular}{lcc}\n"
        "\\toprule\n"
        "Method & Final Reward (last 20\\%) & Cumulative Reward \\\\\n"
        "\\midrule\n"
        + "\n".join(rows_tex) + "\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )
    write_tex(output_dir / "table_formation.tex", tex)


# ──────────────────────────────────────────────────────────────────────
# Table 2: Contamination resistance
# ──────────────────────────────────────────────────────────────────────
def table_contamination(results_dir: Path, output_dir: Path) -> None:
    # Read from contamination benchmark results
    cont_dir = results_dir / "contamination"
    if not (cont_dir / "contamination_results.csv").exists():
        print("  ⚠ contamination_results.csv not found, skipping Table 2")
        return
    rows = read_csv(cont_dir / "contamination_results.csv")
    methods = ["full_memory", "retrieval", "smtr_tci"]
    method_labels = {
        "full_memory": "Full Memory",
        "retrieval": "Retrieval",
        "smtr_tci": "\\textbf{SMTR-TCI}",
    }
    ratios = [0.1, 0.2, 0.3]

    rows_tex: list[str] = []
    for method in methods:
        for ratio in ratios:
            mr = [r for r in rows
                  if r["method"] == method and float(r.get("contamination_ratio", 0)) == ratio]
            if not mr:
                continue
            fr_mean, fr_std = agg(mr, "final_reward")
            rows_tex.append(
                f"{method_labels[method]} & {ratio:.1f} & "
                f"{fmt(fr_mean, fr_std)} \\\\"
            )

    tex = (
        "\\begin{table}[h]\n"
        "\\centering\n"
        "\\caption{Contamination Resistance (100 episodes, 5 seeds)}\n"
        "\\label{tab:contamination}\n"
        "\\begin{tabular}{lcc}\n"
        "\\toprule\n"
        "Method & Ratio & Final Reward \\\\\n"
        "\\midrule\n"
        + "\n".join(rows_tex) + "\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )
    write_tex(output_dir / "table_contamination.tex", tex)


# ──────────────────────────────────────────────────────────────────────
# Table 3: Transfer
# ──────────────────────────────────────────────────────────────────────
def table_transfer(results_dir: Path, output_dir: Path) -> None:
    transfer_dir = results_dir / "transfer"
    if not (transfer_dir / "transfer_results.csv").exists():
        print("  ⚠ transfer_results.csv not found, skipping Table 3")
        return
    rows = read_csv(transfer_dir / "transfer_results.csv")
    methods = ["no_memory", "full_memory", "retrieval", "smtr_tci"]
    method_labels = {
        "no_memory": "No Memory",
        "full_memory": "Full Memory",
        "retrieval": "Retrieval",
        "smtr_tci": "\\textbf{SMTR-TCI}",
    }

    rows_tex: list[str] = []
    for method in methods:
        mr = [r for r in rows if r["method"] == method]
        if not mr:
            continue
        gain_mean, gain_std = agg(mr, "transfer_gain")
        b_mean, b_std = agg(mr, "phase_b_reward")
        rows_tex.append(
            f"{method_labels[method]} & {fmt(gain_mean, gain_std)} & "
            f"{fmt(b_mean, b_std)} \\\\"
        )

    tex = (
        "\\begin{table}[h]\n"
        "\\centering\n"
        "\\caption{Cross-Task Transfer (A→B, 50+50 episodes)}\n"
        "\\label{tab:transfer}\n"
        "\\begin{tabular}{lcc}\n"
        "\\toprule\n"
        "Method & Transfer Gain & Phase-B Reward \\\\\n"
        "\\midrule\n"
        + "\n".join(rows_tex) + "\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )
    write_tex(output_dir / "table_transfer.tex", tex)


# ──────────────────────────────────────────────────────────────────────
# Table 4: Multi-agent propagation
# ──────────────────────────────────────────────────────────────────────
def table_multi_agent(results_dir: Path, output_dir: Path) -> None:
    ma_dir = results_dir / "multi_agent"
    csv_path = ma_dir / "multi_agent_knowledge_results.csv"
    if not csv_path.exists():
        print("  ⚠ multi_agent_knowledge_results.csv not found, skipping Table 4")
        return
    rows = read_csv(csv_path)

    rows_tex: list[str] = []
    for row in rows:
        mode = row["sharing_mode"]
        label = "Naive Sharing" if mode == "naive" else "\\textbf{SMTR Sharing}"
        ep = int(row.get("episodes", 100))
        team = float(row["team_reward_mean"])
        early = float(row.get("early_reward", 0))
        late = float(row.get("late_reward", 0))
        contam = float(row["contamination_propagation"])
        rows_tex.append(
            f"{label} ({ep}ep) & {team:.3f} & {early:.3f} & {late:.3f} & {contam:.3f} \\\\"
        )

    tex = (
        "\\begin{table}[h]\n"
        "\\centering\n"
        "\\caption{Multi-Agent Knowledge Propagation (1 writer + 3 receivers)}\n"
        "\\label{tab:multi_agent}\n"
        "\\begin{tabular}{lcccc}\n"
        "\\toprule\n"
        "Method & Team Reward & Early (20\\%) & Late (20\\%) & Contamination \\\\\n"
        "\\midrule\n"
        + "\n".join(rows_tex) + "\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )
    write_tex(output_dir / "table_multi_agent.tex", tex)


# ──────────────────────────────────────────────────────────────────────
# Table 5: Retention rule ablation
# ──────────────────────────────────────────────────────────────────────
def table_ablation(results_dir: Path, output_dir: Path) -> None:
    abl_dir = results_dir / "ablation" / "retention_rule"
    csv_path = abl_dir / "retention_ablation.csv"
    if not csv_path.exists():
        print("  ⚠ retention_ablation.csv not found, skipping Table 5")
        return
    rows = read_csv(csv_path)

    method_labels = {
        "rule1_single_negative": "Single Negative (strict)",
        "rule2_double_negative": "\\textbf{Double Negative (default)}",
        "rule3_always_retain": "Always Retain",
    }
    scenarios = sorted(set(r["scenario"] for r in rows))

    rows_tex: list[str] = []
    for method in method_labels:
        mr = [r for r in rows if r["method"] == method]
        for sc in scenarios:
            sc_rows = [r for r in mr if r["scenario"] == sc]
            if not sc_rows:
                continue
            fr_mean, fr_std = agg(sc_rows, "final_reward")
            cr_mean, _ = agg(sc_rows, "contamination_rate")
            hr_mean, _ = agg(sc_rows, "harmful_retention")
            rows_tex.append(
                f"{method_labels[method]} & {sc} & "
                f"{fmt(fr_mean, fr_std)} & {cr_mean:.3f} & {hr_mean:.3f} \\\\"
            )

    tex = (
        "\\begin{table}[h]\n"
        "\\centering\n"
        "\\caption{Retention Rule Ablation}\n"
        "\\label{tab:ablation}\n"
        "\\begin{tabular}{llccc}\n"
        "\\toprule\n"
        "Rule & Scenario & Final Reward & Contam. Rate & Harmful Ret. \\\\\n"
        "\\midrule\n"
        + "\n".join(rows_tex) + "\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )
    write_tex(output_dir / "table_ablation.tex", tex)


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results")
    parser.add_argument("--output", default="paper/tables")
    args = parser.parse_args()

    results_dir = Path(args.results)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating paper tables...")
    table_formation(results_dir / "lifelong" / "formation", output_dir)
    table_contamination(results_dir, output_dir)
    table_transfer(results_dir, output_dir)
    table_multi_agent(results_dir, output_dir)
    table_ablation(results_dir, output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
