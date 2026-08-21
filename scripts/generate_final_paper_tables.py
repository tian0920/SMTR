"""Generate unified final paper tables (Task C2).

Produces all tables in paper/tables/final/ with uniform formatting:
  Table 1: MARBLE Main Result
  Table 2: Domain-wise Performance
  Table 3: Contamination
  Table 4: Receiver-conditioned
  Table 5: Cost Analysis
  Table 6: Noise Robustness
  Table 7: Ablation (synthetic baseline performance)

Output: paper/tables/final/
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))

METHOD_LABELS: dict[str, str] = {
    "no_memory": "No Memory",
    "full_memory": "Full Memory",
    "retrieval": "Retrieval",
    "reflexion": "Reflexion",
    "heuristic": "Heuristic",
    "agemem": "AgeMem",
    "smtr_tci": "\\textbf{SMTR-TCI}",
    "random_validation": "Random Val.",
}


def load_csv(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def write_tex(lines: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    print(f"  Written: {path}")


# ──────────────────────────────────────────────────────────────
# Table 1: MARBLE Main Result
# ──────────────────────────────────────────────────────────────

def table1_marble_main(output_dir: Path) -> None:
    csv_path = _PROJECT_ROOT / "results" / "marble" / "main" / "baseline_results.csv"
    if not csv_path.exists():
        print("  SKIP: MARBLE main results not found")
        return
    rows = load_csv(csv_path)
    methods = ["no_memory", "full_memory", "retrieval", "reflexion",
               "heuristic", "agemem", "smtr_tci"]
    lines = [
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Method & Groups & Reward & Injected & Positive \\",
        r"\midrule",
    ]
    for m in methods:
        m_rows = [r for r in rows if r["method"] == m]
        if not m_rows:
            continue
        n = len(m_rows)
        reward = float(np.mean([float(r["method_reward"]) for r in m_rows]))
        injected = float(np.mean([float(r["n_injected"]) for r in m_rows]))
        positive = float(np.mean([float(r["n_positive"]) for r in m_rows]))
        label = METHOD_LABELS.get(m, m)
        if m == "smtr_tci":
            lines.append(rf"{label} & {n} & \textbf{{{reward:.4f}}} & {injected:.1f} & \textbf{{{positive:.2f}}} \\")
        else:
            lines.append(rf"{label} & {n} & {reward:.4f} & {injected:.1f} & {positive:.2f} \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    write_tex(lines, output_dir / "table1_marble_main.tex")


# ──────────────────────────────────────────────────────────────
# Table 2: Domain-wise Performance
# ──────────────────────────────────────────────────────────────

def table2_domain(output_dir: Path) -> None:
    csv_path = _PROJECT_ROOT / "results" / "marble" / "domain_analysis" / "domain_wise_results.csv"
    if not csv_path.exists():
        print("  SKIP: domain analysis not found")
        return
    # Reuse existing table
    src = _PROJECT_ROOT / "paper" / "tables" / "table_marble_domain.tex"
    if src.exists():
        write_tex(src.read_text().splitlines(), output_dir / "table2_domain.tex")


# ──────────────────────────────────────────────────────────────
# Table 3: Contamination
# ──────────────────────────────────────────────────────────────

def table3_contamination(output_dir: Path) -> None:
    csv_path = _PROJECT_ROOT / "results" / "marble" / "contamination" / "contamination_results.csv"
    if not csv_path.exists():
        print("  SKIP: contamination results not found")
        return
    rows = load_csv(csv_path)
    ratios = sorted(set(float(r["ratio"]) for r in rows))
    methods = ["full_memory", "retrieval", "smtr_tci"]

    lines = [
        r"\begin{tabular}{l" + "cc" * len(ratios) + "}",
        r"\toprule",
        r"Method & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{$r={r:.1f}$}}" for r in ratios) + r" \\",
        r" & " + " & ".join(["Reward & Ret."] * len(ratios)) + r" \\",
        r"\midrule",
    ]
    for m in methods:
        cells = []
        for ratio in ratios:
            m_rows = [r for r in rows if r["method"] == m and abs(float(r["ratio"]) - ratio) < 0.01]
            if not m_rows:
                cells.extend(["--", "--"])
            else:
                reward = float(np.mean([float(r["method_reward"]) for r in m_rows]))
                retention = float(np.mean([float(r["harmful_retention"]) for r in m_rows]))
                cells.append(f"{reward:.3f}")
                cells.append(f"{retention:.3f}")
        label = METHOD_LABELS.get(m, m)
        lines.append(rf"{label} & {' & '.join(cells)} \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    write_tex(lines, output_dir / "table3_contamination.tex")


# ──────────────────────────────────────────────────────────────
# Table 4: Receiver-conditioned
# ──────────────────────────────────────────────────────────────

def table4_receiver(output_dir: Path) -> None:
    csv_path = _PROJECT_ROOT / "results" / "marble" / "main" / "receiver_conditioned_results.csv"
    if not csv_path.exists():
        print("  SKIP: receiver analysis not found")
        return
    rows = load_csv(csv_path)
    receivers = sorted(set(r["receiver_agent_id"] for r in rows))

    lines = [
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Receiver & Records & Mean $\tau$ & Positive Rate & Negative Rate \\",
        r"\midrule",
    ]
    for recv in receivers:
        r_rows = [r for r in rows if r["receiver_agent_id"] == recv]
        mean_tau = float(np.mean([float(r["mean_tau"]) for r in r_rows]))
        pos_rate = float(np.mean([float(r["positive_rate"]) for r in r_rows]))
        neg_rate = float(np.mean([float(r["negative_rate"]) for r in r_rows]))
        lines.append(rf"{recv} & {len(r_rows)} & {mean_tau:.4f} & {pos_rate:.4f} & {neg_rate:.4f} \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    write_tex(lines, output_dir / "table4_receiver.tex")


# ──────────────────────────────────────────────────────────────
# Table 5: Cost Analysis
# ──────────────────────────────────────────────────────────────

def table5_cost(output_dir: Path) -> None:
    src = _PROJECT_ROOT / "paper" / "tables" / "table_cost_fair_comparison.tex"
    if src.exists():
        write_tex(src.read_text().splitlines(), output_dir / "table5_cost.tex")
    else:
        print("  SKIP: cost comparison table not found")


# ──────────────────────────────────────────────────────────────
# Table 6: Noise Robustness
# ──────────────────────────────────────────────────────────────

def table6_noise(output_dir: Path) -> None:
    src = _PROJECT_ROOT / "paper" / "tables" / "table_noise_robustness.tex"
    if src.exists():
        write_tex(src.read_text().splitlines(), output_dir / "table6_noise.tex")
    else:
        print("  SKIP: noise table not found")


# ──────────────────────────────────────────────────────────────
# Table 7: Ablation (Synthetic Baseline Performance)
# ──────────────────────────────────────────────────────────────

def table7_ablation(output_dir: Path) -> None:
    src = _PROJECT_ROOT / "paper" / "tables" / "table_baseline_performance.tex"
    if src.exists():
        write_tex(src.read_text().splitlines(), output_dir / "table7_ablation.tex")
    else:
        print("  SKIP: baseline performance table not found")


def main() -> None:
    print("=== Generating Final Paper Tables ===")
    output_dir = _PROJECT_ROOT / "paper" / "tables" / "final"
    output_dir.mkdir(parents=True, exist_ok=True)

    table1_marble_main(output_dir)
    table2_domain(output_dir)
    table3_contamination(output_dir)
    table4_receiver(output_dir)
    table5_cost(output_dir)
    table6_noise(output_dir)
    table7_ablation(output_dir)

    print(f"\nAll tables in: {output_dir}/")


if __name__ == "__main__":
    main()
