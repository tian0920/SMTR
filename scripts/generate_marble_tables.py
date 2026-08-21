"""Generate MARBLE baseline paper tables (Phase 8).

Produces three LaTeX tables:
  1. Table 1: MARBLE Overall Performance
  2. Table 2: Knowledge Quality
  3. Table 3: Receiver-Conditioned Analysis

Output: paper/tables/
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

METHOD_LABELS = {
    "no_memory": "No Memory",
    "full_memory": "Full Memory",
    "retrieval": "Retrieval",
    "reflexion": "Reflexion",
    "heuristic": "Heuristic",
    "agemem": "AgeMem",
    "smtr_tci": "SMTR-TCI",
}


def load_csv(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def generate_main_table(results_dir: Path, output_path: Path) -> None:
    """Table 1: MARBLE Overall Performance."""
    csv_path = results_dir / "baseline_results.csv"
    if not csv_path.exists():
        print(f"  SKIP: {csv_path} not found")
        return
    rows = load_csv(csv_path)

    methods_order = ["no_memory", "full_memory", "retrieval", "reflexion",
                     "heuristic", "agemem", "smtr_tci"]
    lines = [
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Method & Groups & Reward & Injected & Positive \\",
        r"\midrule",
    ]
    for method in methods_order:
        m_rows = [r for r in rows if r["method"] == method]
        if not m_rows:
            continue
        n = len(m_rows)
        reward = float(np.mean([float(r["method_reward"]) for r in m_rows]))
        injected = float(np.mean([float(r["n_injected"]) for r in m_rows]))
        positive = float(np.mean([float(r["n_positive"]) for r in m_rows]))
        label = METHOD_LABELS.get(method, method)
        if method == "smtr_tci":
            label = r"\textbf{" + label + "}"
            lines.append(
                rf"{label} & {n} & \textbf{{{reward:.4f}}} & {injected:.1f} & {positive:.1f} \\"
            )
        else:
            lines.append(rf"{label} & {n} & {reward:.4f} & {injected:.1f} & {positive:.1f} \\")
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
    ])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")
    print(f"  Written: {output_path}")


def generate_quality_table(results_dir: Path, output_path: Path) -> None:
    """Table 2: Knowledge Quality (reuse, transfer, contamination)."""
    csv_path = results_dir / "baseline_results.csv"
    if not csv_path.exists():
        print(f"  SKIP: {csv_path} not found")
        return
    rows = load_csv(csv_path)

    methods_order = ["no_memory", "full_memory", "retrieval", "reflexion",
                     "heuristic", "agemem", "smtr_tci"]
    lines = [
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Method & Injected & Positive & Harmful & Neutral \\",
        r"\midrule",
    ]
    for method in methods_order:
        m_rows = [r for r in rows if r["method"] == method]
        if not m_rows:
            continue
        injected = float(np.mean([float(r["n_injected"]) for r in m_rows]))
        positive = float(np.mean([float(r["n_positive"]) for r in m_rows]))
        harmful = float(np.mean([float(r["n_harmful"]) for r in m_rows]))
        neutral = float(np.mean([float(r["n_neutral"]) for r in m_rows]))
        label = METHOD_LABELS.get(method, method)
        if method == "smtr_tci":
            label = r"\textbf{" + label + "}"
            lines.append(
                rf"{label} & {injected:.1f} & \textbf{{{positive:.2f}}} & \textbf{{{harmful:.2f}}} & {neutral:.2f} \\"
            )
        else:
            lines.append(rf"{label} & {injected:.1f} & {positive:.2f} & {harmful:.2f} & {neutral:.2f} \\")
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
    ])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")
    print(f"  Written: {output_path}")


def generate_contamination_table(results_dir: Path, output_path: Path) -> None:
    """Table 3: Contamination Resilience."""
    csv_path = results_dir / "contamination_results.csv"
    if not csv_path.exists():
        print(f"  SKIP: {csv_path} not found")
        return
    rows = load_csv(csv_path)

    ratios = sorted(set(float(r["ratio"]) for r in rows))
    methods = ["full_memory", "retrieval", "smtr_tci"]

    header_cols = " & ".join(f"r={r:.1f}" for r in ratios)
    lines = [
        r"\begin{tabular}{l" + "cc" * len(ratios) + "}",
        r"\toprule",
        r"Method & " + " & ".join(
            rf"\multicolumn{{2}}{{c}}{{r={r:.1f}}}" for r in ratios
        ) + r" \\",
        r" & " + " & ".join(["Reward & Ret."] * len(ratios)) + r" \\",
        r"\midrule",
    ]
    for method in methods:
        cells = []
        for ratio in ratios:
            m_rows = [r for r in rows if r["method"] == method and float(r["ratio"]) == ratio]
            if not m_rows:
                cells.extend(["--", "--"])
                continue
            reward = float(np.mean([float(r["method_reward"]) for r in m_rows]))
            retention = float(np.mean([float(r["harmful_retention"]) for r in m_rows]))
            cells.append(f"{reward:.4f}")
            cells.append(f"{retention:.4f}")
        label = METHOD_LABELS.get(method, method)
        if method == "smtr_tci":
            label = r"\textbf{" + label + "}"
        lines.append(rf"{label} & {' & '.join(cells)} \\")
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
    ])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")
    print(f"  Written: {output_path}")


def main() -> None:
    print("=== Generating MARBLE Paper Tables ===")

    tables_dir = _PROJECT_ROOT / "paper" / "tables"
    results_main = _PROJECT_ROOT / "results" / "marble" / "main"
    results_contam = _PROJECT_ROOT / "results" / "marble" / "contamination"

    generate_main_table(results_main, tables_dir / "table_marble_main.tex")
    generate_quality_table(results_main, tables_dir / "table_marble_quality.tex")
    generate_contamination_table(results_contam, tables_dir / "table_marble_contamination.tex")


if __name__ == "__main__":
    main()
