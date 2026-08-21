"""Generate MARBLE domain-wise paper table (Task A2).

Output: paper/tables/table_marble_domain.tex
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))

DOMAIN_LABELS = {
    "solo": "Solo (1-2 agents)",
    "small": "Small (3 agents)",
    "medium": "Medium (4-5 agents)",
    "large": "Large (6 agents)",
    "complex": "Complex (7+ agents)",
}

METHOD_LABELS = {
    "no_memory": "No Mem.",
    "full_memory": "Full Mem.",
    "retrieval": "Retrieval",
    "reflexion": "Reflexion",
    "heuristic": "Heuristic",
    "agemem": "AgeMem",
    "smtr_tci": "SMTR",
}

METHOD_COLS = ["full_memory", "retrieval", "reflexion", "heuristic", "smtr_tci"]


def load_csv(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def main() -> None:
    csv_path = _PROJECT_ROOT / "results" / "marble" / "domain_analysis" / "domain_wise_results.csv"
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found")
        return

    rows = load_csv(csv_path)

    # Build domain -> method -> reward mapping
    data: dict[str, dict[str, float]] = {}
    for r in rows:
        domain = r["domain"]
        method = r["method"]
        data.setdefault(domain, {})[method] = float(r["mean_reward"])

    # Compute averages and win counts
    averages: dict[str, float] = {}
    win_count = 0
    domain_order = ["solo", "small", "medium", "large", "complex"]

    lines = [
        r"\begin{tabular}{l" + "c" * (len(METHOD_COLS) + 2) + "}",
        r"\toprule",
        r"Domain & " + " & ".join(METHOD_LABELS.get(m, m) for m in METHOD_COLS)
        + r" & Average & SMTR Win \\",
        r"\midrule",
    ]

    for domain in domain_order:
        d_data = data.get(domain, {})
        if not d_data:
            continue

        cells = []
        rewards = []
        for m in METHOD_COLS:
            val = d_data.get(m)
            if val is None:
                cells.append("--")
            else:
                rewards.append(val)
                cells.append(f"{val:.3f}")

        avg = float(np.mean(rewards)) if rewards else 0.0
        averages[domain] = avg

        smtr_val = d_data.get("smtr_tci", 0.0)
        best_bl = max((d_data.get(m, 0.0) for m in METHOD_COLS if m != "smtr_tci"), default=0.0)
        is_win = smtr_val > best_bl
        if is_win:
            win_count += 1

        # Bold SMTR cell
        smtr_idx = METHOD_COLS.index("smtr_tci")
        if smtr_val is not None:
            cells[smtr_idx] = rf"\textbf{{{smtr_val:.3f}}}"

        label = DOMAIN_LABELS.get(domain, domain)
        win_str = r"\checkmark" if is_win else r"\texttimes"
        lines.append(rf"{label} & {' & '.join(cells)} & {avg:.3f} & {win_str} \\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
    ])

    output_path = _PROJECT_ROOT / "paper" / "tables" / "table_marble_domain.tex"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")
    print(f"Written: {output_path}")
    print(f"SMTR wins: {win_count}/{len(domain_order)}")


if __name__ == "__main__":
    main()
