"""Generate noise robustness paper table (Task B4).

Columns: sigma | SMTR reward | Random reward | SMTR harmful | Random harmful

Output: paper/tables/table_noise_robustness.tex
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))


def main() -> None:
    summary_path = _PROJECT_ROOT / "results" / "noise_robustness" / "noise_summary.json"
    if not summary_path.exists():
        print(f"ERROR: {summary_path} not found")
        return

    summary = json.loads(summary_path.read_text())
    noise_levels = sorted(set(s["noise_sigma"] for s in summary))
    smtr_data = {s["noise_sigma"]: s for s in summary if s["method"] == "smtr_tci"}
    rand_data = {s["noise_sigma"]: s for s in summary if s["method"] == "random_validation"}

    lines = [
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"$\sigma$ & SMTR Reward & Random Reward & SMTR Harmful & Random Harmful \\",
        r"\midrule",
    ]

    for sigma in noise_levels:
        smtr = smtr_data.get(sigma, {})
        rand = rand_data.get(sigma, {})
        smtr_reward = smtr.get("mean_reward", 0.0)
        rand_reward = rand.get("mean_reward", 0.0)
        smtr_harmful = smtr.get("mean_harmful", 0.0)
        rand_harmful = rand.get("mean_harmful", 0.0)

        lines.append(
            f"{sigma:.1f} & \\textbf{{{smtr_reward:.4f}}} & {rand_reward:.4f} "
            f"& \\textbf{{{smtr_harmful:.4f}}} & {rand_harmful:.4f} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
    ])

    output_path = _PROJECT_ROOT / "paper" / "tables" / "table_noise_robustness.tex"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")
    print(f"Written: {output_path}")


if __name__ == "__main__":
    main()
