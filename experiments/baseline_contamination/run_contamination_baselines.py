"""Contamination baseline comparison (Task 7).

Runs all baseline memory controllers under contamination stress:

  Ratios:  0.1, 0.2, 0.3  (false + spurious)
  Outdated: environment change at episode 60 on topics 0-2

Methods: full_memory, retrieval, reflexion, heuristic, agemem, smtr_tci

Output:
  results/baseline_contamination/<variant>/performance.csv
  results/baseline_contamination/<variant>/memory_history.jsonl
  results/baseline_contamination/contamination_baseline_results.csv
  paper/tables/table_contamination_baseline.tex
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))

# Import baseline policies so they register in METHODS
from experiments.lifelong.baseline_policies import BASELINE_METHODS  # noqa: F401
from experiments.lifelong.run_lifelong import ALL_TOPICS, run_experiment

RATIOS = [0.1, 0.2, 0.3]
EPISODES = 100
SEEDS = [0, 1, 2, 3, 4]
# No no_memory (unaffected by contamination) or agile (slow, not informative)
METHODS = ["full_memory", "retrieval", "reflexion",
           "heuristic", "agemem", "smtr_tci"]

OUTDATED_CHANGE_EPISODE = 60
OUTDATED_CHANGED_TOPICS = (0, 1, 2)
OUTDATED_RATIO = 0.2

METHOD_LABELS: dict[str, str] = {
    "full_memory": "Full Memory",
    "retrieval": "Retrieval",
    "reflexion": "Reflexion",
    "heuristic": "Heuristic",
    "agemem": "AgeMem-inspired",
    "smtr_tci": "SMTR-TCI",
}


# ----------------------------------------------------------------------
# Experiment runner
# ----------------------------------------------------------------------
def generate(output_root: Path) -> None:
    """Run contamination benchmark for all ratios + outdated variant."""
    for ratio in RATIOS:
        variant = f"false_spurious_r{ratio}"
        run_experiment(
            experiment=variant,
            output_dir=output_root / variant,
            episodes=EPISODES,
            seeds=SEEDS,
            methods=METHODS,
            contamination_ratio=ratio,
            change_episode=None,
            changed_topics=(),
            topics=ALL_TOPICS,
            topics_after_change=None,
            capacity=None,
        )
    # Outdated variant
    run_experiment(
        experiment="outdated",
        output_dir=output_root / "outdated",
        episodes=EPISODES,
        seeds=SEEDS,
        methods=METHODS,
        contamination_ratio=OUTDATED_RATIO,
        change_episode=OUTDATED_CHANGE_EPISODE,
        changed_topics=OUTDATED_CHANGED_TOPICS,
        topics=ALL_TOPICS,
        topics_after_change=None,
        capacity=None,
    )


# ----------------------------------------------------------------------
# Analysis helpers
# ----------------------------------------------------------------------
def _load_perf(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _load_history(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _reward_by_method_seed(
    perf_rows: list[dict],
) -> dict[tuple[str, int], list[float]]:
    grouped: dict[tuple[str, int], list[tuple[int, float]]] = defaultdict(list)
    for row in perf_rows:
        grouped[(row["method"], int(row["seed"]))].append(
            (int(row["episode"]), float(row["reward"]))
        )
    return {
        key: [r for _, r in sorted(vals)]
        for key, vals in grouped.items()
    }


def _recovery_episodes(rewards: list[float], change_ep: int) -> int:
    """Episodes after the change until rolling mean returns to baseline."""
    baseline = float(np.mean(rewards[max(0, change_ep - 20):change_ep]))
    window = 10
    after = rewards[change_ep:]
    for i in range(window, len(after)):
        if float(np.mean(after[i - window:i])) >= baseline - 0.05:
            return i
    return len(after)


def _retention_rate(history_rows: list[dict], method: str) -> float:
    """Fraction of contaminated memories still retained."""
    contaminated = [
        r for r in history_rows
        if r["method"] == method and r.get("contamination", "none") != "none"
    ]
    if not contaminated:
        return 0.0
    retained = [r for r in contaminated
                if r["status"] in ("validated", "candidate")]
    return len(retained) / len(contaminated)


# ----------------------------------------------------------------------
# Analysis
# ----------------------------------------------------------------------
def analyze(output_root: Path) -> list[dict]:
    """Analyze all contamination variants and return aggregated results."""
    results: list[dict] = []
    for variant_dir in sorted(output_root.iterdir()):
        perf_path = variant_dir / "performance.csv"
        history_path = variant_dir / "memory_history.jsonl"
        if not perf_path.exists():
            continue
        perf_rows = _load_perf(perf_path)
        history_rows = _load_history(history_path)
        rewards = _reward_by_method_seed(perf_rows)
        is_outdated = variant_dir.name == "outdated"

        for method in METHODS:
            finals: list[float] = []
            drops: list[float] = []
            recoveries: list[int] = []
            for (m, _seed), curve in rewards.items():
                if m != method:
                    continue
                n = len(curve)
                finals.append(float(np.mean(curve[-max(1, n // 10):])))
                if is_outdated:
                    pre = float(np.mean(curve[:OUTDATED_CHANGE_EPISODE]))
                    post = float(np.mean(curve[OUTDATED_CHANGE_EPISODE:]))
                    drops.append(pre - post)
                    recoveries.append(
                        _recovery_episodes(curve, OUTDATED_CHANGE_EPISODE)
                    )
            results.append({
                "variant": variant_dir.name,
                "method": method,
                "final_reward_mean": float(np.mean(finals)) if finals else 0.0,
                "final_reward_std": float(np.std(finals)) if finals else 0.0,
                "performance_drop": (
                    float(np.mean(drops)) if drops else None
                ),
                "recovery_episodes": (
                    float(np.mean(recoveries)) if recoveries else None
                ),
                "harmful_retention": _retention_rate(history_rows, method),
            })

    # Save CSV
    csv_path = output_root / "contamination_baseline_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved: {csv_path}")

    # Print summary
    for row in results:
        drop = ("" if row["performance_drop"] is None
                else f" drop={row['performance_drop']:.3f}")
        rec = ("" if row["recovery_episodes"] is None
               else f" recover={row['recovery_episodes']:.0f}ep")
        print(
            f"  {row['variant']:<24} {row['method']:<12}"
            f" final={row['final_reward_mean']:.3f}+-{row['final_reward_std']:.3f}"
            f"{drop}{rec}"
            f" harmful={row['harmful_retention']:.3f}"
        )

    return results


# ----------------------------------------------------------------------
# LaTeX table
# ----------------------------------------------------------------------
def generate_latex_table(results: list[dict]) -> str:
    """Generate contamination baseline comparison LaTeX table."""
    # Group by variant
    variants = sorted(set(r["variant"] for r in results))
    # Filter to main contamination ratios for the main table
    main_variants = [v for v in variants if v.startswith("false_spurious")]
    outdated = "outdated" in variants

    lines = [
        r"\begin{table}[t]",
        r"\caption{Contamination resilience across baselines.}",
        r"\label{tab:contamination_baseline}",
        r"\centering",
    ]

    # Main table: ratios 0.1, 0.2, 0.3
    if main_variants:
        n_cols = 1 + len(main_variants) * 2 + 1  # method + (reward, drop)*n + harmful
        col_spec = "l " + " ".join(["c c"] * len(main_variants)) + " c"
        lines.extend([
            r"\begin{tabular}{" + col_spec + "}",
            r"\toprule",
        ])
        # Header
        header_parts = ["Method"]
        for v in main_variants:
            ratio = v.split("r")[-1]
            header_parts.extend([f"r={ratio} Reward", "Drop"])
        header_parts.append("Harmful Ret.")
        lines.append(" & ".join(header_parts) + r" \\")
        lines.append(r"\midrule")

        for method in METHODS:
            parts = [METHOD_LABELS.get(method, method)]
            for v in main_variants:
                row = next(
                    (r for r in results
                     if r["variant"] == v and r["method"] == method),
                    None,
                )
                if row:
                    parts.append(f"{row['final_reward_mean']:.3f}")
                    parts.append(f"{row['harmful_retention']:.3f}")
                else:
                    parts.extend(["--", "--"])
            # Overall harmful retention (average across variants)
            method_rows = [r for r in results
                           if r["method"] == method
                           and r["variant"] in main_variants]
            avg_harmful = (float(np.mean([r["harmful_retention"]
                                          for r in method_rows]))
                          if method_rows else 0.0)
            parts.append(f"{avg_harmful:.3f}")
            lines.append(" & ".join(parts) + r" \\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
    ])

    # Outdated sub-table
    if outdated:
        lines.extend([
            r"\vspace{4pt}",
            r"\textbf{Outdated variant} (change at episode 60):",
            r"\begin{tabular}{l c c c}",
            r"\toprule",
            r"Method & Final Reward & Perf. Drop & Recovery (ep) \\",
            r"\midrule",
        ])
        for method in METHODS:
            row = next(
                (r for r in results
                 if r["variant"] == "outdated" and r["method"] == method),
                None,
            )
            if row:
                drop = (f"{row['performance_drop']:.3f}"
                        if row["performance_drop"] is not None else "--")
                rec = (f"{row['recovery_episodes']:.0f}"
                       if row["recovery_episodes"] is not None else "--")
                lines.append(
                    f"{METHOD_LABELS.get(method, method)}"
                    f" & {row['final_reward_mean']:.3f}"
                    f" & {drop}"
                    f" & {rec} \\\\"
                )
        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
        ])

    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/baseline_contamination")
    parser.add_argument("--analyze-only", action="store_true",
                        help="Skip experiment run, only analyze existing data")
    args = parser.parse_args()
    output_root = Path(args.output)

    if not args.analyze_only:
        output_root.mkdir(parents=True, exist_ok=True)
        generate(output_root)

    results = analyze(output_root)

    # Generate LaTeX table
    tex_path = Path("paper/tables/table_contamination_baseline.tex")
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.write_text(generate_latex_table(results))
    print(f"\nSaved: {tex_path}")


if __name__ == "__main__":
    main()
