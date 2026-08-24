"""Phase 7: Outcome Discrimination Diagnostics.

Computes the full set of TCI signal diagnostics from ablation results:
1. MOR (Memory Opportunity Rate)
2. ZER (Zero Effect Rate)
3. HER (Harmful Effect Rate)
4. Outcome Resolution
5. PDR (Pairwise Discrimination Rate)
6. Cross-episode Reuse Rate

Input: results/marble/outcome_signal_ablation/ablation_episodes.csv
Output: results/marble/outcome_signal_ablation/metrics.csv
        docs/audit/outcome_discrimination_report.md
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent.parent
ABLATION_CSV = BASE_DIR / "results" / "marble" / "outcome_signal_ablation" / "ablation_episodes.csv"
OUTPUT_DIR = BASE_DIR / "results" / "marble" / "outcome_signal_ablation"


def main() -> None:
    if not ABLATION_CSV.exists():
        print(f"ERROR: Ablation CSV not found: {ABLATION_CSV}")
        print("Run run_outcome_signal_ablation.py first.")
        return

    rows: list[dict[str, Any]] = []
    with open(ABLATION_CSV, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        print("ERROR: No data in ablation CSV")
        return

    signal_types = [
        ("binary_success", "delta_binary", "expose_success", "withhold_success"),
        ("native_final_score", "delta_native", "expose_native_score", "withhold_native_score"),
        ("iteration_improvement", "delta_iteration", None, None),
    ]

    metrics_rows: list[dict[str, Any]] = []

    for signal_name, delta_key, expose_key, withhold_key in signal_types:
        deltas = []
        for r in rows:
            try:
                d = float(r[delta_key])
            except (ValueError, KeyError):
                d = 0.0
            deltas.append(d)

        n_total = len(deltas)
        n_positive = sum(1 for d in deltas if d > 0.001)
        n_zero = sum(1 for d in deltas if abs(d) <= 0.001)
        n_negative = sum(1 for d in deltas if d < -0.001)

        # MOR, ZER, HER
        mor = n_positive / n_total if n_total > 0 else 0.0
        zer = n_zero / n_total if n_total > 0 else 0.0
        her = n_negative / n_total if n_total > 0 else 0.0

        # Outcome Resolution
        unique_values = set()
        if expose_key and withhold_key:
            for r in rows:
                try:
                    ev = r.get(expose_key, "")
                    wv = r.get(withhold_key, "")
                    if ev:
                        unique_values.add(float(ev))
                    if wv:
                        unique_values.add(float(wv))
                except (ValueError, TypeError):
                    pass
        outcome_resolution = len(unique_values)

        # PDR (Pairwise Discrimination Rate)
        n_discriminated = 0
        for d in deltas:
            if abs(d) > 0.001:
                n_discriminated += 1
        pdr = n_discriminated / n_total if n_total > 0 else 0.0

        # Cross-episode reuse (not applicable for this ablation — set to 0)
        cross_episode_reuse = 0

        metrics_rows.append({
            "signal_type": signal_name,
            "n_validations": n_total,
            "positive_delta": n_positive,
            "zero_delta": n_zero,
            "negative_delta": n_negative,
            "MOR": round(mor, 4),
            "ZER": round(zer, 4),
            "HER": round(her, 4),
            "outcome_resolution": outcome_resolution,
            "PDR": round(pdr, 4),
            "cross_episode_reuse": cross_episode_reuse,
            "mean_delta": round(sum(deltas) / len(deltas), 4) if deltas else 0,
            "std_delta": round(_std(deltas), 4) if deltas else 0,
        })

    # Write metrics CSV
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = OUTPUT_DIR / "metrics.csv"
    with open(metrics_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metrics_rows)
    print(f"Written: {metrics_path}")

    # Generate report
    report_lines = [
        "# Outcome Discrimination Report (Phase 7)",
        "",
        f"**Total episodes**: {len(rows)}",
        "",
        "## Metrics by Signal Type",
        "",
        "| Signal Type | MOR | ZER | HER | Resolution | PDR | Cross-Reuse |",
        "|------------|-----|-----|-----|-----------|-----|-------------|",
    ]
    for m in metrics_rows:
        report_lines.append(
            f"| {m['signal_type']} | {m['MOR']:.2%} | {m['ZER']:.2%} | "
            f"{m['HER']:.2%} | {m['outcome_resolution']} | {m['PDR']:.2%} | "
            f"{m['cross_episode_reuse']} |"
        )

    report_lines.extend([
        "",
        "## Interpretation",
        "",
        "**MOR (Memory Opportunity Rate)**: P(delta > 0) — higher is better",
        "**ZER (Zero Effect Rate)**: P(delta = 0) — lower is better (less ceiling effect)",
        "**HER (Harmful Effect Rate)**: P(delta < 0) — should be low",
        "**Outcome Resolution**: Number of unique outcome values — higher means finer discrimination",
        "**PDR (Pairwise Discrimination Rate)**: P(expose != withhold) — higher means more signal",
        "**Cross-episode Reuse**: Number of memories reused across episodes (0 for ablation)",
    ])

    report_path = BASE_DIR / "docs" / "audit" / "outcome_discrimination_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Report saved: {report_path}")


def _std(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return variance ** 0.5


if __name__ == "__main__":
    main()
