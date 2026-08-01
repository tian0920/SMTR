#!/usr/bin/env python3
"""Run B0/B1/M0 comparison experiments across ALL counterfactual scenarios.

Produces per-scenario JSON outputs and a consolidated markdown report.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────
CRITIC_CHECKPOINT = "checkpoints/critic_pi3_v22.joblib"
EPISODES = 20
TASK_SEEDS = [0, 1, 2, 3, 4]
GENERATION_SEEDS = [0, 1]
TRAVERSAL_SEEDS = [0, 1, 2]
TOP_K = 4
MAX_SHARES = 3
BASE_DB = "data/memory_compare_all_scenarios.sqlite"
BASE_OUTPUT = "outputs/b0_b1_m0_all_scenarios"

SCENARIOS = [
    "positive",
    "negative",
    "neutral_success",
    "neutral_failure",
    "prefix_sensitive",
    "flip_pos_to_neg",
    "flip_neg_to_pos",
    "flip_neu_to_neg",
    "flip_neu_to_pos",
]

SCENARIO_LABELS = {
    "positive": "Positive Transfer",
    "negative": "Negative Transfer",
    "neutral_success": "Neutral (Success)",
    "neutral_failure": "Neutral (Failure)",
    "prefix_sensitive": "Prefix-Sensitive",
    "flip_pos_to_neg": "Flip: Pos→Neg",
    "flip_neg_to_pos": "Flip: Neg→Pos",
    "flip_neu_to_neg": "Flip: Neu→Neg",
    "flip_neu_to_pos": "Flip: Neu→Pos",
}


def run_single_scenario(scenario: str) -> dict | None:
    """Run the comparison experiment for a single scenario."""
    output_dir = f"{BASE_OUTPUT}/{scenario}"
    cmd = [
        sys.executable, "-m", "smtr.cli", "compare-routers",
        "--db", BASE_DB,
        "--critic-checkpoint", CRITIC_CHECKPOINT,
        "--episodes", str(EPISODES),
        "--task-seeds", *[str(s) for s in TASK_SEEDS],
        "--generation-seeds", *[str(s) for s in GENERATION_SEEDS],
        "--traversal-seeds", *[str(s) for s in TRAVERSAL_SEEDS],
        "--top-k", str(TOP_K),
        "--max-shares-per-invocation", str(MAX_SHARES),
        "--output-dir", output_dir,
        "--overwrite",
        "--scenario", scenario,
    ]
    print(f"\n{'='*60}")
    print(f"Running scenario: {scenario} ({SCENARIO_LABELS[scenario]})")
    print(f"{'='*60}")

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(Path(__file__).resolve().parent.parent))
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"  ERROR (exit {result.returncode}):")
        print(result.stderr[-500:] if result.stderr else "(no stderr)")
        return None

    # Print key output lines
    for line in result.stdout.strip().splitlines():
        if any(k in line for k in ["success_rate", "avg_selected", "negative_transfer", "share_decision", "output_dir"]):
            print(f"  {line.strip()}")

    # Load summary
    summary_path = Path(output_dir) / "summary.json"
    if not summary_path.exists():
        print(f"  WARNING: summary.json not found at {summary_path}")
        return None

    with open(summary_path, encoding="utf-8") as f:
        summary = json.load(f)

    summary["_scenario"] = scenario
    summary["_label"] = SCENARIO_LABELS[scenario]
    summary["_elapsed_seconds"] = round(elapsed, 2)
    return summary


def generate_report(results: list[dict]) -> str:
    """Generate a consolidated markdown report from all scenario results."""
    lines: list[str] = []
    lines.append("# B0/B1/M0 Full Scenario Experiment Report\n")
    lines.append(f"**Critic checkpoint**: `{CRITIC_CHECKPOINT}`  ")
    lines.append(f"**Episodes per scenario**: {EPISODES}  ")
    lines.append(f"**Task seeds**: {TASK_SEEDS}  ")
    lines.append(f"**Generation seeds**: {GENERATION_SEEDS}  ")
    lines.append(f"**Traversal seeds**: {TRAVERSAL_SEEDS}  ")
    lines.append(f"**Top-k**: {TOP_K}, **Max shares/invocation**: {MAX_SHARES}  ")
    lines.append("")

    # ── Summary table ──────────────────────────────────────────────────
    lines.append("## Cross-Scenario Summary\n")
    lines.append("| Scenario | B0 SR | B1 SR | M0 SR | B1 NegTR | M0 NegTR | B1 PosTR | M0 PosTR | M0 ShareRate |")
    lines.append("|----------|-------|-------|-------|----------|----------|----------|----------|--------------|")

    for r in results:
        sc = r["_scenario"]
        label = r["_label"]
        b0_sr = r["b0"]["success_rate"]
        b1_sr = r["b1"]["success_rate"]
        m0_sr = r["m0"]["success_rate"]
        b1_neg = r["b1"].get("negative_transfer_rate")
        m0_neg = r["m0"].get("negative_transfer_rate")
        b1_pos = r["b1"].get("positive_transfer_rate")
        m0_pos = r["m0"].get("positive_transfer_rate")
        m0_share = r["m0"].get("share_decision_rate")

        def fmt_rate(v):
            return f"{v:.3f}" if v is not None else "—"

        def fmt_pct(v):
            return f"{v*100:.1f}%" if v is not None else "—"

        lines.append(
            f"| {label} | {fmt_rate(b0_sr)} | {fmt_rate(b1_sr)} | {fmt_rate(m0_sr)} "
            f"| {fmt_pct(b1_neg)} | {fmt_pct(m0_neg)} | {fmt_pct(b1_pos)} | {fmt_pct(m0_pos)} "
            f"| {fmt_pct(m0_share)} |"
        )
    lines.append("")

    # ── Per-scenario detail ────────────────────────────────────────────
    lines.append("## Per-Scenario Details\n")

    for r in results:
        sc = r["_scenario"]
        label = r["_label"]
        elapsed = r["_elapsed_seconds"]
        lines.append(f"### {label} (`{sc}`)\n")
        lines.append(f"*Elapsed: {elapsed}s*\n")

        # Per-method breakdown
        for method_key in ["b0", "b1", "m0"]:
            m = r[method_key]
            method_name = {"b0": "B0 (NoMemoryRouter)", "b1": "B1 (RelevanceTopKRouter)", "m0": "M0 (ProductionSequentialRouter)"}[method_key]
            lines.append(f"**{method_name}**:")
            lines.append(f"- Episodes: {m['episode_count']}")
            lines.append(f"- Success rate: {m['success_rate']:.3f}")
            lines.append(f"- Avg selected: {m['avg_selected_size']:.1f}")
            lines.append(f"- All-withhold rate: {m['all_withhold_rate']:.3f}")

            if m.get("positive_transfer_rate") is not None:
                lines.append(f"- Positive transfer: {m['positive_transfer_rate']:.3f}")
                lines.append(f"- Negative transfer: {m['negative_transfer_rate']:.3f}")
                lines.append(f"- Neutral success: {m['neutral_success_rate']:.3f}")
                lines.append(f"- Neutral failure: {m['neutral_failure_rate']:.3f}")
                lines.append(f"- Success delta vs B0: {m['success_delta_vs_b0']:.3f}")

            if m.get("share_decision_rate") is not None:
                lines.append(f"- Share decision rate: {m['share_decision_rate']:.3f}")
                lines.append(f"- τ-LCB rejection rate: {m['tau_lcb_rejection_rate']:.3f}")
                lines.append(f"- Neg-risk UCB rejection rate: {m['negative_risk_ucb_rejection_rate']:.3f}")
                lines.append(f"- Budget rejection rate: {m['share_budget_rejection_rate']:.3f}")
                lines.append(f"- Low-support rejection rate: {m['low_support_rejection_rate']:.3f}")
                if m.get("other_reason_counts"):
                    lines.append(f"- Other rejection reasons: {m['other_reason_counts']}")
            lines.append("")

        # M0 vs B1
        cmp = r.get("m0_vs_b1", {})
        if cmp:
            lines.append("**M0 vs B1**:")
            lines.append(f"- Success difference: {cmp.get('success_difference', 0):.3f}")
            lines.append(f"- Neg-transfer diff: {cmp.get('negative_transfer_rate_difference', 0):.3f}")
            lines.append(f"- Pos-transfer diff: {cmp.get('positive_transfer_rate_difference', 0):.3f}")
            lines.append(f"- Avg selected diff: {cmp.get('average_selected_count_difference', 0):.1f}")
            lines.append("")

        # Bootstrap CI
        bci = r.get("bootstrap_ci", {})
        if bci:
            lines.append("**Bootstrap 95% CI**:")
            for key in ["b0_success_rate", "b1_success_rate", "m0_success_rate"]:
                if key in bci:
                    v = bci[key]
                    lines.append(f"- {key}: mean={v['mean']:.3f} [{v['ci_low']:.3f}, {v['ci_high']:.3f}]")
            lines.append("")

        lines.append("---\n")

    # ── Analysis section ───────────────────────────────────────────────
    lines.append("## Analysis\n")

    # Compute aggregate stats
    b0_avg_sr = sum(r["b0"]["success_rate"] for r in results) / len(results)
    b1_avg_sr = sum(r["b1"]["success_rate"] for r in results) / len(results)
    m0_avg_sr = sum(r["m0"]["success_rate"] for r in results) / len(results)

    b1_neg_transfers = sum(1 for r in results if (r["b1"].get("negative_transfer_rate") or 0) > 0)
    m0_neg_transfers = sum(1 for r in results if (r["m0"].get("negative_transfer_rate") or 0) > 0)
    b1_pos_transfers = sum(1 for r in results if (r["b1"].get("positive_transfer_rate") or 0) > 0)
    m0_pos_transfers = sum(1 for r in results if (r["m0"].get("positive_transfer_rate") or 0) > 0)

    lines.append(f"### Overall Average Success Rates\n")
    lines.append(f"| Method | Avg Success Rate |")
    lines.append(f"|--------|-----------------|")
    lines.append(f"| B0 | {b0_avg_sr:.3f} |")
    lines.append(f"| B1 | {b1_avg_sr:.3f} |")
    lines.append(f"| M0 | {m0_avg_sr:.3f} |")
    lines.append("")

    lines.append(f"### Transfer Occurrence\n")
    lines.append(f"| Metric | B1 | M0 |")
    lines.append(f"|--------|----|----|")
    lines.append(f"| Scenarios with positive transfer | {b1_pos_transfers}/{len(results)} | {m0_pos_transfers}/{len(results)} |")
    lines.append(f"| Scenarios with negative transfer | {b1_neg_transfers}/{len(results)} | {m0_neg_transfers}/{len(results)} |")
    lines.append("")

    return "\n".join(lines)


def main():
    print("=" * 60)
    print("B0/B1/M0 Full Scenario Experiment")
    print(f"Critic: {CRITIC_CHECKPOINT}")
    print(f"Scenarios: {len(SCENARIOS)}")
    print(f"Episodes per scenario: {EPISODES}")
    print(f"Total runs per scenario: {EPISODES * len(GENERATION_SEEDS) * (2 + len(TRAVERSAL_SEEDS))}")
    print("=" * 60)

    results: list[dict] = []
    for scenario in SCENARIOS:
        summary = run_single_scenario(scenario)
        if summary:
            results.append(summary)
        else:
            print(f"  SKIPPED: {scenario}")

    # Generate and write report
    report_path = Path(__file__).resolve().parent.parent / "outputs" / "b0_b1_m0_all_scenarios" / "report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = generate_report(results)
    report_path.write_text(report, encoding="utf-8")
    print(f"\n{'='*60}")
    print(f"Report written to: {report_path}")
    print(f"Scenarios completed: {len(results)}/{len(SCENARIOS)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
