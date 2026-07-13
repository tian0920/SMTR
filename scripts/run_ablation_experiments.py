#!/usr/bin/env python3
"""Run ablation experiment across all 9 counterfactual scenarios.

Runs all 6 methods (B0, B1-Top1, B1-Top3, B1-Matched, A1-NoSet, M0-Full)
across all 9 scenarios and generates consolidated results.

Usage:
    python scripts/run_ablation_experiments.py
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from smtr.experiment.runner import ComparisonRunner
from smtr.experiment.schemas import ExperimentConfig
from smtr.experiment.candidate_diagnostics import (
    compute_all_candidate_diagnostics,
    SCENARIO_TARGET_MEMORY,
    SCENARIO_TARGET_EFFECT,
    SCENARIO_PREFIX_MEMORIES,
)
from smtr.experiment.prefix_trace import compute_all_prefix_traces

SCENARIOS = [
    "positive", "negative", "neutral_success", "neutral_failure",
    "prefix_sensitive", "flip_pos_to_neg", "flip_neg_to_pos",
    "flip_neu_to_neg", "flip_neu_to_pos",
]

METHODS = ["B0", "B1-Top1", "B1-Top3", "B1-Matched", "A1-NoSet", "M0-Full"]

OUTPUT_BASE = Path("outputs/ablation_all_scenarios")


def run_scenario(scenario: str) -> dict:
    """Run all 6 methods for a single scenario."""
    output_dir = OUTPUT_BASE / scenario
    config = ExperimentConfig(
        db_path="data/smtr_memory_v2.sqlite",
        critic_checkpoint="checkpoints/critic_pi3_v22.joblib",
        episodes=20,
        task_seeds=[0, 1, 2, 3, 4],
        generation_seeds=[0, 1],
        traversal_seeds=[0, 1, 2],
        top_k=4,
        max_shares_per_invocation=3,
        output_dir=str(output_dir),
        overwrite=True,
        scenario=scenario,
        methods=METHODS,
        a1_critic_checkpoint="checkpoints/critic_no_selected_set_v1.joblib",
        budget_manifest_path="outputs/budget_manifest.json",
    )

    t0 = time.monotonic()
    runner = ComparisonRunner(config)
    summary = runner.run()
    elapsed = time.monotonic() - t0

    # Load runs for diagnostics
    runs = []
    runs_file = output_dir / "runs.jsonl"
    with open(runs_file) as f:
        for line in f:
            if line.strip():
                runs.append(json.loads(line))

    # Compute candidate diagnostics
    cand_diag = compute_all_candidate_diagnostics(runs, scenario=scenario)

    # Compute prefix trace
    prefix_tr = compute_all_prefix_traces(runs, scenario=scenario)

    # Build result dict
    method_results = {}
    for method_id, ms in summary.methods.items():
        method_results[method_id] = {
            "success_rate": round(ms.success_rate, 4),
            "avg_selected": round(ms.avg_selected_size, 2),
            "negative_transfer_rate": round(ms.negative_transfer_rate, 4) if ms.negative_transfer_rate is not None else None,
            "positive_transfer_rate": round(ms.positive_transfer_rate, 4) if ms.positive_transfer_rate is not None else None,
            "share_decision_rate": round(ms.share_decision_rate, 4) if ms.share_decision_rate is not None else None,
            "tau_lcb_rejection_rate": round(ms.tau_lcb_rejection_rate, 4) if ms.tau_lcb_rejection_rate is not None else None,
            "delta_vs_b0": round(ms.success_delta_vs_b0, 4) if ms.success_delta_vs_b0 is not None else None,
        }

    # Candidate diagnostics per method
    cand_diag_results = {}
    for method_id, cd in cand_diag.items():
        cand_diag_results[method_id] = {
            "positive_target_recall_at_k": cd.positive_target_recall_at_k,
            "negative_target_recall_at_k": cd.negative_target_recall_at_k,
            "router_positive_recall": cd.router_positive_recall,
            "harmful_memory_rejection": cd.harmful_memory_rejection,
            "positive_transfer_precision": cd.positive_transfer_precision,
            "neutral_exposure_rate": cd.neutral_exposure_rate,
            "n_target_in_candidates": cd.n_target_in_candidates,
        }

    # Prefix trace per method (only for prefix scenarios)
    prefix_results = {}
    for method_id, pt in prefix_tr.items():
        prefix_results[method_id] = {
            "prefix_candidate_recall": pt.prefix_candidate_recall,
            "prefix_selection_success_rate": pt.prefix_selection_success_rate,
            "success_given_correct_prefix": pt.success_given_correct_prefix,
            "success_without_correct_prefix": pt.success_without_correct_prefix,
            "n_episodes": pt.n_episodes,
        }

    return {
        "scenario": scenario,
        "runtime_seconds": round(elapsed, 2),
        "methods": method_results,
        "candidate_diagnostics": cand_diag_results,
        "prefix_trace": prefix_results,
    }


def generate_report(all_results: list[dict]) -> str:
    """Generate consolidated markdown report."""
    lines = ["# Ablation Experiment Results\n"]
    lines.append(f"**Methods**: {', '.join(METHODS)}")
    lines.append(f"**Scenarios**: {len(SCENARIOS)}")
    lines.append(f"**Episodes per scenario**: 20 (5 task seeds x 2 gen seeds x 3 trav seeds)")
    lines.append(f"**Critic**: critic_pi3_v22 (M0-Full, A1-NoSet)")
    lines.append("")

    # Cross-scenario summary table
    lines.append("## Cross-Scenario Summary\n")
    lines.append("| Scenario | B0 | B1-Top1 | B1-Top3 | B1-Matched | A1-NoSet | M0-Full |")
    lines.append("|----------|-----|---------|---------|------------|----------|---------|")
    for r in all_results:
        row = f"| {r['scenario']} "
        for m in METHODS:
            sr = r["methods"].get(m, {}).get("success_rate", 0)
            row += f"| {sr:.2f} "
        row += "|"
        lines.append(row)

    # Average success rates
    lines.append("\n## Average Success Rates\n")
    avg_sr = {}
    for m in METHODS:
        rates = [r["methods"].get(m, {}).get("success_rate", 0) for r in all_results]
        avg_sr[m] = sum(rates) / len(rates) if rates else 0
    for m in METHODS:
        lines.append(f"- **{m}**: {avg_sr[m]:.3f}")

    # Negative transfer rates
    lines.append("\n## Negative Transfer Rates\n")
    lines.append("| Scenario | B1-Top1 | B1-Top3 | B1-Matched | A1-NoSet | M0-Full |")
    lines.append("|----------|---------|---------|------------|----------|---------|")
    for r in all_results:
        row = f"| {r['scenario']} "
        for m in METHODS[1:]:  # Skip B0
            ntr = r["methods"].get(m, {}).get("negative_transfer_rate")
            row += f"| {ntr:.2f} " if ntr is not None else "| - "
        row += "|"
        lines.append(row)

    # Candidate diagnostics
    lines.append("\n## Candidate-Level Diagnostics\n")
    lines.append("| Scenario | Method | Recall@K | Router+Recall | Harmful Rej | Precision |")
    lines.append("|----------|--------|----------|---------------|-------------|-----------|")
    for r in all_results:
        for m in ["M0-Full", "A1-NoSet", "B1-Top3"]:
            cd = r["candidate_diagnostics"].get(m, {})
            recall = cd.get("positive_target_recall_at_k") or cd.get("negative_target_recall_at_k")
            rpr = cd.get("router_positive_recall")
            hmr = cd.get("harmful_memory_rejection")
            prec = cd.get("positive_transfer_precision")
            recall_s = f"{recall:.2f}" if recall is not None else "-"
            rpr_s = f"{rpr:.2f}" if rpr is not None else "-"
            hmr_s = f"{hmr:.2f}" if hmr is not None else "-"
            prec_s = f"{prec:.2f}" if prec is not None else "-"
            lines.append(f"| {r['scenario']} | {m} | {recall_s} | {rpr_s} | {hmr_s} | {prec_s} |")

    # Prefix traces
    prefix_scenarios = [s for s in SCENARIOS if SCENARIO_PREFIX_MEMORIES.get(s)]
    if prefix_scenarios:
        lines.append("\n## Prefix Formation Trace\n")
        lines.append("| Scenario | Method | Prefix Recall | Prefix Sel Rate | Success+Prefix | Success-Prefix |")
        lines.append("|----------|--------|---------------|-----------------|----------------|----------------|")
        for r in all_results:
            if r["scenario"] not in prefix_scenarios:
                continue
            for m in ["M0-Full", "A1-NoSet"]:
                pt = r["prefix_trace"].get(m, {})
                pcr = pt.get("prefix_candidate_recall", 0)
                psr = pt.get("prefix_selection_success_rate", 0)
                sgp = pt.get("success_given_correct_prefix", 0)
                swp = pt.get("success_without_correct_prefix", 0)
                lines.append(f"| {r['scenario']} | {m} | {pcr:.2f} | {psr:.2f} | {sgp:.2f} | {swp:.2f} |")

    # Key findings
    lines.append("\n## Key Findings\n")

    # 1. B1-Top1 vs B1-Top3
    b1t1_avg = avg_sr.get("B1-Top1", 0)
    b1t3_avg = avg_sr.get("B1-Top3", 0)
    lines.append(f"1. **B1-Top1 vs B1-Top3**: Top1 avg SR={b1t1_avg:.3f}, Top3 avg SR={b1t3_avg:.3f}. "
                 "Restricting budget to 1 reduces negative transfer from indiscriminate sharing.")

    # 2. B1-Matched safety
    b1m_avg = avg_sr.get("B1-Matched", 0)
    lines.append(f"2. **B1-Matched**: avg SR={b1m_avg:.3f}. Budget-matched relevance baseline "
                 "controls for share count confound.")

    # 3. A1 vs M0
    a1_avg = avg_sr.get("A1-NoSet", 0)
    m0_avg = avg_sr.get("M0-Full", 0)
    lines.append(f"3. **A1-NoSet vs M0-Full**: A1 avg SR={a1_avg:.3f}, M0 avg SR={m0_avg:.3f}. "
                 "Delta measures the value of selected-set conditioning.")

    lines.append("\n## Output Files\n")
    lines.append(f"- Base directory: `{OUTPUT_BASE}/`")
    lines.append(f"- Per-scenario dirs: `{OUTPUT_BASE}/<scenario>/`")
    lines.append("  - `runs.jsonl`: per-run records")
    lines.append("  - `summary.json`: method summaries")
    lines.append("  - `config.json`: experiment configuration")

    return "\n".join(lines)


def main():
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    all_results = []

    for scenario in SCENARIOS:
        print(f"\n{'='*60}")
        print(f"Running scenario: {scenario}")
        print(f"{'='*60}")
        result = run_scenario(scenario)
        all_results.append(result)

        # Print per-scenario summary
        for m in METHODS:
            sr = result["methods"].get(m, {}).get("success_rate", 0)
            print(f"  {m}: SR={sr:.2f}")
        print(f"  Time: {result['runtime_seconds']:.1f}s")

    # Save consolidated results
    results_path = OUTPUT_BASE / "all_results.json"
    results_path.write_text(json.dumps(all_results, indent=2) + "\n")

    # Generate and save report
    report = generate_report(all_results)
    report_path = OUTPUT_BASE / "report.md"
    report_path.write_text(report)

    print(f"\n{'='*60}")
    print(f"All scenarios complete. Results saved to {OUTPUT_BASE}/")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
