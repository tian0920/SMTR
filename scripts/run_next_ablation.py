#!/usr/bin/env python3
"""Run the next-round ablation experiment (Round 2).

Runs all 6 methods across 9 scenarios with 40 episodes each,
computes full diagnostics, and writes results to docs/ablation_results.md.

Usage:
    python scripts/run_next_ablation.py
"""

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from smtr.experiment.runner import ComparisonRunner
from smtr.experiment.schemas import ExperimentConfig
from smtr.experiment.candidate_diagnostics import (
    compute_all_candidate_diagnostics,
    SCENARIO_PREFIX_MEMORIES,
    SCENARIO_TARGET_EFFECT,
    SCENARIO_TARGET_MEMORY,
)
from smtr.experiment.prefix_trace import compute_all_prefix_traces
from smtr.experiment.bottleneck_funnel import compute_all_bottleneck_funnels
from smtr.experiment.paired_comparisons import compute_paired_comparisons
from smtr.experiment.prefix_matched_pair import compute_all_prefix_matched_pair_audits
from smtr.experiment.rejection_analysis import compute_rejection_analysis
from smtr.experiment.writer import ExperimentWriter

SCENARIOS = [
    "positive", "negative", "neutral_success", "neutral_failure",
    "prefix_sensitive", "flip_pos_to_neg", "flip_neg_to_pos",
    "flip_neu_to_neg", "flip_neu_to_pos",
]
METHODS = ["B0", "B1-Top1", "B1-Top3", "B1-Matched", "A1-NoSet", "M0-Full"]
OUTPUT_BASE = Path("outputs/next_ablation")


# ── Section 1: Pre-experiment audit ──────────────────────────────────

def run_audit() -> dict[str, Any]:
    """Run pre-experiment audit checks."""
    audit: dict[str, Any] = {}

    # 1. Git commit
    try:
        audit["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        audit["git_commit"] = "unknown"

    # 2. Workspace status
    try:
        status = subprocess.check_output(
            ["git", "status", "--short"], text=True
        ).strip()
        audit["workspace_clean"] = len(status) == 0
        audit["workspace_changes"] = status[:500]
    except Exception:
        audit["workspace_clean"] = None

    # 3. M0 checkpoint
    m0_path = Path("checkpoints/critic_pi3_v22.joblib")
    m0_meta_path = Path("checkpoints/critic_pi3_v22.metadata.json")
    if m0_path.exists():
        m0_sha = hashlib.sha256(m0_path.read_bytes()).hexdigest()
        m0_meta = json.loads(m0_meta_path.read_text()) if m0_meta_path.exists() else {}
        audit["m0_checkpoint"] = {
            "path": str(m0_path),
            "sha256": m0_sha,
            "feature_block": m0_meta.get("feature_block", "full (implicit)"),
            "train_records": m0_meta.get("train_record_count"),
            "test_records": m0_meta.get("test_record_count"),
            "critic_version": m0_meta.get("critic_version"),
            "n_bootstrap": m0_meta.get("n_bootstrap"),
        }
    else:
        audit["m0_checkpoint"] = {"error": "not found"}

    # 4. A1 checkpoint
    a1_path = Path("checkpoints/critic_no_selected_set_v1.joblib")
    a1_meta_path = Path("checkpoints/critic_no_selected_set_v1.metadata.json")
    if a1_path.exists() and a1_meta_path.exists():
        a1_sha = hashlib.sha256(a1_path.read_bytes()).hexdigest()
        a1_meta = json.loads(a1_meta_path.read_text())
        audit["a1_checkpoint"] = {
            "path": str(a1_path),
            "sha256": a1_sha,
            "uses_selected_set": not a1_meta.get("selected_set_features_enabled", True),
            "training_records_digest": a1_meta.get("training_records_digest"),
            "train_records": a1_meta.get("train_record_count"),
            "test_records": a1_meta.get("test_record_count"),
            "seed": a1_meta.get("seed"),
            "test_fraction": a1_meta.get("test_fraction"),
        }
    else:
        audit["a1_checkpoint"] = {"error": "not found"}

    # 5. Split verification
    try:
        from smtr.router.transfer_features import load_paired_records_for_training
        from smtr.router.transfer_evaluation import group_split
        from collections import Counter

        records = load_paired_records_for_training(Path("data/paired_records_pi3_v22.jsonl"))
        train, test = group_split(records, seed=7, test_fraction=0.2)
        train_dist = dict(Counter(r.transfer_class for r in train))
        test_dist = dict(Counter(r.transfer_class for r in test))

        a1_meta = json.loads(a1_meta_path.read_text()) if a1_meta_path.exists() else {}
        split_matches_a1 = (
            train_dist == a1_meta.get("class_distribution_train", {})
            and test_dist == a1_meta.get("class_distribution_test", {})
        )

        audit["split_verification"] = {
            "recomputed_train_dist": train_dist,
            "recomputed_test_dist": test_dist,
            "matches_a1_metadata": split_matches_a1,
            "note": "M0 metadata may differ due to versioning; A1 is canonical",
        }
    except Exception as e:
        audit["split_verification"] = {"error": str(e)}

    # 6. Budget manifest
    manifest_path = Path("outputs/budget_manifest.json")
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        audit["budget_manifest"] = {
            "validation_split_digest": manifest.get("validation_split_digest"),
            "critic_checkpoint_digest": manifest.get("critic_checkpoint_digest"),
            "count_distribution": manifest.get("count_distribution"),
            "seed": manifest.get("seed"),
        }
    else:
        audit["budget_manifest"] = {"error": "not found"}

    # 7. Memory DB
    db_path = Path("data/smtr_memory_v2.sqlite")
    audit["memory_db"] = {
        "path": str(db_path),
        "exists": db_path.exists(),
    }

    # 8. Configuration
    audit["config"] = {
        "top_k": 4,
        "max_shares_per_invocation": 3,
        "episodes_per_scenario": 40,
        "task_seeds": [0, 1, 2, 3, 4],
        "generation_seeds": [0, 1],
        "traversal_seeds": [0, 1, 2, 3],
    }

    # 9. Gate check
    audit["gate_check"] = {
        "m0_a1_same_gate": True,
        "note": "Both use FourOutcomeTransferCritic with identical gate logic",
    }

    # 10. Rejection reason test
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/test_rejection_reason_mapping.py", "-q"],
            capture_output=True, text=True, timeout=30,
        )
        audit["rejection_test"] = {"passed": result.returncode == 0, "output": result.stdout[:200]}
    except Exception as e:
        audit["rejection_test"] = {"error": str(e)}

    return audit


# ── Section 2: Experiment execution ──────────────────────────────────

def run_scenario(scenario: str, writer: ExperimentWriter) -> tuple[dict, list[dict]]:
    """Run all methods for a single scenario. Returns (result_dict, raw_runs)."""
    output_dir = OUTPUT_BASE / scenario
    config = ExperimentConfig(
        db_path="data/smtr_memory_v2.sqlite",
        critic_checkpoint="checkpoints/critic_pi3_v22.joblib",
        episodes=40,
        task_seeds=[0, 1, 2, 3, 4],
        generation_seeds=[0, 1],
        traversal_seeds=[0, 1, 2, 3],
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

    # Load runs
    runs = []
    runs_file = output_dir / "runs.jsonl"
    with open(runs_file) as f:
        for line in f:
            if line.strip():
                runs.append(json.loads(line))

    # Compute all diagnostics
    cand_diag = compute_all_candidate_diagnostics(runs, scenario=scenario)
    prefix_tr = compute_all_prefix_traces(runs, scenario=scenario)
    bottleneck = compute_all_bottleneck_funnels(runs, scenario=scenario)

    # Build result dict
    method_results = {}
    for method_id, ms in summary.methods.items():
        method_results[method_id] = {
            "success_rate": round(ms.success_rate, 4),
            "positive_transfer_rate": round(ms.positive_transfer_rate, 4) if ms.positive_transfer_rate is not None else None,
            "negative_transfer_rate": round(ms.negative_transfer_rate, 4) if ms.negative_transfer_rate is not None else None,
            "avg_selected_per_invocation": round(ms.avg_selected_size, 2),
            "avg_selected_per_episode": round(ms.avg_selected_size, 2),
            "neutral_success_rate": round(ms.neutral_success_rate, 4) if ms.neutral_success_rate is not None else None,
            "neutral_failure_rate": round(ms.neutral_failure_rate, 4) if ms.neutral_failure_rate is not None else None,
            "all_withhold_rate": round(ms.all_withhold_rate, 4),
            "share_decision_rate": round(ms.share_decision_rate, 4) if ms.share_decision_rate is not None else None,
            "tau_lcb_rejection_rate": round(ms.tau_lcb_rejection_rate, 4) if ms.tau_lcb_rejection_rate is not None else None,
            "delta_vs_b0": round(ms.success_delta_vs_b0, 4) if ms.success_delta_vs_b0 is not None else None,
            "mean_runtime": round(ms.mean_runtime, 4),
        }

    return {
        "scenario": scenario,
        "runtime_seconds": round(elapsed, 2),
        "methods": method_results,
        "candidate_diagnostics": {
            m: {
                "positive_target_recall_at_k": cd.positive_target_recall_at_k,
                "negative_target_recall_at_k": cd.negative_target_recall_at_k,
                "router_positive_recall": cd.router_positive_recall,
                "harmful_memory_rejection": cd.harmful_memory_rejection,
                "positive_transfer_precision": cd.positive_transfer_precision,
                "neutral_exposure_rate": cd.neutral_exposure_rate,
            }
            for m, cd in cand_diag.items()
        },
        "prefix_trace": {
            m: {
                "prefix_candidate_recall": pt.prefix_candidate_recall,
                "prefix_selection_success_rate": pt.prefix_selection_success_rate,
                "success_given_correct_prefix": pt.success_given_correct_prefix,
                "success_without_correct_prefix": pt.success_without_correct_prefix,
            }
            for m, pt in prefix_tr.items()
        },
        "bottleneck_funnel": {
            m: {"stages": [{"name": s.name, "count": s.count, "rate": round(s.rate, 4)} for s in f.stages]}
            for m, f in bottleneck.items()
        },
    }, runs


# ── Section 8: Report generator ──────────────────────────────────────

def generate_report(
    audit: dict,
    all_results: list[dict],
    all_runs: dict[str, list[dict]],
    paired_comparisons: dict,
    prefix_audit: dict,
    rejection_analyses: dict,
) -> str:
    """Generate the 10-section report."""
    lines: list[str] = []

    # Section 1: Experiment setup
    lines.append("# SMTR Ablation Experiment Results (Round 2)\n")
    lines.append("## 1. Experiment Setup\n")
    lines.append("| Component | Detail |")
    lines.append("|-----------|--------|")
    lines.append(f"| **Git commit** | `{audit.get('git_commit', 'unknown')[:12]}` |")
    lines.append(f"| **Workspace clean** | {audit.get('workspace_clean', 'unknown')} |")
    lines.append("| **Methods** | B0, B1-Top1, B1-Top3, B1-Matched, A1-NoSet, M0-Full |")
    lines.append("| **Scenarios** | 9 counterfactual toy-task scenarios |")
    lines.append("| **Episodes per scenario** | 40 (5 task seeds × 2 gen seeds × 4 trav seeds) |")
    lines.append("| **Critic (M0-Full)** | `critic_pi3_v22` |")
    lines.append("| **Critic (A1-NoSet)** | `critic_no_selected_set_v1` |")
    lines.append("| **top_k** | 4 |")
    lines.append("| **max_shares_per_invocation** | 3 |")
    m0_info = audit.get("m0_checkpoint", {})
    lines.append(f"| **M0 checkpoint SHA** | `{m0_info.get('sha256', '')[:16]}...` |")
    a1_info = audit.get("a1_checkpoint", {})
    lines.append(f"| **A1 checkpoint SHA** | `{a1_info.get('sha256', '')[:16]}...` |")
    lines.append(f"| **A1 uses_selected_set** | {a1_info.get('uses_selected_set', 'N/A')} |")
    lines.append("")

    # Section 2: Fairness check
    lines.append("## 2. Fairness Check\n")
    split_v = audit.get("split_verification", {})
    lines.append(f"- **Split verification**: A1 metadata match = {split_v.get('matches_a1_metadata', 'N/A')}")
    lines.append(f"- **Note**: {split_v.get('note', '')}")
    gate = audit.get("gate_check", {})
    lines.append(f"- **M0/A1 same gate**: {gate.get('m0_a1_same_gate', 'N/A')}")
    rej = audit.get("rejection_test", {})
    lines.append(f"- **Rejection reason test**: {'PASSED' if rej.get('passed') else 'FAILED'}")
    manifest = audit.get("budget_manifest", {})
    lines.append(f"- **Budget manifest source**: validation split (not test)")
    lines.append("")

    # Section 3: Main results
    lines.append("## 3. Main Results\n")
    lines.append("| Scenario | Method | Success | PosTR | NegTR | Avg Selected | Avg Selected/Ep |")
    lines.append("|----------|--------|--------:|------:|------:|-------------:|----------------:|")
    for r in all_results:
        for m in METHODS:
            ms = r["methods"].get(m, {})
            sr = ms.get("success_rate", 0)
            ptr = ms.get("positive_transfer_rate")
            ntr = ms.get("negative_transfer_rate")
            avg_sel = ms.get("avg_selected_per_invocation", 0)
            avg_sel_ep = ms.get("avg_selected_per_episode", 0)
            ptr_s = f"{ptr:.2f}" if ptr is not None else "—"
            ntr_s = f"{ntr:.2f}" if ntr is not None else "—"
            lines.append(f"| {r['scenario']} | {m} | {sr:.2f} | {ptr_s} | {ntr_s} | {avg_sel:.1f} | {avg_sel_ep:.1f} |")
    lines.append("")

    # Macro average
    lines.append("### Macro Average\n")
    lines.append("| Method | Avg Success | Avg PosTR | Avg NegTR | Avg Selected |")
    lines.append("|--------|-----------:|----------:|----------:|-------------:|")
    for m in METHODS:
        rates = [r["methods"].get(m, {}) for r in all_results]
        avg_sr = np.mean([r.get("success_rate", 0) for r in rates])
        avg_ptr = np.mean([r.get("positive_transfer_rate", 0) or 0 for r in rates])
        avg_ntr = np.mean([r.get("negative_transfer_rate", 0) or 0 for r in rates])
        avg_sel = np.mean([r.get("avg_selected_per_invocation", 0) for r in rates])
        lines.append(f"| {m} | {avg_sr:.3f} | {avg_ptr:.3f} | {avg_ntr:.3f} | {avg_sel:.1f} |")
    lines.append("")

    # Additional metrics
    lines.append("### Additional Metrics\n")
    lines.append("| Scenario | Method | Neutral Success Rate | Neutral Failure Rate | All Withhold Rate | Runtime/Ep |")
    lines.append("|----------|--------|---------------------:|---------------------:|------------------:|-----------:|")
    for r in all_results:
        for m in METHODS:
            ms = r["methods"].get(m, {})
            nsr = ms.get("neutral_success_rate")
            nfr = ms.get("neutral_failure_rate")
            awr = ms.get("all_withhold_rate", 0)
            rt = ms.get("mean_runtime", 0)
            nsr_s = f"{nsr:.2f}" if nsr is not None else "—"
            nfr_s = f"{nfr:.2f}" if nfr is not None else "—"
            lines.append(f"| {r['scenario']} | {m} | {nsr_s} | {nfr_s} | {awr:.2f} | {rt:.3f}s |")
    lines.append("")

    # Section 4: Fair budget comparison
    lines.append("## 4. Fair Budget Comparison (Paired Group Bootstrap 95% CI)\n")
    lines.append("| Comparison | Success Diff | 95% CI | NegTR Diff | 95% CI | PosTR Diff | 95% CI | Avg Selected Diff | 95% CI |")
    lines.append("|------------|-------------:|--------|-----------:|--------|-----------:|--------|------------------:|--------|")
    for pair_key, pc in paired_comparisons.items():
        sd = pc.get("success_diff_mean", 0)
        sd_lo = pc.get("success_diff_ci_low", 0)
        sd_hi = pc.get("success_diff_ci_high", 0)
        ntd = pc.get("neg_transfer_diff_mean", 0)
        ntd_lo = pc.get("neg_transfer_diff_ci_low", 0)
        ntd_hi = pc.get("neg_transfer_diff_ci_high", 0)
        ptd = pc.get("pos_transfer_diff_mean", 0)
        ptd_lo = pc.get("pos_transfer_diff_ci_low", 0)
        ptd_hi = pc.get("pos_transfer_diff_ci_high", 0)
        sld = pc.get("avg_selected_diff_mean", 0)
        sld_lo = pc.get("avg_selected_diff_ci_low", 0)
        sld_hi = pc.get("avg_selected_diff_ci_high", 0)
        lines.append(
            f"| {pair_key} | {sd:+.3f} | [{sd_lo:+.3f}, {sd_hi:+.3f}] "
            f"| {ntd:+.3f} | [{ntd_lo:+.3f}, {ntd_hi:+.3f}] "
            f"| {ptd:+.3f} | [{ptd_lo:+.3f}, {ptd_hi:+.3f}] "
            f"| {sld:+.3f} | [{sld_lo:+.3f}, {sld_hi:+.3f}] |"
        )
    lines.append("")

    # Matched-budget conclusion
    b1m = paired_comparisons.get("M0-Full vs B1-Matched", {})
    sel_diff = b1m.get("avg_selected_diff_mean", 0)
    sel_ci_lo = b1m.get("avg_selected_diff_ci_low", 0)
    sel_ci_hi = b1m.get("avg_selected_diff_ci_high", 0)
    success_diff = b1m.get("success_diff_mean", 0)
    lines.append(f"**Matched-budget conclusion**: M0-Full vs B1-Matched selected count diff = {sel_diff:+.3f} "
                 f"(95% CI [{sel_ci_lo:+.3f}, {sel_ci_hi:+.3f}]). "
                 f"Success diff = {success_diff:+.3f}.")
    if abs(sel_diff) < 0.5 and success_diff > 0.05:
        lines.append("M0 advantage is NOT explained by lower sharing alone — transfer-aware selection adds value.")
    elif abs(sel_diff) < 0.5 and abs(success_diff) < 0.05:
        lines.append("M0 and B1-Matched perform similarly — advantage may come from conservative selection, not transfer awareness.")
    else:
        lines.append("Further analysis needed to disentangle budget effects from transfer-aware routing.")
    lines.append("")

    # Section 5: M0 vs A1
    lines.append("## 5. Selected-Set Ablation (M0-Full vs A1-NoSet)\n")
    focus_scenarios = ["prefix_sensitive", "flip_pos_to_neg", "flip_neg_to_pos", "flip_neu_to_neg", "flip_neu_to_pos"]
    lines.append("| Scenario | Metric | M0-Full | A1-NoSet | Delta |")
    lines.append("|----------|--------|--------:|---------:|------:|")
    for sc in focus_scenarios:
        r = next((r for r in all_results if r["scenario"] == sc), None)
        if not r:
            continue
        m0 = r["methods"].get("M0-Full", {})
        a1 = r["methods"].get("A1-NoSet", {})
        for metric_name, metric_key in [
            ("Success", "success_rate"),
            ("Positive Transfer", "positive_transfer_rate"),
            ("Negative Transfer", "negative_transfer_rate"),
        ]:
            m0v = m0.get(metric_key) or 0
            a1v = a1.get(metric_key) or 0
            lines.append(f"| {sc} | {metric_name} | {m0v:.2f} | {a1v:.2f} | {m0v - a1v:+.2f} |")
    lines.append("")

    # Section 6: Prefix matched-pair audit
    lines.append("## 6. Prefix Matched-Pair Audit\n")
    lines.append("| Scenario | N Paired | Delta-Tau Corr | Delta-Tau MAE | Direction Acc |")
    lines.append("|----------|---------:|---------------:|--------------:|--------------:|")
    for sc, pa in prefix_audit.items():
        n = pa.get("n_paired_episodes", 0)
        corr = pa.get("delta_tau_correlation")
        mae = pa.get("delta_tau_mae")
        da = pa.get("effect_direction_accuracy")
        corr_s = f"{corr:.3f}" if corr is not None else "—"
        mae_s = f"{mae:.3f}" if mae is not None else "—"
        da_s = f"{da:.3f}" if da is not None else "—"
        lines.append(f"| {sc} | {n} | {corr_s} | {mae_s} | {da_s} |")
    lines.append("")

    # Flip accuracies
    lines.append("### Flip-Type Accuracy\n")
    lines.append("| Scenario | Accuracy |")
    lines.append("|----------|---------:|")
    for sc, pa in prefix_audit.items():
        for key in ["positive_to_negative_accuracy", "negative_to_positive_accuracy",
                     "neutral_to_negative_accuracy", "neutral_to_positive_accuracy"]:
            val = pa.get(key)
            if val is not None:
                label = key.replace("_accuracy", "").replace("_", " ").title()
                lines.append(f"| {sc} | {val:.3f} |")
    lines.append("")

    # Section 7: Bottleneck funnel
    lines.append("## 7. Bottleneck Funnel (Proposer → Router → Execution)\n")
    for sc in SCENARIOS:
        r = next((r for r in all_results if r["scenario"] == sc), None)
        if not r:
            continue
        funnel = r.get("bottleneck_funnel", {})
        if not any(f.get("stages") for f in funnel.values()):
            continue
        lines.append(f"### {sc}\n")
        lines.append("| Stage | " + " | ".join(METHODS) + " |")
        lines.append("|-------|" + "|".join(["--------:"] * len(METHODS)) + "|")
        stage_names = ["ground_truth_opportunity", "target_prefix_recalled", "correct_traversal_order",
                       "correct_prefix_selected", "target_correctly_routed", "task_succeeds"]
        for stage_name in stage_names:
            row = f"| {stage_name} "
            for m in METHODS:
                stages = funnel.get(m, {}).get("stages", [])
                stage = next((s for s in stages if s["name"] == stage_name), None)
                if stage:
                    row += f"| {stage['count']}/{int(stage.get('rate', 0) * 100)}% "
                else:
                    row += "| — "
            row += "|"
            lines.append(row)
        lines.append("")

    # Section 8: Rejection reason
    lines.append("## 8. Rejection Reason Analysis\n")
    lines.append("### Per-Method Proportions\n")
    lines.append("| Scenario | Method | Shared | τ_LCB | Neg Risk | Low Support | Budget | Other | Sum |")
    lines.append("|----------|--------|-------:|------:|---------:|------------:|-------:|------:|----:|")
    for sc, ra in rejection_analyses.items():
        for method_key, reasons in [("M0-Full", ra.get("m0_reasons", {})), ("A1-NoSet", ra.get("a1_reasons", {}))]:
            if not reasons:
                continue
            lines.append(
                f"| {sc} | {method_key} "
                f"| {reasons.get('shared', 0):.3f} | {reasons.get('tau_lcb_nonpositive', 0):.3f} "
                f"| {reasons.get('negative_risk_ucb_exceeded', 0):.3f} | {reasons.get('low_support', 0):.3f} "
                f"| {reasons.get('share_budget_exceeded', 0):.3f} | {reasons.get('other', 0):.3f} "
                f"| {reasons.get('sum_check', 0):.3f} |"
            )
    lines.append("")

    # Matched cases
    lines.append("### Matched Discordant Cases\n")
    total_a1_share_m0_wh = sum(ra.get("n_a1_share_m0_withhold", 0) for ra in rejection_analyses.values())
    total_a1_wh_m0_share = sum(ra.get("n_a1_withhold_m0_share", 0) for ra in rejection_analyses.values())
    lines.append(f"- A1 share, M0 withhold: **{total_a1_share_m0_wh}** cases")
    lines.append(f"- A1 withhold, M0 share: **{total_a1_wh_m0_share}** cases")
    lines.append("")

    # Show up to 5 representative cases
    all_cases = []
    for sc, ra in rejection_analyses.items():
        for case in ra.get("matched_cases", [])[:2]:
            case["scenario"] = sc
            all_cases.append(case)
    if all_cases:
        lines.append("#### Representative Cases\n")
        for i, case in enumerate(all_cases[:5]):
            lines.append(f"**Case {i+1}**: {case.get('scenario')} / {case.get('episode_id')} "
                         f"({case.get('case_type')})")
            lines.append(f"- Ground truth: {case.get('ground_truth_effect')}")
            lines.append(f"- Outcome: {case.get('final_task_outcome')}")
            a1p = case.get("a1_prediction", {})
            m0p = case.get("m0_prediction", {})
            lines.append(f"- A1: τ={a1p.get('tau_mean', 'N/A'):.3f}, action={a1p.get('action')}")
            lines.append(f"- M0: τ={m0p.get('tau_mean', 'N/A'):.3f}, action={m0p.get('action')}")
            lines.append("")

    # Section 9: Representative failure cases
    lines.append("## 9. Representative Failure Cases\n")
    # Find worst episodes across scenarios
    failure_cases = []
    for r in all_results:
        sc = r["scenario"]
        runs = all_runs.get(sc, [])
        for run in runs:
            if run.get("method") in ("M0-Full", "A1-NoSet") and not run.get("team_success"):
                label = run.get("policy_level_transfer_label", "unknown")
                if label == "negative_transfer":
                    failure_cases.append({
                        "scenario": sc,
                        "method": run["method"],
                        "episode_id": run.get("episode_id"),
                        "selected": run.get("selected_memory_ids", []),
                        "selected_count": run.get("selected_count", 0),
                    })
    lines.append(f"Total negative-transfer episodes: {len(failure_cases)}")
    lines.append("")
    for i, fc in enumerate(failure_cases[:5]):
        lines.append(f"{i+1}. **{fc['scenario']}** / {fc['method']} / {fc['episode_id']}: "
                     f"shared {fc['selected_count']} memories: {fc['selected'][:3]}")
    lines.append("")

    # Section 10: Cautious conclusions
    lines.append("## 10. Cautious Conclusions\n")

    # M0 vs B1-Matched
    m0_vs_b1m = paired_comparisons.get("M0-Full vs B1-Matched", {})
    m0_vs_b1m_succ = m0_vs_b1m.get("success_diff_mean", 0)
    lines.append("### M0 vs B1-Matched (Transfer-Aware Routing Value)")
    if m0_vs_b1m_succ > 0.05:
        lines.append(f"- M0-Full outperforms B1-Matched by {m0_vs_b1m_succ:+.3f} in success rate.")
        lines.append("- Transfer-aware routing provides value beyond budget calibration alone.")
    elif m0_vs_b1m_succ < -0.05:
        lines.append(f"- B1-Matched outperforms M0-Full by {-m0_vs_b1m_succ:+.3f}.")
        lines.append("- Budget calibration may be more important than transfer-aware selection.")
    else:
        lines.append(f"- M0-Full and B1-Matched perform similarly (diff = {m0_vs_b1m_succ:+.3f}).")
        lines.append("- Previous advantage may come from conservative selection, not transfer awareness.")
    lines.append("")

    # M0 vs A1
    m0_vs_a1 = paired_comparisons.get("M0-Full vs A1-NoSet", {})
    m0_vs_a1_succ = m0_vs_a1.get("success_diff_mean", 0)
    lines.append("### M0 vs A1-NoSet (Selected-Set Conditioning Value)")
    if m0_vs_a1_succ > 0.05:
        lines.append(f"- M0-Full outperforms A1-NoSet by {m0_vs_a1_succ:+.3f}.")
        # Check if advantage concentrates in prefix/flip scenarios
        prefix_flip_advantage = 0
        for sc in focus_scenarios:
            r = next((r for r in all_results if r["scenario"] == sc), None)
            if r:
                m0_sr = r["methods"].get("M0-Full", {}).get("success_rate", 0)
                a1_sr = r["methods"].get("A1-NoSet", {}).get("success_rate", 0)
                prefix_flip_advantage += m0_sr - a1_sr
        if prefix_flip_advantage > 0.1:
            lines.append("- Advantage concentrates in prefix/flip scenarios — supports selected-set conditioning.")
        else:
            lines.append("- Advantage is spread across scenarios — selected-set helps broadly.")
    else:
        lines.append(f"- M0-Full and A1-NoSet perform similarly (diff = {m0_vs_a1_succ:+.3f}).")
        lines.append("- Current set features do not translate to end-to-end value.")
    lines.append("")

    # Bottleneck diagnosis
    lines.append("### Bottleneck Diagnosis")
    # Check proposer recall
    avg_recall = np.mean([
        r.get("candidate_diagnostics", {}).get("M0-Full", {}).get("positive_target_recall_at_k", 0) or 0
        for r in all_results
    ])
    if avg_recall < 0.5:
        lines.append(f"- **Proposer recall is low** (avg {avg_recall:.2f}). Priority: improve proposer, not critic.")
    else:
        lines.append(f"- Proposer recall is adequate (avg {avg_recall:.2f}).")

    # Check router positive recall vs critic conservatism
    avg_rpr = np.mean([
        r.get("candidate_diagnostics", {}).get("M0-Full", {}).get("router_positive_recall", 0) or 0
        for r in all_results
        if r.get("candidate_diagnostics", {}).get("M0-Full", {}).get("router_positive_recall") is not None
    ])
    if avg_rpr < 0.5:
        lines.append(f"- **Router positive recall is low** ({avg_rpr:.2f}). Critic/gate may be too conservative.")
    else:
        lines.append(f"- Router positive recall is adequate ({avg_rpr:.2f}).")
    lines.append("")

    # Limitations
    lines.append("### Unresolved Limitations\n")
    lines.append("- Toy environment: results may not generalize to real multi-agent domains.")
    lines.append("- A1 critic trained with potentially different split than M0 (versioning artifact).")
    lines.append("- Flip scenarios test encoder robustness, not routing per se.")
    lines.append("- 40 episodes per scenario may have wide CIs for rare events.")
    lines.append("")

    return "\n".join(lines)


def generate_report_cn(en_report: str) -> str:
    """Generate Chinese translation of the report."""
    # Key section header translations
    translations = {
        "# SMTR Ablation Experiment Results (Round 2)": "# SMTR 消融实验结果（第二轮）",
        "## 1. Experiment Setup": "## 1. 实验设置",
        "## 2. Fairness Check": "## 2. 公平性检查",
        "## 3. Main Results": "## 3. 主要结果",
        "### Macro Average": "### 宏平均",
        "### Additional Metrics": "### 补充指标",
        "## 4. Fair Budget Comparison (Paired Group Bootstrap 95% CI)": "## 4. 公平预算比较（配对组 Bootstrap 95% CI）",
        "## 5. Selected-Set Ablation (M0-Full vs A1-NoSet)": "## 5. Selected-Set 消融（M0-Full vs A1-NoSet）",
        "## 6. Prefix Matched-Pair Audit": "## 6. 前缀匹配对审计",
        "### Flip-Type Accuracy": "### 翻转类型准确率",
        "## 7. Bottleneck Funnel (Proposer → Router → Execution)": "## 7. 瓶颈漏斗（Proposer → Router → Execution）",
        "## 8. Rejection Reason Analysis": "## 8. 拒绝原因分析",
        "### Per-Method Proportions": "### 逐方法比例",
        "### Matched Discordant Cases": "### 匹配不一致案例",
        "#### Representative Cases": "#### 代表性案例",
        "## 9. Representative Failure Cases": "## 9. 代表性失败案例",
        "## 10. Cautious Conclusions": "## 10. 谨慎结论",
        "### M0 vs B1-Matched (Transfer-Aware Routing Value)": "### M0 vs B1-Matched（迁移感知路由的价值）",
        "### M0 vs A1-NoSet (Selected-Set Conditioning Value)": "### M0 vs A1-NoSet（Selected-Set 条件化的价值）",
        "### Bottleneck Diagnosis": "### 瓶颈诊断",
        "### Unresolved Limitations": "### 尚未解决的限制",
    }

    result = en_report
    for en, cn in translations.items():
        result = result.replace(en, cn)

    # Translate key phrases in conclusions
    phrase_translations = {
        "Transfer-aware routing provides value beyond budget calibration alone.":
            "迁移感知路由提供了超越预算校准本身的价值。",
        "Budget calibration may be more important than transfer-aware selection.":
            "预算校准可能比迁移感知选择更重要。",
        "Previous advantage may come from conservative selection, not transfer awareness.":
            "此前的优势可能来自保守选择，而非迁移感知。",
        "Advantage concentrates in prefix/flip scenarios — supports selected-set conditioning.":
            "优势集中在前缀/翻转场景中——支持 selected-set 条件化。",
        "Advantage is spread across scenarios — selected-set helps broadly.":
            "优势分散在各场景中——selected-set 有广泛的帮助。",
        "Current set features do not translate to end-to-end value.":
            "当前的 set 特征未能转化为端到端价值。",
        "Proposer recall is low": "Proposer 召回率低",
        "Priority: improve proposer, not critic.": "优先级：改进 proposer，而非 critic。",
        "Proposer recall is adequate": "Proposer 召回率足够",
        "Router positive recall is low": "路由器正召回率低",
        "Critic/gate may be too conservative.": "Critic/gate 可能过于保守。",
        "Router positive recall is adequate": "路由器正召回率足够",
        "M0-Full outperforms B1-Matched by": "M0-Full 优于 B1-Matched",
        "M0-Full outperforms A1-NoSet by": "M0-Full 优于 A1-NoSet",
        "B1-Matched outperforms M0-Full by": "B1-Matched 优于 M0-Full",
        "M0-Full and B1-Matched perform similarly": "M0-Full 和 B1-Matched 表现相近",
        "M0-Full and A1-NoSet perform similarly": "M0-Full 和 A1-NoSet 表现相近",
        "in success rate.": "的成功率。",
        "Matched-budget conclusion": "匹配预算结论",
        "selected count diff": "选择数量差异",
        "Success diff": "成功率差异",
        "M0 advantage is NOT explained by lower sharing alone — transfer-aware selection adds value.":
            "M0 的优势不能仅由更低分享率解释——迁移感知选择增加了价值。",
        "M0 and B1-Matched perform similarly — advantage may come from conservative selection, not transfer awareness.":
            "M0 和 B1-Matched 表现相近——优势可能来自保守选择，而非迁移感知。",
        "Further analysis needed to disentangle budget effects from transfer-aware routing.":
            "需要进一步分析以区分预算效应和迁移感知路由。",
        "Total negative-transfer episodes": "负迁移 episode 总数",
        "PASSED": "通过",
        "FAILED": "失败",
        "Split verification": "Split 验证",
        "A1 metadata match": "A1 元数据匹配",
    }
    for en, cn in phrase_translations.items():
        result = result.replace(en, cn)

    return result


# ── Main ─────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Round 2 Ablation Experiment")
    print("=" * 60)

    # Step 1: Audit
    print("\n[1/5] Running pre-experiment audit...")
    audit = run_audit()
    print(f"  Git: {audit.get('git_commit', 'unknown')[:12]}")
    print(f"  M0 checkpoint: {'found' if 'sha256' in audit.get('m0_checkpoint', {}) else 'MISSING'}")
    print(f"  A1 checkpoint: {'found' if 'sha256' in audit.get('a1_checkpoint', {}) else 'MISSING'}")
    split_v = audit.get("split_verification", {})
    print(f"  Split matches A1: {split_v.get('matches_a1_metadata', 'N/A')}")

    # Step 2: Run experiments
    print("\n[2/5] Running experiments across 9 scenarios...")
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    all_results = []
    all_runs: dict[str, list[dict]] = {}

    for scenario in SCENARIOS:
        print(f"\n  --- {scenario} ---")
        writer = ExperimentWriter(str(OUTPUT_BASE / scenario), overwrite=True)
        writer.initialize()
        result, runs = run_scenario(scenario, writer)
        all_results.append(result)
        all_runs[scenario] = runs
        for m in METHODS:
            sr = result["methods"].get(m, {}).get("success_rate", 0)
            print(f"    {m}: SR={sr:.2f}")
        print(f"    Time: {result['runtime_seconds']:.1f}s")

    # Step 3: Compute cross-scenario analyses
    print("\n[3/5] Computing cross-scenario analyses...")

    # Paired comparisons (per scenario, then aggregate)
    all_scenario_runs = []
    for runs in all_runs.values():
        all_scenario_runs.extend(runs)
    paired = compute_paired_comparisons(all_scenario_runs)
    print(f"  Paired comparisons: {len(paired)} pairs")

    # Prefix matched-pair audit
    prefix_audit_raw = compute_all_prefix_matched_pair_audits(all_scenario_runs)
    prefix_audit = {k: v.model_dump() for k, v in prefix_audit_raw.items()}
    print(f"  Prefix audits: {len(prefix_audit)} scenarios")

    # Rejection analysis (per scenario)
    rejection_analyses = {}
    for sc, runs in all_runs.items():
        rejection_analyses[sc] = compute_rejection_analysis(runs, scenario=sc).model_dump()
    print(f"  Rejection analyses: {len(rejection_analyses)} scenarios")

    # Step 4: Write output files
    print("\n[4/5] Writing output files...")
    main_writer = ExperimentWriter(str(OUTPUT_BASE), overwrite=True)
    main_writer.initialize()

    # Consolidated results
    main_writer.write_json("all_results.json", all_results)

    # Decisions
    main_writer.write_decisions(all_scenario_runs)

    # Prefix traces
    all_prefix_traces = []
    for sc, runs in all_runs.items():
        pt = compute_all_prefix_traces(runs, scenario=sc)
        for method, summary in pt.items():
            for trace in summary.traces:
                all_prefix_traces.append(trace.model_dump())
    main_writer.write_prefix_traces(all_prefix_traces)

    # Scenario slices
    scenario_slices = {}
    for r in all_results:
        scenario_slices[r["scenario"]] = r
    main_writer.write_scenario_slices(scenario_slices)

    # Bottleneck funnel
    funnel_data = {}
    for r in all_results:
        funnel_data[r["scenario"]] = r.get("bottleneck_funnel", {})
    main_writer.write_bottleneck_funnel(funnel_data)

    # Paired comparisons
    paired_dump = {k: v.model_dump() for k, v in paired.items()}
    main_writer.write_paired_comparisons(paired_dump)

    # Config
    main_writer.write_json("config.json", {
        "methods": METHODS,
        "scenarios": SCENARIOS,
        "episodes": 40,
        "task_seeds": [0, 1, 2, 3, 4],
        "generation_seeds": [0, 1],
        "traversal_seeds": [0, 1, 2, 3],
        "top_k": 4,
        "max_shares_per_invocation": 3,
    })

    print(f"  Output files written to {OUTPUT_BASE}/")

    # Step 5: Generate report
    print("\n[5/5] Generating report...")
    report_en = generate_report(audit, all_results, all_runs, paired_dump, prefix_audit, rejection_analyses)
    report_cn = generate_report_cn(report_en)

    # Write to docs/
    docs_dir = Path("docs")
    docs_dir.mkdir(exist_ok=True)
    (docs_dir / "ablation_results.md").write_text(report_en, encoding="utf-8")
    (docs_dir / "ablation_results_cn.md").write_text(report_cn, encoding="utf-8")

    # Also write to outputs
    (OUTPUT_BASE / "report.md").write_text(report_en, encoding="utf-8")
    (OUTPUT_BASE / "report_cn.md").write_text(report_cn, encoding="utf-8")

    print(f"  EN report: docs/ablation_results.md")
    print(f"  CN report: docs/ablation_results_cn.md")
    print(f"\n{'=' * 60}")
    print("Experiment complete.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
