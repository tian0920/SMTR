"""Audit 5: Final Report Aggregator.

Reads results from all 4 audits and produces the final signal_audit report.

Output:
  - reports/signal_audit.json
  - reports/signal_audit.md

Decision tree:
  Case A: Oracle PASS, Representation FAIL → REPRESENTATION_FAILURE
  Case B: Oracle FAIL (ranking ≈ 0.5)     → ENVIRONMENT_MISMATCH
  Case C: ICC < 0.1                       → SIGNAL_NOISE
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

_THIS_DIR = Path(__file__).parent
_PROJECT_ROOT = _THIS_DIR.parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))


def _load_config() -> dict:
    with open(_THIS_DIR / "config.yaml") as f:
        return yaml.safe_load(f)


def _load_json(path: Path) -> dict | None:
    if path.exists():
        return json.loads(path.read_text())
    return None


def main() -> None:
    config = _load_config()
    stability_cfg = config["audit"]["stability"]
    oracle_cfg = config["audit"]["oracle"]

    print("=" * 60)
    print("MARBLE Causal Signal Audit — Final Report")
    print("=" * 60)

    reports_dir = _THIS_DIR / "reports"

    # ── Load audit results ──
    stability = _load_json(reports_dir / "stability_summary.json")
    oracle = _load_json(reports_dir / "oracle_probe.json")
    repr_probe = _load_json(reports_dir / "representation_probe.json")
    cases = _load_json(reports_dir / "cases.json")

    if not stability:
        print("  ERROR: stability_summary.json not found. "
              "Run repeat_intervention.py first.")
        return
    if not oracle:
        print("  ERROR: oracle_probe.json not found. "
              "Run oracle_probe.py first.")
        return

    # ── Extract key metrics ──
    icc = stability.get("icc", 0)
    oracle_ranking = oracle.get("ranking", 0)
    oracle_verdict = oracle.get("verdict", "UNKNOWN")

    repr_a = 0.5
    repr_b = 0.5
    repr_c = 0.5
    repr_verdict = "UNKNOWN"
    if repr_probe:
        fsets = repr_probe.get("feature_sets", {})
        repr_a = fsets.get("A_current_smtr", {}).get("ranking", 0.5)
        repr_b = fsets.get("B_plus_memory_meta", {}).get("ranking", 0.5)
        repr_c = fsets.get("C_plus_execution", {}).get("ranking", 0.5)
        repr_verdict = repr_probe.get("verdict", "UNKNOWN")

    # ── Decision tree ──
    icc_pass_thr = stability_cfg["icc_pass_threshold"]
    icc_fail_thr = stability_cfg["icc_fail_threshold"]
    oracle_thr = oracle_cfg["ranking_pass_threshold"]

    if icc < icc_fail_thr:
        conclusion = "SIGNAL_NOISE"
        explanation = (
            f"ICC = {icc:.4f} < {icc_fail_thr}. "
            "τ is not stable across replicates. "
            "The measured transfer effect is primarily environmental noise."
        )
        next_steps = [
            "MARBLE current subset has no stable causal transfer signal.",
            "Consider: collect more replicates per (task, memory, receiver).",
            "Consider: switch to a different MARBLE task subset.",
        ]
    elif oracle_ranking > oracle_thr:
        if repr_a > 0.65:
            conclusion = "SIGNAL_EXISTS"
            explanation = (
                f"Oracle ranking = {oracle_ranking:.4f} > {oracle_thr}, "
                f"Current SMTR ranking = {repr_a:.4f} > 0.65. "
                "Both oracle and current representation succeed."
            )
            next_steps = [
                "MARBLE is suitable for SMTR.",
                "Proceed to scale experiments.",
            ]
        else:
            conclusion = "REPRESENTATION_FAILURE"
            explanation = (
                f"Oracle ranking = {oracle_ranking:.4f} > {oracle_thr}, "
                f"but current SMTR ranking = {repr_a:.4f} ≤ 0.65. "
                "Signal exists in MARBLE but SMTR features cannot capture it."
            )
            next_steps = [
                "MARBLE is suitable for SMTR.",
                "Modify memory/trajectory representation.",
                f"Feature B (memory meta) ranking: {repr_b:.4f}",
                f"Feature C (execution) ranking: {repr_c:.4f}",
            ]
    else:
        conclusion = "ENVIRONMENT_MISMATCH"
        explanation = (
            f"Oracle ranking = {oracle_ranking:.4f} ≤ {oracle_thr}. "
            "Even with full execution information, no model can predict τ. "
            "MARBLE current subset does not contain stable transfer structure."
        )
        next_steps = [
            "MARBLE current subset is unsuitable for transfer prediction.",
            "Consider: different MARBLE task subset with stronger effects.",
            "Consider: different benchmark with clearer causal structure.",
        ]

    # ── Case analysis summary ──
    case_summary = {}
    if cases:
        patterns = cases.get("patterns", {})
        for label, pat in patterns.items():
            case_summary[label] = {
                "n_cases": pat.get("n_cases", 0),
                "prediction_change_rate": pat.get("prediction_change_rate", 0),
                "tasks": pat.get("task_ids", [])[:5],
            }

    # ── Print report ──
    print("\n" + "=" * 60)
    print("1. Intervention Stability")
    print("=" * 60)
    print(f"   ICC: {icc:.4f}")
    print(f"   Verdict: {stability.get('verdict', 'UNKNOWN')}")
    print(f"   Exact agreement: "
          f"{stability.get('statistics', {}).get('exact_agreement_rate', 0):.2%}")

    print(f"\n{'=' * 60}")
    print("2. Oracle Probe")
    print("=" * 60)
    print(f"   Ranking: {oracle_ranking:.4f}")
    print(f"   Pearson r: {oracle.get('pearson_r', 0):.4f}")
    print(f"   Sign accuracy: {oracle.get('sign_accuracy', 0):.4f}")
    print(f"   Verdict: {oracle_verdict}")

    print(f"\n{'=' * 60}")
    print("3. Representation Probe")
    print("=" * 60)
    print(f"   A (current SMTR):     ranking = {repr_a:.4f}")
    print(f"   B (+ memory meta):    ranking = {repr_b:.4f}")
    print(f"   C (+ execution):      ranking = {repr_c:.4f}")
    if repr_probe:
        print(f"   Verdict: {repr_verdict}")

    print(f"\n{'=' * 60}")
    print("4. Case Analysis")
    print("=" * 60)
    for label, summary in case_summary.items():
        print(f"   {label}:")
        print(f"     cases: {summary['n_cases']}")
        print(f"     prediction_change_rate: "
              f"{summary['prediction_change_rate']:.2%}")
        print(f"     tasks: {summary['tasks']}")

    print(f"\n{'=' * 60}")
    print(f"CONCLUSION: {conclusion}")
    print("=" * 60)
    print(f"   {explanation}")
    print(f"\n   Next steps:")
    for step in next_steps:
        print(f"     - {step}")

    # ── Save JSON ──
    report_json = {
        "audit_version": "1.0",
        "conclusion": conclusion,
        "explanation": explanation,
        "next_steps": next_steps,
        "results": {
            "stability": {
                "icc": icc,
                "verdict": stability.get("verdict", "UNKNOWN"),
                "exact_agreement_rate": stability.get(
                    "statistics", {}
                ).get("exact_agreement_rate", 0),
                "n_groups": stability.get("statistics", {}).get("n_groups", 0),
                "n_unstable": stability.get(
                    "statistics", {}
                ).get("n_unstable", 0),
            },
            "oracle": {
                "ranking": oracle_ranking,
                "pearson_r": oracle.get("pearson_r", 0),
                "sign_accuracy": oracle.get("sign_accuracy", 0),
                "verdict": oracle_verdict,
            },
            "representation": {
                "A_current_smtr": repr_a,
                "B_plus_memory_meta": repr_b,
                "C_plus_execution": repr_c,
                "verdict": repr_verdict,
            } if repr_probe else None,
            "case_analysis": case_summary,
        },
    }

    json_path = reports_dir / "signal_audit.json"
    with open(json_path, "w") as f:
        json.dump(report_json, f, indent=2)
    print(f"\n  Saved: {json_path}")

    # ── Save Markdown ──
    md = []
    md.append("# MARBLE Causal Signal Audit Report\n")
    md.append(f"**Conclusion: {conclusion}**\n")
    md.append(f"{explanation}\n")

    md.append("## 1. Intervention Stability\n")
    md.append(f"| Metric | Value |")
    md.append(f"|--------|-------|")
    md.append(f"| ICC | {icc:.4f} |")
    md.append(f"| Verdict | {stability.get('verdict', 'UNKNOWN')} |")
    md.append(f"| Exact agreement | "
              f"{stability.get('statistics', {}).get('exact_agreement_rate', 0):.2%} |")
    md.append(f"| Groups | "
              f"{stability.get('statistics', {}).get('n_groups', 0)} |")
    md.append(f"| Unstable | "
              f"{stability.get('statistics', {}).get('n_unstable', 0)} |")
    md.append("")

    md.append("## 2. Oracle Probe\n")
    md.append(f"| Metric | Value |")
    md.append(f"|--------|-------|")
    md.append(f"| Ranking | {oracle_ranking:.4f} |")
    md.append(f"| Pearson r | {oracle.get('pearson_r', 0):.4f} |")
    md.append(f"| Sign accuracy | {oracle.get('sign_accuracy', 0):.4f} |")
    md.append(f"| Verdict | {oracle_verdict} |")
    md.append("")

    if repr_probe:
        md.append("## 3. Representation Probe\n")
        md.append(f"| Feature Set | Ranking |")
        md.append(f"|-------------|---------|")
        fsets = repr_probe.get("feature_sets", {})
        for name, res in fsets.items():
            md.append(f"| {name} | {res.get('ranking', 0):.4f} |")
        md.append("")

    if case_summary:
        md.append("## 4. Case Analysis\n")
        for label, summary in case_summary.items():
            md.append(f"### {label}\n")
            md.append(f"- Cases: {summary['n_cases']}")
            md.append(f"- Prediction change rate: "
                      f"{summary['prediction_change_rate']:.2%}")
            md.append(f"- Tasks: {summary['tasks']}")
            md.append("")

    md.append("## Next Steps\n")
    for step in next_steps:
        md.append(f"- {step}")
    md.append("")

    md_path = reports_dir / "signal_audit.md"
    md_path.write_text("\n".join(md))
    print(f"  Saved: {md_path}")


if __name__ == "__main__":
    main()
