"""Generate receiver heterogeneity stress test report.

Reads evaluation results and produces:
  - receiver_heterogeneity_report.json
  - receiver_heterogeneity_report.md

Acceptance criteria:
  1. SMTR Pearson ≥ 0.75
  2. SMTR improvement over Global (Pearson) ≥ 0.20
  3. Receiver permutation drop ≥ 20%
  4. SMTR pairwise ranking ≥ 0.85
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

_THIS_DIR = Path(__file__).parent
sys.path.insert(0, str(_THIS_DIR.parent.parent.parent / "src"))
sys.path.insert(0, str(_THIS_DIR.parent.parent.parent))


def _load_config() -> dict:
    with open(_THIS_DIR / "config.yaml") as f:
        return yaml.safe_load(f)


def main() -> None:
    config = _load_config()
    acceptance = config["acceptance"]
    artifacts_dir = _THIS_DIR / "artifacts"

    print("=" * 60)
    print("Receiver Heterogeneity Stress Test — Report Generation")
    print("=" * 60)

    # Load evaluation results.
    results_path = artifacts_dir / "evaluation_results.npz"
    if not results_path.exists():
        print(f"  ERROR: {results_path} not found.")
        print("  Run evaluate.py first.")
        sys.exit(1)

    data = np.load(results_path)

    # Extract metrics.
    global_pearson = float(data["global_pearson"])
    global_sign = float(data["global_sign"])
    global_ranking = float(data["global_ranking"])

    smtr_pearson = float(data["smtr_pearson"])
    smtr_sign = float(data["smtr_sign"])
    smtr_ranking = float(data["smtr_ranking"])

    pearson_imp = float(data["pearson_imp"])
    sign_imp = float(data["sign_imp"])
    ranking_imp = float(data["ranking_imp"])

    perm_normal_pearson = float(data["perm_normal_pearson"])
    perm_shuffled_pearson = float(data["perm_shuffled_pearson"])
    perm_pearson_drop = float(data["perm_pearson_drop"])
    perm_pearson_drop_std = float(data["perm_pearson_drop_std"])
    perm_normal_sign = float(data["perm_normal_sign"])
    perm_shuffled_sign = float(data["perm_shuffled_sign"])
    perm_sign_drop = float(data["perm_sign_drop"])

    # Check acceptance criteria.
    checks = {
        "smtr_pearson_min": {
            "description": "SMTR Pearson ≥ 0.75",
            "value": smtr_pearson,
            "threshold": acceptance["smtr_pearson_min"],
            "passed": smtr_pearson >= acceptance["smtr_pearson_min"],
        },
        "improvement_min": {
            "description": "SMTR improvement over Global ≥ 0.20",
            "value": pearson_imp,
            "threshold": acceptance["improvement_min"],
            "passed": pearson_imp >= acceptance["improvement_min"],
        },
        "permutation_drop_min": {
            "description": "Receiver permutation drop ≥ 20%",
            "value": perm_pearson_drop,
            "threshold": acceptance["permutation_drop_min"],
            "passed": perm_pearson_drop >= acceptance["permutation_drop_min"],
        },
        "smtr_ranking_min": {
            "description": "SMTR pairwise ranking ≥ 0.85",
            "value": smtr_ranking,
            "threshold": acceptance["smtr_ranking_min"],
            "passed": smtr_ranking >= acceptance["smtr_ranking_min"],
        },
    }

    all_passed = all(c["passed"] for c in checks.values())
    verdict = "PASS" if all_passed else "FAIL"

    # Build report data.
    report = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "global": {
            "pearson": round(global_pearson, 4),
            "sign_accuracy": round(global_sign, 4),
            "ranking": round(global_ranking, 4),
        },
        "smtr": {
            "pearson": round(smtr_pearson, 4),
            "sign_accuracy": round(smtr_sign, 4),
            "ranking": round(smtr_ranking, 4),
        },
        "improvement": {
            "pearson": round(pearson_imp, 4),
            "sign": round(sign_imp, 4),
            "ranking": round(ranking_imp, 4),
        },
        "receiver_shuffle": {
            "normal_pearson": round(perm_normal_pearson, 4),
            "shuffled_pearson": round(perm_shuffled_pearson, 4),
            "pearson_drop": round(perm_pearson_drop, 4),
            "pearson_drop_std": round(perm_pearson_drop_std, 4),
            "normal_sign": round(perm_normal_sign, 4),
            "shuffled_sign": round(perm_shuffled_sign, 4),
            "sign_drop": round(perm_sign_drop, 4),
        },
        "acceptance_criteria": {
            k: {
                "description": v["description"],
                "value": round(v["value"], 4),
                "threshold": v["threshold"],
                "passed": v["passed"],
            }
            for k, v in checks.items()
        },
        "verdict": verdict,
    }

    # Save JSON report.
    json_path = _THIS_DIR / "receiver_heterogeneity_report.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  JSON report: {json_path}")

    # Generate Markdown report.
    md_lines = [
        "# Receiver Heterogeneity Stress Test Report",
        "",
        f"Generated: {report['generated']}",
        "",
        "---",
        "",
        "## Purpose",
        "",
        "Validate the core SMTR hypothesis: "
        "**τ(m, r₁) ≠ τ(m, r₂)**",
        "",
        "Same memory has different causal effects for different receivers.",
        "",
        "---",
        "",
        "## Environment",
        "",
        f"- Memories: {config['environment']['n_memories']}",
        f"- Receivers: {config['environment']['n_receivers']}",
        f"- Embedding dim: {config['environment']['embedding_dim']}",
        f"- Ground truth: τ(m,r) = sign(z_m^T W z_r)",
        f"- Noise: ε ~ N(0, {config['environment']['noise_std']})",
        f"- Train samples: {config['data']['n_train']}",
        f"- Test samples: {config['data']['n_test']}",
        "",
        "---",
        "",
        "## Results",
        "",
        "### Global Model τ̂(m)",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Pearson | {global_pearson:.4f} |",
        f"| Sign accuracy | {global_sign:.4f} |",
        f"| Pairwise ranking | {global_ranking:.4f} |",
        "",
        "### SMTR Receiver-Conditioned Model τ̂(m, r)",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Pearson | {smtr_pearson:.4f} |",
        f"| Sign accuracy | {smtr_sign:.4f} |",
        f"| Pairwise ranking | {smtr_ranking:.4f} |",
        "",
        "### SMTR Improvement over Global",
        "",
        f"| Metric | Improvement |",
        f"|--------|-------------|",
        f"| Pearson | +{pearson_imp:.4f} |",
        f"| Sign accuracy | +{sign_imp:.4f} |",
        f"| Pairwise ranking | +{ranking_imp:.4f} |",
        "",
        "---",
        "",
        "## Receiver Permutation Test",
        "",
        "Shuffle receiver identity while keeping memory fixed.",
        "If the model truly depends on receiver, performance should drop.",
        "",
        f"| Metric | Normal | Shuffled | Drop |",
        f"|--------|--------|----------|------|",
        f"| Pearson | {perm_normal_pearson:.4f} | "
        f"{perm_shuffled_pearson:.4f} | "
        f"{perm_pearson_drop:.4f} ± {perm_pearson_drop_std:.4f} |",
        f"| Sign | {perm_normal_sign:.4f} | "
        f"{perm_shuffled_sign:.4f} | "
        f"{perm_sign_drop:.4f} |",
        "",
        "---",
        "",
        "## Acceptance Criteria",
        "",
    ]

    for k, v in checks.items():
        status = "✅ PASS" if v["passed"] else "❌ FAIL"
        md_lines.append(
            f"{status} **{v['description']}**: "
            f"{v['value']:.4f} (threshold: {v['threshold']})"
        )

    md_lines.extend([
        "",
        "---",
        "",
        f"## Conclusion: **{verdict}**",
        "",
    ])

    if all_passed:
        md_lines.extend([
            "All acceptance criteria met. "
            "The SMTR receiver-conditioning hypothesis is validated: "
            "memory transfer effects are receiver-dependent.",
            "",
            "### Key Findings",
            "",
            "1. **Receiver conditioning is essential**: "
            f"SMTR (Pearson={smtr_pearson:.4f}) significantly outperforms "
            f"Global (Pearson={global_pearson:.4f}) "
            f"by +{pearson_imp:.4f}.",
            "",
            "2. **Receiver identity matters**: "
            f"Permuting receiver causes a {perm_pearson_drop:.4f} "
            "Pearson drop, confirming the model uses receiver information.",
            "",
            "3. **Pairwise ranking is accurate**: "
            f"SMTR achieves {smtr_ranking:.4f} pairwise ranking accuracy, "
            "demonstrating correct ordering of transfer effects.",
        ])
    else:
        failed = [
            v["description"]
            for v in checks.values()
            if not v["passed"]
        ]
        md_lines.extend([
            "Some acceptance criteria were not met:",
            "",
        ])
        for desc in failed:
            md_lines.append(f"- ❌ {desc}")
        md_lines.extend([
            "",
            "**Action required**: Review the theory/estimand "
            "before scaling experiments.",
        ])

    md_content = "\n".join(md_lines)

    # Save Markdown report.
    md_path = _THIS_DIR / "receiver_heterogeneity_report.md"
    md_path.write_text(md_content, encoding="utf-8")
    print(f"  Markdown report: {md_path}")

    # Print summary.
    print(f"\n  {'='*50}")
    print(f"  VERDICT: {verdict}")
    print(f"  {'='*50}")
    print(f"\n  Global:  Pearson={global_pearson:.4f}, "
          f"Sign={global_sign:.4f}, Ranking={global_ranking:.4f}")
    print(f"  SMTR:    Pearson={smtr_pearson:.4f}, "
          f"Sign={smtr_sign:.4f}, Ranking={smtr_ranking:.4f}")
    print(f"  Permutation drop: {perm_pearson_drop:.4f}")

    passed_count = sum(1 for c in checks.values() if c["passed"])
    print(f"\n  Acceptance: {passed_count}/{len(checks)} passed")
    print(f"\nDone.")


if __name__ == "__main__":
    main()
