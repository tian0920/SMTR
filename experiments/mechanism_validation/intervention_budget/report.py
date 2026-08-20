"""Generate intervention budget efficiency report.

Reads evaluation and cost analysis results and produces:
  - reports/intervention_budget_summary.json
  - reports/intervention_budget_summary.md

Acceptance criteria:
  1. 50% budget ranking ≥ 0.90
  2. 25% budget ranking ≥ 0.80
  3. At least one non-100% budget efficiency > Full
  4. Shared control cost reduction ≥ 80%
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


class _NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def main() -> None:
    config = _load_config()
    artifacts_dir = _THIS_DIR / "artifacts"
    reports_dir = _THIS_DIR / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Intervention Budget — Report Generation")
    print("=" * 60)

    # Load evaluation results.
    with open(artifacts_dir / "budget_evaluation.json") as f:
        eval_results = json.load(f)

    # Load cost analysis.
    with open(artifacts_dir / "cost_analysis.json") as f:
        cost_results = json.load(f)

    # Build summary.
    budget_results = {}
    for ratio_key, res in eval_results.items():
        budget_results[ratio_key] = {
            "pearson": round(res["avg_pearson"], 4),
            "sign_accuracy": round(res["avg_sign"], 4),
            "ranking": round(res["avg_ranking"], 4),
            "cost": round(res["cost"], 2),
        }

    shared_control = cost_results["shared_control"]
    acceptance = cost_results["acceptance_criteria"]
    verdict = cost_results["verdict"]

    summary = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "budget_results": budget_results,
        "shared_control": shared_control,
        "efficiency": cost_results["efficiency"],
        "acceptance_criteria": acceptance,
        "verdict": verdict,
    }

    # Save JSON.
    json_path = reports_dir / "intervention_budget_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, cls=_NumpyEncoder)
    print(f"  Saved: {json_path}")

    # Build Markdown.
    lines: list[str] = []
    lines.append("# Intervention Budget Efficiency Report\n")
    lines.append(f"Generated: {summary['generated']}\n")
    lines.append("---\n")
    lines.append("## Purpose\n")
    lines.append("Validate SMTR performance under varying intervention budgets.\n")
    lines.append("Test: can SMTR recover memory causal utility with limited "
                 "intervention coverage?\n")
    lines.append("---\n")
    lines.append("## Environment\n")
    env = config["environment"]
    data = config["data"]
    lines.append(f"- Memories: {env['n_memories']}")
    lines.append(f"- Receivers: {env['n_receivers']}")
    lines.append(f"- Embedding dim: {env['embedding_dim']}")
    lines.append(f"- Ground truth: τ(m,r) = sign(z_m^T W z_r)")
    lines.append(f"- Noise: ε ~ N(0, {env['noise_std']})")
    lines.append(f"- Train samples: {data['n_train']}")
    lines.append(f"- Test samples: {data['n_test']}")
    lines.append(f"- Seeds per budget: {config['budget']['seeds']}\n")
    lines.append("---\n")

    # Results table.
    lines.append("## Results by Budget\n")
    lines.append("| Budget | Pearson | Sign | Ranking | Cost |")
    lines.append("|--------|---------|------|---------|------|")
    for ratio_key in sorted(budget_results.keys(),
                            key=lambda x: float(x)):
        r = budget_results[ratio_key]
        pct = f"{float(ratio_key):.0%}"
        lines.append(
            f"| {pct} | {r['pearson']:.4f} | "
            f"{r['sign_accuracy']:.4f} | "
            f"{r['ranking']:.4f} | {r['cost']:.2f} |"
        )
    lines.append("")
    lines.append("---\n")

    # Cost efficiency.
    lines.append("## Cost Efficiency\n")
    lines.append("Efficiency = Ranking / Cost\n")
    lines.append("| Budget | Ranking | Cost | Efficiency |")
    lines.append("|--------|---------|------|------------|")
    for ratio_key in sorted(cost_results["efficiency"].keys(),
                            key=lambda x: float(x)):
        e = cost_results["efficiency"][ratio_key]
        pct = f"{float(ratio_key):.0%}"
        eff_str = (f"{e['efficiency']:.2f}"
                   if e['cost'] > 0 else "∞")
        lines.append(
            f"| {pct} | {e['ranking']:.4f} | "
            f"{e['cost']:.2f} | {eff_str} |"
        )
    lines.append("")
    lines.append("---\n")

    # Shared control.
    lines.append("## Shared Control Ablation\n")
    lines.append("| Approach | Cost |")
    lines.append("|----------|------|")
    lines.append(f"| Naive (N_m × N_r) | "
                 f"{shared_control['naive_cost']} |")
    lines.append(f"| Shared (N_r) | {shared_control['shared_cost']} |")
    lines.append(f"| **Reduction** | "
                 f"**{shared_control['reduction']:.1%}** |")
    lines.append("")
    lines.append("---\n")

    # Acceptance.
    lines.append("## Acceptance Criteria\n")
    for name, check in acceptance.items():
        status = "PASS" if check["passed"] else "FAIL"
        icon = "✅" if check["passed"] else "❌"
        lines.append(
            f"{icon} {status} **{check['description']}**: "
            f"{check['value']:.4f} (threshold: {check['threshold']})"
        )
    lines.append("")
    lines.append("---\n")

    # Conclusion.
    lines.append(f"## Conclusion: **{verdict}**\n")
    if verdict == "PASS":
        lines.append(
            "All acceptance criteria met. SMTR achieves high ranking "
            "accuracy with limited intervention budget."
        )
        lines.append("")

        # Key findings.
        lines.append("### Key Findings\n")
        r50 = budget_results.get("0.50", {})
        r25 = budget_results.get("0.25", {})
        r100 = budget_results.get("1.00", {})

        lines.append(
            f"1. **50% budget ≈ Full**: "
            f"ranking={r50.get('ranking', 0):.4f} vs "
            f"full={r100.get('ranking', 0):.4f}. "
            f"Half the intervention budget achieves comparable performance."
        )
        lines.append(
            f"2. **25% budget is viable**: "
            f"ranking={r25.get('ranking', 0):.4f}. "
            f"Quarter budget still provides useful ranking signal."
        )
        lines.append(
            f"3. **Shared control saves "
            f"{shared_control['reduction']:.0%}**: "
            f"Reusing control rollouts across memories reduces cost from "
            f"{shared_control['naive_cost']} to "
            f"{shared_control['shared_cost']}."
        )
    else:
        lines.append(
            "Some acceptance criteria not met. "
            "See details above."
        )

    md_content = "\n".join(lines) + "\n"
    md_path = reports_dir / "intervention_budget_summary.md"
    with open(md_path, "w") as f:
        f.write(md_content)
    print(f"  Saved: {md_path}")

    # Print summary.
    print(f"\n  Verdict: {verdict}")
    print(f"\n  Budget  | Pearson | Sign   | Ranking | Cost")
    print(f"  --------|---------|--------|---------|-----")
    for rk in sorted(budget_results.keys(), key=lambda x: float(x)):
        r = budget_results[rk]
        print(f"  {float(rk):6.0%}  | {r['pearson']:.4f}  | "
              f"{r['sign_accuracy']:.4f} | {r['ranking']:.4f}  | "
              f"{r['cost']:.2f}")

    print(f"\n  Shared Control: "
          f"{shared_control['naive_cost']} → "
          f"{shared_control['shared_cost']} "
          f"(reduction: {shared_control['reduction']:.1%})")
    print("\nDone.")


if __name__ == "__main__":
    main()
