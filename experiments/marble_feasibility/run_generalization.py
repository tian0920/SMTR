"""Run generalization diagnostic across three data splits.

Orchestrates:
  1. generate_splits.py    (create Split A/B/C)
  2. For each split:
     a. train_smtr_probe.py --split <name>
     b. evaluate_signal.py  --split <name>
  3. Aggregate results into generalization_report.json + .md

Acceptance criteria:
  1. In-distribution ranking > 0.65
  2. Memory holdout ranking > random
  3. Task holdout ranking > random
  4. SMTR > outcome-only by at least +5%

Outputs:
  - reports/generalization_report.json
  - reports/generalization_report.md
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml

_THIS_DIR = Path(__file__).parent
_PROJECT_ROOT = _THIS_DIR.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))

SPLIT_NAMES = ["in_distribution", "memory_holdout", "task_holdout"]


def _load_config() -> dict:
    with open(_THIS_DIR / "config.yaml") as f:
        return yaml.safe_load(f)


def _run_script(script_name: str, extra_args: list[str] | None = None) -> bool:
    """Run a Python script and return True if successful."""
    script_path = _THIS_DIR / script_name
    cmd = [sys.executable, str(script_path)]
    if extra_args:
        cmd.extend(extra_args)
    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print('='*60)

    result = subprocess.run(cmd, cwd=_THIS_DIR, capture_output=False)
    return result.returncode == 0


def _load_split_results(split_name: str) -> dict | None:
    """Load evaluation results for a split."""
    results_path = _THIS_DIR / "splits" / split_name / "evaluation_results.json"
    if not results_path.exists():
        return None
    with open(results_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _check_acceptance_criteria(
    results: dict[str, dict | None],
) -> dict:
    """Check all acceptance criteria and return results."""
    config = _load_config()
    acceptance = config["acceptance"]

    checks = {}

    # Criterion 1: In-distribution ranking > 0.65
    id_results = results.get("in_distribution")
    if id_results:
        id_ranking = id_results.get("smtr_probe", {}).get("informative_ranking", 0.0)
        checks["in_distribution_ranking"] = {
            "description": "In-distribution ranking > 0.65",
            "passed": id_ranking > 0.65,
            "value": f"{id_ranking:.4f}",
            "threshold": ">0.65",
        }
    else:
        checks["in_distribution_ranking"] = {
            "description": "In-distribution ranking > 0.65",
            "passed": False,
            "value": "N/A (no results)",
            "threshold": ">0.65",
        }

    # Criterion 2: Memory holdout > random
    mem_results = results.get("memory_holdout")
    if mem_results:
        mem_ranking = mem_results.get("smtr_probe", {}).get("informative_ranking", 0.0)
        mem_random = mem_results.get("random_baseline", {}).get("pairwise_ranking", 0.5)
        checks["memory_holdout_vs_random"] = {
            "description": "Memory holdout ranking > random",
            "passed": mem_ranking > mem_random,
            "value": f"{mem_ranking:.4f} (random={mem_random:.4f})",
            "threshold": ">random",
        }
    else:
        checks["memory_holdout_vs_random"] = {
            "description": "Memory holdout ranking > random",
            "passed": False,
            "value": "N/A (no results)",
            "threshold": ">random",
        }

    # Criterion 3: Task holdout > random
    task_results = results.get("task_holdout")
    if task_results:
        task_ranking = task_results.get("smtr_probe", {}).get("informative_ranking", 0.0)
        task_random = task_results.get("random_baseline", {}).get("pairwise_ranking", 0.5)
        checks["task_holdout_vs_random"] = {
            "description": "Task holdout ranking > random",
            "passed": task_ranking > task_random,
            "value": f"{task_ranking:.4f} (random={task_random:.4f})",
            "threshold": ">random",
        }
    else:
        checks["task_holdout_vs_random"] = {
            "description": "Task holdout ranking > random",
            "passed": False,
            "value": "N/A (no results)",
            "threshold": ">random",
        }

    # Criterion 4: SMTR > outcome-only by at least +5%
    # Check on in-distribution split
    if id_results:
        id_smtr = id_results.get("smtr_probe", {}).get("informative_ranking", 0.0)
        id_outcome = id_results.get("outcome_only_baseline", {}).get("pairwise_ranking", 0.0)
        diff = id_smtr - id_outcome
        checks["smtr_vs_outcome_only"] = {
            "description": "SMTR > outcome-only by at least +5%",
            "passed": diff >= 0.05,
            "value": f"SMTR={id_smtr:.4f}, outcome-only={id_outcome:.4f}, diff={diff:+.4f}",
            "threshold": ">=+0.05",
        }
    else:
        checks["smtr_vs_outcome_only"] = {
            "description": "SMTR > outcome-only by at least +5%",
            "passed": False,
            "value": "N/A (no results)",
            "threshold": ">=+0.05",
        }

    all_passed = all(check["passed"] for check in checks.values())
    return {
        "checks": checks,
        "all_passed": all_passed,
        "verdict": "PASS" if all_passed else "FAIL",
    }


def _generate_report(
    results: dict[str, dict | None],
    acceptance_results: dict,
) -> None:
    """Generate JSON and Markdown reports."""
    reports_dir = _THIS_DIR / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # JSON report
    json_report = {
        "split_results": {
            name: res for name, res in results.items() if res is not None
        },
        "acceptance_criteria": acceptance_results,
    }
    json_path = reports_dir / "generalization_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_report, f, indent=2)
    print(f"\n  Saved JSON report: {json_path}")

    # Markdown report
    md_lines = [
        "# MARBLE Generalization Diagnostic Report",
        "",
        "## Split Summary",
        "",
    ]

    for split_name in SPLIT_NAMES:
        res = results.get(split_name)
        if res is None:
            md_lines.append(f"### {split_name}: N/A (no results)")
            md_lines.append("")
            continue

        smtr_ranking = res.get("smtr_probe", {}).get("informative_ranking", 0.0)
        random_ranking = res.get("random_baseline", {}).get("pairwise_ranking", 0.5)
        outcome_ranking = res.get("outcome_only_baseline", {}).get("pairwise_ranking", 0.0)
        pred_dist = res.get("prediction_distribution", {})
        test_records = res.get("test_records", 0)
        valid_records = res.get("valid_records", 0)

        md_lines.extend([
            f"### {split_name}",
            f"- **Test records:** {test_records} ({valid_records} valid)",
            f"- **SMTR informative ranking:** {smtr_ranking:.4f}",
            f"- **Random baseline:** {random_ranking:.4f}",
            f"- **Outcome-only baseline:** {outcome_ranking:.4f}",
            f"- **SMTR vs random:** {smtr_ranking - random_ranking:+.4f}",
            f"- **SMTR vs outcome-only:** {smtr_ranking - outcome_ranking:+.4f}",
            f"- **τ pred std:** {pred_dist.get('std', 0):.4f}",
            "",
        ])

    md_lines.extend([
        "## Acceptance Criteria",
        "",
    ])

    for check_name, check_data in acceptance_results.get("checks", {}).items():
        status = "PASS" if check_data["passed"] else "FAIL"
        icon = "✅" if check_data["passed"] else "❌"
        md_lines.append(f"### {icon} {status} {check_data['description']}")
        md_lines.append(f"- Value: {check_data['value']}")
        if "threshold" in check_data:
            md_lines.append(f"- Threshold: {check_data['threshold']}")
        md_lines.append("")

    verdict = acceptance_results.get("verdict", "FAIL")
    md_lines.extend([
        "---",
        "",
        f"## Conclusion: **{verdict}**",
        "",
    ])

    # Add interpretation
    id_ranking = results.get("in_distribution", {}).get("smtr_probe", {}).get("informative_ranking", 0.0)
    mem_ranking = results.get("memory_holdout", {}).get("smtr_probe", {}).get("informative_ranking", 0.0)
    task_ranking = results.get("task_holdout", {}).get("smtr_probe", {}).get("informative_ranking", 0.0)

    md_lines.append("## Interpretation")
    md_lines.append("")

    if id_ranking > 0.65 and mem_ranking > 0.55 and task_ranking > 0.55:
        md_lines.extend([
            "**Case A: Method is viable.**",
            "",
            "All generalization criteria met. The critic learns transferable patterns.",
            "Ready to enter scale experiments.",
        ])
    elif id_ranking > 0.65 and (mem_ranking < 0.55 or task_ranking < 0.55):
        md_lines.extend([
            "**Case B: Needs representation enhancement.**",
            "",
            "In-distribution performance is good, but generalization to unseen "
            "memories/tasks is limited. Consider:",
            "- Enhancing memory/receiver feature representations",
            "- Collecting more diverse training examples",
            "- Exploring transfer learning or meta-learning approaches",
        ])
    else:
        md_lines.extend([
            "**Case C: Insufficient signal in current data.**",
            "",
            "The critic cannot learn meaningful patterns even in-distribution. "
            "Possible causes:",
            "- Memory effects too sparse or noisy in MARBLE",
            "- Feature representation inadequate for the task",
            "- Need significantly more training data",
        ])

    md_lines.append("")

    md_content = "\n".join(md_lines)
    md_path = reports_dir / "generalization_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"  Saved Markdown report: {md_path}")


def main() -> None:
    print("="*60)
    print("MARBLE Generalization Diagnostic")
    print("="*60)

    # Step 1: Generate splits
    if not _run_script("generate_splits.py"):
        print("\nERROR: generate_splits.py failed")
        sys.exit(1)

    # Step 2: Train + evaluate each split
    results: dict[str, dict | None] = {}
    for split_name in SPLIT_NAMES:
        print(f"\n{'='*60}")
        print(f"Processing split: {split_name}")
        print('='*60)

        # Train
        if not _run_script("train_smtr_probe.py", ["--split", split_name]):
            print(f"\nERROR: train_smtr_probe.py --split {split_name} failed")
            results[split_name] = None
            continue

        # Evaluate
        if not _run_script("evaluate_signal.py", ["--split", split_name]):
            print(f"\nERROR: evaluate_signal.py --split {split_name} failed")
            results[split_name] = None
            continue

        # Load results
        results[split_name] = _load_split_results(split_name)

    # Step 3: Check acceptance criteria
    print(f"\n{'='*60}")
    print("Checking Acceptance Criteria")
    print('='*60)

    acceptance_results = _check_acceptance_criteria(results)

    print("\n  Acceptance Criteria Summary:")
    for check_name, check_data in acceptance_results["checks"].items():
        status = "PASS" if check_data["passed"] else "FAIL"
        print(f"    [{status}] {check_data['description']}: {check_data['value']}")

    verdict = acceptance_results["verdict"]
    print(f"\n  Overall Verdict: {verdict}")

    # Step 4: Generate report
    print(f"\n{'='*60}")
    print("Generating Report")
    print('='*60)
    _generate_report(results, acceptance_results)

    print(f"\n{'='*60}")
    print(f"Generalization Diagnostic Complete: {verdict}")
    print('='*60)

    if verdict == "PASS":
        print("\n✅ All generalization criteria met.")
        print("   SMTR can generalize to unseen memories and tasks.")
    else:
        print("\n❌ Some criteria not met.")
        print("   Review the report for diagnostics and recommendations.")

    sys.exit(0 if verdict == "PASS" else 1)


if __name__ == "__main__":
    main()
