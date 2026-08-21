"""Run critic training stabilization across 3 modes x 3 splits.

For each training mode (regression, ranking, hybrid):
  1. generate_splits.py (create splits if not present)
  2. For each split: train + evaluate
  3. Select best mode by in-distribution ranking
  4. Check acceptance criteria on best mode

Acceptance criteria:
  1. In-distribution ranking >= 0.75
  2. Task split ranking >= 0.65
  3. Memory split ranking > random
  4. SMTR > outcome-only by +10%

Outputs:
  - reports/stabilization_report.json
  - reports/stabilization_report.md
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
TRAINING_MODES = ["regression", "ranking", "hybrid"]


def _load_config() -> dict:
    with open(_THIS_DIR / "config.yaml") as f:
        return yaml.safe_load(f)


def _run_script(script_name: str, extra_args: list[str] | None = None) -> bool:
    script_path = _THIS_DIR / script_name
    cmd = [sys.executable, str(script_path)]
    if extra_args:
        cmd.extend(extra_args)
    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print('='*60)
    result = subprocess.run(cmd, cwd=_THIS_DIR, capture_output=False)
    return result.returncode == 0


def _load_eval_results(split_name: str, mode: str) -> dict | None:
    path = _THIS_DIR / "splits" / split_name / f"eval_{mode}.json"
    if not path.exists():
        path = _THIS_DIR / "splits" / split_name / "evaluation_results.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def _check_acceptance(results: dict[str, dict | None]) -> dict:
    config = _load_config()
    acceptance = config["acceptance"]

    checks = {}

    # Criterion 1: In-distribution ranking >= 0.75
    id_res = results.get("in_distribution")
    if id_res:
        id_ranking = id_res.get("smtr_probe", {}).get("informative_ranking", 0.0)
        checks["in_distribution_ranking"] = {
            "description": f"In-distribution ranking >= {acceptance['in_distribution_ranking_min']}",
            "passed": id_ranking >= acceptance["in_distribution_ranking_min"],
            "value": f"{id_ranking:.4f}",
            "threshold": f">={acceptance['in_distribution_ranking_min']}",
        }
    else:
        checks["in_distribution_ranking"] = {
            "description": "In-distribution ranking",
            "passed": False, "value": "N/A", "threshold": ">=0.75",
        }

    # Criterion 2: Task split ranking >= 0.65
    task_res = results.get("task_holdout")
    if task_res:
        task_ranking = task_res.get("smtr_probe", {}).get("informative_ranking", 0.0)
        checks["task_holdout_ranking"] = {
            "description": f"Task split ranking >= {acceptance['task_holdout_ranking_min']}",
            "passed": task_ranking >= acceptance["task_holdout_ranking_min"],
            "value": f"{task_ranking:.4f}",
            "threshold": f">={acceptance['task_holdout_ranking_min']}",
        }
    else:
        checks["task_holdout_ranking"] = {
            "description": "Task split ranking",
            "passed": False, "value": "N/A", "threshold": ">=0.65",
        }

    # Criterion 3: Memory split > random
    mem_res = results.get("memory_holdout")
    if mem_res:
        mem_ranking = mem_res.get("smtr_probe", {}).get("informative_ranking", 0.0)
        mem_random = mem_res.get("random_baseline", {}).get("pairwise_ranking", 0.5)
        checks["memory_holdout_vs_random"] = {
            "description": "Memory split ranking > random",
            "passed": mem_ranking > mem_random,
            "value": f"{mem_ranking:.4f} (random={mem_random:.4f})",
            "threshold": ">random",
        }
    else:
        checks["memory_holdout_vs_random"] = {
            "description": "Memory split ranking",
            "passed": False, "value": "N/A", "threshold": ">random",
        }

    # Criterion 4: SMTR > outcome-only by +10% (full ranking)
    margin = acceptance["smtr_vs_outcome_margin"]
    if id_res:
        id_smtr = id_res.get("smtr_probe", {}).get("full_ranking", 0.0)
        id_outcome = id_res.get("outcome_only_baseline", {}).get("pairwise_ranking", 0.0)
        # Use full_ranking for SMTR and outcome-only full ranking
        # outcome-only informative ranking is trivially 1.0 (uses share label)
        id_outcome_full = id_res.get("outcome_only_baseline", {}).get("full_ranking",
                                 id_res.get("outcome_only_baseline", {}).get("pairwise_ranking", 0.0))
        diff = id_smtr - id_outcome_full
        checks["smtr_vs_outcome_only"] = {
            "description": f"SMTR > outcome-only by +{margin:.0%}",
            "passed": diff >= margin,
            "value": f"SMTR={id_smtr:.4f}, outcome_full={id_outcome_full:.4f}, diff={diff:+.4f}",
            "threshold": f">=+{margin:.2f}",
        }
    else:
        checks["smtr_vs_outcome_only"] = {
            "description": f"SMTR > outcome-only by +{margin:.0%}",
            "passed": False, "value": "N/A", "threshold": f">=+{margin:.2f}",
        }

    all_passed = all(c["passed"] for c in checks.values())
    return {"checks": checks, "all_passed": all_passed,
            "verdict": "PASS" if all_passed else "FAIL"}


def _generate_report(
    all_mode_results: dict[str, dict[str, dict | None]],
    best_mode: str,
    best_acceptance: dict,
) -> None:
    reports_dir = _THIS_DIR / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    json_report = {
        "best_mode": best_mode,
        "best_mode_results": {
            name: res for name, res in all_mode_results[best_mode].items()
            if res is not None
        },
        "all_modes_summary": {},
        "acceptance_criteria": best_acceptance,
    }

    for mode, results in all_mode_results.items():
        mode_summary = {}
        for split_name, res in results.items():
            if res:
                mode_summary[split_name] = {
                    "ranking": res.get("smtr_probe", {}).get("informative_ranking", 0.0),
                    "tau_corr": res.get("smtr_probe", {}).get("tau_correlation", 0.0),
                    "sign_acc": res.get("sign_classifier", {}).get("accuracy"),
                }
        json_report["all_modes_summary"][mode] = mode_summary

    json_path = reports_dir / "stabilization_report.json"
    with open(json_path, "w") as f:
        json.dump(json_report, f, indent=2)

    # Markdown
    md = ["# MARBLE Critic Training Stabilization Report\n"]
    md.append(f"**Best training mode: {best_mode}**\n")
    md.append(f"**Verdict: {best_acceptance['verdict']}**\n")

    md.append("## Mode Comparison\n")
    md.append("| Mode | In-dist | Task | Memory | Sign Acc |")
    md.append("|------|---------|------|--------|----------|")

    for mode in TRAINING_MODES:
        results = all_mode_results.get(mode, {})
        id_r = results.get("in_distribution", {})
        task_r = results.get("task_holdout", {})
        mem_r = results.get("memory_holdout", {})

        id_rank = (id_r.get("smtr_probe", {}).get("informative_ranking", 0.0)
                   if id_r else 0.0)
        task_rank = (task_r.get("smtr_probe", {}).get("informative_ranking", 0.0)
                     if task_r else 0.0)
        mem_rank = (mem_r.get("smtr_probe", {}).get("informative_ranking", 0.0)
                    if mem_r else 0.0)
        sign_acc = (id_r.get("sign_classifier", {}).get("accuracy", "N/A")
                    if id_r else "N/A")

        marker = " **" if mode == best_mode else ""
        marker_end = "**" if mode == best_mode else ""
        md.append(f"| {marker}{mode}{marker_end} | {id_rank:.4f} | "
                  f"{task_rank:.4f} | {mem_rank:.4f} | {sign_acc} |")

    md.append("\n## Acceptance Criteria (best mode)\n")
    for name, check in best_acceptance.get("checks", {}).items():
        icon = "✅" if check["passed"] else "❌"
        md.append(f"### {icon} {check['description']}")
        md.append(f"- Value: {check['value']}")
        md.append(f"- Threshold: {check.get('threshold', '')}")
        md.append("")

    verdict = best_acceptance["verdict"]
    md.append("---\n")
    md.append(f"## Conclusion: **{verdict}**\n")

    id_ranking = (all_mode_results[best_mode]
                  .get("in_distribution", {})
                  .get("smtr_probe", {}).get("informative_ranking", 0.0) if all_mode_results[best_mode].get("in_distribution") else 0.0)

    if verdict == "PASS":
        md.append("All criteria met. The critic achieves near-oracle ranking "
                  "with standard SMTR features. Ready for scale experiments.")
    elif id_ranking >= 0.65:
        md.append("Partial success. In-distribution ranking is reasonable "
                  "but not all criteria met. Review individual split results.")
    else:
        md.append("Critic training still unstable. The representation probe "
                  "showed ranking=0.6989 is achievable — training procedure "
                  "may need further tuning.")

    md_path = reports_dir / "stabilization_report.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md))
    print(f"\n  Saved: {json_path}")
    print(f"  Saved: {md_path}")


def main() -> None:
    print("=" * 60)
    print("MARBLE Critic Training Stabilization")
    print("=" * 60)

    config = _load_config()
    training_cfg = config.get("training", {})
    modes = training_cfg.get("modes", TRAINING_MODES)

    # Step 1: Generate splits (if not present)
    splits_dir = _THIS_DIR / "splits"
    if not (splits_dir / "in_distribution" / "train.jsonl").exists():
        print("\n  Generating splits...")
        if not _run_script("generate_splits.py"):
            print("ERROR: generate_splits.py failed")
            sys.exit(1)

    # Step 2: For each mode, train + evaluate all splits
    all_mode_results: dict[str, dict[str, dict | None]] = {}

    for mode in modes:
        print(f"\n{'='*60}")
        print(f"Training mode: {mode}")
        print('='*60)

        mode_results: dict[str, dict | None] = {}

        for split_name in SPLIT_NAMES:
            print(f"\n  --- {split_name} / {mode} ---")

            # Train
            if not _run_script("train_smtr_probe.py",
                               ["--split", split_name, "--training_mode", mode]):
                print(f"  ERROR: train failed for {split_name}/{mode}")
                mode_results[split_name] = None
                continue

            # Evaluate
            if not _run_script("evaluate_signal.py",
                               ["--split", split_name, "--mode", mode]):
                print(f"  ERROR: evaluate failed for {split_name}/{mode}")
                mode_results[split_name] = None
                continue

            # Load results
            res = _load_eval_results(split_name, mode)
            mode_results[split_name] = res

        all_mode_results[mode] = mode_results

    # Step 3: Select best mode by in-distribution ranking
    best_mode = None
    best_ranking = -1.0
    for mode, results in all_mode_results.items():
        id_res = results.get("in_distribution")
        if id_res:
            ranking = id_res.get("smtr_probe", {}).get("informative_ranking", 0.0)
            if ranking > best_ranking:
                best_ranking = ranking
                best_mode = mode

    if best_mode is None:
        best_mode = "hybrid"
    print(f"\n  Best mode: {best_mode} (in-dist ranking={best_ranking:.4f})")

    # Step 4: Check acceptance on best mode
    acceptance = _check_acceptance(all_mode_results[best_mode])

    print(f"\n{'='*60}")
    print("Acceptance Criteria Summary")
    print('='*60)
    for name, check in acceptance["checks"].items():
        status = "PASS" if check["passed"] else "FAIL"
        print(f"  [{status}] {check['description']}: {check['value']}")
    print(f"\n  Verdict: {acceptance['verdict']}")

    # Step 5: Generate report
    _generate_report(all_mode_results, best_mode, acceptance)

    print(f"\n{'='*60}")
    print(f"Stabilization Complete: {acceptance['verdict']}")
    print(f"{'='*60}")

    sys.exit(0 if acceptance["verdict"] == "PASS" else 1)


if __name__ == "__main__":
    main()
