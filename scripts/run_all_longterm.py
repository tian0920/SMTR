"""Unified long-term memory lifecycle experiment launcher (Task 9).

Usage:
    python scripts/run_all_longterm.py --experiment lifelong
    python scripts/run_all_longterm.py --experiment contamination
    python scripts/run_all_longterm.py --experiment transfer
    python scripts/run_all_longterm.py --experiment multi_agent
    python scripts/run_all_longterm.py --experiment budget
    python scripts/run_all_longterm.py --experiment all
    python scripts/run_all_longterm.py --experiment baseline_comparison

Baseline comparison (memory_controller):
    python scripts/run_all_longterm.py --experiment lifelong --memory_controller reflexion
    python scripts/run_all_longterm.py --experiment lifelong --memory_controller agile
    python scripts/run_all_longterm.py --experiment lifelong --memory_controller heuristic
    python scripts/run_all_longterm.py --experiment lifelong --memory_controller agemem
    python scripts/run_all_longterm.py --experiment lifelong --memory_controller smtr

Full baseline benchmark (all methods at once):
    python scripts/run_all_longterm.py --experiment baseline_comparison
    python scripts/run_all_longterm.py --experiment baseline_comparison --config configs/baseline_comparison.yaml

Each run automatically saves config, seeds, results and logs under
results/<experiment>/ (plus figures/ where applicable).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

try:
    import yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))

PYTHON = sys.executable

MEMORY_CONTROLLERS: dict[str, list[str]] = {
    "full": ["full_memory"],
    "retrieval": ["retrieval"],
    "reflexion": ["no_memory", "full_memory", "retrieval", "reflexion", "smtr_tci"],
    "agile": ["no_memory", "full_memory", "retrieval", "agile", "smtr_tci"],
    "heuristic": ["no_memory", "full_memory", "retrieval", "heuristic", "smtr_tci"],
    "agemem": ["no_memory", "full_memory", "retrieval", "agemem", "smtr_tci"],
    "smtr": ["no_memory", "full_memory", "retrieval", "smtr_tci"],
    "all_baselines": [
        "no_memory", "full_memory", "retrieval",
        "reflexion", "agile", "heuristic", "agemem",
        "smtr_tci",
    ],
}

EXPERIMENTS: dict[str, list[list[str]]] = {
    "lifelong": [
        [
            "experiments/lifelong/run_lifelong.py",
            "--experiment", "formation",
            "--episodes", "100",
            "--seeds", "0", "1", "2", "3", "4",
            "--contamination-ratio", "0.2",
            "--output", "results/lifelong",
        ],
        ["experiments/lifelong/analyze_formation.py"],
    ],
    "contamination": [
        ["experiments/contamination/contamination_generator.py",
         "--output", "results/contamination"],
    ],
    "transfer": [
        ["experiments/lifelong/run_transfer.py", "--output", "results/transfer"],
    ],
    "multi_agent": [
        ["experiments/lifelong/run_multi_agent.py", "--output", "results/multi_agent"],
    ],
    "budget": [
        ["experiments/lifelong/run_budget.py", "--output", "results/budget"],
    ],
    "baseline_comparison": [
        [
            "experiments/lifelong/run_lifelong.py",
            "--experiment", "formation",
            "--episodes", "100",
            "--seeds", "0", "1", "2", "3", "4",
            "--contamination-ratio", "0.2",
            "--methods",
            "no_memory", "full_memory", "retrieval",
            "reflexion", "agile", "heuristic", "agemem",
            "smtr_tci",
            "--output", "results/baseline_comparison",
        ],
        ["scripts/generate_baseline_performance_table.py",
         "--results", "results/baseline_comparison/formation"],
        ["scripts/plot_baseline_longterm.py",
         "--results", "results/baseline_comparison/formation"],
        ["scripts/run_baseline_audit.py",
         "--results", "results/baseline_comparison/formation"],
    ],
    "baseline_contamination": [
        ["experiments/baseline_contamination/run_contamination_baselines.py",
         "--output", "results/baseline_contamination"],
    ],
}


def run_step(cmd: list[str], log_dir: Path) -> int:
    print(f"\n$ {PYTHON} {' '.join(cmd)}")
    log_path = log_dir / f"{int(time.time())}_{Path(cmd[0]).stem}.log"
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(
            [PYTHON, *cmd], cwd=_PROJECT_ROOT,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        log.write(process.stdout or "")
        sys.stdout.write(process.stdout or "")
    return process.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default="all",
                        choices=[*EXPERIMENTS, "all"])
    parser.add_argument(
        "--memory-controller", default=None,
        choices=list(MEMORY_CONTROLLERS),
        help="Select baseline memory controller.  Each choice defines "
             "a fixed set of --methods for the lifelong experiment "
             "so all baselines share the same task stream, seed and "
             "evaluation.",
    )
    parser.add_argument(
        "--config", default=None,
        help="Path to YAML config file (e.g. configs/baseline_comparison.yaml). "
             "If provided, overrides default episodes/seeds/output.",
    )
    args = parser.parse_args()

    # Load YAML config if provided
    yaml_config: dict = {}
    if args.config is not None:
        if not _HAS_YAML:
            print("ERROR: pyyaml not installed. Run: pip install pyyaml")
            sys.exit(1)
        with Path(args.config).open() as f:
            yaml_config = yaml.safe_load(f)
        # Override defaults from YAML
        yaml_episodes = yaml_config.get("episodes", 100)
        yaml_seeds = yaml_config.get("seeds", [0, 1, 2, 3, 4])
        yaml_output = yaml_config.get("output", {}).get("dir", "results/baseline_comparison")
        yaml_methods = yaml_config.get("methods", list(MEMORY_CONTROLLERS["all_baselines"]))
        yaml_contam = yaml_config.get("evaluation", {}).get("contamination_ratio", 0.2)
        # Rewrite baseline_comparison experiment
        EXPERIMENTS["baseline_comparison"] = [
            [
                "experiments/lifelong/run_lifelong.py",
                "--experiment", "formation",
                "--episodes", str(yaml_episodes),
                "--seeds", *[str(s) for s in yaml_seeds],
                "--contamination-ratio", str(yaml_contam),
                "--methods", *yaml_methods,
                "--output", yaml_output,
            ],
            ["scripts/generate_baseline_performance_table.py",
             "--results", f"{yaml_output}/formation"],
            ["scripts/plot_baseline_longterm.py",
             "--results", f"{yaml_output}/formation"],
            ["scripts/run_baseline_audit.py",
             "--results", f"{yaml_output}/formation"],
        ]

    # If --memory-controller is set, override the lifelong experiment
    # commands to use the corresponding method list.
    if args.memory_controller is not None:
        methods = MEMORY_CONTROLLERS[args.memory_controller]
        EXPERIMENTS["lifelong"] = [
            [
                "experiments/lifelong/run_lifelong.py",
                "--experiment", "formation",
                "--episodes", "100",
                "--seeds", "0", "1", "2", "3", "4",
                "--contamination-ratio", "0.2",
                "--methods", *methods,
                "--output", f"results/baselines/{args.memory_controller}",
            ],
            ["experiments/lifelong/analyze_formation.py"],
        ]

    selected = list(EXPERIMENTS) if args.experiment == "all" else [args.experiment]
    summary: list[dict] = []
    for name in selected:
        started = datetime.now(UTC).isoformat()
        log_dir = Path("results") / name / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        failures = 0
        for cmd in EXPERIMENTS[name]:
            rc = run_step(cmd, log_dir)
            if rc != 0:
                failures += 1
                print(f"!! step failed (rc={rc}): {cmd[0]}")
        summary.append({
            "experiment": name,
            "started_at": started,
            "finished_at": datetime.now(UTC).isoformat(),
            "status": "PASS" if failures == 0 else f"FAIL({failures})",
        })

    manifest_path = Path("results") / f"longterm_manifest_{int(time.time())}.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps({
        "launcher": "scripts/run_all_longterm.py",
        "requested": args.experiment,
        "runs": summary,
    }, indent=2))
    print(f"\nManifest: {manifest_path}")
    for row in summary:
        print(f"  {row['experiment']:<14} {row['status']}")
    if any(r["status"] != "PASS" for r in summary):
        sys.exit(1)


if __name__ == "__main__":
    main()
