"""Unified long-term memory lifecycle experiment launcher (Task 9).

Usage:
    python scripts/run_all_longterm.py --experiment lifelong
    python scripts/run_all_longterm.py --experiment contamination
    python scripts/run_all_longterm.py --experiment transfer
    python scripts/run_all_longterm.py --experiment multi_agent
    python scripts/run_all_longterm.py --experiment budget
    python scripts/run_all_longterm.py --experiment all

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

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))

PYTHON = sys.executable

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
    args = parser.parse_args()

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
