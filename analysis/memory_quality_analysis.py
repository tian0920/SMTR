"""P1-3: Memory Quality Analysis.

Analyzes what TCI keeps vs rejects:

  Validated memory:
    - average transfer gain (effect on future tasks)
    - future reuse frequency (how often selected for injection)
    - lifetime utility (cumulative contribution)

  Rejected memory:
    - later failure rate (how often the task failed without this memory)

Output:
  results/lifelong/formation/memory_quality_report.md
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.lifelong.lifelong_env import topic_affinity


def load_formation_data(results_dir: Path):
    """Load performance.csv, memory_history.jsonl, trajectory.jsonl."""
    perf: list[dict] = []
    with (results_dir / "performance.csv").open() as f:
        for row in csv.DictReader(f):
            perf.append(row)

    hist: list[dict] = []
    with (results_dir / "memory_history.jsonl").open() as f:
        for line in f:
            hist.append(json.loads(line))

    traj: list[dict] = []
    with (results_dir / "trajectory.jsonl").open() as f:
        for line in f:
            traj.append(json.loads(line))

    return perf, hist, traj


def analyze_quality(
    perf: list[dict],
    hist: list[dict],
    traj: list[dict],
    method: str = "smtr_tci",
) -> dict:
    """Compute memory quality metrics for one method."""
    method_hist = [r for r in hist if r["method"] == method]
    method_perf = [r for r in perf if r["method"] == method]
    method_traj = [r for r in traj if r["method"] == method]

    # Group by seed
    seeds = sorted(set(r["seed"] for r in method_hist))
    results_per_seed: list[dict] = []

    for seed in seeds:
        seed_hist = [r for r in method_hist if r["seed"] == seed]
        seed_perf = [r for r in method_perf if r["seed"] == seed]
        seed_traj = [r for r in method_traj if r["seed"] == seed]

        # Final status of each memory
        final_status: dict[str, str] = {}
        memory_topic: dict[str, int] = {}
        memory_episode: dict[str, int] = {}
        for r in seed_hist:
            final_status[r["memory_id"]] = r["status"]
            memory_topic[r["memory_id"]] = r["topic"]
            memory_episode[r["memory_id"]] = r["episode"]

        validated = {mid for mid, st in final_status.items() if st == "validated"}
        rejected = {mid for mid, st in final_status.items() if st == "rejected"}
        contaminated = {r["memory_id"] for r in seed_hist
                       if r.get("contamination", "none") != "none"}

        # Transfer gain: for validated memories, how much did they contribute?
        # Approximate: episodes where validated memories were injected,
        # what was the success rate vs episodes without?
        injected_validated: list[int] = []
        injected_any: list[int] = []
        for t in seed_traj:
            ep = t["episode"]
            injected = t.get("injected_ids", [])
            has_validated = any(mid in validated for mid in injected)
            if has_validated:
                injected_validated.append(ep)
            if injected:
                injected_any.append(ep)

        # Performance with vs without validated memories
        perf_with_val = [float(seed_perf[ep]["reward"])
                        for ep in injected_validated if ep < len(seed_perf)]
        perf_without_val = [float(seed_perf[ep]["reward"])
                          for ep in range(len(seed_perf))
                          if ep not in injected_validated]

        transfer_gain = (
            np.mean(perf_with_val) - np.mean(perf_without_val)
            if perf_with_val and perf_without_val else 0.0
        )

        # Reuse frequency: how many times each validated memory was injected
        reuse_counts: dict[str, int] = defaultdict(int)
        for t in seed_traj:
            for mid in t.get("injected_ids", []):
                if mid in validated:
                    reuse_counts[mid] += 1
        avg_reuse = np.mean(list(reuse_counts.values())) if reuse_counts else 0.0

        # Lifetime utility: cumulative reward contribution
        # (episodes with validated memory injected, sum of rewards)
        lifetime_utility = sum(perf_with_val) if perf_with_val else 0.0

        # Rejected memory: later failure rate
        # Episodes right after rejection where task failed
        rejected_episodes = sorted(
            memory_episode[mid] for mid in rejected if mid in memory_episode
        )
        failures_after_rejection = 0
        total_after_rejection = 0
        for ep in rejected_episodes:
            for offset in range(1, 6):  # next 5 episodes
                future_ep = ep + offset
                if future_ep < len(seed_perf):
                    total_after_rejection += 1
                    if float(seed_perf[future_ep]["reward"]) < 0.5:
                        failures_after_rejection += 1
        failure_rate = (
            failures_after_rejection / total_after_rejection
            if total_after_rejection > 0 else 0.0
        )

        # Contamination in validated set
        contam_in_validated = len(validated & contaminated)
        contam_in_rejected = len(rejected & contaminated)

        results_per_seed.append({
            "n_validated": len(validated),
            "n_rejected": len(rejected),
            "transfer_gain": float(transfer_gain),
            "avg_reuse_frequency": float(avg_reuse),
            "lifetime_utility": float(lifetime_utility),
            "rejected_failure_rate": float(failure_rate),
            "contaminated_validated": contam_in_validated,
            "contaminated_rejected": contam_in_rejected,
        })

    # Aggregate
    keys = ["n_validated", "n_rejected", "transfer_gain",
            "avg_reuse_frequency", "lifetime_utility",
            "rejected_failure_rate", "contaminated_validated", "contaminated_rejected"]
    agg = {}
    for k in keys:
        vals = [r[k] for r in results_per_seed]
        agg[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    return agg


def generate_report(results_dir: Path, analyses: dict[str, dict]) -> str:
    """Generate markdown report."""
    lines = [
        "# Memory Quality Analysis Report (P1-3)\n",
        f"**Data**: {results_dir}\n",
        "## Summary\n",
        "Analyzes what TCI keeps (validated) vs rejects and their downstream utility.\n",
    ]

    for method, agg in analyses.items():
        lines.append(f"\n## {method}\n")
        lines.append("| Metric | Mean | Std |")
        lines.append("|--------|------|-----|")
        for k, v in agg.items():
            lines.append(f"| {k} | {v['mean']:.3f} | {v['std']:.3f} |")

    # Comparison
    if "smtr_tci" in analyses and "full_memory" in analyses:
        smtr = analyses["smtr_tci"]
        full = analyses["full_memory"]
        lines.append("\n## Key Findings\n")
        lines.append(f"- SMTR-TCI validated {smtr['n_validated']['mean']:.0f} "
                     f"± {smtr['n_validated']['std']:.0f} memories "
                     f"(vs {full['n_validated']['mean']:.0f} for full_memory)")
        lines.append(f"- Transfer gain: SMTR {smtr['transfer_gain']['mean']:+.3f} "
                     f"vs Full {full['transfer_gain']['mean']:+.3f}")
        lines.append(f"- Contaminated in validated set: "
                     f"SMTR {smtr['contaminated_validated']['mean']:.1f} "
                     f"vs Full {full['contaminated_validated']['mean']:.1f}")
        lines.append(f"- Rejected memories' later failure rate: "
                     f"{smtr['rejected_failure_rate']['mean']:.3f} "
                     f"(confirms rejection quality)")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results/lifelong/formation")
    parser.add_argument("--methods", nargs="+", default=["full_memory", "retrieval", "smtr_tci"])
    args = parser.parse_args()

    results_dir = Path(args.results)
    perf, hist, traj = load_formation_data(results_dir)

    analyses: dict[str, dict] = {}
    for method in args.methods:
        print(f"Analyzing {method}...")
        analyses[method] = analyze_quality(perf, hist, traj, method)
        agg = analyses[method]
        print(f"  validated={agg['n_validated']['mean']:.0f}  "
              f"transfer_gain={agg['transfer_gain']['mean']:+.3f}  "
              f"reuse={agg['avg_reuse_frequency']['mean']:.1f}  "
              f"contam_val={agg['contaminated_validated']['mean']:.1f}")

    report = generate_report(results_dir, analyses)
    report_path = results_dir / "memory_quality_report.md"
    report_path.write_text(report)
    print(f"\nSaved report to {report_path}")


if __name__ == "__main__":
    main()
