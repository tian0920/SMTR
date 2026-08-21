"""Memory quality analysis across all baseline controllers.

Computes six quality dimensions for each method:

  1. Memory retention       — number of stored memories at end of run
  2. Memory reuse           — how often stored memories are retrieved later
  3. Knowledge transfer     — fraction of helpful memories used cross-topic
  4. Harmful retention      — ratio of contaminated memories still retained
  5. Late-stage gain        — late reward minus no-memory baseline
  6. Knowledge Quality Score (MQS) = reuse × (1 + transfer) / (1 + harmful)

Output:
  results/memory_quality/memory_quality.csv
  paper/tables/table_memory_quality.tex
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

METHOD_LABELS: dict[str, str] = {
    "no_memory": "No Memory",
    "full_memory": "Full Memory",
    "retrieval": "Retrieval",
    "reflexion": "Reflexion",
    "agile": "AGILE-inspired",
    "heuristic": "Heuristic",
    "agemem": "AgeMem-inspired",
    "smtr_tci": "SMTR-TCI",
}

ROW_ORDER = [
    "full_memory", "retrieval", "reflexion", "agile",
    "heuristic", "agemem", "smtr_tci",
]


def load_data(results_dir: Path):
    perf: list[dict] = []
    with (results_dir / "performance.csv").open() as f:
        for row in csv.DictReader(f):
            perf.append(row)

    hist: list[dict] = []
    with (results_dir / "memory_history.jsonl").open() as f:
        for line in f:
            if line.strip():
                hist.append(json.loads(line))

    traj: list[dict] = []
    with (results_dir / "trajectory.jsonl").open() as f:
        for line in f:
            if line.strip():
                traj.append(json.loads(line))

    return perf, hist, traj


def _no_memory_baseline(perf: list[dict]) -> dict[int, float]:
    """Compute average reward per seed for no_memory method.

    Falls back to BASE_SUCCESS (0.40) if no_memory data is absent.
    """
    nm = [r for r in perf if r["method"] == "no_memory"]
    if not nm:
        # no_memory not in results — use theoretical baseline
        seeds = sorted(set(int(r["seed"]) for r in perf))
        return {s: 0.40 for s in seeds}
    result: dict[int, float] = {}
    for seed in sorted(set(int(r["seed"]) for r in nm)):
        rows = [r for r in nm if int(r["seed"]) == seed]
        result[seed] = float(np.mean([float(r["reward"]) for r in rows]))
    return result


def analyze_method(
    perf: list[dict], hist: list[dict], traj: list[dict], method: str,
    no_mem_baseline: dict[int, float] | None = None,
) -> dict:
    """Compute quality metrics for one method across all seeds."""
    m_hist = [r for r in hist if r["method"] == method]
    m_perf = [r for r in perf if r["method"] == method]
    m_traj = [r for r in traj if r["method"] == method]

    seeds = sorted(set(int(r["seed"]) for r in m_hist))
    per_seed: list[dict] = []

    for seed in seeds:
        s_hist = [r for r in m_hist if int(r["seed"]) == seed]
        s_perf = [r for r in m_perf if int(r["seed"]) == seed]
        s_traj = [r for r in m_traj if int(r["seed"]) == seed]

        # --- 1. Memory retention ---
        final_status: dict[str, str] = {}
        memory_topic: dict[str, int] = {}
        memory_episode: dict[str, int] = {}
        memory_contamination: dict[str, str] = {}
        for r in s_hist:
            final_status[r["memory_id"]] = r["status"]
            memory_topic[r["memory_id"]] = r["topic"]
            memory_episode[r["memory_id"]] = r["episode"]
            memory_contamination[r["memory_id"]] = r.get("contamination", "none")

        stored = {mid for mid, st in final_status.items()
                  if st in ("validated", "candidate")}
        n_stored = len(stored)

        # --- 2. Memory reuse ---
        reuse_counts: dict[str, int] = defaultdict(int)
        for t in s_traj:
            for mid in t.get("injected_ids", []):
                if mid in stored:
                    reuse_counts[mid] += 1
        avg_reuse = float(np.mean(list(reuse_counts.values()))) if reuse_counts else 0.0
        # useful_rate: fraction of stored memories used at least once
        useful_rate = len(reuse_counts) / n_stored if n_stored > 0 else 0.0

        # --- 3. Knowledge transfer (cross-topic utilization) ---
        # A helpful memory (contamination=none) that is injected on a
        # cross-topic task (affinity > 0, different topic) counts as a
        # successful knowledge transfer.
        helpful_stored = {mid for mid in stored
                          if memory_contamination.get(mid, "none") == "none"}
        same_topic_used: set[str] = set()
        cross_topic_used: set[str] = set()
        for t in s_traj:
            task_topic = t["topic"]
            for mid in t.get("injected_ids", []):
                if mid not in helpful_stored:
                    continue
                mem_t = memory_topic.get(mid, -1)
                if mem_t == task_topic:
                    same_topic_used.add(mid)
                elif topic_affinity(mem_t, task_topic) > 0:
                    cross_topic_used.add(mid)

        n_helpful = len(helpful_stored) if helpful_stored else 1
        same_topic_rate = len(same_topic_used) / n_helpful
        cross_topic_rate = len(cross_topic_used) / n_helpful
        transfer_gain = cross_topic_rate  # direct measure of knowledge transfer

        # --- 4. Harmful retention ---
        contaminated_stored = sum(
            1 for mid in stored
            if memory_contamination.get(mid, "none") != "none"
        )
        harmful_ratio = contaminated_stored / n_stored if n_stored > 0 else 0.0

        # --- 5. Late-stage gain ---
        # Sort by episode number to ensure we get the actual last 20 episodes
        sorted_perf = sorted(s_perf, key=lambda r: int(r["episode"]))
        late_rewards = [float(r["reward"]) for r in sorted_perf[-20:]]
        late_reward = float(np.mean(late_rewards)) if late_rewards else 0.0
        baseline_reward = (no_mem_baseline or {}).get(seed, 0.4)
        late_stage_gain = late_reward - baseline_reward

        # --- 6. Knowledge Quality Score ---
        # Quality-based: penalises hoarding (low useful_rate) and
        # rewards low harmful retention + cross-topic transfer.
        mqs = useful_rate * (1.0 + transfer_gain) / (1.0 + harmful_ratio)

        per_seed.append({
            "n_stored": n_stored,
            "avg_reuse": avg_reuse,
            "useful_rate": useful_rate,
            "same_topic_rate": same_topic_rate,
            "cross_topic_rate": cross_topic_rate,
            "transfer_gain": transfer_gain,
            "harmful_retention": harmful_ratio,
            "late_stage_gain": late_stage_gain,
            "mqs": mqs,
        })

    # Aggregate across seeds
    keys = ["n_stored", "avg_reuse", "useful_rate", "same_topic_rate",
            "cross_topic_rate", "transfer_gain", "harmful_retention",
            "late_stage_gain", "mqs"]
    agg: dict[str, dict] = {}
    for key in keys:
        vals = [r[key] for r in per_seed]
        agg[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    return agg


CSV_COLUMNS = [
    "method", "method_key", "n_stored", "avg_reuse", "useful_rate",
    "same_topic_rate", "cross_topic_rate", "transfer_gain",
    "harmful_retention", "late_stage_gain", "mqs",
]


def generate_quality_csv(all_metrics: dict[str, dict]) -> list[dict]:
    rows = []
    for method_key in ROW_ORDER:
        m = all_metrics.get(method_key, {})
        rows.append({
            "method": METHOD_LABELS.get(method_key, method_key),
            "method_key": method_key,
            "n_stored": f"{m.get('n_stored', {}).get('mean', 0):.1f}",
            "avg_reuse": f"{m.get('avg_reuse', {}).get('mean', 0):.2f}",
            "useful_rate": f"{m.get('useful_rate', {}).get('mean', 0):.3f}",
            "same_topic_rate": f"{m.get('same_topic_rate', {}).get('mean', 0):.3f}",
            "cross_topic_rate": f"{m.get('cross_topic_rate', {}).get('mean', 0):.3f}",
            "transfer_gain": f"{m.get('transfer_gain', {}).get('mean', 0):.3f}",
            "harmful_retention": f"{m.get('harmful_retention', {}).get('mean', 0):.3f}",
            "late_stage_gain": f"{m.get('late_stage_gain', {}).get('mean', 0):.3f}",
            "mqs": f"{m.get('mqs', {}).get('mean', 0):.3f}",
        })
    return rows


def generate_quality_latex(all_metrics: dict[str, dict]) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\caption{Memory quality comparison across baselines.}",
        r"\label{tab:memory_quality}",
        r"\centering",
        r"\begin{tabular}{l c c c c c c}",
        r"\toprule",
        r"Method & Stored & Useful\% & Cross-topic & Harmful Ret. & Late Gain & MQS \\",
        r"\midrule",
    ]
    for method_key in ROW_ORDER:
        label = METHOD_LABELS.get(method_key, method_key)
        m = all_metrics.get(method_key, {})
        ns = f"{m.get('n_stored', {}).get('mean', 0):.0f}$\\pm${m.get('n_stored', {}).get('std', 0):.0f}"
        ur = f"{m.get('useful_rate', {}).get('mean', 0):.3f}"
        ct = f"{m.get('cross_topic_rate', {}).get('mean', 0):.3f}"
        hr = f"{m.get('harmful_retention', {}).get('mean', 0):.3f}"
        lg = f"{m.get('late_stage_gain', {}).get('mean', 0):.3f}"
        mqs = f"{m.get('mqs', {}).get('mean', 0):.3f}"
        lines.append(f"{label} & {ns} & {ur} & {ct} & {hr} & {lg} & {mqs} \\\\")
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results/baseline_comparison/formation")
    parser.add_argument("--output-dir", default="results/memory_quality")
    args = parser.parse_args()

    results_dir = Path(args.results)
    if not (results_dir / "performance.csv").exists():
        print(f"ERROR: {results_dir}/performance.csv not found.")
        sys.exit(1)

    perf, hist, traj = load_data(results_dir)

    # Compute no-memory baseline for late-stage gain
    no_mem_baseline = _no_memory_baseline(perf)

    all_metrics: dict[str, dict] = {}
    for method_key in ROW_ORDER:
        # Check if method exists in data
        if any(r["method"] == method_key for r in hist):
            print(f"Analyzing {method_key}...")
            all_metrics[method_key] = analyze_method(
                perf, hist, traj, method_key, no_mem_baseline,
            )
            m = all_metrics[method_key]
            print(f"  stored={m['n_stored']['mean']:.0f}  "
                  f"useful={m['useful_rate']['mean']:.3f}  "
                  f"cross={m['cross_topic_rate']['mean']:.3f}  "
                  f"harmful={m['harmful_retention']['mean']:.3f}  "
                  f"late_gain={m['late_stage_gain']['mean']:.3f}  "
                  f"MQS={m['mqs']['mean']:.3f}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # CSV
    csv_rows = generate_quality_csv(all_metrics)
    csv_path = output_dir / "memory_quality.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\nWrote {csv_path}")

    # LaTeX
    tex_path = Path("paper/tables/table_memory_quality.tex")
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.write_text(generate_quality_latex(all_metrics))
    print(f"Wrote {tex_path}")


if __name__ == "__main__":
    main()
