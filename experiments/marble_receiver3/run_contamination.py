"""MARBLE Receiver=3 Contamination Experiment (Phase 7).

Tests: Does receiver-conditioned TCI prevent contamination propagation
in multi-agent memory sharing?

Contamination types:
  1. false_procedural: neutral_failure records marked as contaminated
  2. spurious_success: neutral_success records marked as contaminated
  3. outdated: random fraction of all non-positive records

For each contamination type and ratio:
  - full_memory: inject all (propagates contamination)
  - retrieval: inject top-k (partial propagation)
  - smtr_uniform: TCI aggregate (prevents positive-only)
  - smtr_receiver: TCI per-receiver (prevents + receiver-specific)

Metrics:
  - receiver contamination rate (fraction of contaminated memories injected)
  - team reward
  - harmful propagation depth (how many receivers affected)
  - recovery episodes (episodes where contaminated memories were rejected)

Output: results/marble/receiver3/contamination/
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.marble_receiver3.pilot.run_pilot import (
    RECEIVER_IDS,
    load_paired_records,
    simulate_receiver_outcome,
)


def run_contamination(
    *,
    records: list[dict],
    contamination_types: list[str],
    ratios: list[float],
    seeds: list[int],
    methods: list[str],
) -> list[dict]:
    valid = [r for r in records if r.get("valid")]

    groups: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    for r in valid:
        key = (r["task_id"], r["receiver_agent_id"], r["generation_seed"])
        groups[key].append(r)

    rows: list[dict] = []

    for contam_type in contamination_types:
        for ratio in ratios:
            for (task_id, _orig_receiver, seed), group_records in sorted(groups.items()):
                if seed not in seeds:
                    continue

                rng = np.random.RandomState(hash((task_id, seed, contam_type, ratio)) % (2**31))

                # Identify contamination targets
                if contam_type == "false_procedural":
                    contam_candidates = [
                        r for r in group_records if r.get("label") == "neutral_failure"
                    ]
                elif contam_type == "spurious_success":
                    contam_candidates = [
                        r for r in group_records if r.get("label") == "neutral_success"
                    ]
                else:  # outdated
                    contam_candidates = [
                        r for r in group_records
                        if r.get("label") in ("neutral_failure", "neutral_success", "negative_transfer")
                    ]

                n_contaminate = int(len(contam_candidates) * ratio)
                contam_indices = (
                    rng.choice(len(contam_candidates), size=n_contaminate, replace=False)
                    if n_contaminate > 0 else []
                )
                contaminated_mids = {
                    contam_candidates[i]["candidate_memory_id"]
                    for i in contam_indices
                }

                # Simulate per-receiver outcomes
                receiver_outcomes: dict[str, dict[str, tuple[float, float]]] = {}
                for rid in RECEIVER_IDS:
                    r_outcomes: dict[str, tuple[float, float]] = {}
                    for r in group_records:
                        mid = r["candidate_memory_id"]
                        exp, wh = simulate_receiver_outcome(r, rid, rng)
                        r_outcomes[mid] = (exp, wh)
                    receiver_outcomes[rid] = r_outcomes

                for method in methods:
                    # Per-receiver selection
                    per_receiver_selected: dict[str, list[str]] = {}
                    for rid in RECEIVER_IDS:
                        if method == "full_memory":
                            selected = [r["candidate_memory_id"] for r in group_records]
                        elif method == "retrieval":
                            ranked = sorted(group_records, key=lambda r: r.get("candidate_rank", 0))
                            selected = [r["candidate_memory_id"] for r in ranked[:3]]
                        elif method == "smtr_uniform":
                            # Aggregate delta > 0
                            selected = []
                            for r in group_records:
                                mid = r["candidate_memory_id"]
                                deltas = []
                                for r2_id in RECEIVER_IDS:
                                    if mid in receiver_outcomes[r2_id]:
                                        e, w = receiver_outcomes[r2_id][mid]
                                        deltas.append(e - w)
                                mean_d = float(np.mean(deltas)) if deltas else 0.0
                                if mean_d > 0:
                                    selected.append(mid)
                        elif method == "smtr_receiver":
                            # Per-receiver delta > 0
                            r_out = receiver_outcomes[rid]
                            selected = [
                                mid for mid, (e, w) in r_out.items()
                                if e - w > 0
                            ]
                        else:
                            selected = []
                        per_receiver_selected[rid] = selected

                    # Compute metrics
                    total_reward = 0.0
                    total_contam_injected = 0
                    total_injected = 0
                    receivers_contaminated = 0

                    for rid in RECEIVER_IDS:
                        selected = per_receiver_selected[rid]
                        r_out = receiver_outcomes[rid]
                        r_reward = 0.0
                        r_contam = 0
                        for mid in selected:
                            if mid in r_out:
                                e, w = r_out[mid]
                                r_reward += e - w
                            if mid in contaminated_mids:
                                r_contam += 1
                        total_reward += r_reward
                        total_contam_injected += r_contam
                        total_injected += len(selected)
                        if r_contam > 0:
                            receivers_contaminated += 1

                    avg_reward = total_reward / len(RECEIVER_IDS)
                    contam_rate = total_contam_injected / max(len(contaminated_mids) * len(RECEIVER_IDS), 1)
                    propagation_depth = receivers_contaminated / len(RECEIVER_IDS)

                    rows.append({
                        "contamination_type": contam_type,
                        "ratio": ratio,
                        "method": method,
                        "task_id": task_id,
                        "seed": seed,
                        "n_contaminated_total": len(contaminated_mids),
                        "n_contaminated_injected": total_contam_injected,
                        "contamination_rate": round(contam_rate, 4),
                        "team_reward": round(avg_reward, 4),
                        "total_injected": total_injected,
                        "propagation_depth": round(propagation_depth, 4),
                    })

    return rows


def main() -> None:
    paired_path = _PROJECT_ROOT / "artifacts" / "marble" / "paired" / "train" / "paired_records.jsonl"
    records = load_paired_records(paired_path)

    contamination_types = ["false_procedural", "spurious_success", "outdated"]
    ratios = [0.1, 0.2, 0.3]
    seeds = [0, 1, 2, 3, 4]
    methods = ["full_memory", "retrieval", "smtr_uniform", "smtr_receiver"]

    print("=== MARBLE Receiver=3 Contamination Experiment ===")
    print(f"  types: {contamination_types}")
    print(f"  ratios: {ratios}")
    print(f"  seeds: {seeds}")
    print(f"  methods: {methods}")
    print()

    rows = run_contamination(
        records=records,
        contamination_types=contamination_types,
        ratios=ratios,
        seeds=seeds,
        methods=methods,
    )

    # Write CSV
    output_dir = _PROJECT_ROOT / "results" / "marble" / "receiver3" / "contamination"
    output_dir.mkdir(parents=True, exist_ok=True)

    if rows:
        csv_path = output_dir / "contamination_results.csv"
        fieldnames = list(rows[0].keys())
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Written: {csv_path} ({len(rows)} rows)")

    # Print summary
    print()
    print("=" * 100)
    print(f"{'Type':<20} {'Ratio':>5} {'Method':<18} {'Groups':>7} {'Reward':>8} {'ContamRate':>10} {'PropDepth':>10}")
    print("-" * 100)
    for contam_type in contamination_types:
        for ratio in ratios:
            for method in methods:
                m_rows = [
                    r for r in rows
                    if r["contamination_type"] == contam_type
                    and r["ratio"] == ratio
                    and r["method"] == method
                ]
                if not m_rows:
                    continue
                mean_reward = float(np.mean([r["team_reward"] for r in m_rows]))
                mean_contam = float(np.mean([r["contamination_rate"] for r in m_rows]))
                mean_prop = float(np.mean([r["propagation_depth"] for r in m_rows]))
                print(
                    f"{contam_type:<20} {ratio:>5.1f} {method:<18} {len(m_rows):>7} "
                    f"{mean_reward:>8.4f} {mean_contam:>10.4f} {mean_prop:>10.4f}"
                )
            print()

    # Summary JSON
    summary: list[dict] = []
    for contam_type in contamination_types:
        for ratio in ratios:
            for method in methods:
                m_rows = [
                    r for r in rows
                    if r["contamination_type"] == contam_type
                    and r["ratio"] == ratio
                    and r["method"] == method
                ]
                if not m_rows:
                    continue
                summary.append({
                    "contamination_type": contam_type,
                    "ratio": ratio,
                    "method": method,
                    "n_groups": len(m_rows),
                    "mean_team_reward": float(np.mean([r["team_reward"] for r in m_rows])),
                    "mean_contamination_rate": float(np.mean([r["contamination_rate"] for r in m_rows])),
                    "mean_propagation_depth": float(np.mean([r["propagation_depth"] for r in m_rows])),
                })

    summary_path = output_dir / "contamination_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Written: {summary_path}")


if __name__ == "__main__":
    main()
